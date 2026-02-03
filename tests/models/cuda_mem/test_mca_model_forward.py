import gc
import os
import time

from megatron.core.models.common.embeddings import RotaryEmbedding
from megatron.core.parallel_state import get_global_memory_buffer
from megatron.core.pipeline_parallel import get_forward_backward_func
import transformer_engine.pytorch.module.base as te_base
import transformer_engine.pytorch.attention as te_attention
from megatron.core.transformer.moe.legacy_a2a_token_dispatcher import MoEAlltoAllSEQTokenDispatcher
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher, MoEAllGatherTokenDispatcher
from transformers import AutoTokenizer
from mcore_adapter import TrainingArguments
from mcore_adapter.initialize import initialize_megatron
from mcore_adapter.models import AutoModel
import torch
import torch.distributed as dist

from roll.configs.model_args import ModelArguments
from roll.utils.functionals import pad_to_length
from tests.models.cuda_mem.utils import log_gpu_memory_usage

path = "Qwen/Qwen2.5-7B-Instruct"
seq_len = 16384

vp = 2

megatron_train_args = TrainingArguments(
    output_dir="./output",
    use_distributed_optimizer=False,
    bf16=True,
    # moe_grouped_gemm=True,
    tensor_model_parallel_size=2,
    sequence_parallel=True,
    pipeline_model_parallel_size=2,
    context_parallel_size=2,
    virtual_pipeline_model_parallel_size=vp,
    expert_model_parallel_size=1,
)
model_args = ModelArguments(model_name_or_path=path, attn_implementation="fa2", dtype="bf16")

MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000
torch.cuda.memory._record_memory_history(max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT, stacks="python")


initialize_megatron(args=megatron_train_args)
forward_backward_func = get_forward_backward_func()

model = AutoModel.from_pretrained(model_args.model_name_or_path, megatron_train_args)
for module in model.get_models():
    module.requires_grad_(False)

tokenizer = AutoTokenizer.from_pretrained(
    path, trust_remote_code=True, use_fast=True, split_special_tokens=True, padding_side="left", padding=True
)

prompts = [
    "Compared with Google, Microsoft",
    "据悉，美国总统",
]
batch = tokenizer(prompts, return_tensors="pt", padding=True)
model = model.to("cpu")
torch.cuda.empty_cache()
log_gpu_memory_usage(head="initialize offload")

model = model.to("cuda")


def forward_func(output_tensor):
    return torch.Tensor([1]).cuda(), {}


def forward_step_func(data_iterator, module):
    input_ids = batch["input_ids"].to("cuda")
    input_ids = pad_to_length(tensor=input_ids, length=seq_len, pad_value=tokenizer.pad_token_id)
    output = module(input_ids=input_ids, attention_mask=None, position_ids=None)
    return output, forward_func


with torch.no_grad():
    forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[iter([1])] * (1 if vp is None else vp),
        model=model.get_models(),
        num_microbatches=1 if vp is None else vp,
        seq_length=seq_len,
        micro_batch_size=2,
        forward_only=True,
    )

model = model.to("cpu")
get_global_memory_buffer().buffer.clear()
for model_chunk in model.get_models():
    setattr(model_chunk.decoder, "input_tensor", None)
    for layer in model_chunk.decoder.layers:
        if isinstance(layer.mlp, MoELayer):
            if isinstance(layer.mlp.token_dispatcher, MoEAlltoAllTokenDispatcher | MoEAlltoAllSEQTokenDispatcher):
                layer.mlp.token_dispatcher.probs = None
                layer.mlp.token_dispatcher.routing_map = None
                layer.mlp.token_dispatcher.hidden_shape = None
                layer.mlp.token_dispatcher.reversed_local_input_permutation_mapping = None
                layer.mlp.token_dispatcher.input_splits = None
                layer.mlp.token_dispatcher.output_splits = None
                layer.mlp.token_dispatcher.output_splits_tp = None
                layer.mlp.token_dispatcher.num_global_tokens_per_local_expert_cpu = None
                layer.mlp.token_dispatcher.num_out_tokens = None
                layer.mlp.token_dispatcher.capacity = None
            elif isinstance(layer.mlp.token_dispatcher, MoEAllGatherTokenDispatcher):
                layer.mlp.token_dispatcher.hidden_shape = None
                layer.mlp.token_dispatcher.local_map = None
                layer.mlp.token_dispatcher.local_probs = None
                layer.mlp.token_dispatcher.reversed_local_input_permutation_mapping = None
t0 = torch.randint(0, 100, (1024, 1024, 1024), device="cuda")
del model, t0

RotaryEmbedding.forward.cache_clear()
te_base._cublas_workspace = None
te_attention._cu_seqlens_cache.clear()
gc.collect()
torch.cuda.empty_cache()
log_gpu_memory_usage(head=f"forward offload rank: {dist.get_rank()}")

print(f"dist.get_rank(): {dist.get_rank()}")

dump_file = f"./memory_dump/mem_debug_{os.environ['RANK']}.pickle"
os.makedirs(os.path.dirname(dump_file), exist_ok=True)
torch.cuda.memory._dump_snapshot(dump_file)
torch.cuda.memory._record_memory_history(enabled=None)

time.sleep(600)
"""
RANK=0 WORLD_SIZE=1 MASTER_ADDR='127.0.0.1' MASTER_PORT=54893 python tests/models/cuda_mem/test_mca_model_forward.py

torchrun --standalone --nnodes=1 --nproc-per-node=2 tests/models/cuda_mem/test_mca_model_forward.py
"""
