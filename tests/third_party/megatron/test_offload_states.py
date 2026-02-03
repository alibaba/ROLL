import os
from typing import Optional, List

import pytest
import torch
import torch.distributed as dist

from roll.platforms import current_platform
from megatron.core import DistributedDataParallel
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import (
    OptimizerConfig,
    MegatronOptimizer,
    ChainedOptimizer,
    DistributedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from torch.optim.lr_scheduler import ChainedScheduler
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from transformers import get_scheduler, PreTrainedTokenizer

from mcore_adapter import TrainingArguments
from mcore_adapter.initialize import initialize_megatron
from roll.configs import ModelArguments, DataArguments
from roll.models.model_providers import default_tokenizer_provider, default_actor_model_provider
from roll.third_party.megatron.offload_states_patch import (
    bind_megatron_offload_states_func,
    MegatronOffloadStateType,
    offload_megatron_no_grad_module,
    reload_megatron_no_grad_module,
)
from roll.third_party.megatron.optimizer import get_megatron_optimizer


class McaModelCreator:

    def __init__(self, optimizer_type, model_name="/data/cpfs_0/common/models/Qwen2.5-0.5B-Instruct"):
        self.model_name = model_name
        if optimizer_type is None:
            self.megatron_train_args = TrainingArguments(
                output_dir="./output",
                use_distributed_optimizer=False,
                bf16=True,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                expert_model_parallel_size=1,
            )
            self.model_args = ModelArguments(model_name_or_path=self.model_name, attn_implementation="fa2", dtype="bf16")
            self.create_model = self.create_mca_infer_only
        elif optimizer_type == "dist_optimizer":
            self.megatron_train_args = TrainingArguments(
                output_dir="./output",
                use_distributed_optimizer=True,
                bf16=True,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                expert_model_parallel_size=1,
            )
            self.model_args = ModelArguments(model_name_or_path=self.model_name, attn_implementation="fa2", dtype="bf16")
            self.create_model = self.create_mca_model
        elif optimizer_type == "fp16":
            self.megatron_train_args = TrainingArguments(
                output_dir="./output",
                use_distributed_optimizer=False,
                bf16=True,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                expert_model_parallel_size=1,
            )
            self.model_args = ModelArguments(model_name_or_path=self.model_name, attn_implementation="fa2", dtype="bf16")
            self.create_model = self.create_mca_model
        elif optimizer_type == "fp32":
            self.megatron_train_args = TrainingArguments(
                output_dir="./output",
                use_distributed_optimizer=False,
                bf16=False,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                expert_model_parallel_size=1,
            )
            self.model_args = ModelArguments(model_name_or_path=self.model_name, attn_implementation="fa2", dtype="fp32")
            self.create_model = self.create_mca_model
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        self.data_args = DataArguments(template="qwen2_5")
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model = None
        self.optimizer: Optional[MegatronOptimizer] = None
        self.scheduler = None
        self.forward_backward_func = None
        self.batch_size = 4

        self.create_model()

        self.prompts = [
            "Compared with Google, Microsoft",
            "据悉，美国总统",
            "接天莲叶无穷碧，",
            "中国的首都是北京，而美国的",
            "Artificial intelligence is transforming industries such as",
            "在过去的十年中，科技的快速发展使得",
            "The Great Wall of China is a famous landmark that",
            "COVID-19 pandemic has impacted global economies, leading to",
            "Machine learning algorithms can improve efficiency in",
            "近年来，全球气候变化引发了人们对",
            "The exploration of Mars is a significant step for",
            "在文化交流中，中西方的差异让人们",
            "Sustainable energy sources are crucial for combating",
            "在文学的创作中，诗歌常常与",
            "The rise of social media has changed how we connect with",
            "科技在日常生活中扮演着越来越重要的角色，例如",
        ] * 2
        self.tokenized_prompts = self.tokenizer(self.prompts, return_tensors="pt", padding=True)
        dataset = TensorDataset(
            self.tokenized_prompts["input_ids"],
            self.tokenized_prompts["attention_mask"],
        )
        self.data_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, sampler=DistributedSampler(dataset)
        )

    def create_mca_infer_only(self):
        initialize_megatron(args=self.megatron_train_args)

        self.forward_backward_func = get_forward_backward_func()

        self.tokenizer = default_tokenizer_provider(model_args=self.model_args)
        self.model = default_actor_model_provider(
            tokenizer=self.tokenizer, training_args=self.megatron_train_args, model_args=self.model_args
        )
        for module in self.model.get_models():
            module.requires_grad_(False)

    def create_mca_model(self):

        initialize_megatron(args=self.megatron_train_args)

        self.forward_backward_func = get_forward_backward_func()

        self.tokenizer = default_tokenizer_provider(model_args=self.model_args)
        self.model = default_actor_model_provider(
            tokenizer=self.tokenizer, training_args=self.megatron_train_args, model_args=self.model_args
        )

        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=self.megatron_train_args.accumulate_allreduce_grads_in_fp32,
            overlap_grad_reduce=self.megatron_train_args.overlap_grad_reduce,
            use_distributed_optimizer=self.megatron_train_args.use_distributed_optimizer,
            check_for_nan_in_grad=self.megatron_train_args.check_for_nan_in_loss_and_grad,
            bucket_size=self.megatron_train_args.ddp_bucket_size,
        )
        models_wrapped = [
            DistributedDataParallel(
                config=m.config,
                ddp_config=ddp_config,
                module=m,
                # Turn off bucketing for model_chunk 2 onwards, since communication for these
                # model chunks is overlapped with compute anyway.
                disable_bucketing=(model_index > 0),
            )
            for model_index, m in enumerate(self.model.get_models())
        ]
        self.models_unwrapped = self.model.get_models()
        self.model.models = models_wrapped

        params_dtype = (
            torch.float16
            if self.megatron_train_args.fp16
            else torch.bfloat16 if self.megatron_train_args.bf16 else torch.float32
        )
        optimizer_config = OptimizerConfig(
            optimizer=self.megatron_train_args.optimizer,
            lr=self.megatron_train_args.learning_rate,
            weight_decay=self.megatron_train_args.weight_decay,
            adam_beta1=self.megatron_train_args.adam_beta1,
            adam_beta2=self.megatron_train_args.adam_beta2,
            adam_eps=self.megatron_train_args.adam_epsilon,
            fp16=self.megatron_train_args.fp16,
            bf16=self.megatron_train_args.bf16,
            params_dtype=params_dtype,
            use_distributed_optimizer=self.megatron_train_args.use_distributed_optimizer,
            clip_grad=self.megatron_train_args.max_grad_norm,
        )
        self.optimizer: MegatronOptimizer = get_megatron_optimizer(optimizer_config, models_wrapped)

        bind_megatron_offload_states_func(optimizer=self.optimizer)

        if not isinstance(self.optimizer, ChainedOptimizer):
            self.scheduler = get_scheduler(
                "cosine",
                optimizer=self.optimizer if self.optimizer is None else self.optimizer.optimizer,
                num_warmup_steps=self.megatron_train_args.get_warmup_steps(self.megatron_train_args.max_steps),
                num_training_steps=self.megatron_train_args.max_steps,
                scheduler_specific_kwargs=self.megatron_train_args.lr_scheduler_kwargs,
            )
        else:
            lr_schedulers = []
            for opt in self.optimizer.chained_optimizers:
                sch = get_scheduler(
                    "cosine",
                    optimizer=opt if opt is None else opt.optimizer,
                    num_warmup_steps=self.megatron_train_args.get_warmup_steps(self.megatron_train_args.max_steps),
                    num_training_steps=self.megatron_train_args.max_steps,
                    scheduler_specific_kwargs=self.megatron_train_args.lr_scheduler_kwargs,
                )
                lr_schedulers.append(sch)

            self.scheduler = ChainedScheduler(lr_schedulers)


"""
torchrun --standalone --nnodes=1 --nproc-per-node=2 -m pytest -s tests/third_party/megatron/test_offload_states.py
-s 显示stdout/err
"""


def test_megatron_init_memory():
    MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000
    torch.cuda.memory._record_memory_history(
        max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT,
    )

    mca_model = McaModelCreator(optimizer_type="dist_optimizer")

    # buffer_data = []
    # for buffer in mca_model.optimizer.buffers:
    #     buffer_data.append(buffer.param_data.data.storage().data_ptr())

    mca_model.optimizer.offload_states(include=[MegatronOffloadStateType.other_params], pin_memory=True)

    t0 = torch.randint(0, 100, (1024, 1024, 1024), device="cuda")
    del t0

    mca_model.optimizer.reload_states(include=[MegatronOffloadStateType.model_params])
    if dist.get_rank() == 0:
        t0 = torch.randint(0, 100, (1024, 1024, 1024), device="cuda")
        dump_file = f"./memory_dump/snapshot_megatron_init_offload_{os.environ['RANK']}.pickle"
        os.makedirs(os.path.dirname(dump_file), exist_ok=True)
        torch.cuda.memory._dump_snapshot(dump_file)
        torch.cuda.memory._record_memory_history(enabled=None)

    # tensors_group_by_data_ptr = defaultdict(list)
    # tensors = objgraph.by_type('Tensor')
    # print(f"len(tensor)={len(tensors)}")
    # for tensor in tensors:
    #     tensors_group_by_data_ptr[tensor.storage().data_ptr()].append(tensor)
    #
    # for buffer in buffer_data:
    #     objgraph.show_backrefs(tensors_group_by_data_ptr[buffer], max_depth=10,
    #                            extra_ignore=[id(locals())],
    #                            filename=f'/checkpoint/binary/ScaleAligner/memory_dump/buffer_data_group_tensors.param_data_{datetime.now().strftime("%Y%m%d-%H%M%S")}.png')


def test_megatron_init_ddp_memory():
    MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000
    torch.cuda.memory._record_memory_history(
        max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT,
    )

    mca_model = McaModelCreator(optimizer_type=None)

    offload_megatron_no_grad_module(model_chunks=mca_model.model.get_models())

    t0 = torch.randint(0, 100, (1024, 1024, 1024), device="cuda")
    del t0

    reload_megatron_no_grad_module(model_chunks=mca_model.model.get_models())

    if dist.get_rank() == 0:
        t0 = torch.randint(0, 100, (1024, 1024, 1024), device="cuda")
        dump_file = f"./memory_dump/snapshot_megatron_init_ddp_offload_{os.environ['RANK']}.pickle"
        os.makedirs(os.path.dirname(dump_file), exist_ok=True)
        torch.cuda.memory._dump_snapshot(dump_file)
        torch.cuda.memory._record_memory_history(enabled=None)


def check_devices(tensors: List[torch.Tensor], target_device) -> None:
    target_device = torch.device(target_device)
    for tensor in tensors:
        assert tensor.device == target_device


def check_tensors(expected_tensors: List[torch.Tensor], tensors: List[torch.Tensor]) -> None:
    for tensor_expected, tensor_restored in zip(expected_tensors, tensors):
        assert torch.equal(tensor_expected, tensor_restored)


def run_model_infer(mca_model: McaModelCreator, included_state, pin_memory, non_blocking):
    with torch.no_grad():
        for batch in mca_model.data_loader:
            input_ids, attention_mask = batch
            input_ids = input_ids.to("cuda")
            attention_mask = attention_mask.to("cuda")
            position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)

            models = mca_model.model.get_models()
            for model in models:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)

            model_params_expected = [p.clone() for model in models for p in model.parameters()]

            alloc_before_offload = current_platform.memory_allocated()
            offload_megatron_no_grad_module(model_chunks=models, pin_memory=pin_memory, non_blocking=non_blocking)
            alloc_after_offload = current_platform.memory_allocated()
            assert (
                alloc_after_offload < alloc_before_offload
            ), f"Allocated memory should decrease after offload, ({alloc_before_offload}, {alloc_after_offload})"
            check_devices(tensors=[p for model in models for p in model.parameters()], target_device="cpu")

            reload_megatron_no_grad_module(model_chunks=models, non_blocking=non_blocking)
            alloc_after_reload = current_platform.memory_allocated()
            assert (
                alloc_after_offload < alloc_after_reload
            ), f"Allocated memory should increase after offload back, ({alloc_after_offload}, {alloc_after_reload})"

            model_params_restored = [p for model in models for p in model.parameters()]
            for param_expected, param_restored in zip(model_params_expected, model_params_restored):
                assert torch.equal(param_expected, param_restored)

            check_devices(
                tensors=[p for model in models for p in model.parameters()],
                target_device=f"{current_platform.device_type}:{current_platform.current_device()}",
            )


def run_model_dist_optimizer(mca_model: McaModelCreator, included_state, pin_memory, non_blocking):
    assert isinstance(mca_model.optimizer, DistributedOptimizer)

    for batch in mca_model.data_loader:
        input_ids, attention_mask = batch
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)

        models = mca_model.model.get_models()
        for model in models:
            output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
            output.mean().backward()

        mca_model.optimizer.step()
        mca_model.optimizer.zero_grad()

        model_params_expected = [p.clone() for model in models for p in model.parameters() if p.requires_grad]
        buffer_params_expected = [buffer.param_data.clone() for buffer in mca_model.optimizer.buffers]
        bucket_params_expected = [
            bucket.param_data.clone() for buffer in mca_model.optimizer.buffers for bucket in buffer.buckets
        ]

        shard_float16_group_params_expected = [
            param.clone() for group in mca_model.optimizer.shard_float16_groups for param in group
        ]
        shard_fp32_groups_params_expected = [
            param.clone() for group in mca_model.optimizer.shard_fp32_groups for param in group
        ]

        main_grad_params_expected = [
            p.main_grad.clone() for model in models for p in model.parameters() if p.requires_grad
        ]
        buffer_grads_expected = [buffer.grad_data.clone() for buffer in mca_model.optimizer.buffers]
        bucket_grads_expected = [
            bucket.grad_data.clone() for buffer in mca_model.optimizer.buffers for bucket in buffer.buckets
        ]
        shard_fp32_from_float16_groups_params_expected = [
            param.clone() for group in mca_model.optimizer.shard_fp32_from_float16_groups for param in group
        ]
        adam_exp_avg_expected = [
            mca_model.optimizer.optimizer.state.get(param).get("exp_avg").clone()
            for group in mca_model.optimizer.param_groups
            for param in group["params"]
        ]
        adam_exp_avg_sq_expected = [
            mca_model.optimizer.optimizer.state.get(param).get("exp_avg_sq").clone()
            for group in mca_model.optimizer.param_groups
            for param in group["params"]
        ]

        alloc_before_offload = current_platform.memory_allocated()
        offload_megatron_no_grad_module(model_chunks=models, pin_memory=pin_memory, non_blocking=non_blocking)
        mca_model.optimizer.offload_states(include=included_state, pin_memory=pin_memory, non_blocking=non_blocking)

        alloc_after_offload = current_platform.memory_allocated()
        print(f"alloc_after_offload < alloc_before_offload: {alloc_after_offload}, {alloc_before_offload}")
        # assert alloc_after_offload < alloc_before_offload, f"Allocated memory should decrease after offload, ({alloc_before_offload}, {alloc_after_offload})"

        if included_state is None or MegatronOffloadStateType.model_params in included_state:
            check_devices(tensors=[p for model in models for p in model.parameters()], target_device="cpu")
            check_devices([buffer.param_data for buffer in mca_model.optimizer.buffers], "cpu")
            check_devices(
                [bucket.param_data for buffer in mca_model.optimizer.buffers for bucket in buffer.buckets], "cpu"
            )
            check_devices([param for group in mca_model.optimizer.shard_float16_groups for param in group], "cpu")
            check_devices([param for group in mca_model.optimizer.shard_fp32_groups for param in group], "cpu")

        if included_state is None or MegatronOffloadStateType.other_params in included_state:
            check_devices([p.main_grad for model in models for p in model.parameters() if p.requires_grad], "cpu")
            check_devices([buffer.grad_data for buffer in mca_model.optimizer.buffers], "cpu")
            check_devices(
                [bucket.grad_data for buffer in mca_model.optimizer.buffers for bucket in buffer.buckets], "cpu"
            )
            check_devices(
                [param for group in mca_model.optimizer.shard_fp32_from_float16_groups for param in group], "cpu"
            )
        if included_state is None or MegatronOffloadStateType.optimizer_states in included_state:
            check_devices(
                [
                    mca_model.optimizer.optimizer.state.get(param).get("exp_avg")
                    for group in mca_model.optimizer.param_groups
                    for param in group["params"]
                ],
                "cpu",
            )
            check_devices(
                [
                    mca_model.optimizer.optimizer.state.get(param).get("exp_avg_sq")
                    for group in mca_model.optimizer.param_groups
                    for param in group["params"]
                ],
                "cpu",
            )

        reload_megatron_no_grad_module(model_chunks=models, non_blocking=non_blocking)
        mca_model.optimizer.reload_states(include=included_state, non_blocking=non_blocking)
        alloc_after_reload = current_platform.memory_allocated()
        print(f"alloc_after_offload < alloc_after_reload: {alloc_after_offload}, {alloc_after_reload}")

        # assert alloc_after_offload < alloc_after_reload, f"Allocated memory should increase after offload back, ({alloc_after_offload}, {alloc_after_reload})"

        model_params_restored = [p for model in models for p in model.parameters()]
        main_grad_params_restored = [p.main_grad for model in models for p in model.parameters() if p.requires_grad]
        buffer_params_restored = [buffer.param_data for buffer in mca_model.optimizer.buffers]
        buffer_grads_restored = [buffer.grad_data for buffer in mca_model.optimizer.buffers]
        bucket_params_restored = [
            bucket.param_data for buffer in mca_model.optimizer.buffers for bucket in buffer.buckets
        ]
        bucket_grads_restored = [
            bucket.grad_data for buffer in mca_model.optimizer.buffers for bucket in buffer.buckets
        ]
        shard_float16_group_params_restored = [
            param for group in mca_model.optimizer.shard_float16_groups for param in group
        ]
        shard_fp32_groups_params_restored = [
            param for group in mca_model.optimizer.shard_fp32_groups for param in group
        ]
        shard_fp32_from_float16_groups_params_restored = [
            param for group in mca_model.optimizer.shard_fp32_from_float16_groups for param in group
        ]
        adam_exp_avg_restored = [
            mca_model.optimizer.optimizer.state.get(param).get("exp_avg")
            for group in mca_model.optimizer.param_groups
            for param in group["params"]
        ]
        adam_exp_avg_sq_restored = [
            mca_model.optimizer.optimizer.state.get(param).get("exp_avg_sq")
            for group in mca_model.optimizer.param_groups
            for param in group["params"]
        ]

        if included_state is None or MegatronOffloadStateType.model_params in included_state:
            check_tensors(model_params_expected, model_params_restored)
            check_tensors(buffer_params_expected, buffer_params_restored)
            check_tensors(bucket_params_expected, bucket_params_restored)
            check_tensors(shard_float16_group_params_expected, shard_float16_group_params_restored)
            check_tensors(shard_fp32_groups_params_expected, shard_fp32_groups_params_restored)

        if included_state is None or MegatronOffloadStateType.other_params in included_state:
            # check_tensors(main_grad_params_expected, main_grad_params_restored)
            # check_tensors(buffer_grads_expected, buffer_grads_restored)
            # check_tensors(bucket_grads_expected, bucket_grads_restored)
            check_tensors(
                shard_fp32_from_float16_groups_params_expected, shard_fp32_from_float16_groups_params_restored
            )

        if included_state is None or MegatronOffloadStateType.optimizer_states in included_state:
            check_tensors(adam_exp_avg_expected, adam_exp_avg_restored)
            check_tensors(adam_exp_avg_sq_expected, adam_exp_avg_sq_restored)

        if included_state is None or MegatronOffloadStateType.model_params in included_state:
            check_devices([p for model in models for p in model.parameters()], f"{current_platform.device_type}:{current_platform.current_device()}")
            check_devices(
                [buffer.param_data for buffer in mca_model.optimizer.buffers], f"{current_platform.device_type}:{current_platform.current_device()}"
            )
            check_devices(
                [bucket.param_data for buffer in mca_model.optimizer.buffers for bucket in buffer.buckets],
                f"{current_platform.device_type}:{current_platform.current_device()}",
            )
            check_devices(
                [param for group in mca_model.optimizer.shard_float16_groups for param in group],
                f"{current_platform.device_type}:{current_platform.current_device()}",
            )
            check_devices(
                [param for group in mca_model.optimizer.shard_fp32_groups for param in group],
                f"{current_platform.device_type}:{current_platform.current_device()}",
            )

        if included_state is None or MegatronOffloadStateType.other_params in included_state:
            check_devices(
                [p.main_grad for model in models for p in model.parameters() if p.requires_grad],
                f"{current_platform.device_type}:{current_platform.current_device()}",
            )
            check_devices(
                [buffer.grad_data for buffer in mca_model.optimizer.buffers], f"{current_platform.device_type}:{current_platform.current_device()}"
            )
            check_devices(
                [bucket.grad_data for buffer in mca_model.optimizer.buffers for bucket in buffer.buckets],
                f"{current_platform.device_type}:{current_platform.current_device()}",
            )
            check_devices(
                [param for group in mca_model.optimizer.shard_fp32_from_float16_groups for param in group],
                f"{current_platform.device_type}:{current_platform.current_device()}",
            )

        if included_state is None or MegatronOffloadStateType.optimizer_states in included_state:
            check_devices(
                [
                    mca_model.optimizer.optimizer.state.get(param).get("exp_avg")
                    for group in mca_model.optimizer.param_groups
                    for param in group["params"]
                ],
                f"{current_platform.device_type}:{current_platform.current_device()}",
            )
            check_devices(
                [
                    mca_model.optimizer.optimizer.state.get(param).get("exp_avg_sq")
                    for group in mca_model.optimizer.param_groups
                    for param in group["params"]
                ],
                f"{current_platform.device_type}:{current_platform.current_device()}",
            )


def run_model_fp16_optimizer(mca_model: McaModelCreator, included_state, pin_memory, non_blocking):
    assert isinstance(mca_model.optimizer, Float16OptimizerWithFloat16Params)

    for batch in mca_model.data_loader:
        input_ids, attention_mask = batch
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)

        models = mca_model.model.get_models()
        for model in models:
            output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
            output.mean().backward()

        mca_model.optimizer.step()
        mca_model.optimizer.zero_grad()

        model_params_expected = [p.clone() for model in models for p in model.parameters() if p.requires_grad]
        float16_groups_expected = [param.clone() for group in mca_model.optimizer.float16_groups for param in group]
        float32_groups_expected = [
            param.clone() for group in mca_model.optimizer.fp32_from_fp32_groups for param in group
        ]

        main_grad_params_expected = [
            p.main_grad.clone() for model in models for p in model.parameters() if p.requires_grad
        ]
        buffer_grads_expected = [buffer.grad_data.clone() for buffer in mca_model.optimizer.buffers]
        bucket_grads_expected = [
            bucket.grad_data.clone() for buffer in mca_model.optimizer.buffers for bucket in buffer.buckets
        ]
        fp32_from_float16_groups_expected = [
            param.clone() for group in mca_model.optimizer.fp32_from_float16_groups for param in group
        ]
        adam_exp_avg_expected = [
            mca_model.optimizer.optimizer.state.get(param).get("exp_avg").clone()
            for group in mca_model.optimizer.param_groups
            for param in group["params"]
        ]
        adam_exp_avg_sq_expected = [
            mca_model.optimizer.optimizer.state.get(param).get("exp_avg_sq").clone()
            for group in mca_model.optimizer.param_groups
            for param in group["params"]
        ]

        alloc_before_offload = current_platform.memory_allocated()
        offload_megatron_no_grad_module(model_chunks=models, pin_memory=pin_memory, non_blocking=non_blocking)
        mca_model.optimizer.offload_states(include=included_state, pin_memory=pin_memory, non_blocking=non_blocking)

        alloc_after_offload = current_platform.memory_allocated()
        print(f"alloc_after_offload < alloc_before_offload: {alloc_after_offload}, {alloc_before_offload}")
        # assert alloc_after_offload < alloc_before_offload, f"Allocated memory should decrease after offload, ({alloc_before_offload}, {alloc_after_offload})"

        if included_state is None or MegatronOffloadStateType.model_params in included_state:
            check_devices([p for model in models for p in model.parameters() if p.requires_grad], "cpu")
            check_devices([param for group in mca_model.optimizer.float16_groups for param in group], "cpu")
            check_devices([param for group in mca_model.optimizer.fp32_from_fp32_groups for param in group], "cpu")
        if included_state is None or MegatronOffloadStateType.other_params in included_state:
            check_devices([p.main_grad for model in models for p in model.parameters() if p.requires_grad], "cpu")
            check_devices([buffer.grad_data for buffer in mca_model.optimizer.buffers], "cpu")
            check_devices(
                [bucket.grad_data for buffer in mca_model.optimizer.buffers for bucket in buffer.buckets], "cpu"
            )
            check_devices(
                [param for group in mca_model.optimizer.fp32_from_float16_groups for param in group], "cpu"
            )

        if included_state is None or MegatronOffloadStateType.optimizer_states in included_state:
            check_devices(
                [
                    mca_model.optimizer.optimizer.state.get(param).get("exp_avg")
                    for group in mca_model.optimizer.param_groups
                    for param in group["params"]
                ],
                "cpu",
            )
            check_devices(
                [
                    mca_model.optimizer.optimizer.state.get(param).get("exp_avg_sq")
                    for group in mca_model.optimizer.param_groups
                    for param in group["params"]
                ],
                "cpu",
            )

        reload_megatron_no_grad_module(model_chunks=models, non_blocking=non_blocking)
        mca_model.optimizer.reload_states(include=included_state, non_blocking=non_blocking)
        alloc_after_reload = current_platform.memory_allocated()
        print(f"alloc_after_offload < alloc_after_reload: {alloc_after_offload}, {alloc_after_reload}")

        # assert alloc_after_offload < alloc_after_reload, f"Allocated memory should increase after offload back, ({alloc_after_offload}, {alloc_after_reload})"

        model_params_restored = [p for model in models for p in model.parameters() if p.requires_grad]
        float16_groups_restored = [param for group in mca_model.optimizer.float16_groups for param in group]
        float32_groups_restored = [param for group in mca_model.optimizer.fp32_from_fp32_groups for param in group]

        main_grad_params_restored = [p.main_grad for model in models for p in model.parameters() if p.requires_grad]
        buffer_grads_restored = [buffer.grad_data for buffer in mca_model.optimizer.buffers]
        bucket_grads_restored = [
            bucket.grad_data for buffer in mca_model.optimizer.buffers for bucket in buffer.buckets
        ]
        fp32_from_float16_groups_restored = [
            param for group in mca_model.optimizer.fp32_from_float16_groups for param in group
        ]
        adam_exp_avg_restored = [
            mca_model.optimizer.optimizer.state.get(param).get("exp_avg")
            for group in mca_model.optimizer.param_groups
            for param in group["params"]
        ]
        adam_exp_avg_sq_restored = [
            mca_model.optimizer.optimizer.state.get(param).get("exp_avg_sq")
            for group in mca_model.optimizer.param_groups
            for param in group["params"]
        ]

        if included_state is None or MegatronOffloadStateType.model_params in included_state:
            check_tensors(model_params_expected, model_params_restored)
            check_tensors(float16_groups_expected, float16_groups_restored)
            check_tensors(float32_groups_expected, float32_groups_restored)
        if included_state is None or MegatronOffloadStateType.other_params in included_state:
            # check_tensors(main_grad_params_expected, main_grad_params_restored)
            # check_tensors(buffer_grads_expected, buffer_grads_restored)
            # check_tensors(bucket_grads_expected, bucket_grads_restored)
            check_tensors(fp32_from_float16_groups_expected, fp32_from_float16_groups_restored)

        if included_state is None or MegatronOffloadStateType.optimizer_states in included_state:
            check_tensors(adam_exp_avg_expected, adam_exp_avg_restored)
            check_tensors(adam_exp_avg_sq_expected, adam_exp_avg_sq_restored)

        if included_state is None or MegatronOffloadStateType.model_params in included_state:
            check_devices(
                [p for model in models for p in model.parameters() if p.requires_grad],
                f"{current_platform.device_type}:{current_platform.current_device()}",
            )
            check_devices(
                [param for group in mca_model.optimizer.float16_groups for param in group],
                f"{current_platform.device_type}:{current_platform.current_device()}",
            )
            check_devices(
                [param for group in mca_model.optimizer.fp32_from_fp32_groups for param in group],
                f"{current_platform.device_type}:{current_platform.current_device()}",
            )
        if included_state is None or MegatronOffloadStateType.other_params in included_state:
            check_devices(
                [p.main_grad for model in models for p in model.parameters() if p.requires_grad],
                f"{current_platform.device_type}:{current_platform.current_device()}",
            )
            check_devices(
                [buffer.grad_data for buffer in mca_model.optimizer.buffers], f"{current_platform.device_type}:{current_platform.current_device()}"
            )
            check_devices(
                [bucket.grad_data for buffer in mca_model.optimizer.buffers for bucket in buffer.buckets],
                f"{current_platform.device_type}:{current_platform.current_device()}",
            )
            check_devices(
                [param for group in mca_model.optimizer.fp32_from_float16_groups for param in group],
                f"{current_platform.device_type}:{current_platform.current_device()}",
            )
        if included_state is None or MegatronOffloadStateType.optimizer_states in included_state:
            check_devices(
                [
                    mca_model.optimizer.optimizer.state.get(param).get("exp_avg")
                    for group in mca_model.optimizer.param_groups
                    for param in group["params"]
                ],
                f"{current_platform.device_type}:{current_platform.current_device()}",
            )
            check_devices(
                [
                    mca_model.optimizer.optimizer.state.get(param).get("exp_avg_sq")
                    for group in mca_model.optimizer.param_groups
                    for param in group["params"]
                ],
                f"{current_platform.device_type}:{current_platform.current_device()}",
            )


def run_model_fp32_optimizer(mca_model: McaModelCreator, included_state, pin_memory, non_blocking):
    assert isinstance(mca_model.optimizer, FP32Optimizer)

    for batch in mca_model.data_loader:
        input_ids, attention_mask = batch
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)

        models = mca_model.model.get_models()
        for model in models:
            output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
            output.mean().backward()

        mca_model.optimizer.step()
        mca_model.optimizer.zero_grad()

        model_params_expected = [p.clone() for model in models for p in model.parameters() if p.requires_grad]
        float32_groups_expected = [
            param.clone() for sub_group in mca_model.optimizer.param_groups for param in sub_group["params"]
        ]

        main_grad_params_expected = [
            p.main_grad.clone() for model in models for p in model.parameters() if p.requires_grad
        ]
        buffer_grads_expected = [buffer.grad_data.clone() for buffer in mca_model.optimizer.buffers]
        bucket_grads_expected = [
            bucket.grad_data.clone() for buffer in mca_model.optimizer.buffers for bucket in buffer.buckets
        ]
        adam_exp_avg_expected = [
            mca_model.optimizer.optimizer.state.get(param).get("exp_avg").clone()
            for group in mca_model.optimizer.param_groups
            for param in group["params"]
        ]
        adam_exp_avg_sq_expected = [
            mca_model.optimizer.optimizer.state.get(param).get("exp_avg_sq").clone()
            for group in mca_model.optimizer.param_groups
            for param in group["params"]
        ]

        alloc_before_offload = current_platform.memory_allocated()
        offload_megatron_no_grad_module(model_chunks=models, pin_memory=pin_memory, non_blocking=non_blocking)
        mca_model.optimizer.offload_states(include=included_state, pin_memory=pin_memory, non_blocking=non_blocking)

        alloc_after_offload = current_platform.memory_allocated()
        print(f"alloc_after_offload < alloc_before_offload: {alloc_after_offload}, {alloc_before_offload}")
        # assert alloc_after_offload < alloc_before_offload, f"Allocated memory should decrease after offload, ({alloc_before_offload}, {alloc_after_offload})"

        if included_state is None or MegatronOffloadStateType.model_params in included_state:
            check_devices([p for model in models for p in model.parameters() if p.requires_grad], "cpu")
            check_devices(
                [param.clone() for sub_group in mca_model.optimizer.param_groups for param in sub_group["params"]],
                "cpu",
            )
        if included_state is None or MegatronOffloadStateType.other_params in included_state:
            check_devices([p.main_grad for model in models for p in model.parameters() if p.requires_grad], "cpu")
            check_devices([buffer.grad_data for buffer in mca_model.optimizer.buffers], "cpu")
            check_devices(
                [bucket.grad_data for buffer in mca_model.optimizer.buffers for bucket in buffer.buckets], "cpu"
            )

        if included_state is None or MegatronOffloadStateType.optimizer_states in included_state:
            check_devices(
                [
                    mca_model.optimizer.optimizer.state.get(param).get("exp_avg")
                    for group in mca_model.optimizer.param_groups
                    for param in group["params"]
                ],
                "cpu",
            )
            check_devices(
                [
                    mca_model.optimizer.optimizer.state.get(param).get("exp_avg_sq")
                    for group in mca_model.optimizer.param_groups
                    for param in group["params"]
                ],
                "cpu",
            )

        reload_megatron_no_grad_module(model_chunks=models, non_blocking=non_blocking)
        mca_model.optimizer.reload_states(include=included_state, non_blocking=non_blocking)
        alloc_after_reload = current_platform.memory_allocated()
        print(f"alloc_after_offload < alloc_after_reload: {alloc_after_offload}, {alloc_after_reload}")

        # assert alloc_after_offload < alloc_after_reload, f"Allocated memory should increase after offload back, ({alloc_after_offload}, {alloc_after_reload})"

        model_params_restored = [p for model in models for p in model.parameters() if p.requires_grad]
        float32_groups_restored = [
            param for sub_group in mca_model.optimizer.param_groups for param in sub_group["params"]
        ]

        main_grad_params_restored = [p.main_grad for model in models for p in model.parameters() if p.requires_grad]
        buffer_grads_restored = [buffer.grad_data for buffer in mca_model.optimizer.buffers]
        bucket_grads_restored = [
            bucket.grad_data for buffer in mca_model.optimizer.buffers for bucket in buffer.buckets
        ]
        adam_exp_avg_restored = [
            mca_model.optimizer.optimizer.state.get(param).get("exp_avg")
            for group in mca_model.optimizer.param_groups
            for param in group["params"]
        ]
        adam_exp_avg_sq_restored = [
            mca_model.optimizer.optimizer.state.get(param).get("exp_avg_sq")
            for group in mca_model.optimizer.param_groups
            for param in group["params"]
        ]

        if included_state is None or MegatronOffloadStateType.model_params in included_state:
            check_tensors(model_params_expected, model_params_restored)
            check_tensors(float32_groups_expected, float32_groups_restored)
        if included_state is None or MegatronOffloadStateType.other_params in included_state:
            # check_tensors(main_grad_params_expected, main_grad_params_restored)
            # check_tensors(buffer_grads_expected, buffer_grads_restored)
            # check_tensors(bucket_grads_expected, bucket_grads_restored)
            pass
        if included_state is None or MegatronOffloadStateType.optimizer_states in included_state:
            check_tensors(adam_exp_avg_expected, adam_exp_avg_restored)
            check_tensors(adam_exp_avg_sq_expected, adam_exp_avg_sq_restored)

        if included_state is None or MegatronOffloadStateType.model_params in included_state:
            check_devices(
                [p for model in models for p in model.parameters() if p.requires_grad],
                f"{current_platform.device_type}:{current_platform.current_device()}",
            )
            check_devices(
                [param for sub_group in mca_model.optimizer.param_groups for param in sub_group["params"]],
                f"{current_platform.device_type}:{current_platform.current_device()}",
            )
        if included_state is None or MegatronOffloadStateType.other_params in included_state:
            check_devices(
                [p.main_grad for model in models for p in model.parameters() if p.requires_grad],
                f"{current_platform.device_type}:{current_platform.current_device()}",
            )
            check_devices(
                [buffer.grad_data for buffer in mca_model.optimizer.buffers], f"{current_platform.device_type}:{current_platform.current_device()}"
            )
            check_devices(
                [bucket.grad_data for buffer in mca_model.optimizer.buffers for bucket in buffer.buckets],
                f"{current_platform.device_type}:{current_platform.current_device()}",
            )
        if included_state is None or MegatronOffloadStateType.optimizer_states in included_state:
            check_devices(
                [
                    mca_model.optimizer.optimizer.state.get(param).get("exp_avg")
                    for group in mca_model.optimizer.param_groups
                    for param in group["params"]
                ],
                f"{current_platform.device_type}:{current_platform.current_device()}",
            )
            check_devices(
                [
                    mca_model.optimizer.optimizer.state.get(param).get("exp_avg_sq")
                    for group in mca_model.optimizer.param_groups
                    for param in group["params"]
                ],
                f"{current_platform.device_type}:{current_platform.current_device()}",
            )


# @pytest.mark.parametrize("included_state", [MegatronOffloadStateType.model_params])
# @pytest.mark.parametrize("pin_memory", [True])
# @pytest.mark.parametrize("non_blocking", [True])
# @pytest.mark.parametrize("optimizer_type", ['dist_optimizer'])
@pytest.mark.parametrize(
    "included_state",
    [
        MegatronOffloadStateType.model_params,
        MegatronOffloadStateType.other_params,
        MegatronOffloadStateType.optimizer_states,
        None,
    ],
)
@pytest.mark.parametrize("pin_memory", [True, False])
@pytest.mark.parametrize("non_blocking", [True, False])
@pytest.mark.parametrize("optimizer_type", [None, "dist_optimizer", "fp16", "fp32"])
def test_megatron_offload_states(included_state, pin_memory, non_blocking, optimizer_type):
    """
    有四块非optimizer的显存未释放:
    /opt/conda/envs/python3.10.13/lib/python3.10/site-packages/transformer_engine/pytorch/module/base.py:58:get_workspace
    /root/.local/lib/python3.10/site-packages/megatron/core/tensor_parallel/layers.py:413:forward
    /root/.local/lib/python3.10/site-packages/megatron/core/models/gpt/gpt_model.py:249:forward
    /root/.local/lib/python3.10/site-packages/megatron/core/tensor_parallel/layers.py:450:backward
    """
    # MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000
    # torch.cuda.memory._record_memory_history(
    #     max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT,
    #     stacks='python'
    # )

    mca_model = McaModelCreator(optimizer_type=optimizer_type)

    include = None if included_state is None else [included_state]
    if optimizer_type is None:
        run_model_infer(mca_model, include, pin_memory, non_blocking)
    elif optimizer_type == "dist_optimizer":
        run_model_dist_optimizer(mca_model, include, pin_memory, non_blocking)
    elif optimizer_type == "fp16":
        run_model_fp16_optimizer(mca_model, include, pin_memory, non_blocking)
    elif optimizer_type == "fp32":
        run_model_fp32_optimizer(mca_model, include, pin_memory, non_blocking)

    # print(f"dist.get_rank(): {dist.get_rank()}")
    # if dist.get_rank() == 0:
    #     t0 = torch.randint(0, 100, (1024, 1024, 1024), device="cuda")
    #     dump_file = f"./memory_dump/snapshot_test_megatron_offload_states_offload_{os.environ['RANK']}.pickle"
    #     os.makedirs(os.path.dirname(dump_file), exist_ok=True)
    #     torch.cuda.memory._dump_snapshot(dump_file)
    #     torch.cuda.memory._record_memory_history(enabled=None)
