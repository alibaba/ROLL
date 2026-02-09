from collections import defaultdict
from concurrent import futures
from datetime import timedelta
from typing import Callable, Dict, List, Optional, Tuple

import deepspeed
import torch
import torch.distributed as dist
from accelerate import cpu_offload_with_hook
from accelerate.hooks import UserCpuOffloadHook
from torch.nn.utils.rnn import pad_sequence
from transformers import set_seed

from roll.datasets.collator import collate_fn_to_dict_list
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.strategy import InferenceStrategy
from roll.models.func_providers import log_probs_forward_step_func
from roll.models.model_providers import default_tokenizer_provider
from roll.platforms import current_platform
from roll.utils.collective import collective
from roll.utils.cuda_ipc_utils import MultiprocessingSerializer
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType, load_hf_model, offload_hf_model
from roll.utils.send_recv_utils import monkey_patch_torch_reductions, named_tensors_from_bucket

logger = get_logger()


class HfInferStrategy(InferenceStrategy):
    strategy_name = "hf_infer"

    def __init__(self, worker: "Worker"):
        super().__init__(worker)
        self.executor: futures.ThreadPoolExecutor = futures.ThreadPoolExecutor(max_workers=1)
        self.generate_config = None
        self.buffer_cache = None

    def initialize(self, model_provider):
        set_seed(seed=self.worker.pipeline_config.seed)
        dist.init_process_group(
            backend=current_platform.communication_backend,
            timeout=timedelta(minutes=self.worker_config.backend_timeout),
        )
        dist.all_reduce(torch.zeros(1).to(current_platform.device_type))

        self.worker.rank_info.dp_rank = dist.get_rank()
        self.worker.rank_info.dp_size = dist.get_world_size()

        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)

        self.model = model_provider(
            tokenizer=self.tokenizer, model_args=self.worker_config.model_args, is_trainable=False
        )
        logger.info(f"{self.model}")

    def forward_step(
        self,
        batch: DataProto,
        forward_func: Callable[[DataProto, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        batch_size = batch.batch.batch_size[0]
        micro_batch_size = batch.meta_info["micro_batch_size"]
        num_microbatches = max(batch_size // micro_batch_size, 1)
        micro_batches = batch.chunk(chunks=num_microbatches)
        losses_reduced = []
        for data in micro_batches:
            input_ids = data.batch["input_ids"]
            attention_mask = data.batch["attention_mask"]
            position_ids = data.batch["position_ids"]
            forward_args = data.meta_info.get("forward_args", {})
            if position_ids.dim() == 3:
                # qwen-vl mrope-style 3D position_ids stored in DataProto as (bsz, C, seqlen)
                # transpose to (C, bsz, seqlen) for model forward.
                position_ids = position_ids.transpose(0, 1)  # (bsz, C, seqlen) -> (C, bsz, seqlen)
            if "multi_modal_inputs" in data.non_tensor_batch:
                multi_modal_inputs = data.non_tensor_batch["multi_modal_inputs"]
                multi_modal_data = defaultdict(list)
                # mm inputs of some samples would be empty to allow text and mm
                # mixed data
                for sample_mm_inputs in multi_modal_inputs:
                    for key in sample_mm_inputs.keys():
                        multi_modal_data[key].append(sample_mm_inputs[key])
                for key in multi_modal_data.keys():
                    assert key not in forward_args
                    # DataProto.to('cuda') in upper frame not work for non_tensor_batch
                    forward_args[key] = torch.concat(multi_modal_data[key], dim=0).to(input_ids.device)
            # in Qwen2-vl/Qwen2.5-vl, use_cache=False should be set manually to
            # to avoid error in _update_causal_mask, otherwise past_key_values
            # is not None (would init as DynamicCache when use_cache) and requires
            # left-padding when using fa2
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                **forward_args,
            )
            loss, loss_reduced = forward_func(data, output.logits)
            losses_reduced.append(loss_reduced)
        results = collate_fn_to_dict_list(losses_reduced)
        return results

    def generate(self, batch: DataProto, generation_config):
        generation_config.pop("logprobs", None)
        if self.generate_config is None:
            self.generate_config = generation_config
            logger.info(f"generate_config: {self.generate_config}")

        batch_size = batch.batch.batch_size[0]
        micro_batch_size = batch.meta_info["micro_batch_size"]
        num_microbatches = max(batch_size // micro_batch_size, 1)
        micro_batches = batch.chunk(chunks=num_microbatches)

        output_list = []
        for data in micro_batches:
            input_ids = data.batch["input_ids"]  # (bs, prompt_length)
            attention_mask = data.batch["attention_mask"]  # left-padded attention_mask
            forward_args = data.meta_info.get("forward_args", {})
            if "multi_modal_inputs" in data.non_tensor_batch:
                multi_modal_inputs = data.non_tensor_batch["multi_modal_inputs"]
                multi_modal_data = defaultdict(list)
                # mm inputs of some samples would be empty to allow text and mm
                # mixed data
                for sample_mm_inputs in multi_modal_inputs:
                    for key in sample_mm_inputs.keys():
                        multi_modal_data[key].append(sample_mm_inputs[key])
                for key in multi_modal_data.keys():
                    assert key not in forward_args
                    # DataProto.to('cuda') in upper frame not work for non_tensor_batch
                    forward_args[key] = torch.concat(multi_modal_data[key], dim=0).to(input_ids.device)
            output = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=True, **forward_args, **generation_config
            )
            [output_list.append(output_tensor) for output_tensor in output]
        output = pad_sequence(output_list, batch_first=True, padding_value=generation_config["pad_token_id"])

        return output

    def unwrap_model(self):
        return self.model

    def broadcast_parameter(self, names, dtypes, shapes, group_name, is_lora=False):
        assert (
            self.worker_config.num_gpus_per_worker == 1
        ), "hf generate only support on device, please use vllm instead."
        assert not is_lora

        weights_and_handles = []
        for name, dtype, shape in zip(names, dtypes, shapes):
            target_dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
            weight = torch.empty(shape, dtype=target_dtype, device=self.device)
            handle = collective.broadcast(tensor=weight, src_rank=0, group_name=group_name, async_op=True)
            weights_and_handles.append((name, weight, handle))

        def weights_iter():
            for name, weight, handle in weights_and_handles:
                handle.wait()
                yield name, weight

        for name, weight in weights_iter():
            self.update_parameter(name, weight)

    def update_parameter(self, parameter_name, weight):
        param = self.model.get_parameter(parameter_name)
        param.data.copy_(weight)
        del weight

    def update_parameter_in_bucket(self, serialized_named_tensors, is_lora=False):
        # TODO: add lora
        assert not is_lora

        monkey_patch_torch_reductions()
        bucket_with_meta = MultiprocessingSerializer.deserialize(serialized_named_tensors[0])
        named_params = named_tensors_from_bucket(**bucket_with_meta)
        for name, weight in named_params:
            self.update_parameter(name, weight)

    # offload/load 相关接口
    def load_states(self, *args, **kwargs):
        load_hf_model(model=self.model)

    def offload_states(self, include=None, non_blocking=False):
        if include is None or OffloadStateType.model_params in include:
            offload_hf_model(model=self.model)
        current_platform.empty_cache()
