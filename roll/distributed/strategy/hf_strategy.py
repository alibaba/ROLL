from concurrent import futures
from collections import defaultdict
from datetime import timedelta
from typing import List, Optional, Callable, Dict, Tuple

import deepspeed
import torch
import torch.distributed as dist
from accelerate import cpu_offload_with_hook
from accelerate.hooks import UserCpuOffloadHook
from roll.utils.collective import collective
from torch.nn.utils.rnn import pad_sequence
from transformers import set_seed

from roll.datasets.collator import collate_fn_to_dict_list
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.strategy import InferenceStrategy
from roll.models.func_providers import log_probs_forward_step_func
from roll.models.model_providers import default_tokenizer_provider
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType, offload_hf_model, load_hf_model

logger = get_logger()


class HfInferStrategy(InferenceStrategy):
    strategy_name = "hf_infer"

    def __init__(self, worker: "Worker"):
        super().__init__(worker)
        self.executor: futures.ThreadPoolExecutor = futures.ThreadPoolExecutor(max_workers=1)
        self.generate_config = None

    def initialize(self, model_provider):
        set_seed(seed=self.worker.pipeline_config.seed)
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=self.worker_config.backend_timeout))
        dist.all_reduce(torch.zeros(1).cuda())

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
                # qwen2vl mrope, maybe use a placeholder and let model generate position_ids
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)
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

    # 参数同步相关接口
    def broadcast_bucket(self, model_update_name, src_pp_rank, meta_infos, bucket_size):
        if src_pp_rank not in self.model_update_comm_plan[model_update_name]:
            return
        comm_plan = self.model_update_comm_plan[model_update_name][src_pp_rank]
        buffer = torch.empty(bucket_size, dtype=torch.int8, device="cuda")
        collective.broadcast(tensor=buffer, src_rank=0, group_name=comm_plan["group_name"])
        self.update_parameter_in_bucket(model_update_name, meta_infos, buffer, [dist.get_rank()])

    def broadcast_parameter(self, model_update_name, src_pp_rank, dtype, shape, parameter_name, is_lora=False):
        assert (
            self.worker_config.num_gpus_per_worker == 1
        ), "hf generate only support on device, please use vllm instead."
        if src_pp_rank not in self.model_update_comm_plan[model_update_name]:
            return
        comm_plan = self.model_update_comm_plan[model_update_name][src_pp_rank]
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        collective.broadcast(tensor=weight, src_rank=0, group_name=comm_plan["group_name"])
        self.update_parameter(model_update_name, parameter_name, weight, [dist.get_rank()])

    def update_parameter(self, model_update_name, parameter_name, weight, ranks_in_worker):
        if dist.get_rank() not in ranks_in_worker:
            return
        param = self.model.get_parameter(parameter_name)
        param.data.copy_(weight)
        del weight

    def update_parameter_in_bucket(self, model_update_name, meta_infos, buffer, ranks_in_worker):
        if dist.get_rank() not in ranks_in_worker:
            return
        from mcore_adapter.models.converter.convert_utils import RecvBucketManager

        self.recv_manager = getattr(self, "recv_manager", RecvBucketManager())
        named_params = self.recv_manager.process_bucket(meta_infos, buffer)
        del buffer
        for name, weight in named_params.items():
            self.update_parameter(model_update_name, name, weight, ranks_in_worker)

    # offload/load 相关接口
    def load_states(self, *args, **kwargs):
        load_hf_model(model=self.model)

    def offload_states(self, include=None, non_blocking=False):
        if include is None or OffloadStateType.model_params in include:
            offload_hf_model(model=self.model)
        torch.cuda.empty_cache()
