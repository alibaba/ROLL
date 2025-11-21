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
from roll.platforms import current_platform

logger = get_logger()


class MockInferStrategy(InferenceStrategy):
    strategy_name = "mock_infer"

    def __init__(self, worker: "Worker"):
        super().__init__(worker)
        self.executor: futures.ThreadPoolExecutor = futures.ThreadPoolExecutor(max_workers=1)
        self.generate_config = None

    def initialize(self, model_provider):
        set_seed(seed=self.worker.pipeline_config.seed)
        dist.init_process_group(backend=current_platform.communication_backend, timeout=timedelta(minutes=self.worker_config.backend_timeout))
        dist.all_reduce(torch.zeros(1).to(current_platform.device_type))

        self.worker.rank_info.dp_rank = dist.get_rank()
        self.worker.rank_info.dp_size = dist.get_world_size()

        # 是否最少存个tokenizer
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        # TODO：是否需要model？
        # logger.info(f"{self.model}")

    def forward_step(
        self,
        batch: DataProto,
        forward_func: Callable[[DataProto, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        # TODO 补充一下results的格式
        input_ids = batch.batch["input_ids"]
        # 创建 placeholder log_probs，形状与 input_ids 相同
        seq_len = input_ids.size(1)
        target_len = max(seq_len - 1, 0)
        log_probs = torch.zeros(
            input_ids.size(0), target_len, dtype=torch.float32, device=input_ids.device
        )
        entropy = torch.zeros(
            input_ids.size(0), target_len, dtype=torch.float32, device=input_ids.device
        )
        results = {"log_probs": log_probs, "entropy": entropy}
        return results

    def generate(self, batch: DataProto, generation_config):
        # TODO 补充一下output的格式
        input_ids = batch.batch["input_ids"]
        batch_size = input_ids.shape[0]
        input_length = input_ids.shape[1]
        # 获取生成的最大新token数，如果没有则使用默认值
        max_new_tokens = generation_config.get("max_new_tokens", generation_config.get("max_length", 50))
        # 生成的序列长度 = 输入长度 + 新生成的token数
        output_length = input_length + max_new_tokens
        # 创建 placeholder output，形状为 (batch_size, output_length)
        output = torch.zeros(batch_size, output_length, dtype=input_ids.dtype, device=input_ids.device)
        return output

    def unwrap_model(self):
        # return self.model
        raise NotImplementedError

    def update_parameter(self, model_update_name, parameter_name, weight, ranks_in_worker):
        logger.warning(f"update_parameter method is not implemented in {self.strategy_name} strategy")

    def update_parameter_in_bucket(self, model_update_name, meta_infos, buffer, ranks_in_worker):
        logger.warning(f"update_parameter_in_bucket method is not implemented in {self.strategy_name} strategy")

    # offload/load 相关接口
    def load_states(self, *args, **kwargs):
        logger.warning(f"load_states method is not implemented in {self.strategy_name} strategy")

    def offload_states(self, include=None, non_blocking=False):
        logger.warning(f"offload_states method is not implemented in {self.strategy_name} strategy")
