import gc
import time
from collections import OrderedDict
from typing import Optional

import torch
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.v1.worker.gpu_worker import Worker

from roll.third_party.vllm.vllm_utils import TensorLoRARequest, patch_vllm_lora_manager
from roll.third_party.vllm.worker_helper import WorkerHelper
from roll.utils.logging import get_logger
from roll.utils.send_recv_utils import RecvBucketManager
from roll.platforms import current_platform


logger = get_logger()


class Worker0102(WorkerHelper, Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_params = OrderedDict()
        patch_vllm_lora_manager()

    def update_parameter(self, parameter_name, weight, ranks_in_worker, is_lora):
        weight_dict = weight
        weight = torch.tensor(weight_dict["weight"], dtype=weight_dict["dtype"]).to(current_platform.device_type)
        super().update_parameter(parameter_name, weight, ranks_in_worker, is_lora)

    def broadcast_bucket(self, src_pp_rank, meta_infos, bucket_size):
        RecvBucketManager.dict_to_meta(meta_infos)
        super().broadcast_bucket(src_pp_rank, meta_infos, bucket_size)

    def update_parameter_in_bucket(self, meta_infos, buffer, ranks_in_worker):
        RecvBucketManager.dict_to_meta(meta_infos)
        buffer = torch.tensor(buffer, dtype=torch.int8, device=current_platform.device_type)
        super().update_parameter_in_bucket(meta_infos, buffer, ranks_in_worker)

    def add_lora(self, peft_config) -> bool:
        lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
        lora_request = TensorLoRARequest(
            lora_name=f"{lora_int_id}",
            lora_int_id=lora_int_id,
            lora_path="dummy_lora_path",
            peft_config=peft_config,
            lora_tensors=self.lora_params,
        )
        del self.lora_params
        self.lora_params = OrderedDict()
        super().reload_model()
        return self.model_runner.add_lora(lora_request)
