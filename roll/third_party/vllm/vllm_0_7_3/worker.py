import gc
import time
from collections import OrderedDict
from typing import Optional

import torch
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.worker.worker import Worker

from roll.third_party.vllm.vllm_utils import TensorLoRARequest, patch_vllm_lora_manager
from roll.third_party.vllm.worker_helper import WorkerHelper
from roll.utils.logging import get_logger


logger = get_logger()


class Worker073(WorkerHelper, Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_params = OrderedDict()
        patch_vllm_lora_manager()

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)

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
