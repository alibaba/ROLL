import gc
import hashlib
import json
from collections import OrderedDict
from typing import Iterable, Tuple

import torch

from roll.platforms import current_platform
from roll.third_party.vllm.vllm_utils import (
    TensorLoRARequest,
    patch_vllm_lora_manager,
    patch_vllm_moe_model_weight_loader,
)
from roll.utils.collective import collective
from roll.utils.cuda_ipc_utils import MultiprocessingSerializer
from roll.utils.fp8 import is_mxfp8_ascend
from roll.utils.logging import get_logger
from roll.utils.send_recv_utils import monkey_patch_torch_reductions, named_tensors_from_bucket

logger = get_logger()


def _restore_mxfp8_weights(model: torch.nn.Module) -> None:
    """Restore vLLM-Ascend MXFP8 weights to checkpoint layout before refit."""
    restored = 0
    for module in model.modules():
        if not getattr(module, "_mxfp8_transformed", False):
            continue
        quant_method = getattr(module, "quant_method", None)
        quant_method = getattr(quant_method, "quant_method", quant_method)
        restore = getattr(quant_method, "restore_weights_for_rl_loading", None)
        if restore is not None:
            restore(module)
            restored += 1
    if restored:
        logger.info("MXFP8: restored %d modules before weight update", restored)


class TensorLoraManager:
    def __init__(self):
        self.lora_params = OrderedDict()
        self.add_lora_count = 0

    def add_weight(self, name: str, weight: torch.Tensor):
        self.lora_params[name] = weight

    def build_request(self, peft_config: dict) -> TensorLoRARequest:
        """Build the same LoRA ID on every TP rank from the PEFT config."""
        self.add_lora_count += 1
        peft_config["add_lora_count"] = self.add_lora_count
        peft_config_str = json.dumps(peft_config, sort_keys=True)
        lora_int_id = int(hashlib.sha256(peft_config_str.encode()).hexdigest(), 16) % 0x7FFFFFFF
        request = TensorLoRARequest(
            lora_name=str(lora_int_id),
            lora_int_id=lora_int_id,
            lora_path="dummy_lora_path",
            peft_config=peft_config,
            lora_tensors=self.lora_params,
        )
        self.lora_params = OrderedDict()
        return request


class WorkerBase:
    def custom_init_worker(self, *args, **kwargs):
        self.weight_loaded = True
        self.kv_cache_loaded = True
        self.tensor_lora_manager = TensorLoraManager()
        self._is_mxfp8_model = is_mxfp8_ascend(self.vllm_config.quant_config)
        self._model_update_in_progress = False

    def reload_model(self):
        if not self.weight_loaded:
            self.wake_up(["weights"])
            self.weight_loaded = True

    def begin_model_update(self):
        if not self._is_mxfp8_model:
            return
        if self._model_update_in_progress:
            raise RuntimeError("MXFP8 model update is already in progress")
        self.reload_model()
        _restore_mxfp8_weights(self.model_runner.model)
        self._model_update_in_progress = True

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        self.reload_model()
        auto_finalize = self._is_mxfp8_model and not self._model_update_in_progress
        if auto_finalize:
            self.begin_model_update()

        weights_list = list(weights)
        patch_vllm_moe_model_weight_loader(self.model_runner.model)
        self.model_runner.model.load_weights(weights=weights_list)

        drafter = getattr(self.model_runner, "drafter", None)
        if hasattr(drafter, "model"):
            logger.info("Updating drafter (MTP/EAGLE) model weights...")
            patch_vllm_moe_model_weight_loader(drafter.model)
            drafter.model.load_weights(weights=weights_list)

        if auto_finalize:
            self.process_weights_after_loading()

    def load_states(self):
        self.reload_model()
        if not self.kv_cache_loaded:
            self.wake_up(["kv_cache"])
            self.kv_cache_loaded = True

    def offload_states(self, level):
        assert (self.weight_loaded and self.kv_cache_loaded) or (
            not self.weight_loaded and not self.kv_cache_loaded
        )
        if not self.weight_loaded:
            return
        self.sleep(level)
        self.weight_loaded = False
        self.kv_cache_loaded = False
        if hasattr(self, "recv_manager"):
            self.recv_manager.clear()
        gc.collect()
        current_platform.empty_cache()

    def setup_collective_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        group_rank = self.rank + rank_offset
        collective.init_collective_group(
            world_size,
            rank=group_rank,
            backend=backend,
            group_name=group_name,
            master_addr=master_address,
            master_port=master_port,
        )
        logger.info(f"setup_collective_group: {group_name} rank: {group_rank} world_size: {world_size}")

    def broadcast_parameter(self, names, dtypes, shapes, group_name, is_lora=False):
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

        if is_lora:
            for name, weight in weights_iter():
                self.tensor_lora_manager.add_weight(name, weight)
            return
        self.load_weights(weights=weights_iter())

    def update_parameter_in_bucket(self, serialized_named_tensors, is_lora=False):
        monkey_patch_torch_reductions()
        bucket_with_meta = MultiprocessingSerializer.deserialize(serialized_named_tensors[self.rank])
        named_params = named_tensors_from_bucket(**bucket_with_meta)
        if is_lora:
            for name, weight in named_params:
                self.tensor_lora_manager.add_weight(name, weight)
            return
        self.load_weights(named_params)

    def process_weights_after_loading(self):
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.utils import process_weights_after_loading
        from vllm.utils.torch_utils import set_default_torch_dtype

        load_device = self.vllm_config.load_config.device or self.device_config.device
        with set_default_torch_dtype(self.model_config.dtype), set_current_vllm_config(self.vllm_config):
            process_weights_after_loading(
                self.model_runner.model,
                self.model_config,
                torch.device(load_device),
            )
        self._model_update_in_progress = False


class WorkerV1(WorkerBase):
    def custom_init_worker(self, *args, **kwargs):
        super().custom_init_worker(*args, **kwargs)
        patch_vllm_lora_manager()

    # Use custom prefix because worker_extension_cls can not has
    # conflicting method name with vllm worker.
    def custom_add_lora(self, peft_config) -> bool:
        lora_request = self.tensor_lora_manager.build_request(peft_config)
        super().reload_model()
        return self.model_runner.add_lora(lora_request)
