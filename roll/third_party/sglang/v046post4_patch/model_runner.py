import logging
from dataclasses import dataclass
import torch
import torch.distributed as dist
import datetime

from roll.platforms import current_platform

from sglang.srt.model_executor.model_runner import ModelRunner, UNBALANCED_MODEL_LOADING_TIMEOUT_S
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.quantization import monkey_patch_isinstance_for_vllm_base_layer
from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state
from sglang.srt.model_loader import get_model
from sglang.srt.utils import (
    get_available_gpu_memory,
    monkey_patch_vllm_gguf_config,
    set_cuda_arch,
)

from roll.utils.collective import collective
from roll.utils.functionals import get_dist_info_from_comm_plan
from roll.platforms import current_platform

logger = logging.getLogger(__name__)


class ModelRunnerSA(ModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model(self):
        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Load weight begin. avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        # This can reduce thread conflicts and speed up weight loading.
        if self.device != "cpu":
            torch.set_num_threads(1)
        if self.device == current_platform.device_type:
            if current_platform.get_device_capability()[0] < 8:
                if self.should_log:
                    logger.info(
                        "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
                    )
                self.server_args.dtype = "float16"
                self.model_config.dtype = torch.float16
                if current_platform.get_device_capability()[1] < 5:
                    raise RuntimeError("SGLang only supports sm75 and above.")

        set_cuda_arch()

        # Prepare the model config
        self.load_config = LoadConfig(
            load_format=self.server_args.load_format,
            download_dir=self.server_args.download_dir,
        )
        if self.server_args.load_format == "gguf":
            monkey_patch_vllm_gguf_config()

        # Load the model
        # Remove monkey_patch when linear.py quant remove dependencies with vllm
        monkey_patch_vllm_parallel_state()
        monkey_patch_isinstance_for_vllm_base_layer()

        self.model = get_model(
            model_config=self.model_config,
            load_config=self.load_config,
            device_config=DeviceConfig(self.device),
        )
        monkey_patch_vllm_parallel_state(reverse=True)
        monkey_patch_isinstance_for_vllm_base_layer(reverse=True)

        if self.server_args.kv_cache_dtype == "fp8_e4m3":
            if self.server_args.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    self.model.load_kv_cache_scales(
                        self.server_args.quantization_param_path
                    )
                    if self.should_log:
                        logger.info(
                            "Loaded KV cache scaling factors from %s",
                            self.server_args.quantization_param_path,
                        )
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        "model %s does not support loading scaling factors.",
                        self.model.__class__,
                    )
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors "
                    "provided. Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!"
                )

        # Parse other args
        self.sliding_window_size = (
            self.model.get_attention_sliding_window_size()
            if hasattr(self.model, "get_attention_sliding_window_size")
            else None
        )
        self.dtype = self.model_config.dtype

        after_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Load weight end. "
            f"type={type(self.model).__name__}, "
            f"dtype={self.dtype}, "
            f"avail mem={after_avail_memory:.2f} GB, "
            f"mem usage={(before_avail_memory - after_avail_memory):.2f} GB."
        )

        # Handle the case where some ranks do not finish loading.
        try:
            dist.monitored_barrier(
                group=get_tp_group().cpu_group,
                timeout=datetime.timedelta(seconds=UNBALANCED_MODEL_LOADING_TIMEOUT_S),
                wait_all_ranks=True,
            )
        except RuntimeError:
            raise ValueError(
                f"TP rank {self.tp_rank} could finish the model loading, but there are other ranks that didn't finish loading. It is likely due to unexpected failures (e.g., OOM) or a slow node."
            ) from None
        
    def setup_collective_group(self, comm_plan, backend, rank_in_cluster):
        self.model_update_comm_plan = getattr(self, "model_update_comm_plan", {})
        rank, comm_plan_args = get_dist_info_from_comm_plan(comm_plan, rank_in_cluster=rank_in_cluster,
                                                            rank_in_worker=dist.get_rank())
        if rank is None:
            logger.info(f"no comm_plan found for rank {rank_in_cluster}/{dist.get_rank()}")
            return True, "Succeeded to setup_collective_group."
        
        group_name = comm_plan_args["group_name"]
        master_addr = comm_plan_args["master_addr"]
        master_port = comm_plan_args["master_port"]
        world_size = len(comm_plan_args["tgt_devices"]) + 1
        src_pp_rank = comm_plan_args["src_pp_rank"]
        collective.init_collective_group(world_size, rank, backend=backend, group_name=group_name,
                                         master_addr=master_addr, master_port=master_port)
        # A small all_reduce for warmup.
        collective.allreduce(torch.zeros(1).to(current_platform.device_type), group_name=group_name)
        self.model_update_comm_plan[src_pp_rank] = dict(rank=rank,
                                                        world_size=world_size,
                                                        src_pp_rank=src_pp_rank,
                                                        group_name=group_name,
                                                        comm_plan=comm_plan,
                                                        comm_plan_args=comm_plan_args)
        logger.info(f"warmup setup_collective_group: {group_name} rank: {rank} world_size: {world_size}")
        return True, "Succeeded to setup_collective_group."

    def broadcast_bucket(self, src_pp_rank, meta_infos, bucket_size):
        if src_pp_rank not in self.model_update_comm_plan:
            return True, "Succeeded to broadcast_bucket."

        comm_plan = self.model_update_comm_plan[src_pp_rank]
        buffer = torch.empty(bucket_size, dtype=torch.int8, device=current_platform.device_type)
        collective.broadcast(tensor=buffer, src_rank=0, group_name=comm_plan["group_name"])
        self.update_parameter_in_bucket(meta_infos, buffer, [dist.get_rank()])
        return True, "Succeeded to broadcast_bucket."

    def broadcast_parameter(self, src_pp_rank, dtype, shape, parameter_name):
        if src_pp_rank not in self.model_update_comm_plan:
            return True, "Succeeded to broadcast_parameter."
        comm_plan = self.model_update_comm_plan[src_pp_rank]
        weight = torch.empty(shape, dtype=dtype, device=current_platform.device_type)
        collective.broadcast(tensor=weight, src_rank=0, group_name=comm_plan["group_name"])
        self.update_parameter(parameter_name, weight, [dist.get_rank()])
        return True, "Succeeded to broadcast_parameter."


    def update_parameter(self, parameter_name, weight, ranks_in_worker):
        if dist.get_rank() not in ranks_in_worker:
            return True, "Succeeded to update_parameter."
        self.model.load_weights([(parameter_name, weight)])
        del weight
        return True, "Succeeded to update_parameter."

    def update_parameter_in_bucket(self, meta_infos, buffer, ranks_in_worker):
        if dist.get_rank() not in ranks_in_worker:
            return True, "Succeeded to update_parameter_in_bucket."
        from mcore_adapter.models.converter.convert_utils import RecvBucketManager
        self.recv_manager = getattr(self, "recv_manager", RecvBucketManager())
        named_params = self.recv_manager.process_bucket(meta_infos, buffer)
        del buffer
        self.model.load_weights([(name, weight) for name, weight in named_params.items()])
        return True, "Succeeded to update_parameter_in_bucket."