import os
import asyncio
from typing import (Tuple, List, Dict, Optional, Union, Any,
                    Callable, Dict, List, Optional)

from vllm import envs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.llm import _R
from vllm.usage.usage_lib import UsageContext
from vllm.v1.executor.abstract import Executor

from roll.utils.logging import get_logger
from roll.utils.send_recv_utils import SendBucketManager

logger = get_logger()

class AsyncLLM084(AsyncLLM):

    def __init__(self, resource_placement_groups, **kwargs):
        assert envs.VLLM_USE_V1

        engine_args = AsyncEngineArgs(
            **kwargs,
        )
        engine_args.enable_sleep_mode = True
        vllm_config = engine_args.create_engine_config(UsageContext.ENGINE_CONTEXT)

        parallel_config = vllm_config.parallel_config
        assert len(resource_placement_groups) == parallel_config.world_size
        parallel_config.placement_group = resource_placement_groups

        assert not vllm_config.scheduler_config.is_multi_step
        assert not vllm_config.speculative_config
        parallel_config.worker_cls = "roll.third_party.vllm.vllm_0_8_4.v1.worker.Worker084"

        executor_class = Executor.get_class(vllm_config)
        if parallel_config.distributed_executor_backend == "ray":
            from roll.third_party.vllm.vllm_0_8_4.v1.ray_distributed_executor import (
                CustomRayDistributedExecutor as V1CustomeRayDistributedExecutor)
            executor_class = V1CustomeRayDistributedExecutor

        # https://github.com/vllm-project/vllm/pull/14189/files
        # TODO do not override other options in PYTORCH_CUDA_ALLOC_CONF
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

        # Default fork method is not compatible with ScaleAligner.
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

        logger.info(f"Using AsyncLLM")
        logger.info(f"Using executor_class: {executor_class}")
        logger.info(f"Using worker cls: {parallel_config.worker_cls}")
        return super().__init__(
            vllm_config=vllm_config,
            executor_class=executor_class,
            start_engine_loop=True,
            log_requests=True,
            log_stats=True,
            usage_context=UsageContext.ENGINE_CONTEXT,
        )

    def collective_rpc(self,
                       method: Union[str, Callable[..., _R]],
                       timeout: Optional[float] = None,
                       args: Tuple = (),
                       kwargs: Optional[Dict[str, Any]] = None) -> List[_R]:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.engine_core.collective_rpc_async(method, timeout, args, kwargs))

    def load_states(self):
        self.collective_rpc(method="load_states")

    def offload_states(self, level=1):
        self.reset_prefix_cache()
        self.collective_rpc(method="offload_states", args=(level,))

    # 参数同步接口
    def setup_collective_group(self, *args, **kwargs):
        self.collective_rpc(method="setup_collective_group", args=args, kwargs=kwargs)

    def broadcast_bucket(self, src_pp_rank, meta_infos, bucket_size):
        if envs.VLLM_USE_V1:
            SendBucketManager.meta_to_dict(meta_infos)
        self.collective_rpc(method="broadcast_bucket", args=(src_pp_rank, meta_infos, bucket_size))

    def broadcast_parameter(self, *args, **kwargs):
        self.collective_rpc(method="broadcast_parameter", args=args, kwargs=kwargs)

    def update_parameter(self, *args, **kwargs):
        self.collective_rpc(method="update_parameter", args=args, kwargs=kwargs)

    def update_parameter_in_bucket(self, meta_infos, buffer, ranks_in_worker):
        if envs.VLLM_USE_V1:
            SendBucketManager.meta_to_dict(meta_infos)
        self.collective_rpc(method="update_parameter_in_bucket", args=(meta_infos, buffer, ranks_in_worker))

    def add_lora(self, *args, **kwargs):
        self.collective_rpc(method="add_lora", args=args, kwargs=kwargs)
