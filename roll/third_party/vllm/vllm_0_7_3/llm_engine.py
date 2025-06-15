from typing import Dict, Optional, Type

from vllm import LLMEngine, EngineArgs, envs
from vllm.config import VllmConfig
from vllm.engine.metrics_types import StatLoggerBase
from vllm.executor.executor_base import ExecutorBase
from vllm.usage.usage_lib import UsageContext
from roll.utils.logging import get_logger

logger = get_logger()


class LLMEngine073(LLMEngine):

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_config = engine_args.create_engine_config(usage_context)

        parallel_config = engine_config.parallel_config
        resource_placement_groups = getattr(engine_args, "resource_placement_groups")
        assert len(resource_placement_groups) == parallel_config.world_size
        parallel_config.placement_group = resource_placement_groups

        # change worker cls to custom
        cls.update_worker_cls_config(engine_config)

        executor_class = cls._get_executor_cls(engine_config)

        logger.info(f"Using executor_class: {executor_class}")
        logger.info(f"Using worker cls: {parallel_config.worker_cls}")
        # Create the LLM engine.
        engine = cls(
            vllm_config=engine_config,
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

        return engine

    @classmethod
    def update_worker_cls_config(cls, vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config

        if scheduler_config.is_multi_step:
            if envs.VLLM_USE_V1:
                raise NotImplementedError(
                    "Multi-step scheduling is not supported (and not "
                    "needed) on VLLM V1. Please launch without "
                    "--num-scheduler-steps.")
            else:
                parallel_config.worker_cls = \
                    "vllm.worker.multi_step_worker.MultiStepWorker"
        elif vllm_config.speculative_config:
            # TODO: Speculative sampling
            if envs.VLLM_USE_V1:
                parallel_config.worker_cls = \
                    "vllm.v1.worker.gpu_worker.Worker"
            else:
                parallel_config.worker_cls = \
                    "vllm.spec_decode.spec_decode_worker.create_spec_worker"
                parallel_config.sd_worker_cls = \
                    "vllm.worker.worker.Worker"
        else:
            if envs.VLLM_USE_V1:
                # TODO: Implement v1
                parallel_config.worker_cls = \
                    "vllm.v1.worker.gpu_worker.Worker"
            else:
                parallel_config.worker_cls = "roll.third_party.vllm.vllm_0_7_3.worker.Worker073"

    @classmethod
    def _get_executor_cls(cls,
                          engine_config: VllmConfig) -> Type[ExecutorBase]:
        # distributed_executor_backend must be set in VllmConfig.__post_init__
        distributed_executor_backend = (
            engine_config.parallel_config.distributed_executor_backend)
        # Initialize the cluster and specify the executor class.
        if isinstance(distributed_executor_backend, type):
            if not issubclass(distributed_executor_backend, ExecutorBase):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"ExecutorBase. Got {distributed_executor_backend}.")
            executor_class = distributed_executor_backend
        elif distributed_executor_backend == "ray":
            from roll.third_party.vllm.vllm_0_7_3.ray_distributed_executor import (
                CustomRayDistributedExecutor as V0CustomRayDistributedExecutor)
            executor_class = V0CustomRayDistributedExecutor
        elif distributed_executor_backend == "mp":
            from vllm.executor.mp_distributed_executor import (
                MultiprocessingDistributedExecutor)
            assert not envs.VLLM_USE_RAY_SPMD_WORKER, (
                "multiprocessing distributed executor backend does not "
                "support VLLM_USE_RAY_SPMD_WORKER=1")
            executor_class = MultiprocessingDistributedExecutor
        elif distributed_executor_backend == "uni":
            # JAX-style, single-process, multi-device executor.
            from vllm.executor.uniproc_executor import UniProcExecutor
            executor_class = UniProcExecutor
        elif distributed_executor_backend == "external_launcher":
            # executor with external launcher
            from vllm.executor.uniproc_executor import (  # noqa
                ExecutorWithExternalLauncher)
            executor_class = ExecutorWithExternalLauncher
        else:
            raise ValueError("unrecognized distributed_executor_backend: "
                             f"{distributed_executor_backend}")
        return executor_class
