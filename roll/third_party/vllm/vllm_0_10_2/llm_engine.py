from typing import Dict, Optional, Type

from vllm import LLMEngine, EngineArgs, envs
from vllm.config import VllmConfig
from vllm.usage.usage_lib import UsageContext
from vllm.engine.metrics_types import StatLoggerBase

import roll.third_party.vllm.fp8 as fp8
from roll.utils.logging import get_logger

logger = get_logger()


class LLMEngine0102(LLMEngine):

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        disable_log_stats: bool = False,
    ) -> "LLMEngine":
        parallel_config = vllm_config.parallel_config

        executor_class = cls._get_executor_cls(vllm_config)
        if parallel_config.distributed_executor_backend == "ray":
            from roll.third_party.vllm.vllm_0_10_0.ray_distributed_executor import (
                CustomRayDistributedExecutor as V0CustomRayDistributedExecutor)
            executor_class = V0CustomRayDistributedExecutor

        logger.info(f"Using executor_class: {executor_class}")
        logger.info(f"Using worker cls: {parallel_config.worker_cls}")
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=(not disable_log_stats),
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        parallel_config = vllm_config.parallel_config

        resource_placement_groups = getattr(engine_args, "resource_placement_groups")
        assert len(resource_placement_groups) == parallel_config.world_size
        parallel_config.placement_group = resource_placement_groups

        # change worker cls to custom
        cls.update_worker_cls_config(vllm_config)

        fp8.update_quant_config(vllm_config)

        engine_cls = cls
        if envs.VLLM_USE_V1:
            from roll.third_party.vllm.vllm_0_10_2.v1.llm_engine import (
                LLMEngine0102 as V1LLMEngine0102)
            engine_cls = V1LLMEngine0102

        return engine_cls.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            disable_log_stats=engine_args.disable_log_stats,
        )

    @classmethod
    def update_worker_cls_config(cls, vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config

        assert parallel_config.worker_cls != "auto"
        if vllm_config.speculative_config:
            pass
        else:
            if envs.VLLM_USE_V1:
                parallel_config.worker_cls = "roll.third_party.vllm.vllm_0_10_2.v1.worker.Worker0102"
            else:
                parallel_config.worker_cls = "roll.third_party.vllm.vllm_0_10_2.worker.Worker0102"
