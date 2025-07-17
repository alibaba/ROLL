import copy
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import ray
from transformers import PreTrainedTokenizer

from roll.agentic.rollout.env_manager import EnvManager
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider, default_processor_provider
from roll.pipeline.agentic.agentic_config import EnvManagerConfig


class EnvironmentWorker(Worker):
    """
    一个group内env状态一致，通过一致的seed来实现
    env并行方式调整成进程+线程并行：目的解决一个env占用一个进程对系统资源的开销
      - 一个EnvironmentWorker持有n个EnvStateManager
      - EnvStateManager管理一个env的rollout loop
      - EnvStateManager.run_rollout_loop,运行在n个线程里
    TODO: GiGPO: https://arxiv.org/abs/2505.10978
    """

    def __init__(self, worker_config: EnvManagerConfig):
        super().__init__(worker_config)
        self.worker_config: EnvManagerConfig = worker_config
        self.env_managers: Dict[int, EnvManager] = {}
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.env_configs: Dict[int, Dict] = worker_config.env_configs[self.rank]
        self.thread_lock = threading.Lock()
        self.output_queue = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    async def initialize(self,
                   pipeline_config,
                   generate_scheduler,
                   output_queue,
                   collator: Optional[callable] = None,
                   mode: str = "train"):
        super().initialize(pipeline_config)
        self.output_queue = output_queue
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.processor = default_processor_provider(model_args=self.worker_config.model_args)
        for env_id, env_config in self.env_configs.items():
            self.env_managers[env_id] = EnvManager(worker_config=self.worker_config,
                                                   pipeline_config=pipeline_config,
                                                   env_config=env_config,
                                                   tokenizer=copy.deepcopy(self.tokenizer), # https://github.com/huggingface/tokenizers/issues/537
                                                   generate_scheduler=generate_scheduler,
                                                   output_queue=output_queue,
                                                   thread_lock=self.thread_lock,
                                                   processor=copy.deepcopy(self.processor),
                                                   collator=collator,
                                                   mode=mode)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    async def run_rollout_loop(self, current_step, seed):
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=len(self.env_managers)) as pool:
            try:
                await asyncio.gather(
                    *[
                        loop.run_in_executor(pool, env_manager.run_rollout_loop, current_step, seed)
                        for env_manager in self.env_managers.values()
                    ]
                )
            except Exception as e:
                self.logger.error(f"EnvManager run with except: {e}", exc_info=True)
                ref = self.output_queue.put_exception.remote(e)
                await asyncio.wrap_future(ref.future())
                raise e

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    async def update_step(self, global_step):
        for env_manager in self.env_managers.values():
            env_manager.update_step(global_step)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    async def stop(self):
        for env_manager in self.env_managers.values():
            env_manager.stop()
