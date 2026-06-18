import asyncio
import copy
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

from transformers import PreTrainedTokenizer, ProcessorMixin

from roll.pipeline.agentic.proxy.router import create_proxy_router
from roll.utils.context_managers import local_profiler

from roll.pipeline.agentic.env_manager.base_env_manager import BaseEnvManager
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider, default_processor_provider, get_extra_data_provider
from roll.pipeline.agentic.agentic_config import EnvManagerConfig
from roll.utils.checkpoint_manager import download_model
from roll.utils.import_utils import safe_import_class
from roll.pipeline.agentic.proxy.proxy_server import ProxyServer


class EnvironmentWorker(Worker):
    """
      Within a group, all environments share identical states by using the same seed.
      To reduce the overhead of dedicating one process per environment, parallelism is redesigned as **process + threads** :
      - One `EnvironmentWorker` holds multiple `EnvStateManager`s.
      - Each `EnvStateManager` manages the rollout loop for a single environment.
      - `EnvStateManager.run_rollout_loop` runs inside dedicated threads.
        TODO: GiGPO: https://arxiv.org/abs/2505.10978
    """

    def __init__(self, worker_config: EnvManagerConfig):
        super().__init__(worker_config)
        self.worker_config: EnvManagerConfig = worker_config
        self.env_managers: Dict[int, BaseEnvManager] = {}
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.processor: Optional[ProcessorMixin] = None
        self.env_configs: Dict[int, Dict] = worker_config.env_configs[self.rank]
        self.thread_lock = threading.Lock()
        self.output_queue = None
        self.mode = "train"
        self.proxy_server: Optional[ProxyServer] = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    async def initialize(self,
                   pipeline_config,
                   generate_scheduler,
                   output_queue,
                   collator: Optional[callable] = None,
                   mode: str = "train"):
        super().initialize(pipeline_config)

        self.output_queue = output_queue
        self.mode = mode
        
        needs_proxy = any(
            "proxy_env_manager" in cfg.get('env_manager_cls', '') for cfg in self.env_configs.values()
        )
        has_rocknative = any(cfg.get('env_type') == "rocknative" for cfg in self.env_configs.values())
        # Pull-mode runners poll the sandbox directly and do not need ALB/ingress registration.
        needs_proxy_router = has_rocknative and not any(
            "PullModeRunner" in cfg.get('agent_runner_cls', '') for cfg in self.env_configs.values()
        )

        if needs_proxy:
            # Create and start proxy server for any env that uses ProxyEnvManager
            proxy_port = self.get_free_port()
            self.proxy_server = ProxyServer(host="0.0.0.0", port=proxy_port)
            self.proxy_server.start(timeout=30)
            self.logger.info(f"Proxy server started at: {self.proxy_server.url}")

            for env_id, env_config in self.env_configs.items():
                env_config['proxy_port'] = proxy_port

        if needs_proxy_router:
            # Register proxy router only for push-mode rocknative environments
            proxy_url = self.proxy_server.url if self.proxy_server else None
            if proxy_url is None:
                raise RuntimeError("rocknative env_type requires a proxy server, but no ProxyEnvManager is configured")
            backend = create_proxy_router()
            result = backend.register_servers(
                addresses=[proxy_url],
                job_id=os.environ.get('TASK_ID'),
                port=proxy_port,
            )
            self.logger.info(f'[Network Register] Auto registration result: {result}')
        model_name_or_path = download_model(self.worker_config.model_args.model_name_or_path)
        self.tokenizer = default_tokenizer_provider(self.worker_config.model_args, model_name_or_path)
        self.processor = default_processor_provider(self.worker_config.model_args, model_name_or_path)
        def create_env_manager(env_id, env_config):
            if env_id == 0:
                self.logger.info(f"use env_manager_cls: {env_config['env_manager_cls']}")
            env_manager_cls = safe_import_class(env_config["env_manager_cls"])

            assert env_manager_cls is not None
            tokenizer = copy.deepcopy(self.tokenizer)
            processor = copy.deepcopy(self.processor)
            extra_data_provider = None
            if processor is not None and isinstance(processor, ProcessorMixin):
                extra_data_provider = get_extra_data_provider(model_name_or_path, processor=processor)
            return env_id, env_manager_cls(
                worker_config=self.worker_config,
                pipeline_config=pipeline_config,
                env_config=env_config,
                tokenizer=tokenizer,  # https://github.com/huggingface/tokenizers/issues/537
                processor=processor,
                generate_scheduler=generate_scheduler,
                output_queue=output_queue,
                thread_lock=self.thread_lock,
                mode=mode,
                extra_data_provider=extra_data_provider,
            )
        with ThreadPoolExecutor(max_workers=min(len(self.env_configs), 64)) as executor:
            futures = [
                executor.submit(create_env_manager, env_id, env_config)
                for env_id, env_config in self.env_configs.items()
            ]
            for future in as_completed(futures):
                try:
                    env_id, env_manager = future.result()
                    self.env_managers[env_id] = env_manager
                except Exception as e:
                    self.logger.error(f"Failed to initialize env_manager: {e}", exc_info=True)
                    raise e

        # Register handlers for each env_manager
        if self.proxy_server:
            for env_id, env_manager in self.env_managers.items():
                if hasattr(env_manager, 'process_request'):
                    self.proxy_server.register_handler(env_id, env_manager.process_request)
                    self.logger.info(f"Registered env_id={env_id} with proxy server")
                else:
                    self.logger.warning(f"env_manager for env_id={env_id} does not have process_request method")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    async def run_rollout_loop(self, seed):
        # Set environment variables for profiler context
        os.environ["roll_EXEC_FUNC_NAME"] = "run_rollout_loop"
        os.environ["WORKER_NAME"] = f"EnvironmentWorker_{self.rank}"
        
        loop = asyncio.get_event_loop()
        pool = ThreadPoolExecutor(max_workers=len(self.env_managers))
        
        def run_with_profiler(env_manager, data_proto):
            with local_profiler():
                return env_manager.run_rollout_loop(data_proto)
        
        def run_without_profiler(env_manager, data_proto):
            return env_manager.run_rollout_loop(data_proto)
        
        tasks = []
        for env_id, env_manager in self.env_managers.items():
            # Only profile the first env_manager (env_id=0) on rank=0
            run_func = run_without_profiler
            if self.rank == 0 and env_id == 0:
                run_func = run_with_profiler
            tasks.append(loop.run_in_executor(pool, run_func, env_manager, DataProto(meta_info={"seed": seed})))
        
        await asyncio.gather(*tasks)
        pool.shutdown()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    async def update_step(self, global_step, trace_meta):
        for env_manager in self.env_managers.values():
            env_manager.update_step(global_step, trace_meta)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    async def stop(self):
        for env_manager in self.env_managers.values():
            env_manager.stop()

        # Stop proxy server
        if self.proxy_server:
            self.proxy_server.stop()
            self.logger.info("Proxy server stopped")
