import os
from typing import List, Type, Dict, Union, Any

import ray
from ray._private.worker import RemoteFunctionNoArgs
from ray.runtime_env import RuntimeEnv
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker, RankInfo
from roll.distributed.scheduler.decorator import (
    BIND_WORKER_METHOD_FLAG,
    Dispatch,
    get_predefined_dispatch_fn,
    func_generator,
    get_predefined_execute_fn,
    collect_all_to_all,
    dispatch_one_to_all,
)
from roll.platforms import current_platform
from roll.utils.constants import RAY_NAMESPACE
from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.utils.import_utils import safe_import_class
from roll.utils.logging import get_logger

logger = get_logger()


class Cluster:

    def __init__(
        self,
        name,
        worker_cls: Union[RemoteFunctionNoArgs[Worker], Type[Worker], str],
        resource_manager: ResourceManager,
        worker_config: WorkerConfig,
    ):

        self.cluster_name = name
        if isinstance(worker_cls, str):
            worker_cls = safe_import_class(worker_cls)

        if not hasattr(worker_cls, "__ray_actor_class__"):
            logger.info(f"wrap {worker_cls.__name__} to ray.remote()")
            self.worker_cls = ray.remote(worker_cls)
        else:
            self.worker_cls = worker_cls
        self.resource_manager = resource_manager
        self.worker_config = worker_config

        self.workers: List[Any] = []

        self.master_addr = None
        self.master_port = None
        self.world_size = self.worker_config.world_size

        self._create_workers()
        self._bind_worker_method()
        self._worker_rank_info = None
        self.initialized = False

        self.rank2worker = {k: self.workers[k] for k in range(len(self.workers))}
        self.worker2rank = {self.workers[k]: k for k in range(len(self.workers))}
        self.rank2devices = dict(zip(map(lambda worker: self.worker2rank[worker], self.workers),
                                     ray.get([worker.get_devices_info.remote() for worker in self.workers])))
        self.worker2nodes = dict(zip(self.workers, ray.get([worker.get_node_ip.remote() for worker in self.workers])))
        logger.debug(f"{self.cluster_name} rank2devices {self.rank2devices}")
        # for cluster object can transfer by ray rpc.
        del self.worker_cls

    @property
    def dp_size(self):
        return self.worker_rank_info[0].dp_size

    @property
    def tp_size(self):
        return self.worker_rank_info[0].tp_size

    @property
    def pp_size(self):
        return self.worker_rank_info[0].pp_size

    @property
    def worker_rank_info(self) -> List[RankInfo]:
        if not self._worker_rank_info or not self.initialized:
            # initialize 后RankInfo不能改变了，使用缓存
            self._worker_rank_info: List[RankInfo] = self.execute_all_sync(method_name="get_rank_info")
        return self._worker_rank_info

    def get_rank_info(self, rank):
        assert 0 <= rank < self.world_size, f"rank must be from [0, world_size), Got {rank}"
        return self.worker_rank_info[rank]

    def _create_workers(self):
        placement_groups: List[List[Dict]] = self.resource_manager.allocate_placement_group(
            device_mapping=self.worker_config.device_mapping, world_size=self.worker_config.world_size
        )
        logger.debug(f"placement_groups: {placement_groups}")

        for rank, pgs in enumerate(placement_groups):
            deploy_pg = pgs[0]
            pg_zero_gpu_ranks = sorted([pg["gpu_rank"] for pg in pgs if pg["node_rank"] == deploy_pg["node_rank"]])
            worker_name = f"{self.cluster_name}-{rank}"
            env_vars = {
                "WORLD_SIZE": str(self.world_size),
                "RANK": str(rank),
                "LOCAL_RANK": str(0),
                "CLUSTER_NAME": self.cluster_name,
                "WORKER_NAME": worker_name,
            }

            if rank != 0:
                env_vars["MASTER_ADDR"] = self.master_addr
                env_vars["MASTER_PORT"] = str(self.master_port)
            if deploy_pg["gpu_rank"] is not None:
                current_platform.update_env_vars_for_visible_devices(env_vars=env_vars, gpu_ranks=pg_zero_gpu_ranks)
            if "ROLL_LOG_DIR" in os.environ:
                env_vars["ROLL_LOG_DIR"] = os.environ["ROLL_LOG_DIR"]
            env_vars.update(self.worker_config.system_envs)

            runtime_env = RuntimeEnv(env_vars=env_vars)
            self.worker_config.resource_placement_groups = pgs

            worker_options = {
                "scheduling_strategy": PlacementGroupSchedulingStrategy(placement_group=deploy_pg["placement_group"]),
                "name": worker_name,
                "namespace": RAY_NAMESPACE,
                "runtime_env": runtime_env,
                "num_cpus": 0.01,
            }

            if current_platform.ray_device_key == "GPU":
                worker_options.update({"num_gpus": 0.01 if self.worker_config.device_mapping else 0})
            elif current_platform.ray_device_key == "NPU":
                worker_options.update(
                    {
                        "num_gpus": 0,
                        "resources": {
                            current_platform.ray_device_key: 0.01 if self.worker_config.device_mapping else 0
                        },
                    }
                )

            worker = self.worker_cls.options(**worker_options).remote(worker_config=self.worker_config)
            self.workers.append(worker)
            if rank == 0:
                self.master_addr, self.master_port = ray.get(worker.get_master_addr_and_port.remote())

    def _bind_worker_method(self):
        """
        magic method: 用Cluster来代理向List[Worker]的请求
        ref: https://github.com/volcengine/verl/blob/27b43eba2b8905fdf18237548e596819e1831fdb/single_controller/base/worker_group.py#L136C9-L136C28
        """
        for method_name in dir(self.worker_cls):
            if method_name.startswith("_"):
                continue
            try:
                method = getattr(self.worker_cls, method_name)
                assert callable(method), f"{method_name} in {self.worker_cls} is not callable"
            except Exception as e:
                logger.debug(str(e))
                continue

            if hasattr(method, BIND_WORKER_METHOD_FLAG):

                attribute = getattr(method, BIND_WORKER_METHOD_FLAG)
                assert isinstance(attribute, Dict), f"attribute must be a dictionary. Got {type(attribute)}"
                assert "dispatch_mode" in attribute, f"attribute must contain dispatch_mode in its key"

                dispatch_mode = attribute["dispatch_mode"]
                execute_mode = attribute["execute_mode"]

                if isinstance(dispatch_mode, Dispatch):
                    fn = get_predefined_dispatch_fn(dispatch_mode=dispatch_mode)
                    dispatch_fn = fn["dispatch_fn"]
                    collect_fn = fn["collect_fn"]
                else:
                    assert isinstance(dispatch_mode, dict)
                    assert "dispatch_fn" in dispatch_mode
                    assert "collect_fn" in dispatch_mode
                    dispatch_fn = dispatch_mode["dispatch_fn"]
                    collect_fn = dispatch_mode["collect_fn"]

                execute_mode = get_predefined_execute_fn(execute_mode=execute_mode)
                execute_fn_name = execute_mode["execute_fn_name"]

                try:
                    execute_fn = getattr(self, execute_fn_name)
                    assert callable(execute_fn), "execute_fn must be callable"
                except Exception as e:
                    logger.warning(str(e))
                    raise

                func = func_generator(
                    self, method_name, dispatch_fn=dispatch_fn, collect_fn=collect_fn, execute_fn=execute_fn
                )
                try:
                    setattr(self, method_name, func)
                except Exception as e:
                    logger.warning(str(e))
                    raise ValueError(f"Fail to set method_name {method_name}")

    def execute_rank_zero_sync(self, method_name: str, *args, **kwargs):
        return ray.get(self.execute_rank_zero_async(method_name, *args, **kwargs))

    def execute_rank_zero_async(self, method_name: str, *args, **kwargs):
        remote_call = getattr(self.workers[0], method_name)
        return remote_call.remote(*args, **kwargs)

    def execute_rank_zero(self, method_name: str, *args, **kwargs):
        return self.execute_rank_zero_async(method_name, *args, **kwargs)

    def execute_all(self, method_name: str, *args, **kwargs):
        return self.execute_all_async(method_name, *args, **kwargs)

    def execute_all_sync(self, method_name: str, *args, **kwargs):
        return ray.get(self.execute_all_async(method_name, *args, **kwargs))

    def execute_all_async(self, method_name: str, *args, **kwargs):
        length = len(self.workers)
        if all(isinstance(arg, list) for arg in args) and all(isinstance(kwarg, list) for kwarg in kwargs.values()):
            if all(len(arg) == length for arg in args) and all(len(kwarg) == length for kwarg in kwargs.values()):
                result = []
                for i in range(length):
                    sliced_args = tuple(arg[i] for arg in args)
                    sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                    remote_call = getattr(self.workers[i], method_name)
                    result.append(remote_call.remote(*sliced_args, **sliced_kwargs))
                return result

        return [getattr(worker, method_name).remote(*args, **kwargs) for worker in self.workers]
