import asyncio
import os
from typing import Optional, List

import msgspec

import ray
from ray.runtime_env import RuntimeEnv
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import envs
from vllm.executor.msgspec_utils import encode_hook
from vllm.executor.ray_distributed_executor import RayDistributedExecutor, RayWorkerMetaData
from vllm.executor.ray_utils import RayWorkerWrapper
from vllm.v1.outputs import SamplerOutput
from vllm.platforms import current_platform
from vllm.ray.ray_env import get_env_vars_to_copy
from vllm.utils import make_async, get_ip, get_distributed_init_method, get_open_port
from roll.platforms import current_platform as roll_current_platform

from roll.utils.logging import get_logger

logger = get_logger()

def initialize_ray_cluster(ray_address: Optional[str] = None):
    if ray.is_initialized():
        return
    ray.init(address=ray_address)

class CustomRayDistributedExecutor(RayDistributedExecutor):

    def _init_executor(self) -> None:
        self.forward_dag: Optional[ray.dag.CompiledDAG] = None
        if envs.VLLM_USE_V1:
            # V1 uses SPMD worker and compiled DAG
            os.environ["VLLM_USE_RAY_SPMD_WORKER"] = "1"
            os.environ["VLLM_USE_RAY_COMPILED_DAG"] = "1"
            assert not current_platform.is_tpu()

        # If the env var is set, it uses the Ray's compiled DAG API
        # which optimizes the control plane overhead.
        # Run vLLM with VLLM_USE_RAY_COMPILED_DAG=1 to enable it.
        # Currently, this requires USE_RAY_SPMD_WORKER=True.
        self.use_ray_compiled_dag = envs.VLLM_USE_RAY_COMPILED_DAG
        # If the env var is set, then we do not distinguish between the
        # "driver worker" vs other workers. Also, the rank 0 worker will
        # be executed in a remote Ray worker. Currently this requires
        # USE_RAY_COMPILED_DAG=True.
        self.use_ray_spmd_worker = envs.VLLM_USE_RAY_SPMD_WORKER
        if self.use_ray_compiled_dag:
            assert self.use_ray_spmd_worker, (
                "VLLM_USE_RAY_COMPILED_DAG=1 requires "
                "VLLM_USE_RAY_SPMD_WORKER=1")
        if self.use_ray_spmd_worker:
            assert self.use_ray_compiled_dag, (
                "VLLM_USE_RAY_SPMD_WORKER=1 requires "
                "VLLM_USE_RAY_COMPILED_DAG=1")

        placement_group = self.parallel_config.placement_group
        assert self.uses_ray
        assert len(placement_group) > 0
        initialize_ray_cluster(placement_group[0]['ray_address'])
        assert ray.is_initialized()

        # Disable Ray usage stats collection.
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

        # Create the parallel GPU workers.
        self._init_workers_ray(placement_group)

        self.input_encoder = msgspec.msgpack.Encoder(enc_hook=encode_hook)
        self.output_decoder = msgspec.msgpack.Decoder(
            Optional[List[SamplerOutput]])
        self.use_v1 = envs.VLLM_USE_V1

        self.pp_locks: Optional[List[asyncio.Lock]] = None
        if not self.use_ray_compiled_dag:
            self.driver_exec_method = make_async(
                self.driver_worker.execute_method)

    def _init_workers_ray(self, placement_group: "PlacementGroup",
                          **ray_remote_kwargs):
        assert len(placement_group) == self.parallel_config.world_size
    
        # The driver dummy worker does not actually use any resources.
        # It holds the resource for the driver worker.
        self.driver_dummy_worker: Optional[RayWorkerWrapper] = None
        # The remaining workers are the actual ray actors.
        self.workers: List[RayWorkerWrapper] = []

        # Used in ray compiled DAG: indexed first by PP rank,
        # and then TP rank. In other words, the inner list is
        # the TP group of workers for a PP rank.
        self.pp_tp_workers: List[List[RayWorkerWrapper]] = []

        if self.parallel_config.ray_workers_use_nsight:
            ray_remote_kwargs = self._configure_ray_workers_use_nsight(
                ray_remote_kwargs)

        logger.info("use_ray_spmd_worker: %s", self.use_ray_spmd_worker)

        # Create the workers.
        worker_metadata: List[RayWorkerMetaData] = []
        driver_ip = get_ip()
        for rank in range(self.parallel_config.world_size):
            pg = placement_group[rank]['placement_group']
            gpu_rank = placement_group[rank]['gpu_rank']
            env_vars = {}
            env_vars.update(roll_current_platform.get_custom_env_vars())
            env_vars.update(roll_current_platform.get_vllm_run_time_env_vars(gpu_rank))
            runtime_env = RuntimeEnv(env_vars=env_vars)
            assert current_platform.ray_device_key == "GPU"
            # NV+AMD GPUs, and Intel XPUs
            if current_platform.ray_device_key == "GPU":
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=0.01,
                    runtime_env=runtime_env,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, ),
                    **ray_remote_kwargs,
                )(RayWorkerWrapper).remote(vllm_config=self.vllm_config,
                                            rpc_rank=rank)
            else:
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=0,
                    runtime_env=runtime_env,
                    resources={current_platform.ray_device_key: 0.01},
                    scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, ),
                    **ray_remote_kwargs,
                )(RayWorkerWrapper).remote(vllm_config=self.vllm_config,
                                           rpc_rank=rank)

            worker_metadata.append(
                RayWorkerMetaData(worker=worker, created_rank=rank))

        worker_ips = ray.get([
            each.worker.get_node_ip.remote()  # type: ignore[attr-defined]
            for each in worker_metadata
        ])

        for each, ip in zip(worker_metadata, worker_ips):
            each.ip = ip

        if not self.use_ray_spmd_worker:
            for i, each in enumerate(worker_metadata):
                # find and remove the dummy worker from the list
                worker = each.worker
                worker_ip = each.ip
                if self.driver_dummy_worker is None and worker_ip == driver_ip:
                    # If the worker is on the same node as the driver, we use it
                    # as the resource holder for the driver process.
                    self.driver_dummy_worker = worker
                    self.driver_worker = RayWorkerWrapper(
                        vllm_config=self.vllm_config, rpc_rank=0)
                    worker_metadata.pop(i)
                    break

        logger.debug("workers: %s", worker_metadata)
        logger.debug("driver_dummy_worker: %s", self.driver_dummy_worker)
        if not self.use_ray_spmd_worker and self.driver_dummy_worker is None:
            raise ValueError(
                "Ray does not allocate any GPUs on the driver node."
                f"Driver IP: {driver_ip}, worker IPs: {worker_ips}."
                "Consider adjusting the Ray placement group or running "
                "the driver on a GPU node.")

        # 不需要sorted，按placement_group给定的资源顺序即可
        start_rank = 0 if self.use_ray_spmd_worker else 1
        for i, item in enumerate(worker_metadata):
            item.adjusted_rank = i + start_rank
        self.workers = [item.worker for item in worker_metadata]
        rerank_mapping = {
            item.created_rank: item.adjusted_rank
            for item in worker_metadata
        }
        self._run_workers("adjust_rank", rerank_mapping)

        # Get the set of GPU IDs used on each node.
        worker_node_and_gpu_ids = []
        for worker in [self.driver_dummy_worker] + self.workers:
            if worker is None:
                # driver_dummy_worker can be None when using ray spmd worker.
                continue
            worker_node_and_gpu_ids.append(
                ray.get(worker.get_node_and_gpu_ids.remote()) \
            ) # type: ignore

        # Set environment variables for the driver and workers.
        # 移除了device_control_env_var(CUDA_VISIBLE_DEVICES)设置，原因是我们只为每个worker分配了一个可见gpu
        all_args_to_update_environment_variables = [{} for (node_id, _) in worker_node_and_gpu_ids]
        # Environment variables to copy from driver to workers
        env_vars_to_copy = get_env_vars_to_copy(
            exclude_vars=self.WORKER_SPECIFIC_ENV_VARS,
            additional_vars=set(current_platform.additional_env_vars).union(
                self.ADDITIONAL_ENV_VARS),
            destination="workers")

        # Copy existing env vars to each worker's args
        for args in all_args_to_update_environment_variables:
            for name in env_vars_to_copy:
                if name in os.environ:
                    args[name] = os.environ[name]

        self._env_vars_for_all_workers = (
            all_args_to_update_environment_variables)

        self._run_workers("update_environment_variables",
                          self._get_env_vars_to_be_updated())

        distributed_init_method = get_distributed_init_method(
            driver_ip, get_open_port())

        # Initialize the actual workers inside worker wrapper.
        all_kwargs = []
        for rank, (node_id, _) in enumerate(worker_node_and_gpu_ids):
            local_rank = 0
            kwargs = dict(
                vllm_config=self.vllm_config,
                local_rank=local_rank,
                rank=rank,
                distributed_init_method=distributed_init_method,
                is_driver_worker=(not self.parallel_config)
                or (rank % self.parallel_config.tensor_parallel_size == 0),
            )
            all_kwargs.append(kwargs)
        self._run_workers("init_worker", all_kwargs)

        self._run_workers("init_device")
        self._run_workers("load_model",
                          max_concurrent_workers=self.parallel_config.
                          max_parallel_loading_workers)

        if self.use_ray_spmd_worker:
            for pp_rank in range(self.parallel_config.pipeline_parallel_size):
                self.pp_tp_workers.append([])
                for tp_rank in range(
                        self.parallel_config.tensor_parallel_size):
                    # PP=2, TP=4
                    # pp_tp_workers = [[0, 1, 2, 3], [4, 5, 6, 7]]
                    rank = (pp_rank * self.parallel_config.tensor_parallel_size
                            ) + tp_rank
                    assert len(self.pp_tp_workers[pp_rank]) == tp_rank
                    assert pp_rank < len(self.pp_tp_workers)
                    self.pp_tp_workers[pp_rank].append(self.workers[rank])

        # This is the list of workers that are rank 0 of each TP group EXCEPT
        # global rank 0. These are the workers that will broadcast to the
        # rest of the workers.
        self.tp_driver_workers: List[RayWorkerWrapper] = []
        # This is the list of workers that are not drivers and not the first
        # worker in a TP group. These are the workers that will be
        # broadcasted to.
        self.non_driver_workers: List[RayWorkerWrapper] = []

        # Enforce rank order for correct rank to return final output.
        for index, worker in enumerate(self.workers):
            # The driver worker is rank 0 and not in self.workers.
            rank = index + 1
            if rank % self.parallel_config.tensor_parallel_size == 0:
                self.tp_driver_workers.append(worker)
            else:
                self.non_driver_workers.append(worker)

    def shutdown(self) -> None:
        logger.info(
            "Shutting down Ray distributed executor. If you see error log "
            "from logging.cc regarding SIGTERM received, please ignore because "
            "this is the expected termination process in Ray.")
        if hasattr(self, "forward_dag") and self.forward_dag is not None:
            self.forward_dag.teardown()
            import ray
            for worker in self.workers:
                ray.kill(worker)
            self.forward_dag = None
