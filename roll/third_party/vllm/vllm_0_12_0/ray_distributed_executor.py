import os
from typing import TYPE_CHECKING

import ray
from ray.runtime_env import RuntimeEnv
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm.platforms import current_platform
from vllm.ray.ray_env import get_env_vars_to_copy
from vllm.utils.network_utils import get_distributed_init_method, get_ip, get_open_port
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.executor.ray_executor import RayDistributedExecutor, RayWorkerMetaData
from vllm.v1.executor.ray_utils import RayWorkerWrapper

from roll.platforms import current_platform as roll_current_platform
from roll.utils.logging import get_logger


if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup


logger = get_logger()


def initialize_ray_cluster(ray_address: str | None = None):
    if ray.is_initialized():
        return
    ray.init(address=ray_address)


class CustomRayDistributedExecutor(RayDistributedExecutor):
    def _init_executor(self) -> None:
        self.forward_dag: ray.dag.CompiledDAG | None = None

        assert not current_platform.is_tpu()

        placement_group = self.parallel_config.placement_group
        assert self.uses_ray
        assert len(placement_group) > 0
        initialize_ray_cluster(placement_group[0]["ray_address"])
        assert ray.is_initialized()

        # Disable Ray usage stats collection.
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

        # Create the parallel GPU workers.
        self._init_workers_ray(placement_group)

        # KV connector setup
        self.has_connector = self.vllm_config.kv_transfer_config is not None

        self.uses_sampler = self.vllm_config.model_config.runner_type != "pooling" and (
            self.vllm_config.ec_transfer_config is None or not self.vllm_config.ec_transfer_config.is_ec_producer
        )

        self.scheduler_output: SchedulerOutput | None = None

    def _init_workers_ray(self, placement_group: "PlacementGroup", **ray_remote_kwargs):
        assert len(placement_group) == self.parallel_config.world_size

        # The driver dummy worker does not actually use any resources.
        # It holds the resource for the driver worker.
        self.driver_dummy_worker: RayWorkerWrapper | None = None
        # The remaining workers are the actual ray actors.
        self.workers: list[RayWorkerWrapper] = []

        # Used in ray compiled DAG: indexed first by PP rank,
        # and then TP rank. In other words, the inner list is
        # the TP group of workers for a PP rank.
        self.pp_tp_workers: list[list[RayWorkerWrapper]] = []

        if self.parallel_config.ray_workers_use_nsight:
            ray_remote_kwargs = self._configure_ray_workers_use_nsight(ray_remote_kwargs)

        worker_metadata: list[RayWorkerMetaData] = []
        driver_ip = get_ip()
        for rank in range(self.parallel_config.world_size):
            pg = placement_group[rank]["placement_group"]
            gpu_rank = placement_group[rank]["gpu_rank"]
            env_vars = {}
            env_vars.update(roll_current_platform.get_custom_env_vars())
            env_vars.update(roll_current_platform.get_vllm_run_time_env_vars(gpu_rank))
            runtime_env = RuntimeEnv(env_vars=env_vars)
            assert current_platform.ray_device_key == "GPU"
            # NV+AMD GPUs, and Intel XPUs
            worker = ray.remote(
                num_cpus=0,
                num_gpus=0.01,
                runtime_env=runtime_env,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                ),
                **ray_remote_kwargs,
            )(RayWorkerWrapper).remote(vllm_config=self.vllm_config, rpc_rank=rank)
            worker_metadata.append(RayWorkerMetaData(worker=worker, created_rank=rank))

        worker_ips = ray.get(
            [
                each.worker.get_node_ip.remote()  # type: ignore[attr-defined]
                for each in worker_metadata
            ]
        )

        for each, ip in zip(worker_metadata, worker_ips):
            each.ip = ip

        logger.debug("workers: %s", worker_metadata)
        logger.debug("driver_dummy_worker: %s", self.driver_dummy_worker)

        # No need to sort, just use the given resource order of the placement group
        for i, item in enumerate(worker_metadata):
            item.adjusted_rank = i
        self.workers = [item.worker for item in worker_metadata]
        rerank_mapping = {item.created_rank: item.adjusted_rank for item in worker_metadata}
        self.collective_rpc("adjust_rank", args=(rerank_mapping,))

        # Get the set of GPU IDs used on each node.
        worker_node_and_gpu_ids = []
        for worker in [self.driver_dummy_worker] + self.workers:
            if worker is None:
                # driver_dummy_worker can be None when using ray spmd worker.
                continue
            worker_node_and_gpu_ids.append(ray.get(worker.get_node_and_gpu_ids.remote()))  # type: ignore[attr-defined]

        # Set environment variables for the driver and workers.
        # remove device_control_env_var(CUDA_VISIBLE_DEVICES), for we only allocate one gpu for each worker
        all_args_to_update_environment_variables = [{}] * len(worker_node_and_gpu_ids)

        # Environment variables to copy from driver to workers
        env_vars_to_copy = get_env_vars_to_copy(
            exclude_vars=self.WORKER_SPECIFIC_ENV_VARS,
            additional_vars=set(current_platform.additional_env_vars).union(self.ADDITIONAL_ENV_VARS),
            destination="workers",
        )

        # Copy existing env vars to each worker's args
        for args in all_args_to_update_environment_variables:
            # TODO: refactor platform-specific env vars
            for name in env_vars_to_copy:
                if name in os.environ:
                    args[name] = os.environ[name]

        self._env_vars_for_all_workers = all_args_to_update_environment_variables

        self.collective_rpc("update_environment_variables", args=(self._get_env_vars_to_be_updated(),))

        distributed_init_method = get_distributed_init_method(driver_ip, get_open_port())

        # Initialize the actual workers inside worker wrapper.
        all_kwargs = []
        for rank, (node_id, _) in enumerate(worker_node_and_gpu_ids):
            local_rank = 0
            kwargs = dict(
                vllm_config=self.vllm_config,
                local_rank=local_rank,
                rank=rank,
                distributed_init_method=distributed_init_method,
                is_driver_worker=(not self.parallel_config) or (rank % self.parallel_config.tensor_parallel_size == 0),
            )
            all_kwargs.append(kwargs)
        self.collective_rpc("init_worker", args=(all_kwargs,))

        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

        for pp_rank in range(self.parallel_config.pipeline_parallel_size):
            self.pp_tp_workers.append([])
            for tp_rank in range(self.parallel_config.tensor_parallel_size):
                # PP=2, TP=4
                # pp_tp_workers = [[0, 1, 2, 3], [4, 5, 6, 7]]
                rank = (pp_rank * self.parallel_config.tensor_parallel_size) + tp_rank
                assert len(self.pp_tp_workers[pp_rank]) == tp_rank
                assert pp_rank < len(self.pp_tp_workers)
                self.pp_tp_workers[pp_rank].append(self.workers[rank])

    def shutdown(self) -> None:
        logger.info(
            "Shutting down Ray distributed executor. If you see error log "
            "from logging.cc regarding SIGTERM received, please ignore because "
            "this is the expected termination process in Ray."
        )
        if hasattr(self, "forward_dag") and self.forward_dag is not None:
            self.forward_dag.teardown()
            import ray

            for worker in self.workers:
                ray.kill(worker)
            self.forward_dag = None
