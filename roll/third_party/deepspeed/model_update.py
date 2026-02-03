import ray
import torch.distributed as dist
from deepspeed.runtime.zero import GatheredParameters
from peft import get_peft_model_state_dict

from roll.configs.base_config import PPOConfig
from roll.configs.worker_config import is_actor_infer_overlapping_with_any_cluster
from roll.utils.collective import collective
from roll.utils.logging import get_logger
from roll.utils.network_utils import collect_free_port, get_node_ip
from roll.utils.send_recv_utils import serialize_named_weights


logger = get_logger()


def _get_ds_param_size(param):
    if hasattr(param, "ds_numel"):
        ds_numel = param.ds_numel
    else:
        ds_numel = param.numel()
    return ds_numel * param.element_size()


def _gather_weights(is_zero3, named_params):
    if not is_zero3:
        return [(n, p.data) for n, p in named_params]
    with GatheredParameters([p for _, p in named_params]):
        return [(n, p.data) for n, p in named_params]


def gather_deepspeed_weights(model, ds_config, buffer_size):
    is_zero3 = ds_config.is_zero3()
    named_params = [(name, param) for name, param in model.named_parameters()]

    waiting_params, waiting_params_size = [], 0
    for name, param in named_params:
        if waiting_params and waiting_params_size + _get_ds_param_size(param) > buffer_size:
            yield _gather_weights(is_zero3, waiting_params)
            waiting_params, waiting_params_size = [], 0
        waiting_params_size += _get_ds_param_size(param)
        waiting_params.append((name, param))

    if waiting_params:
        yield _gather_weights(is_zero3, waiting_params)


class DeepSpeedWeightUpdater:
    def __init__(self, pipeline_config: PPOConfig, infer_cluster, worker_config, model_update_name: str, model, ds_config, is_lora):
        self.pipeline_config = pipeline_config
        self.worker_config = worker_config
        self.model_update_name = model_update_name
        self.model = model
        self.ds_config = ds_config
        self.model_update_infer_workers = infer_cluster.workers
        self._model_update_buffer_size = pipeline_config.model_update_buffer_size_mb * 1024 * 1024  # Convert MB to bytes
        self.is_lora = is_lora
        self.infer_worker_config = infer_cluster.worker_config
        self.infer_cluster = infer_cluster
        self.is_colocated = is_actor_infer_overlapping_with_any_cluster(infer_cluster.worker_config, actor_train=worker_config)

        # Colocated mode attributes
        self._infer_parallel_cpu_group = None
        self._co_infer_worker = None
        self._buffer_num = None
        self._broadcast_workers = None

        # Separated mode attributes
        self.model_update_group_name = None
        self._model_update_locker = None

        if self.is_colocated:
            self._setup_colocated_model_update()
        else:
            self._setup_separated_model_update()

    def model_update(self):
        if self.is_colocated:
            return self._colocated_model_update()
        return self._separated_model_update()

    def _setup_colocated_model_update(self):
        logger.info(f"RANK {dist.get_rank()} Setup colocated model update")
        infer_worker_devices_num = self.infer_worker_config.num_gpus_per_worker
        train_world_size = dist.get_world_size()

        device_start_diff = min(self.worker_config.device_mapping) - min(self.infer_worker_config.device_mapping)
        device_end_diff = max(self.worker_config.device_mapping) - max(self.infer_worker_config.device_mapping)

        assert device_start_diff % infer_worker_devices_num == 0
        assert device_end_diff % infer_worker_devices_num == 0

        for start_rank in range(0, train_world_size, infer_worker_devices_num):
            end_rank = start_rank + infer_worker_devices_num
            assert end_rank <= train_world_size
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._infer_parallel_cpu_group = new_group
        infer_worker_idx = dist.get_rank() + device_start_diff // infer_worker_devices_num
        self._co_infer_worker = None
        if 0 <= infer_worker_idx < len(self.model_update_infer_workers):
            self._co_infer_worker = self.model_update_infer_workers[infer_worker_idx]

        # rank0 broadcast to mismatch workers
        if dist.get_rank() == 0 and (device_start_diff > 0 or device_end_diff < 0):
            self._broadcast_workers = []
            if device_start_diff > 0:
                self._broadcast_workers.extend(self.model_update_infer_workers[: device_start_diff // infer_worker_devices_num])
            if device_end_diff < 0:
                self._broadcast_workers.extend(self.model_update_infer_workers[device_end_diff // infer_worker_devices_num :])
            self._setup_broadcast_group()

    def _setup_separated_model_update(self):
        if dist.get_rank() != 0:
            return

        self._broadcast_workers = self.model_update_infer_workers
        self._setup_broadcast_group()

    def _setup_broadcast_group(self):
        if not self._broadcast_workers:
            return
        self.model_update_group_name = f"{self.model_update_name}_deepspeed"
        num_gpus_per_infer_worker = self.infer_worker_config.num_gpus_per_worker
        infer_device_num = num_gpus_per_infer_worker * len(self._broadcast_workers)
        master_address, master_port = get_node_ip(), collect_free_port()

        refs = [
            infer_worker.setup_collective_group.remote(
                master_address=master_address,
                master_port=master_port,
                group_name=self.model_update_group_name,
                rank_offset=i * num_gpus_per_infer_worker + 1,
                world_size=infer_device_num + 1,
            )
            for i, infer_worker in enumerate(self._broadcast_workers)
        ]
        collective.init_collective_group(
            infer_device_num + 1,
            0,
            group_name=self.model_update_group_name,
            master_addr=master_address,
            master_port=master_port,
        )
        ray.get(refs)

        logger.info(f"Init weights update group {self.model_update_group_name}")

    def _colocated_model_update(self):
        refs = []
        for named_weights in gather_deepspeed_weights(
            self.model, self.ds_config, buffer_size=self._model_update_buffer_size
        ):
            serialized_tensors = serialize_named_weights(
                named_weights, infer_strategy=self.infer_worker_config.strategy_args.strategy_name
            )
            infer_parallel_size = dist.get_world_size(self._infer_parallel_cpu_group)
            co_infer_rank = dist.get_rank(self._infer_parallel_cpu_group)
            infer_parallel_tensors = [serialized_tensors]  # tensors for each infer parallel rank
            if infer_parallel_size > 1:
                infer_parallel_tensors = [None] * infer_parallel_size if co_infer_rank == 0 else None
                dist.gather_object(
                    serialized_tensors, infer_parallel_tensors, group_dst=0, group=self._infer_parallel_cpu_group
                )
            if refs:
                ray.get(refs)
                refs = []
            if co_infer_rank == 0 and self._co_infer_worker is not None:
                refs.append(self._co_infer_worker.update_parameter_in_bucket.remote(infer_parallel_tensors))
            if self._broadcast_workers:
                refs.extend(self._broadcast_to_infer_workers(named_weights))
        if refs:
            ray.get(refs)
        return {}

    def _broadcast_to_infer_workers(self, named_weights) -> list[ray.ObjectRef]:
        if not self._broadcast_workers:
            return []
        refs = [
            worker.broadcast_parameter.remote(
                group_name=self.model_update_group_name,
                names=[n for n, _ in named_weights],
                dtypes=[w.dtype for _, w in named_weights],
                shapes=[w.shape for _, w in named_weights],
            )
            for worker in self._broadcast_workers
        ]
        handles = []
        for _, weight in named_weights:
            handles.append(
                collective.broadcast(tensor=weight, src_rank=0, group_name=self.model_update_group_name, async_op=True)
            )
        for handle in handles:
            handle.wait()
        return refs

    def _separated_model_update(self):
        logger.info(f"start broadcast model update {self.model_update_group_name}")
        for named_weights in gather_deepspeed_weights(
            self.model, self.ds_config, buffer_size=self._model_update_buffer_size
        ):
            refs = self._broadcast_to_infer_workers(named_weights)
            ray.get(refs)
        return {}
