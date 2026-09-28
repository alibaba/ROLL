"""Direct GPU tensor transfer between diffusion worker roles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ray
import torch

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.cluster import Cluster
from roll.distributed.executor.worker import Worker
from roll.platforms import current_platform
from roll.utils.collective import collective
from roll.utils.cuda_ipc_utils import MultiprocessingSerializer
from roll.utils.logging import get_logger

logger = get_logger()


@dataclass(frozen=True)
class _TensorTransferRoute:
    src_rank: int
    tgt_rank: int
    same_gpu: bool
    group_name: str | None


class TensorTransferWorker(Worker):
    """Worker base with GPU slots used by direct cross-role tensor transfer."""

    _gpu_tensor_transfer_enabled = False

    def __init__(self, worker_config: WorkerConfig) -> None:
        super().__init__(worker_config=worker_config)
        self._tensor_transfer_slots: dict[str, torch.Tensor] = {}

    def set_gpu_tensor_transfer_enabled(self, enabled: bool) -> None:
        """Enable or disable direct cross-role GPU tensor transport."""
        self._gpu_tensor_transfer_enabled = enabled

    def setup_tensor_transfer_group(
        self,
        group_name: str,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
    ) -> None:
        """Initialize one persistent process group used for cross-role tensor transfer."""
        collective.init_collective_group(
            world_size=world_size,
            rank=rank,
            group_name=group_name,
            master_addr=master_addr,
            master_port=master_port,
        )
        collective.allreduce(torch.zeros(1, device=self.device), group_name=group_name)

    def receive_tensor_via_ipc(self, tgt_slot: str, tensor_handle: bytes) -> None:
        """Receive a colocated CUDA tensor and take ownership of its storage."""
        tensor = MultiprocessingSerializer.deserialize(tensor_handle)
        self._tensor_transfer_slots[tgt_slot] = tensor.clone()
        current_platform.synchronize()

    def send_tensor_via_ipc(
        self,
        src_slot: str,
        tgt_worker: Any,
        tgt_slot: str,
        clear_source: bool,
    ) -> None:
        """Send a staged tensor to a worker sharing the same physical GPU."""
        tensor = self._tensor_transfer_slots[src_slot]
        current_platform.synchronize()
        tensor_handle = MultiprocessingSerializer.serialize(tensor)
        ray.get(tgt_worker.receive_tensor_via_ipc.remote(tgt_slot=tgt_slot, tensor_handle=tensor_handle))
        if clear_source:
            self._tensor_transfer_slots.pop(src_slot)

    def receive_tensor_via_collective(
        self,
        tgt_slot: str,
        tensor_shape: tuple[int, ...],
        tensor_dtype: torch.dtype,
        group_name: str,
    ) -> None:
        """Receive a tensor into a GPU buffer through a persistent NCCL group."""
        tensor = torch.empty(tensor_shape, dtype=tensor_dtype, device=self.device)
        collective.broadcast(tensor=tensor, src_rank=0, group_name=group_name)
        self._tensor_transfer_slots[tgt_slot] = tensor

    def send_tensor_via_collective(
        self,
        src_slot: str,
        tgt_worker: Any,
        tgt_slot: str,
        group_name: str,
        clear_source: bool,
    ) -> None:
        """Broadcast a staged tensor to a worker on another GPU or node."""
        tensor = self._tensor_transfer_slots[src_slot]
        receive_ref = tgt_worker.receive_tensor_via_collective.remote(
            tgt_slot=tgt_slot,
            tensor_shape=tuple(tensor.shape),
            tensor_dtype=tensor.dtype,
            group_name=group_name,
        )
        collective.broadcast(tensor=tensor, src_rank=0, group_name=group_name)
        ray.get(receive_ref)
        if clear_source:
            self._tensor_transfer_slots.pop(src_slot)


class TensorTransferGroup:
    """Move identically sharded tensors between two topology-compatible clusters.

    Each source rank is paired with the target rank that has the same DP/TP/PP/CP
    coordinates. CUDA IPC is used when the pair shares a physical GPU; otherwise
    a persistent two-rank NCCL group is used.
    """

    def __init__(self, src_cluster: Cluster, tgt_cluster: Cluster, name: str) -> None:
        error = self.compatibility_error(src_cluster, tgt_cluster)
        if error is not None:
            raise ValueError(error)

        self.src_cluster = src_cluster
        self.tgt_cluster = tgt_cluster
        self.name = name
        self.routes = self._build_routes()
        self._initialize_collective_groups()

        ipc_routes = sum(route.same_gpu for route in self.routes)
        logger.info(
            f"Tensor transfer '{name}' initialized with {ipc_routes} CUDA IPC routes and "
            f"{len(self.routes) - ipc_routes} NCCL routes"
        )

    @staticmethod
    def compatibility_error(src_cluster: Cluster, tgt_cluster: Cluster) -> str | None:
        """Return why direct transfer is unsupported, or ``None`` when supported."""
        src_ranks = {
            (rank_info.dp_rank, rank_info.tp_rank, rank_info.pp_rank, rank_info.cp_rank)
            for rank_info in src_cluster.worker_rank_info
        }
        tgt_ranks = {
            (rank_info.dp_rank, rank_info.tp_rank, rank_info.pp_rank, rank_info.cp_rank)
            for rank_info in tgt_cluster.worker_rank_info
        }
        if src_ranks != tgt_ranks or src_cluster.world_size != tgt_cluster.world_size:
            return (
                f"clusters '{src_cluster.cluster_name}' and '{tgt_cluster.cluster_name}' have different "
                "DP/TP/PP/CP layouts"
            )

        for cluster in (src_cluster, tgt_cluster):
            if any(
                len(cluster.rank2devices[rank]) != 1
                or cluster.rank2devices[rank][0]["gpu_rank"] is None
                for rank in range(cluster.world_size)
            ):
                return f"cluster '{cluster.cluster_name}' does not use exactly one GPU per worker"
        return None

    def _build_routes(self) -> list[_TensorTransferRoute]:
        tgt_rank_by_layout = {
            (rank_info.dp_rank, rank_info.tp_rank, rank_info.pp_rank, rank_info.cp_rank): rank
            for rank, rank_info in enumerate(self.tgt_cluster.worker_rank_info)
        }
        routes: list[_TensorTransferRoute] = []
        for src_rank, rank_info in enumerate(self.src_cluster.worker_rank_info):
            layout = (rank_info.dp_rank, rank_info.tp_rank, rank_info.pp_rank, rank_info.cp_rank)
            tgt_rank = tgt_rank_by_layout[layout]
            src_device = self.src_cluster.rank2devices[src_rank][0]
            tgt_device = self.tgt_cluster.rank2devices[tgt_rank][0]
            same_gpu = (
                src_device["node_rank"] == tgt_device["node_rank"]
                and src_device["gpu_rank"] == tgt_device["gpu_rank"]
            )
            routes.append(
                _TensorTransferRoute(
                    src_rank=src_rank,
                    tgt_rank=tgt_rank,
                    same_gpu=same_gpu,
                    group_name=None if same_gpu else f"tensor_transfer_{self.name}_{src_rank}_{tgt_rank}",
                )
            )
        return routes

    def _initialize_collective_groups(self) -> None:
        for route in self.routes:
            if route.group_name is None:
                continue
            src_worker = self.src_cluster.rank2worker[route.src_rank]
            tgt_worker = self.tgt_cluster.rank2worker[route.tgt_rank]
            master_addr = ray.get(src_worker.get_node_ip.remote())
            master_port = ray.get(src_worker.get_free_port.remote())
            ray.get(
                [
                    src_worker.setup_tensor_transfer_group.remote(
                        group_name=route.group_name,
                        rank=0,
                        world_size=2,
                        master_addr=master_addr,
                        master_port=master_port,
                    ),
                    tgt_worker.setup_tensor_transfer_group.remote(
                        group_name=route.group_name,
                        rank=1,
                        world_size=2,
                        master_addr=master_addr,
                        master_port=master_port,
                    ),
                ]
            )

    def transfer(self, src_slot: str, tgt_slot: str, clear_source: bool = True) -> None:
        """Transfer one staged tensor from every source rank to its paired target rank."""
        refs: list[Any] = []
        for route in self.routes:
            src_worker = self.src_cluster.rank2worker[route.src_rank]
            tgt_worker = self.tgt_cluster.rank2worker[route.tgt_rank]
            if route.same_gpu:
                refs.append(
                    src_worker.send_tensor_via_ipc.remote(
                        src_slot=src_slot,
                        tgt_worker=tgt_worker,
                        tgt_slot=tgt_slot,
                        clear_source=clear_source,
                    )
                )
            else:
                refs.append(
                    src_worker.send_tensor_via_collective.remote(
                        src_slot=src_slot,
                        tgt_worker=tgt_worker,
                        tgt_slot=tgt_slot,
                        group_name=route.group_name,
                        clear_source=clear_source,
                    )
                )
        ray.get(refs)
