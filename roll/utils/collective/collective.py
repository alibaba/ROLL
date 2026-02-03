from typing import Union, Optional

from torch._C._distributed_c10d import ReduceOp
from torch.distributed import Backend
import torch.distributed as dist

from roll.platforms import current_platform
from roll.utils.collective.pg_utils import init_custom_process_group
from roll.utils.logging import get_logger

logger = get_logger()


class GroupManager:

    def __init__(self):
        """
        基于torch ProcessGroup 实现
        ref: https://github.com/ray-project/ray/blob/master/python/ray/util/collective/collective.py
        """
        self._name_group_map = {}
        self._group_name_map = {}

    def create_collective_group(self, backend, world_size, rank, master_addr: str, master_port: int, group_name, global_ranks=None):
        self._name_group_map[group_name] = init_custom_process_group(
            backend=backend,
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=group_name,
            global_ranks=global_ranks
        )

        return self._name_group_map[group_name]

    def is_group_exist(self, group_name):
        return group_name in self._name_group_map

    def get_group_by_name(self, group_name):
        """Get the collective group handle by its name."""
        if not self.is_group_exist(group_name):
            logger.warning("The group '{}' is not initialized.".format(group_name))
            return None
        return self._name_group_map[group_name]

    def destroy_collective_group(self, group_name):
        """Group destructor."""
        if not self.is_group_exist(group_name):
            logger.warning("The group '{}' does not exist.".format(group_name))
            return

        # release the collective group resource
        g = self._name_group_map[group_name]
        # clean up the dicts
        del self._group_name_map[g]
        del self._name_group_map[group_name]


_group_mgr = GroupManager()


def init_collective_group(
    world_size: int,
    rank: int,
    master_addr: str,
    master_port: int,
    backend: Union[str, Backend] = current_platform.communication_backend,
    group_name: str = "default",
    global_ranks: Optional[list] = None,
):
    global _group_mgr
    if not group_name:
        raise ValueError("group_name '{}' needs to be a string.".format(group_name))

    if _group_mgr.is_group_exist(group_name):
        raise RuntimeError("Trying to initialize a group twice.")

    assert world_size > 0
    assert rank >= 0
    assert rank < world_size
    _group_mgr.create_collective_group(backend, world_size, rank, master_addr, master_port, group_name, global_ranks=global_ranks)


def allreduce(tensor, group_name: str = "default", op=ReduceOp.SUM):
    global _group_mgr
    dist.all_reduce(tensor, op=op, group=_group_mgr.get_group_by_name(group_name))


def broadcast(tensor, src_rank: int = 0, group_name: str = "default", async_op=False):
    global _group_mgr
    return dist.broadcast(tensor, src=src_rank, group=_group_mgr.get_group_by_name(group_name), async_op=async_op)

def barrier(group_name):
    global _group_mgr
    dist.barrier(group=_group_mgr.get_group_by_name(group_name), device_ids=[0])

def all_gather_object(object_list, obj, group_name):
    global _group_mgr
    dist.all_gather_object(object_list, obj, group=_group_mgr.get_group_by_name(group_name))

def broadcast_object_list(object_list, src=None, group_name="default", device=None, group_src=None):
    global _group_mgr
    assert (src is not None and group_src is None) or (src is None and group_src is not None),\
        ("Either src or group_src must be set, but they cannot be set simultaneously.")
    dist.broadcast_object_list(object_list, src=src, group_src=group_src, group=_group_mgr.get_group_by_name(group_name))
