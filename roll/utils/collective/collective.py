# Copyright (c) 2025, ALIBABA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Union

from torch._C._distributed_c10d import ReduceOp
from torch.distributed import Backend
import torch.distributed as dist

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

    def create_collective_group(self, backend, world_size, rank, master_addr: str, master_port: int, group_name):
        self._name_group_map[group_name] = init_custom_process_group(
            backend=backend,
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=group_name,
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
    backend: Union[str, Backend] = "nccl",
    group_name: str = "default",
):
    global _group_mgr
    if not group_name:
        raise ValueError("group_name '{}' needs to be a string.".format(group_name))

    if _group_mgr.is_group_exist(group_name):
        raise RuntimeError("Trying to initialize a group twice.")

    assert world_size > 0
    assert rank >= 0
    assert rank < world_size
    _group_mgr.create_collective_group(backend, world_size, rank, master_addr, master_port, group_name)


def allreduce(tensor, group_name: str = "default", op=ReduceOp.SUM):
    global _group_mgr
    dist.all_reduce(tensor, op=op, group=_group_mgr.get_group_by_name(group_name))


def broadcast(tensor, src_rank: int = 0, group_name: str = "default"):
    global _group_mgr
    dist.broadcast(tensor, src=src_rank, group=_group_mgr.get_group_by_name(group_name))


def barrier(group_name):
    global _group_mgr
    dist.barrier(group=_group_mgr.get_group_by_name(group_name), device_ids=[0])
