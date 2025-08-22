# This file is modified from https://github.com/feifeibear/long-context-attention
# Implementation refers to USP Paper: https://arxiv.org/abs/2405.07719

import torch.distributed as dist


class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance


class ProcessGroupManager(Singleton):
    def __init__(self):
        self.ulysses_group = None
        self.ulysses_seqlen = None
        self.ulysses_size = None


PROCESS_GROUP_MANAGER = ProcessGroupManager()


def set_upg_manager(ulysses_size: int, rank: int, world_size: int):
    """
    Set the process group manager for Ulysses parallelism.
    """

    assert ulysses_size <= world_size
    assert world_size % ulysses_size == 0, f"world_size={world_size} must be divisible by ulysses_size={ulysses_size}."

    up_group_num = world_size // ulysses_size
    for i in range(up_group_num):
        sp_ranks = list(range(i * ulysses_size, (i + 1) * ulysses_size))
        group = dist.new_group(sp_ranks)
        if rank in sp_ranks:
            up_group = group

    PROCESS_GROUP_MANAGER.ulysses_group = up_group
    PROCESS_GROUP_MANAGER.ulysses_size = ulysses_size


def get_upg_manager():
    return PROCESS_GROUP_MANAGER


def get_ulysses_group():
    """Get the overall Ulysses parallel process group."""
    return PROCESS_GROUP_MANAGER.ulysses_group


def get_ulysses_seqlen():
    """Get the size of the Ulysses sequence parallel group."""
    return PROCESS_GROUP_MANAGER.ulysses_seqlen


def set_ulysses_seqlen(seqlen):
    """Get the size of the Ulysses sequence parallel group."""
    PROCESS_GROUP_MANAGER.ulysses_seqlen = seqlen


def get_ulysses_size():
    """Get the size of the Ulysses sequence parallel group."""
    return PROCESS_GROUP_MANAGER.ulysses_size
