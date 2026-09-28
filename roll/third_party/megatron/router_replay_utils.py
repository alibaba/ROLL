"""
Router Replay Utilities
Utilities for handling router replay functionality in Megatron models.
ref from https://github.com/verl-project/verl/blob/cb236075dbf1f9b89660d5e2f28e30f3268ec7ee/verl/utils/megatron/router_replay_utils.py
"""

import inspect
from typing import Optional

import torch
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel.schedules import get_schedule_table
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region, scatter_to_sequence_parallel_region
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.transformer.transformer_block import get_num_layers_to_build

# from roll.third_party.megatron.router_replay_patch import RouterReplay, RouterReplayAction
from megatron.core.transformer.moe.router_replay import (
    RouterReplay,
    RouterReplayAction,
)

from roll.utils.logging import get_logger

logger = get_logger()


def get_expert_dtype(num_moe_experts: int) -> torch.dtype:
    """Statically select minimal dtype for expert indices based on model config.

    Uses tf_config.num_moe_experts (unified Megatron field) to determine the
    minimal dtype without a runtime .max().item() GPU sync. This is safe because
    router top-k indices are always in [0, num_moe_experts - 1].
    """
    if num_moe_experts <= 256:
        return torch.uint8
    elif num_moe_experts <= 65536:
        return torch.uint16
    return torch.uint32


def get_device_name() -> str:
    """Get the device type string based on available accelerators.

    Detects the available accelerator and returns the corresponding PyTorch
    device type string. Currently supports CUDA, Ascend NPU, and CPU.

    Returns:
        str: Device type string ('cuda', or 'cpu').
    """
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device


def is_moe_layer(tf_config, layer_idx):
    moe_layer_freq = getattr(tf_config, "moe_layer_freq", None)

    if moe_layer_freq is None:
        return True
    elif isinstance(moe_layer_freq, int):
        return layer_idx % moe_layer_freq == 0
    elif isinstance(moe_layer_freq, list):
        return moe_layer_freq[layer_idx] == 1
    else:
        raise ValueError(f"Unsupported moe_layer_freq type: {type(moe_layer_freq)}")


def get_moe_num_layers_to_build(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> int:
    """Count the number of MoE layers assigned to the current rank.
    When ``moe_layer_freq`` is 1 or unset, every transformer layer is an MoE
    layer, so the count equals the total layer count. Otherwise only layers
    whose global index satisfies the frequency predicate are counted.
    Args:
        config: Megatron TransformerConfig providing layer layout information.
        vp_stage: Virtual-pipeline stage index (None defaults to current).
        pp_rank: Pipeline-parallel rank (None defaults to current).
    Returns:
        Number of MoE layers on the specified rank/stage.
    """
    total_layers = get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)

    sig = inspect.signature(get_transformer_layer_offset)
    # core 0.12.1 is not support vp_stage and pp_rank as parameters
    if "vp_stage" in sig.parameters and "pp_rank" in sig.parameters:
        layer_offset = get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank)
    elif "pp_rank" in sig.parameters:
        layer_offset = get_transformer_layer_offset(config, pp_rank=pp_rank)
    else:
        layer_offset = get_transformer_layer_offset(config)

    local_global_indices = range(layer_offset, layer_offset + total_layers)

    num_moe_layers = sum(1 for idx in local_global_indices if is_moe_layer(config, idx))

    return num_moe_layers


def collect_r2_router_indices(tf_config, vp_rank: int) -> Optional[torch.Tensor]:
    """Collect recorded router top-k indices for R2 mode after model forward.

    Gathers recorded_topk_idx from all router instances for the current VP chunk,
    performs dtype optimization, stack, permute, and SP gather. Returns the raw
    gathered tensor for the caller to unpack/reshape as needed.

    Args:
        tf_config: Megatron TransformerConfig providing layer layout information.
        vp_rank: Virtual pipeline stage rank for this micro-batch.

    Returns:
        Tensor of shape [1, tokens_all, moe_layers_in_vp, topk] on the current
        device, or None if no router instances are found.
    """
    router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
    if not router_instances_list:
        return None

    expert_dtype = get_expert_dtype(tf_config.num_moe_experts)

    # Stack recorded indices: [moe_layers_in_vp, tokens_local, topk]
    layers_topk_idx = torch.stack([r.recorded_topk_idx.to(expert_dtype) for r in router_instances_list])
    # -> [tokens_local, moe_layers_in_vp, topk]
    layers_topk_idx = layers_topk_idx.permute(1, 0, 2).to(get_device_name())

    # SP gather -> [tokens_all, moe_layers_in_vp, topk], unsqueeze -> [1, tokens_all, moe_layers_in_vp, topk]
    layers_topk_idx = (
        gather_from_sequence_parallel_region(layers_topk_idx, tensor_parallel_output_grad=False)
        .unsqueeze(0)
        .contiguous()
    )
    return layers_topk_idx


def finalize_r2_routed_experts(
    router_topk_indices_list: list,
    tf_config,
    num_microbatches: int,
) -> torch.Tensor:
    """Finalize R2 routed experts by merging VPP layers and gathering across PP ranks.

    Handles VPP reordering (if vp_size > 1) and PP all-gather to produce the
    global routed_experts tensor. Sequence-packing restore_results_order is
    left to the caller.

    Args:
        router_topk_indices_list: List of per-micro-batch tensors, each of shape
            [mbs, seq_length, moe_layers_in_vp, topk].
        tf_config: Megatron TransformerConfig providing VPP and PP info.
        num_microbatches: Number of microbatches in this forward step.

    Returns:
        Tensor of shape [bs, max_seq_len, total_moe_layers, topk] on CPU.
    """
    vp_size = tf_config.virtual_pipeline_model_parallel_size
    if vp_size is not None and vp_size > 1:
        microbatch_group_size_per_vp_stage = tf_config.microbatch_group_size_per_vp_stage
        layers_topk_idx = reorder_and_merge_vpp_layers(
            router_topk_indices_list, num_microbatches, vp_size, microbatch_group_size_per_vp_stage
        )
    else:
        layers_topk_idx = torch.cat(router_topk_indices_list, dim=0)

    layers_topk_idx = pp_gather(layers_topk_idx, tf_config)
    return layers_topk_idx


def set_router_replay_data(layers_topk_idx, tf_config, vp_rank=None):
    """
    Scatter the packed router top-k indices back to sequence-parallel ranks and update each local
    RouterReplay instance with target indices for replay mode.

    Args:
        layers_topk_idx (torch.Tensor): Router top-k indices with shape [bs, max_seq_len, layer_num, topk].
        tf_config: Megatron/Transformer engine configuration object.
        vp_rank (Optional[int]): Virtual pipeline stage rank override. If None, the current VP rank from
            Megatron parallel state will be used.
    """
    with torch.no_grad():
        local_rank_info = get_current_rank_layer_info(tf_config, vp_rank)
        offset = local_rank_info["start"]

        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)

        layers_topk_idx = layers_topk_idx.permute(1, 0, 2, 3).contiguous()
        layers_topk_idx = scatter_to_sequence_parallel_region(layers_topk_idx.to(get_device_name()))

        layers_topk_idx = layers_topk_idx.reshape(-1, layers_topk_idx.shape[2], layers_topk_idx.shape[3])
        layers_topk_idx = layers_topk_idx.permute(1, 0, 2).contiguous()

        num_layers_in_tensor = layers_topk_idx.shape[0]
        index_by_layer = (num_layers_in_tensor == tf_config.num_layers)
        end = local_rank_info["end"]
        moe_idx = sum(1 for i in range(offset) if is_moe_layer(tf_config, i))
        router_offset = 0
        for layer_idx in range(offset, end):
            if not is_moe_layer(tf_config, layer_idx):
                continue
            idx = layer_idx if index_by_layer else moe_idx
            raw = layers_topk_idx[idx].to(torch.int64)
            # Sanitize invalid expert indices (e.g. -1 sentinels from SGLang, or 255 after
            # uint8 wrap). Out-of-range rows get dropped from routing_map, which desyncs
            # the MoE all-to-all split sizes against the dispatched token count.
            target = raw.clamp(0, tf_config.num_moe_experts - 1)
            router_instances_list[router_offset].set_target_indices(target)
            router_offset += 1
            moe_idx += 1
        if router_offset != len(router_instances_list):
            error_msg = (
                f"[RouterReplay] set {router_offset} routers but expected "
                f"{len(router_instances_list)}; unset routers keep stale targets"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)


def reorder_and_merge_vpp_layers(
    micro_batch_tensor_list,
    num_microbatches: int,
    vpp_size: int,
    microbatch_group_size_per_vp_stage: int,
) -> torch.Tensor:
    """
    Reorder and merge per-VPP layer blocks into a contiguous layer dimension.

    Given a tensor shaped as [bs*vpp_size, max_token_len, layer_num_per_vpp, topk], this function:
    1) Builds the schedule table for virtual microbatches and reorders the first dimension so that entries
       belonging to the same model chunk (VPP stage) become contiguous.
    2) Reshapes and merges the (vpp_size, layer_num_per_vpp) into a single layer dimension, producing
       [bs, max_token_len, layer_num, topk].

    Args:
        micro_batch_tensor_list : the list of Input tensor.
        num_microbatches (int): Number of microbatches per pipeline stage (bs).
        vpp_size (int): Virtual pipeline parallel size (number of model chunks).
        microbatch_group_size_per_vp_stage (int): Number of consecutive microbatches processed per VPP stage.

    Returns:
        torch.Tensor: Output tensor of shape [bs, max_token_len, layer_num, topk].

    Raises:
        ValueError: If input tensor dimensionality or expected sizes do not match.
        RuntimeError: If the computed output shape is unexpected or the schedule length mismatches.
    """
    # 1) Build schedule table: map each virtual_microbatch_id -> (microbatch_id, model_chunk_id)
    schedule_table = get_schedule_table(num_microbatches, vpp_size, microbatch_group_size_per_vp_stage)

    # 2) Group by model_chunk_id to build reorder indices so entries of the same chunk become contiguous along dim 0
    tensor_by_chunk = [[] for _ in range(vpp_size)]
    mini_tensor_list = []

    for vidx, (_mb, chunk_id) in enumerate(schedule_table):
        tensor_by_chunk[chunk_id].append(micro_batch_tensor_list[vidx])

    for chunk_id in range(vpp_size):
        mini_tensor_list.append(torch.cat(tensor_by_chunk[chunk_id], dim=0))

    out = torch.cat(mini_tensor_list, dim=2)
    return out


def get_current_rank_layer_info(tf_config, vp_rank=None):
    # When vp_rank is None, default to the current VP rank (or 0 if VP is disabled).
    """Return the local layer range/count for the current process and the full assignment table.

    Args:
        tf_config: Configuration object used by compute_pipeline_layer_assignment.
        vp_rank (Optional[int]): Explicit virtual pipeline stage rank to query. If None, uses
            mpu.get_virtual_pipeline_model_parallel_rank() when VP is enabled; otherwise 0.

    Returns:
        Tuple[dict, dict]: A tuple of (local_assignment, all_assignments) where local_assignment contains
        keys {"start", "end", "count"} for the current (pp_rank, vp_stage).
    """
    if vp_rank is None:
        vp_rank = 0
    num_layers_to_build = get_num_layers_to_build(tf_config, vp_stage=vp_rank)
    offset = get_transformer_layer_offset(tf_config, vp_stage=vp_rank)
    local = {}
    local["start"] = offset
    local["end"] = offset + num_layers_to_build
    local["count"] = num_layers_to_build
    return local


def _compute_moe_layers_per_pp(tf_config):
    """Compute total MoE layer count for each PP rank from config (no communication needed)."""
    pp_size = tf_config.pipeline_model_parallel_size
    vp_size = tf_config.virtual_pipeline_model_parallel_size
    moe_layers_per_pp = []
    for pp_rank in range(pp_size):
        if vp_size is not None:
            total = sum(get_moe_num_layers_to_build(tf_config, vp_stage, pp_rank) for vp_stage in range(vp_size))
        else:
            total = get_moe_num_layers_to_build(tf_config, pp_rank=pp_rank)
        moe_layers_per_pp.append(total)
    return moe_layers_per_pp


def pp_gather(local_layers_router_map, tf_config):
    """
    Gather local router maps from all PP ranks into a global router map.

    Supports non-uniform MoE layer distribution across PP ranks by padding
    the layer dimension to the maximum before all_gather, then slicing back.

    Args:
        local_layers_router_map (torch.Tensor): Local router map of shape
            [bs, max_seq_len, local_moe_layers, topk].
        tf_config: Configuration providing pipeline_model_parallel_size.

    Returns:
        torch.Tensor: Global router map of shape [bs, max_seq_len, total_moe_layers, topk] on CPU.
    """
    pp_size = tf_config.pipeline_model_parallel_size
    if pp_size <= 1:
        return local_layers_router_map

    pp_group = mpu.get_pipeline_model_parallel_group()
    world_size = torch.distributed.get_world_size(pp_group)
    local_layers_router_map = local_layers_router_map.to(get_device_name())

    moe_layers_per_pp = _compute_moe_layers_per_pp(tf_config)
    max_moe_layers = max(moe_layers_per_pp)
    local_moe_layers = local_layers_router_map.shape[2]

    # Pre-allocate max-size buffer and copy in to avoid temporary doubling from F.pad
    if local_moe_layers < max_moe_layers:
        padded = torch.zeros(
            (*local_layers_router_map.shape[:2], max_moe_layers, local_layers_router_map.shape[3]),
            dtype=local_layers_router_map.dtype,
            device=local_layers_router_map.device,
        )
        padded[:, :, :local_moe_layers, :] = local_layers_router_map
        del local_layers_router_map
        local_layers_router_map = padded

    layers_topk_idx_global_list = [
        torch.empty(
            size=local_layers_router_map.shape,
            dtype=local_layers_router_map.dtype,
            device=local_layers_router_map.device,
        )
        for _ in range(world_size)
    ]
    torch.distributed.all_gather(
        tensor=local_layers_router_map,
        tensor_list=layers_topk_idx_global_list,
        group=pp_group,
        async_op=False,
    )

    # Slice each rank's tensor back to its actual MoE layer count
    for i in range(world_size):
        actual = moe_layers_per_pp[i]
        if actual < max_moe_layers:
            layers_topk_idx_global_list[i] = layers_topk_idx_global_list[i][:, :, :actual, :]

    vp_size = tf_config.virtual_pipeline_model_parallel_size
    if vp_size is not None:
        vpp_router_map_offset = [[] for _ in range(pp_size)]
        for pp_stage in range(pp_size):
            vpp_router_map_offset[pp_stage].append(0)
            for vp_stage in range(vp_size):
                num_layers_to_build = get_moe_num_layers_to_build(tf_config, vp_stage, pp_stage)
                vpp_router_map_offset[pp_stage].append(num_layers_to_build + vpp_router_map_offset[pp_stage][-1])
        layers_topk_idx_global = []
        for vp_stage in range(vp_size):
            for pp_stage in range(pp_size):
                piece = slice(vpp_router_map_offset[pp_stage][vp_stage], vpp_router_map_offset[pp_stage][vp_stage + 1])
                layers_topk_idx_global.append(layers_topk_idx_global_list[pp_stage][:, :, piece, :])
        global_router_map = torch.cat(layers_topk_idx_global, dim=2).to("cpu")
    else:
        global_router_map = torch.cat(layers_topk_idx_global_list, dim=2).to("cpu")

    return global_router_map


class RouterReplayHelper:
    """Helper class to query router replay state and locate local RouterReplay instances."""

    @staticmethod
    def get_micro_batch_router_list(tf_config, vp_rank=None):
        """
        Return the list of RouterReplay instances corresponding to the current micro-batch and local
        (pp_rank, vp_stage) layer range.

        When virtual pipeline (VPP) is enabled, the local range for the PP rank is expanded to include
        all VP stages by multiplying the per-VP count by vp_size. The returned slice is taken from the
        global RouterReplay.router_instances list.

        Args:
            tf_config: Configuration object used to compute layer assignments.
            vp_rank (Optional[int]): Explicit virtual pipeline stage to query. If None, the current VP
                rank from Megatron parallel state is used when available.
        Returns:
            list: A contiguous sublist of RouterReplay.router_instances for the local layer range.
        """
        vp_size = tf_config.virtual_pipeline_model_parallel_size
        if vp_size is not None:
            vp_rank = 0 if vp_rank is None else vp_rank
            offset = 0
            for pre_vp_stage in range(vp_size):
                if pre_vp_stage == vp_rank:
                    break
                offset += get_moe_num_layers_to_build(tf_config, pre_vp_stage)
        else:
            offset = 0

        num_layers_to_build = get_moe_num_layers_to_build(tf_config, vp_rank)
        router_instances_list = RouterReplay.global_router_replay_instances[offset: offset + num_layers_to_build]
        return router_instances_list

    @staticmethod
    def is_r2_record_action(tf_config, vp_rank=None) -> bool:
        """Return True if the current router_replay_action is RECORD (R2) for the local router instances.

        This inspects the first local RouterReplay instance's router_replay_action and compares it to
        RouterReplayAction.RECORD.
        """
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return router_instances_list and router_instances_list[0].router_replay_action == RouterReplayAction.RECORD

    @staticmethod
    def is_replay_forward_action(tf_config, vp_rank=None) -> bool:
        """Return True if the current router_replay_action is REPLAY_FORWARD for the local router instances.

        This inspects the first local RouterReplay instance's router_replay_action and compares it to
        RouterReplayAction.REPLAY_FORWARD.
        """
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return (
            router_instances_list and router_instances_list[0].router_replay_action == RouterReplayAction.REPLAY_FORWARD
        )

    @staticmethod
    def is_replay_backward_action(tf_config, vp_rank=None) -> bool:
        """Return True if the current router_replay_action is REPLAY_BACKWARD for the local router instances.

        This inspects the first local RouterReplay instance's router_replay_action and compares it to
        RouterReplayAction.REPLAY_BACKWARD.
        """
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return (
            router_instances_list
            and router_instances_list[0].router_replay_action == RouterReplayAction.REPLAY_BACKWARD
        )
