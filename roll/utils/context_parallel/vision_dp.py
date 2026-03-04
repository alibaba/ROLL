# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Alibaba Inc.
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
"""
Vision Data Parallel utilities for distributing ViT computation across Ulysses SP ranks.

Distribute whole images across SP ranks, not patches within images.
Each rank runs ViT on its assigned images, then all-gather combines embeddings.
Backward all_reduce(SUM) recovers complete gradients before slicing by assignment.

Ported from verl (https://github.com/verl-project/verl/pull/5230).
"""

import torch
import torch.distributed as dist
from torch.autograd import Function

from roll.utils.context_parallel.globals import get_ulysses_group, get_ulysses_size


def get_image_patch_counts(grid_thw: torch.Tensor) -> list[int]:
    """Return [t*h*w for each image] from a [num_images, 3] grid_thw tensor."""
    if grid_thw.numel() == 0:
        return []
    return (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()


def get_image_embedding_counts(grid_thw: torch.Tensor, spatial_merge_size: int = 1) -> list[int]:
    """Return per-image embedding counts after spatial merging: t * (h/merge) * (w/merge)."""
    if grid_thw.numel() == 0:
        return []
    if spatial_merge_size == 1:
        return get_image_patch_counts(grid_thw)
    t = grid_thw[:, 0]
    h = grid_thw[:, 1] // spatial_merge_size
    w = grid_thw[:, 2] // spatial_merge_size
    return (t * h * w).tolist()


def assign_images_to_dp_ranks(
    patch_counts: list[int],
    dp_size: int,
) -> tuple[list[list[int]], list[int]]:
    """Assign whole images to DP ranks via greedy contiguous bin-packing.

    Returns (image_assignments, rank_patch_counts). Images are kept contiguous
    so the gather result needs no reordering.
    """
    num_images = len(patch_counts)
    if num_images == 0:
        return [[] for _ in range(dp_size)], [0] * dp_size

    image_assignments: list[list[int]] = [[] for _ in range(dp_size)]
    rank_loads = [0] * dp_size

    remaining_patches = sum(patch_counts)
    img_idx = 0
    for rank in range(dp_size):
        remaining_ranks = dp_size - rank
        remaining_images = num_images - img_idx

        if remaining_images <= 0:
            break

        # Dynamic target: distribute remaining patches evenly among remaining ranks
        target = remaining_patches / remaining_ranks

        # Must leave at least 1 image for each remaining rank
        max_images = remaining_images - (remaining_ranks - 1)

        # Greedily add images until we reach the target load or hit the max
        count = 0
        while img_idx < num_images and count < max_images:
            image_assignments[rank].append(img_idx)
            rank_loads[rank] += patch_counts[img_idx]
            img_idx += 1
            count += 1

            # Stop early once we've reached the target (always take at least 1)
            if rank_loads[rank] >= target:
                break

        remaining_patches -= rank_loads[rank]

    return image_assignments, rank_loads


def prepare_local_vision_inputs(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    image_assignments: list[list[int]],
    dp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Extract pixel values and grid_thw for this DP rank's assigned images.

    Exploits contiguous assignment: a single slice instead of per-image cat.
    """
    local_indices = image_assignments[dp_rank]

    if len(local_indices) == 0:
        return (
            torch.empty(
                (0, pixel_values.shape[1]) if pixel_values.dim() > 1 else (0,),
                dtype=pixel_values.dtype,
                device=pixel_values.device,
            ),
            torch.empty((0, 3), dtype=grid_thw.dtype, device=grid_thw.device),
            [],
        )

    # local_indices are contiguous (e.g. [2, 3, 4]), so use tensor slicing
    first_img_idx = local_indices[0]
    last_img_idx = local_indices[-1]

    # Compute patch offsets using cumsum
    patch_counts = get_image_patch_counts(grid_thw)
    patch_counts_tensor = torch.tensor(patch_counts, device=grid_thw.device, dtype=torch.long)
    offsets = torch.cat(
        (
            torch.tensor([0], device=grid_thw.device, dtype=torch.long),
            torch.cumsum(patch_counts_tensor, dim=0),
        )
    )

    start_patch = offsets[first_img_idx].item()
    end_patch = offsets[last_img_idx + 1].item()

    local_pixel_values = pixel_values[start_patch:end_patch]
    local_grid_thw = grid_thw[first_img_idx : last_img_idx + 1]

    # Cross-check: verify extracted slice matches independently computed patch counts
    independent_counts = get_image_patch_counts(local_grid_thw)
    expected_patches = sum(independent_counts)
    assert local_pixel_values.shape[0] == expected_patches, (
        f"[Vision DP] Local patch count mismatch: "
        f"extracted={local_pixel_values.shape[0]}, expected={expected_patches}, "
        f"local_indices={local_indices}"
    )

    return local_pixel_values, local_grid_thw, local_indices


class GatherVisionEmbeddings(Function):
    """All-gather vision embeddings with gradient support.

    Contiguous assignment means simple concat without reordering.
    Backward: all_reduce(SUM) to aggregate gradients from all sequence shards,
              then slice to extract this rank's image gradients.
    """

    @staticmethod
    def forward(ctx, local_embeddings, dp_group, all_counts: list[int]):
        dp_size = dist.get_world_size(dp_group)
        dp_rank = dist.get_rank(dp_group)
        ctx.dp_size = dp_size
        ctx.dp_group = dp_group
        ctx.all_counts = all_counts
        ctx.dp_rank = dp_rank

        if dp_size == 1:
            return local_embeddings

        max_count = max(all_counts) if all_counts else 0
        if max_count == 0:
            return local_embeddings

        hidden_size = local_embeddings.shape[1] if local_embeddings.dim() > 1 else 1

        if local_embeddings.shape[0] < max_count:
            pad_size = max_count - local_embeddings.shape[0]
            padding = torch.zeros(
                (pad_size, hidden_size),
                dtype=local_embeddings.dtype,
                device=local_embeddings.device,
            )
            local_padded = torch.cat([local_embeddings, padding], dim=0)
        else:
            local_padded = local_embeddings

        gathered = [torch.empty_like(local_padded) for _ in range(dp_size)]
        dist.all_gather(gathered, local_padded, group=dp_group)

        result_chunks = [gathered[r][: all_counts[r]] for r in range(dp_size)]
        result = torch.cat(result_chunks, dim=0)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        dp_size = ctx.dp_size

        if dp_size == 1:
            return grad_output, None, None

        all_counts = ctx.all_counts
        dp_rank = ctx.dp_rank
        dp_group = ctx.dp_group

        # all_reduce(SUM) aggregates partial gradients from all SP ranks:
        # each rank only has non-zero grad for vision tokens in its sequence shard.
        # NCCL all_reduce requires contiguous tensors — defensive guard.
        grad = grad_output.contiguous()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=dp_group)

        start = sum(all_counts[:dp_rank])
        end = start + all_counts[dp_rank]
        local_grad = grad[start:end]
        return local_grad, None, None


def gather_vision_embeddings(local_embeddings, dp_group, all_counts: list[int]):
    """All-gather vision embeddings from all DP ranks with gradient support."""
    dp_group = get_ulysses_group() if dp_group is None else dp_group
    if dp_group is None or dist.get_world_size(dp_group) == 1:
        return local_embeddings
    return GatherVisionEmbeddings.apply(local_embeddings, dp_group, all_counts)


def create_dp_vision_forward(original_forward):
    """Wrap VisionTransformer.forward for Vision DP (Data Parallel across SP ranks).

    Strategy:
    1. Distribute whole images to SP ranks (not patches within images)
    2. Each rank processes its assigned images independently
    3. All-gather embeddings at the end (contiguous assignment, no reordering)

    Gradient correctness: after all-gather in forward, each SP rank's inputs_embeds
    contains vision tokens from ALL images. But Ulysses gives each rank only its
    sequence shard. In backward, each rank only has non-zero gradient for vision
    tokens in its own shard. The all_reduce(SUM) in GatherVisionEmbeddings.backward
    aggregates partial gradients from all ranks, recovering the complete gradient.
    """

    def dp_vision_forward(self, hidden_states, grid_thw, **kwargs):
        dp_size = get_ulysses_size()
        if dp_size is None or dp_size <= 1:
            return original_forward(self, hidden_states, grid_thw, **kwargs)

        dp_group = get_ulysses_group()
        dp_rank = dist.get_rank(dp_group)

        # Move grid_thw to CPU once to avoid repeated GPU->CPU syncs in
        # metadata helpers (grid_thw is a tiny [num_images, 3] tensor).
        grid_thw_cpu = grid_thw.cpu()

        # Step 1: Get image assignment
        patch_counts = get_image_patch_counts(grid_thw_cpu)
        total_patches = sum(patch_counts)
        assert hidden_states.shape[0] == total_patches

        spatial_merge_size = 1
        if hasattr(self, "merger") and hasattr(self.merger, "spatial_merge_size"):
            spatial_merge_size = self.merger.spatial_merge_size
        elif hasattr(self, "spatial_merge_size"):
            spatial_merge_size = self.spatial_merge_size

        embedding_counts = get_image_embedding_counts(grid_thw_cpu, spatial_merge_size)
        total_embeddings = sum(embedding_counts)

        image_assignments, _ = assign_images_to_dp_ranks(patch_counts, dp_size)

        # Step 2: Extract local inputs
        local_pixels, local_grid_thw, local_indices = prepare_local_vision_inputs(
            hidden_states, grid_thw, image_assignments, dp_rank
        )

        # Detect Qwen3-VL deepstack: model attribute, not return type,
        # because empty ranks don't call original_forward and can't inspect the return.
        has_deepstack = hasattr(self, "deepstack_merger_list")

        # Step 3: Process local images
        if local_pixels.shape[0] > 0:
            local_embeddings = original_forward(self, local_pixels, local_grid_thw, **kwargs)
        else:
            # This rank has no images, create empty tensor with correct hidden size
            hidden_size = getattr(getattr(self, "config", None), "out_hidden_size", None)
            if hidden_size is None:
                raise RuntimeError(
                    f"Cannot determine hidden_size: self.config.out_hidden_size not found. "
                    f"Model type: {type(self).__name__}"
                )

            local_embeddings = torch.empty(
                (0, hidden_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            # Empty rank must participate in autograd for backward all_reduce
            local_embeddings.requires_grad_()

        # Unpack Qwen3-VL deepstack: forward returns (embeddings, list[3 × Tensor])
        local_deepstack = None
        if has_deepstack:
            if isinstance(local_embeddings, tuple):
                local_embeddings, local_deepstack = local_embeddings[0], local_embeddings[1]
            else:
                # Empty rank: create matching empty deepstack tensors
                num_deepstack = len(self.deepstack_merger_list)
                h = local_embeddings.shape[1]
                local_deepstack = [
                    torch.empty(
                        (0, h), dtype=hidden_states.dtype, device=hidden_states.device
                    )
                    for _ in range(num_deepstack)
                ]

        # Step 4: All-gather
        # Compute per-rank embedding counts locally (grid_thw is replicated on all ranks)
        all_counts = [sum(embedding_counts[i] for i in image_assignments[r]) for r in range(dp_size)]
        all_embeddings = gather_vision_embeddings(local_embeddings, dp_group, all_counts)
        assert all_embeddings.shape[0] == total_embeddings

        # Step 5: All-gather deepstack embeddings (all ranks must participate)
        if local_deepstack is not None:
            gathered_deepstack = [
                gather_vision_embeddings(ds, dp_group, all_counts)
                for ds in local_deepstack
            ]
            return all_embeddings, gathered_deepstack

        return all_embeddings

    return dp_vision_forward
