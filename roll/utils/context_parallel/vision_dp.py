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

Ported from verl (https://github.com/verl-project/verl/pull/5230).

Strategy: Distribute whole images across DP ranks, not patches within images.
This avoids breaking cu_seqlens semantics while parallelizing ViT computation.

Key difference from text SP:
- Text SP: Split sequence within attention layers, all-to-all per layer
- Vision DP: Split images across ranks, all_gather once at the end
"""

import torch
import torch.distributed as dist
from torch.autograd import Function

from roll.utils.context_parallel.globals import get_ulysses_group, get_ulysses_size


def get_image_patch_counts(grid_thw: torch.Tensor) -> list[int]:
    """Compute number of patches per image from grid_thw.

    Args:
        grid_thw: Tensor of shape (num_images, 3) where each row is [t, h, w].

    Returns:
        List of patch counts per image.
    """
    if grid_thw.numel() == 0:
        return []
    return (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()


def get_image_embedding_counts(grid_thw: torch.Tensor, spatial_merge_size: int = 1) -> list[int]:
    """Compute number of embeddings per image after spatial merging.

    Args:
        grid_thw: Tensor of shape (num_images, 3) where each row is [t, h, w].
        spatial_merge_size: Spatial merge factor (typically 2 for Qwen-VL).

    Returns:
        List of embedding counts per image.
    """
    if grid_thw.numel() == 0:
        return []
    if spatial_merge_size == 1:
        return (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
    t = grid_thw[:, 0]
    h = grid_thw[:, 1] // spatial_merge_size
    w = grid_thw[:, 2] // spatial_merge_size
    return (t * h * w).tolist()


def assign_images_to_dp_ranks(
    patch_counts: list[int],
    dp_size: int,
) -> tuple[list[list[int]], list[int]]:
    """Assign whole images to DP ranks using contiguous distribution.

    Rank 0 gets images [0, 1, ...], rank 1 gets next chunk, etc.
    This ensures no reordering is needed after all-gather.

    Args:
        patch_counts: Number of patches per image.
        dp_size: Number of DP ranks.

    Returns:
        Tuple of (image_assignments, rank_loads) where:
        - image_assignments[rank] = list of image indices assigned to that rank
        - rank_loads[rank] = total patches assigned to that rank
    """
    num_images = len(patch_counts)
    if num_images == 0:
        return [[] for _ in range(dp_size)], [0] * dp_size

    image_assignments: list[list[int]] = [[] for _ in range(dp_size)]
    rank_loads = [0] * dp_size

    base_size = num_images // dp_size
    remainder = num_images % dp_size

    start = 0
    for rank in range(dp_size):
        chunk_size = base_size + (1 if rank < remainder else 0)
        end = start + chunk_size
        for img_idx in range(start, end):
            image_assignments[rank].append(img_idx)
            rank_loads[rank] += patch_counts[img_idx]
        start = end

    return image_assignments, rank_loads


def prepare_local_vision_inputs(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    image_assignments: list[list[int]],
    dp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Extract pixel values and grid_thw for this DP rank's assigned images.

    Args:
        pixel_values: All pixel values concatenated, shape (total_patches, dim).
        grid_thw: Grid dimensions per image, shape (num_images, 3).
        image_assignments: Per-rank image index assignments.
        dp_rank: This rank's index in the DP group.

    Returns:
        Tuple of (local_pixel_values, local_grid_thw, local_indices).
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

    patch_counts = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
    cumsum = [0]
    for c in patch_counts:
        cumsum.append(cumsum[-1] + c)

    local_patches = []
    local_grids = []
    for idx in local_indices:
        start, end = cumsum[idx], cumsum[idx + 1]
        local_patches.append(pixel_values[start:end])
        local_grids.append(grid_thw[idx : idx + 1])

    local_pixel_values = torch.cat(local_patches, dim=0)
    local_grid_thw = torch.cat(local_grids, dim=0)

    expected_patches = sum(patch_counts[idx] for idx in local_indices)
    assert local_pixel_values.shape[0] == expected_patches

    return local_pixel_values, local_grid_thw, local_indices


class GatherVisionEmbeddings(Function):
    """All-gather vision embeddings with gradient support.

    Contiguous assignment means simple concat without reordering.
    Backward: scales gradients by dp_size to compensate for partial processing.
    """

    @staticmethod
    def forward(ctx, local_embeddings, dp_group, grad_scaler=True):
        ctx.grad_scaler = grad_scaler
        dp_size = dist.get_world_size(dp_group)
        dp_rank = dist.get_rank(dp_group)
        ctx.dp_size = dp_size

        if dp_size == 1:
            return local_embeddings

        local_count = torch.tensor(
            [local_embeddings.shape[0]], dtype=torch.long, device=local_embeddings.device
        )
        all_counts = [torch.zeros_like(local_count) for _ in range(dp_size)]
        dist.all_gather(all_counts, local_count, group=dp_group)
        all_counts = [c.item() for c in all_counts]
        ctx.all_counts = all_counts
        ctx.dp_rank = dp_rank

        max_count = max(all_counts) if all_counts else 0
        if max_count == 0:
            return local_embeddings

        hidden_size = local_embeddings.shape[1] if local_embeddings.dim() > 1 else 1
        ctx.hidden_size = hidden_size

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
        grad_scaler = ctx.grad_scaler

        if dp_size == 1:
            return grad_output, None, None

        all_counts = ctx.all_counts
        dp_rank = ctx.dp_rank

        if grad_scaler:
            grad_output = grad_output * dp_size

        start = sum(all_counts[:dp_rank])
        end = start + all_counts[dp_rank]
        local_grad = grad_output[start:end]
        return local_grad, None, None


def gather_vision_embeddings(local_embeddings, dp_group=None, grad_scaler=True):
    """All-gather vision embeddings from all DP ranks.

    Args:
        local_embeddings: This rank's vision embeddings.
        dp_group: Process group for all-gather. Defaults to Ulysses group.
        grad_scaler: Whether to scale gradients in backward pass.

    Returns:
        All-gathered embeddings concatenated across ranks.
    """
    dp_group = get_ulysses_group() if dp_group is None else dp_group
    if dp_group is None or dist.get_world_size(dp_group) == 1:
        return local_embeddings
    return GatherVisionEmbeddings.apply(local_embeddings, dp_group, grad_scaler)


def create_dp_vision_forward(original_forward):
    """Wrap VisionTransformer.forward for Vision DP.

    Model-agnostic wrapper for any VisionTransformer with
    ``forward(self, hidden_states, grid_thw, **kwargs) -> Tensor`` signature.

    When Ulysses SP size > 1, distributes images across SP ranks and
    all-gathers the embeddings after ViT computation.

    Args:
        original_forward: The original VisionTransformer.forward method.

    Returns:
        Wrapped forward method with Vision DP support.
    """

    def dp_vision_forward(self, hidden_states, grid_thw, **kwargs):
        dp_size = get_ulysses_size()
        if dp_size is None or dp_size <= 1:
            return original_forward(self, hidden_states, grid_thw, **kwargs)

        dp_group = get_ulysses_group()
        dp_rank = dist.get_rank(dp_group)

        # Step 1: Get image assignment
        patch_counts = get_image_patch_counts(grid_thw)
        total_patches = sum(patch_counts)
        assert hidden_states.shape[0] == total_patches

        spatial_merge_size = 1
        if hasattr(self, "merger") and hasattr(self.merger, "spatial_merge_size"):
            spatial_merge_size = self.merger.spatial_merge_size
        elif hasattr(self, "spatial_merge_size"):
            spatial_merge_size = self.spatial_merge_size

        embedding_counts = get_image_embedding_counts(grid_thw, spatial_merge_size)
        total_embeddings = sum(embedding_counts)

        image_assignments, rank_loads = assign_images_to_dp_ranks(patch_counts, dp_size)

        # Step 2: Extract local inputs
        local_pixels, local_grid_thw, local_indices = prepare_local_vision_inputs(
            hidden_states, grid_thw, image_assignments, dp_rank
        )

        # Step 3: Process local images
        if local_pixels.shape[0] > 0:
            local_embeddings = original_forward(self, local_pixels, local_grid_thw, **kwargs)
        else:
            # Determine hidden_size for empty tensor
            if hasattr(self, "merger") and hasattr(self.merger, "ln_q"):
                ln_q = self.merger.ln_q
                if hasattr(ln_q, "normalized_shape"):
                    hidden_size = ln_q.normalized_shape[0]
                elif hasattr(ln_q, "weight"):
                    hidden_size = ln_q.weight.shape[0]
                else:
                    raise RuntimeError(
                        "Cannot determine hidden_size from merger.ln_q: "
                        "no 'normalized_shape' or 'weight' attribute found"
                    )
            elif hasattr(self, "out_hidden_size"):
                hidden_size = self.out_hidden_size
            elif hasattr(self, "config") and hasattr(self.config, "hidden_size"):
                hidden_size = self.config.hidden_size
            else:
                raise RuntimeError(
                    "Cannot determine hidden_size for empty Vision DP output. "
                    "Expected one of: self.merger.ln_q, self.out_hidden_size, self.config.hidden_size"
                )

            local_embeddings = torch.empty(
                (0, hidden_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        # Handle Qwen3-VL which returns (embeddings, deepstack_embeddings)
        deepstack_outputs = None
        if isinstance(local_embeddings, tuple):
            local_embeddings, deepstack_outputs = local_embeddings[0], local_embeddings[1:]

        # Step 4: All-gather
        all_embeddings = gather_vision_embeddings(local_embeddings, dp_group)
        assert all_embeddings.shape[0] == total_embeddings

        if deepstack_outputs is not None:
            # All-gather deepstack embeddings too
            gathered_deepstack = []
            for ds_emb in deepstack_outputs:
                if isinstance(ds_emb, list):
                    # List of tensors (one per deepstack layer)
                    gathered_list = []
                    for single_emb in ds_emb:
                        gathered_list.append(gather_vision_embeddings(single_emb, dp_group))
                    gathered_deepstack.append(gathered_list)
                elif isinstance(ds_emb, torch.Tensor):
                    gathered_deepstack.append(gather_vision_embeddings(ds_emb, dp_group))
                else:
                    gathered_deepstack.append(ds_emb)
            return (all_embeddings, *gathered_deepstack)

        return all_embeddings

    return dp_vision_forward
