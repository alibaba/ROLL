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
Unit tests for Vision Data Parallel utilities.
Ported from verl (https://github.com/verl-project/verl/pull/5230).
"""

import pytest
import torch

from roll.utils.context_parallel.vision_dp import (
    assign_images_to_dp_ranks,
    get_image_patch_counts,
    prepare_local_vision_inputs,
)


class TestGetImagePatchCounts:
    """Tests for get_image_patch_counts function."""

    def test_basic_patch_counts(self):
        grid_thw = torch.tensor([
            [2, 4, 4],  # 2*4*4 = 32
            [1, 2, 2],  # 1*2*2 = 4
            [1, 8, 8],  # 1*8*8 = 64
        ])
        counts = get_image_patch_counts(grid_thw)
        assert counts == [32, 4, 64]

    def test_single_image(self):
        grid_thw = torch.tensor([[1, 4, 4]])  # 16 patches
        counts = get_image_patch_counts(grid_thw)
        assert counts == [16]

    def test_empty_input(self):
        grid_thw = torch.empty((0, 3), dtype=torch.long)
        counts = get_image_patch_counts(grid_thw)
        assert counts == []

    def test_video_frames(self):
        grid_thw = torch.tensor([[4, 4, 4]])  # 4 frames, 4*4 patches each = 64
        counts = get_image_patch_counts(grid_thw)
        assert counts == [64]


class TestAssignImagesToDpRanks:
    """Tests for assign_images_to_dp_ranks function."""

    def test_balanced_assignment(self):
        patch_counts = [100, 100, 100, 100]
        assignments, loads = assign_images_to_dp_ranks(patch_counts, dp_size=2)
        assert len(assignments[0]) == 2
        assert len(assignments[1]) == 2
        assert loads[0] == 200
        assert loads[1] == 200

    def test_imbalanced_images(self):
        patch_counts = [500, 100, 100, 100]
        assignments, loads = assign_images_to_dp_ranks(patch_counts, dp_size=2)
        total_assigned = sum(len(a) for a in assignments)
        assert total_assigned == 4
        assert 0 in assignments[0] or 0 in assignments[1]

    def test_fewer_images_than_ranks(self):
        patch_counts = [100, 200]
        assignments, loads = assign_images_to_dp_ranks(patch_counts, dp_size=4)
        non_empty_ranks = sum(1 for a in assignments if len(a) > 0)
        assert non_empty_ranks == 2
        all_assigned = set()
        for a in assignments:
            all_assigned.update(a)
        assert all_assigned == {0, 1}

    def test_empty_input(self):
        patch_counts = []
        assignments, loads = assign_images_to_dp_ranks(patch_counts, dp_size=4)
        assert all(len(a) == 0 for a in assignments)
        assert all(load == 0 for load in loads)

    def test_single_rank(self):
        patch_counts = [100, 200, 300]
        assignments, loads = assign_images_to_dp_ranks(patch_counts, dp_size=1)
        assert assignments == [[0, 1, 2]]
        assert loads == [600]

    def test_equal_images_equal_size(self):
        patch_counts = [100, 100, 100, 100, 100, 100]  # 6 images
        assignments, loads = assign_images_to_dp_ranks(patch_counts, dp_size=3)
        assert all(len(a) == 2 for a in assignments)
        assert all(load == 200 for load in loads)

    def test_image_order_preserved(self):
        patch_counts = [10, 20, 30, 40, 50]
        assignments, _ = assign_images_to_dp_ranks(patch_counts, dp_size=2)
        for rank_assignment in assignments:
            assert rank_assignment == sorted(rank_assignment)


class TestPrepareLocalVisionInputs:
    """Tests for prepare_local_vision_inputs function."""

    def test_basic_extraction(self):
        pixel_values = torch.randn(100, 768)
        grid_thw = torch.tensor([
            [1, 6, 6],  # 36 patches (indices 0-35)
            [1, 8, 8],  # 64 patches (indices 36-99)
        ])
        image_assignments = [[0], [1]]

        local_pix, local_grid, local_indices = prepare_local_vision_inputs(
            pixel_values, grid_thw, image_assignments, dp_rank=0
        )
        assert local_pix.shape[0] == 36
        assert local_grid.shape[0] == 1
        assert local_indices == [0]
        assert torch.allclose(local_pix, pixel_values[:36])

        local_pix, local_grid, local_indices = prepare_local_vision_inputs(
            pixel_values, grid_thw, image_assignments, dp_rank=1
        )
        assert local_pix.shape[0] == 64
        assert local_grid.shape[0] == 1
        assert local_indices == [1]
        assert torch.allclose(local_pix, pixel_values[36:100])

    def test_multiple_images_per_rank(self):
        pixel_values = torch.randn(200, 768)
        grid_thw = torch.tensor([
            [1, 5, 10],  # 50 patches
            [1, 5, 10],  # 50 patches
            [1, 5, 10],  # 50 patches
            [1, 5, 10],  # 50 patches
        ])
        image_assignments = [[0, 2], [1, 3]]

        local_pix, local_grid, local_indices = prepare_local_vision_inputs(
            pixel_values, grid_thw, image_assignments, dp_rank=0
        )
        assert local_pix.shape[0] == 100
        assert local_grid.shape[0] == 2
        assert local_indices == [0, 2]
        expected = torch.cat([pixel_values[0:50], pixel_values[100:150]], dim=0)
        assert torch.allclose(local_pix, expected)

    def test_empty_rank(self):
        pixel_values = torch.randn(100, 768)
        grid_thw = torch.tensor([[1, 10, 10]])
        image_assignments = [[0], []]

        local_pix, local_grid, local_indices = prepare_local_vision_inputs(
            pixel_values, grid_thw, image_assignments, dp_rank=1
        )
        assert local_pix.shape[0] == 0
        assert local_grid.shape[0] == 0
        assert local_indices == []

    def test_grid_thw_preserved(self):
        pixel_values = torch.randn(150, 768)
        grid_thw = torch.tensor([
            [1, 5, 5],   # 25 patches
            [2, 5, 5],   # 50 patches
            [3, 5, 5],   # 75 patches
        ])
        image_assignments = [[0, 2], [1]]

        _, local_grid, _ = prepare_local_vision_inputs(
            pixel_values, grid_thw, image_assignments, dp_rank=0
        )
        assert local_grid.shape == (2, 3)
        assert torch.equal(local_grid[0], grid_thw[0])
        assert torch.equal(local_grid[1], grid_thw[2])


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_workflow(self):
        grid_thw = torch.tensor([
            [1, 4, 4],   # 16 patches
            [1, 8, 8],   # 64 patches
            [1, 4, 4],   # 16 patches
            [1, 6, 6],   # 36 patches
            [1, 4, 4],   # 16 patches
        ])
        total_patches = 16 + 64 + 16 + 36 + 16  # 148
        pixel_values = torch.randn(total_patches, 768)

        patch_counts = get_image_patch_counts(grid_thw)
        assert patch_counts == [16, 64, 16, 36, 16]

        assignments, loads = assign_images_to_dp_ranks(patch_counts, dp_size=2)
        all_assigned = []
        for a in assignments:
            all_assigned.extend(a)
        assert sorted(all_assigned) == [0, 1, 2, 3, 4]

        total_local_patches = 0
        for rank in range(2):
            local_pix, local_grid, local_indices = prepare_local_vision_inputs(
                pixel_values, grid_thw, assignments, dp_rank=rank
            )
            expected_patches = sum(patch_counts[i] for i in local_indices)
            assert local_pix.shape[0] == expected_patches
            assert local_grid.shape[0] == len(local_indices)
            total_local_patches += local_pix.shape[0]

        assert total_local_patches == total_patches

    def test_same_size_images(self):
        num_images = 50
        patch_per_image = 64
        grid_thw = torch.tensor([[1, 8, 8]] * num_images)
        total_patches = num_images * patch_per_image
        _ = torch.randn(total_patches, 768)

        patch_counts = get_image_patch_counts(grid_thw)
        assert all(c == 64 for c in patch_counts)

        assignments, loads = assign_images_to_dp_ranks(patch_counts, dp_size=4)
        for rank in range(4):
            assert 12 <= len(assignments[rank]) <= 13
        for load in loads:
            assert load in [768, 832]
