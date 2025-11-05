"""Tests for automatic padding functionality in DynamicSamplingScheduler."""

import pytest
import torch
from unittest.mock import Mock, MagicMock

from roll.distributed.scheduler.generate_scheduler import DynamicSamplingScheduler
from roll.distributed.scheduler.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto


class TestDynamicSamplingSchedulerPadding:
    """Test cases for padding functionality in DynamicSamplingScheduler."""

    @pytest.fixture
    def mock_scheduler(self):
        """Create a mock DynamicSamplingScheduler for testing."""
        scheduler = Mock(spec=DynamicSamplingScheduler)
        scheduler.actor_cluster = Mock()
        scheduler.actor_cluster.dp_size = 4
        scheduler.actor_cluster.generate = Mock()
        scheduler.generation_config = {"num_return_sequences": 1}
        scheduler.is_val = False
        scheduler.batch_size = 7
        scheduler.collect_fn = Mock()
        scheduler.get_next_dataset_item = Mock()
        scheduler.reward_scheduler = Mock()
        scheduler.reward_clusters = []
        scheduler.pipeline_config = {}
        scheduler.query_filter_fn = Mock(return_value=True)
        scheduler.query_filter_count = 0
        scheduler.response_filter_count = 0
        scheduler.reset_status = Mock()
        return scheduler

    def test_padding_when_batch_not_divisible_by_dp_size(self, mock_scheduler):
        """Test padding when batch_size is not divisible by dp_size."""
        # Create test data
        batch_size = 7
        dp_size = 4
        
        # Create actual DataProto for testing
        test_data = DataProto.from_single_dict({
            "input_ids": torch.randint(0, 1000, (batch_size, 10)),
            "attention_mask": torch.ones(batch_size, 10),
            "position_ids": torch.arange(10).unsqueeze(0).repeat(batch_size, 1)
        })
        
        # Test padding logic with actual data
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_data, dp_size)
        
        # Verify padding was applied
        assert pad_size == 1  # 7 % 4 = 3, so pad_size = 4 - 3 = 1
        assert len(gen_batch_padded) == 8

    def test_no_padding_when_batch_divisible_by_dp_size(self, mock_scheduler):
        """Test no padding when batch_size is already divisible by dp_size."""
        batch_size = 8
        dp_size = 4
        
        # Create actual DataProto for testing
        test_data = DataProto.from_single_dict({
            "input_ids": torch.randint(0, 1000, (batch_size, 10)),
            "attention_mask": torch.ones(batch_size, 10),
            "position_ids": torch.arange(10).unsqueeze(0).repeat(batch_size, 1)
        })
        
        # Test padding logic with actual data
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_data, dp_size)
        
        # Verify no padding was needed
        assert pad_size == 0  # 8 % 4 = 0, so no padding needed
        assert len(gen_batch_padded) == 8

    def test_unpadding_restores_original_size(self):
        """Test that unpadding restores the original batch size."""
        # Create test data
        original_size = 7
        pad_size = 1
        
        # Create actual batch with padding
        original_data = DataProto.from_single_dict({
            "input_ids": torch.randint(0, 1000, (original_size, 10)),
            "attention_mask": torch.ones(original_size, 10)
        })
        
        # Pad the data
        padded_data, _ = pad_dataproto_to_divisor(original_data, 4)
        
        # Test unpadding
        result = unpad_dataproto(padded_data, pad_size)
        
        # Verify unpadding was applied
        assert len(result) == original_size

    def test_padding_preserves_data_integrity(self):
        """Test that padding preserves data integrity."""
        # Create test DataProto
        test_data = DataProto.from_single_dict({
            "input_ids": torch.randint(0, 1000, (3, 10)),
            "attention_mask": torch.ones(3, 10)
        })
        
        # Apply padding
        padded_data, pad_size = pad_dataproto_to_divisor(test_data, 4)
        
        # Verify padded data size
        assert len(padded_data) == 4
        assert pad_size == 1
        
        # Verify data integrity using proper TensorDict methods
        assert "input_ids" in padded_data.batch.keys()
        assert "attention_mask" in padded_data.batch.keys()
        assert padded_data.batch["input_ids"].shape[0] == 4
        assert padded_data.batch["attention_mask"].shape[0] == 4

    def test_edge_case_empty_batch(self):
        """Test padding behavior with empty batch."""
        # Create empty DataProto
        empty_data = DataProto.from_single_dict({
            "input_ids": torch.empty(0, 10),
            "attention_mask": torch.empty(0, 10)
        })
        
        # Apply padding
        padded_data, pad_size = pad_dataproto_to_divisor(empty_data, 4)
        
        # Verify empty batch handling
        assert pad_size == 0
        assert len(padded_data) == 0

    def test_edge_case_single_item_batch(self):
        """Test padding behavior with single item batch."""
        # Create single item DataProto
        single_data = DataProto.from_single_dict({
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10)
        })
        
        # Apply padding
        padded_data, pad_size = pad_dataproto_to_divisor(single_data, 4)
        
        # Verify padding
        assert pad_size == 3
        assert len(padded_data) == 4

    def test_backward_compatibility(self, mock_scheduler):
        """Test that padding doesn't break existing functionality."""
        batch_size = 8  # Already divisible by dp_size
        dp_size = 4
        
        # Create actual DataProto for testing
        test_data = DataProto.from_single_dict({
            "input_ids": torch.randint(0, 1000, (batch_size, 10)),
            "attention_mask": torch.ones(batch_size, 10),
            "position_ids": torch.arange(10).unsqueeze(0).repeat(batch_size, 1)
        })
        
        # Test that existing flow still works
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_data, dp_size)
        
        # Verify no padding was applied and flow is unchanged
        assert pad_size == 0
        assert len(gen_batch_padded) == batch_size