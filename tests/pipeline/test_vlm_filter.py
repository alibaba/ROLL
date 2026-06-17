"""Unit tests for VLM filter_overlong_prompts function."""

import pytest
from unittest.mock import MagicMock, patch
import datasets


class TestFilterOverlongPrompts:
    """Tests for filter_overlong_prompts function."""

    @pytest.fixture
    def mock_processor(self):
        """Create a mock processor for testing."""
        processor = MagicMock()
        processor.apply_chat_template = MagicMock(return_value="formatted prompt")
        processor.return_value = {"input_ids": [[1, 2, 3, 4, 5]]}
        return processor

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        return datasets.Dataset.from_dict({
            "prompt": ["short prompt", "medium length prompt here", "this is a very long prompt that exceeds the limit"],
            "images": [[None], [None], [None]],
            "image_flag": [False, False, False],
            "tag": ["test", "test", "test"],
        })

    def test_filter_empty_dataset(self, mock_processor):
        """Test that empty dataset is handled correctly."""
        from roll.pipeline.rlvr.rlvr_vlm_pipeline import filter_overlong_prompts

        empty_dataset = datasets.Dataset.from_dict({
            "prompt": [],
            "images": [],
            "image_flag": [],
        })

        result = filter_overlong_prompts(
            dataset=empty_dataset,
            processor=mock_processor,
            max_prompt_length=10,
        )

        assert len(result) == 0

    def test_filter_keeps_short_prompts(self, mock_processor, sample_dataset):
        """Test that prompts within limit are kept."""
        from roll.pipeline.rlvr.rlvr_vlm_pipeline import filter_overlong_prompts

        # All prompts return 5 tokens, max is 10, so all should pass
        result = filter_overlong_prompts(
            dataset=sample_dataset,
            processor=mock_processor,
            max_prompt_length=10,
            num_workers=1,
        )

        assert len(result) == 3

    def test_filter_removes_long_prompts(self, mock_processor, sample_dataset):
        """Test that prompts exceeding limit are filtered out."""
        from roll.pipeline.rlvr.rlvr_vlm_pipeline import filter_overlong_prompts

        # All prompts return 5 tokens, max is 3, so all should be filtered
        result = filter_overlong_prompts(
            dataset=sample_dataset,
            processor=mock_processor,
            max_prompt_length=3,
            num_workers=1,
        )

        assert len(result) == 0

    def test_filter_handles_parse_errors(self, mock_processor, sample_dataset):
        """Test that parse errors result in filtering."""
        from roll.pipeline.rlvr.rlvr_vlm_pipeline import filter_overlong_prompts

        # Make processor raise an exception for some samples
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise ValueError("Parse error")
            return {"input_ids": [[1, 2, 3]]}

        mock_processor.return_value = side_effect

        result = filter_overlong_prompts(
            dataset=sample_dataset,
            processor=mock_processor,
            max_prompt_length=10,
            num_workers=1,
        )

        # Second sample should be filtered due to error, 2 remain
        assert len(result) == 2

    def test_filter_with_custom_keys(self, mock_processor):
        """Test that custom key names are respected."""
        from roll.pipeline.rlvr.rlvr_vlm_pipeline import filter_overlong_prompts

        dataset = datasets.Dataset.from_dict({
            "custom_prompt": ["test prompt"],
            "custom_images": [[None]],
            "custom_image_flag": [False],
        })

        result = filter_overlong_prompts(
            dataset=dataset,
            processor=mock_processor,
            max_prompt_length=10,
            prompt_key="custom_prompt",
            image_key="custom_images",
            image_flag_key="custom_image_flag",
            num_workers=1,
        )

        assert len(result) == 1

    def test_filter_with_valid_images(self, mock_processor):
        """Test that samples with valid images are processed correctly."""
        from roll.pipeline.rlvr.rlvr_vlm_pipeline import filter_overlong_prompts

        mock_image = MagicMock()
        dataset = datasets.Dataset.from_dict({
            "prompt": ["prompt with image"],
            "images": [[mock_image]],
            "image_flag": [True],
        })

        result = filter_overlong_prompts(
            dataset=dataset,
            processor=mock_processor,
            max_prompt_length=10,
            num_workers=1,
        )

        # Verify processor was called with images
        mock_processor.assert_called()
        assert len(result) == 1

    def test_filter_boundary_condition(self, mock_processor):
        """Test that prompts exactly at max_prompt_length are kept."""
        from roll.pipeline.rlvr.rlvr_vlm_pipeline import filter_overlong_prompts

        dataset = datasets.Dataset.from_dict({
            "prompt": ["exact length"],
            "images": [[None]],
            "image_flag": [False],
        })

        # Token length is 5, max is 5, should be kept (<=)
        result = filter_overlong_prompts(
            dataset=dataset,
            processor=mock_processor,
            max_prompt_length=5,
            num_workers=1,
        )

        assert len(result) == 1

    def test_filter_single_image_not_in_list(self, mock_processor):
        """Test handling of single image not wrapped in list."""
        from roll.pipeline.rlvr.rlvr_vlm_pipeline import filter_overlong_prompts

        mock_image = MagicMock()
        dataset = datasets.Dataset.from_dict({
            "prompt": ["prompt"],
            "images": [mock_image],  # Single image, not in list
            "image_flag": [True],
        })

        result = filter_overlong_prompts(
            dataset=dataset,
            processor=mock_processor,
            max_prompt_length=10,
            num_workers=1,
        )

        assert len(result) == 1


class TestVLMFilterConfig:
    """Tests for VLMFilterConfig dataclass."""

    def test_default_values(self):
        """Test that default values are correct."""
        from roll.pipeline.rlvr.rlvr_config import VLMFilterConfig

        config = VLMFilterConfig()

        assert config.enable is True
        assert config.num_workers is None
        assert config.prompt_key == "prompt"
        assert config.image_key == "images"
        assert config.image_flag_key == "image_flag"

    def test_custom_values(self):
        """Test that custom values can be set."""
        from roll.pipeline.rlvr.rlvr_config import VLMFilterConfig

        config = VLMFilterConfig(
            enable=False,
            num_workers=4,
            prompt_key="custom_prompt",
            image_key="custom_images",
            image_flag_key="custom_flag",
        )

        assert config.enable is False
        assert config.num_workers == 4
        assert config.prompt_key == "custom_prompt"
        assert config.image_key == "custom_images"
        assert config.image_flag_key == "custom_flag"


class TestGetVLMDataset:
    """Tests for get_vlm_dataset function integration with VLMFilterConfig."""

    @pytest.fixture
    def mock_data_args(self):
        """Create mock data args."""
        data_args = MagicMock()
        data_args.cache_path = None
        data_args.preprocessing_num_workers = 1
        return data_args

    def test_filter_disabled(self, mock_data_args):
        """Test that filtering can be disabled via config."""
        from roll.pipeline.rlvr.rlvr_vlm_pipeline import get_vlm_dataset
        from roll.pipeline.rlvr.rlvr_config import VLMFilterConfig

        vlm_filter = VLMFilterConfig(enable=False)

        # Mock the dependencies
        with patch('roll.pipeline.rlvr.rlvr_vlm_pipeline.get_dataset') as mock_get_dataset:
            mock_get_dataset.return_value = datasets.Dataset.from_dict({
                "prompt": ["test"],
                "images": [[None]],
                "reward_model": [{"ground_truth": "answer"}],
                "data_source": ["test"],
            })

            with patch('roll.pipeline.rlvr.rlvr_vlm_pipeline.encode_function') as mock_encode:
                mock_encode.return_value = {
                    "tag": "test",
                    "images": [[None]],
                    "prompt": ["test"],
                    "ground_truth": ["answer"],
                    "reward_model": [{"ground_truth": "answer"}],
                    "image_flag": [False],
                }

                mock_processor = MagicMock()
                mock_processor.tokenizer = MagicMock()
                mock_processor.tokenizer.pad_token = "<pad>"

                result = get_vlm_dataset(
                    data_args=mock_data_args,
                    encode_function=lambda *args, **kwargs: mock_encode.return_value,
                    processor=mock_processor,
                    get_eval=False,
                    max_prompt_length=5,
                    vlm_filter=vlm_filter,
                )

                # When disabled, dataset should not be filtered
                assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
