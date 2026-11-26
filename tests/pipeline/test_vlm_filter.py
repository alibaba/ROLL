"""Unit tests for VLM filter_overlong_prompts function."""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import datasets
from PIL import Image


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
        from roll.datasets.vlm_dataset_utils import filter_overlong_prompts

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
        from roll.datasets.vlm_dataset_utils import filter_overlong_prompts

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
        from roll.datasets.vlm_dataset_utils import filter_overlong_prompts

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
        from roll.datasets.vlm_dataset_utils import filter_overlong_prompts

        # Make processor raise an exception for some samples
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise ValueError("Parse error")
            return {"input_ids": [[1, 2, 3]]}

        mock_processor.side_effect = side_effect

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
        from roll.datasets.vlm_dataset_utils import filter_overlong_prompts

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
        from roll.datasets.vlm_dataset_utils import filter_overlong_prompts

        mock_image = Image.new("RGB", (28, 28))
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
        from roll.datasets.vlm_dataset_utils import filter_overlong_prompts

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
        from roll.datasets.vlm_dataset_utils import filter_overlong_prompts

        mock_image = Image.new("RGB", (28, 28))
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
        data_args.image_min_pixels = 784
        data_args.image_max_pixels = 3136
        return data_args

    def test_filter_disabled(self, mock_data_args):
        """Test that filtering can be disabled via config."""
        from roll.datasets.vlm_dataset_utils import get_vlm_dataset
        from roll.pipeline.rlvr.rlvr_config import VLMFilterConfig

        vlm_filter = VLMFilterConfig(enable=False)

        # Mock the dependencies
        with patch('roll.datasets.vlm_dataset_utils.get_dataset') as mock_get_dataset:
            mock_get_dataset.return_value = datasets.Dataset.from_dict({
                "prompt": ["test"],
                "images": [[None]],
                "reward_model": [{"ground_truth": "answer"}],
                "data_source": ["test"],
            })

            with patch('roll.datasets.vlm_dataset_utils.encode_function') as mock_encode:
                mock_encode.return_value = {
                    "tag": ["test"],
                    "images": [[None]],
                    "prompt": ["test"],
                    "ground_truth": ["answer"],
                    "reward_model": [{"ground_truth": "answer"}],
                    "image_flag": [False],
                }

                mock_processor = MagicMock()
                mock_processor.tokenizer = MagicMock()
                mock_processor.tokenizer.pad_token = "<pad>"
                mock_processor.return_value = {"input_ids": [list(range(10))]}

                result = get_vlm_dataset(
                    data_args=mock_data_args,
                    encode_function=lambda *args, **kwargs: mock_encode.return_value,
                    processor=mock_processor,
                    get_eval=False,
                    max_prompt_length=5,
                    vlm_filter=vlm_filter,
                )

                # When disabled, dataset should not be filtered
                assert result["prompt"] == ["test"]
                mock_processor.assert_not_called()


class TestPipelineDataKwargs:
    @pytest.mark.parametrize("is_val", [False, True])
    def test_default_getter_forwards_filter_settings(self, is_val):
        from roll.datasets import vlm_dataset_utils as utils
        from roll.pipeline.rlvr.rlvr_config import VLMFilterConfig

        data_args = SimpleNamespace(
            custom_data_kwargs_func=None, image_min_pixels=784, image_max_pixels=3136
        )
        dataset = datasets.Dataset.from_dict({"prompt": ["kept prompt"]})
        tokenizer, processor = MagicMock(), MagicMock()
        config = VLMFilterConfig(num_workers=1)
        with patch.object(utils, "get_vlm_dataset", return_value=dataset) as getter:
            result = utils.create_pipeline_data_kwargs(
                data_args, tokenizer, processor, is_val=is_val,
                max_prompt_length=17, vlm_filter=config,
            )

        getter.assert_called_once_with(
            data_args, utils.encode_function, processor, get_eval=is_val,
            max_prompt_length=17, vlm_filter=config,
        )
        assert result["dataset"] is dataset
        assert result["dataset"]["prompt"] == ["kept prompt"]
        assert result["collect_fn_kwargs"]["image_sample_kwargs"] == {
            "min_pixels": 784, "max_pixels": 3136
        }

    @pytest.mark.parametrize("is_val", [False, True])
    @pytest.mark.parametrize("import_by_name", [False, True])
    def test_custom_getter_preserves_existing_signature(self, is_val, import_by_name):
        from roll.datasets import vlm_dataset_utils as utils
        from roll.pipeline.rlvr.rlvr_config import VLMFilterConfig

        dataset = datasets.Dataset.from_dict({"prompt": ["custom prompt"]})
        tokenizer, processor = MagicMock(), MagicMock()
        calls = []

        # Deliberately has no **kwargs: new filtering options must not reach custom getters.
        def custom_getter(data_args, passed_tokenizer, passed_processor, is_val=False):
            calls.append((data_args, passed_tokenizer, passed_processor, is_val))
            return {"dataset": dataset, "collect_fn_kwargs": {"prompt_key": "prompt"}}

        data_args = SimpleNamespace(
            custom_data_kwargs_func="custom.module.get_data" if import_by_name else custom_getter
        )
        with patch.object(utils, "safe_import_class", return_value=custom_getter) as importer:
            result = utils.create_pipeline_data_kwargs(
                data_args, tokenizer, processor, is_val=is_val,
                max_prompt_length=1, vlm_filter=VLMFilterConfig(num_workers=1),
            )

        assert calls == [(data_args, tokenizer, processor, is_val)]
        assert result["dataset"]["prompt"] == ["custom prompt"]
        if import_by_name:
            importer.assert_called_once_with("custom.module.get_data")
        else:
            importer.assert_not_called()


@pytest.mark.parametrize("get_eval", [False, True])
def test_filter_ignores_existing_cache_and_recomputes_for_changed_limit(tmp_path, get_eval):
    from roll.datasets import vlm_dataset_utils as utils
    from roll.pipeline.rlvr.rlvr_config import VLMFilterConfig

    cache_path = tmp_path / ("val" if get_eval else "train")
    datasets.Dataset.from_dict({"prompt": ["stale cached prompt"]}).save_to_disk(str(cache_path))
    raw_dataset = datasets.Dataset.from_dict({
        "prompt": ["one two", "one two three four"],
        "images": [[], []],
        "reward_model": [{"ground_truth": "answer"}, {"ground_truth": "answer"}],
        "data_source": ["test", "test"],
    })
    data_args = SimpleNamespace(
        cache_path=str(tmp_path), preprocessing_num_workers=1,
        image_min_pixels=784, image_max_pixels=3136,
    )
    processor = MagicMock()
    processor.side_effect = lambda text, **kwargs: {"input_ids": [list(range(len(text.split())))]}

    def encode(data, processor, prompt_getter, ground_truth_getter, image_getter, tag_getter, **kwargs):
        return {
            "prompt": prompt_getter(data), "images": image_getter(data),
            "ground_truth": ground_truth_getter(data), "tag": tag_getter(data),
            "reward_model": data["reward_model"], "image_flag": [False] * len(data["prompt"]),
        }

    with (
        patch.object(utils, "get_dataset", return_value=raw_dataset) as source,
        patch.object(utils, "load_from_disk") as cache_loader,
        patch.object(datasets.Dataset, "save_to_disk") as cache_saver,
    ):
        results = [
            utils.get_vlm_dataset(
                data_args, encode, processor, get_eval=get_eval,
                max_prompt_length=limit, vlm_filter=VLMFilterConfig(num_workers=1),
            )
            for limit in (2, 4)
        ]

    assert results[0]["prompt"] == ["one two"]
    assert results[1]["prompt"] == ["one two", "one two three four"]
    assert source.call_count == 2
    cache_loader.assert_not_called()
    cache_saver.assert_not_called()
    assert datasets.load_from_disk(str(cache_path))["prompt"] == ["stale cached prompt"]


def test_format_prompt_preserves_original_question():
    from roll.datasets.vlm_dataset_utils import format_prompt

    question = "  Answer plainly.\nKeep {braces} and 中文 exactly.  "
    processor = MagicMock()
    processor.apply_chat_template.side_effect = lambda messages, **kwargs: messages[0]["content"][0]["text"]

    result = format_prompt(question, processor, use_image=False)

    assert result == question
    processor.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": [{"type": "text", "text": question}]}],
        tokenize=False, add_generation_prompt=True,
    )


def test_filter_uses_collator_image_sampling_and_disables_second_resize():
    from roll.datasets.vlm_dataset_utils import filter_overlong_prompts

    dataset = datasets.Dataset.from_dict({
        "prompt": ["image prompt", "text prompt"],
        "images": [[Image.new("RGB", (56, 56))], []],
        "image_flag": [True, False],
    })
    resized_image = Image.new("RGB", (28, 28))
    processor = MagicMock()
    processor.image_processor.patch_size = 14
    calls = []

    def process(text, images, **kwargs):
        calls.append((text, images, kwargs))
        # A second resize or use of the original image must fail the length check.
        length = 4 if images is None or (images == [resized_image] and kwargs == {"do_resize": False}) else 10
        return {"input_ids": [list(range(length))]}

    processor.side_effect = process
    with patch("roll.datasets.collator.load_images", return_value=([resized_image], [])) as image_loader:
        result = filter_overlong_prompts(
            dataset, processor, max_prompt_length=4, num_workers=1,
            image_sample_kwargs={"min_pixels": 784, "max_pixels": 3136},
        )

    assert result["prompt"] == ["image prompt", "text prompt"]
    image_loader.assert_called_once()
    args, kwargs = image_loader.call_args
    assert len(args[0]) == 1
    assert args[0][0].size == (56, 56)
    assert args[1] == [{}]
    assert kwargs == {"image_patch_size": 14, "min_pixels": 784, "max_pixels": 3136}
    assert calls == [
        ("image prompt", [resized_image], {"do_resize": False}),
        ("text prompt", None, {}),
    ]


def test_filter_with_real_image_sampling_keeps_exact_token_boundary():
    from roll.datasets.vlm_dataset_utils import filter_overlong_prompts

    dataset = datasets.Dataset.from_dict({
        "prompt": ["boundary image", "oversized image"],
        "images": [[Image.new("RGB", (56, 56))], [Image.new("RGB", (112, 112))]],
        "image_flag": [True, True],
    })
    processor = MagicMock()
    processor.image_processor.patch_size = 14
    sampled_inputs = []

    def process(text, images, **kwargs):
        image = images[0]
        sampled_inputs.append((text, image.size, kwargs))
        image_tokens = image.width * image.height // (28 * 28)
        return {"input_ids": [list(range(image_tokens))]}

    processor.side_effect = process
    result = filter_overlong_prompts(
        dataset, processor, max_prompt_length=4, num_workers=1,
        image_sample_kwargs={"min_pixels": 784, "max_pixels": 7056},
    )

    assert result["prompt"] == ["boundary image"]
    assert sampled_inputs == [
        ("boundary image", (56, 56), {"do_resize": False}),
        ("oversized image", (84, 84), {"do_resize": False}),
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
