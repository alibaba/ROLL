import pytest
import torch
import sys
from unittest.mock import Mock, patch, MagicMock

# Mock vllm modules before importing
mock_vllm = Mock()
mock_vllm.__version__ = "0.8.4"
sys.modules['vllm'] = mock_vllm
sys.modules['vllm.sampling_params'] = Mock()
sys.modules['vllm.beam_search'] = Mock()
sys.modules['vllm.lora'] = Mock()
sys.modules['vllm.lora.request'] = Mock()
sys.modules['vllm.utils'] = Mock()
sys.modules['roll.third_party.vllm'] = Mock()

# Create mock classes
class MockRequestOutput:
    def __init__(self):
        self.request_id = "test_request"
        self.outputs = [Mock()]
        self.outputs[0].token_ids = [100, 200, 300]
        self.outputs[0].finish_reason = "length"
        self.outputs[0].logprobs = None
        self.finished = True

class MockSamplingParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.n = kwargs.get('n', 1)
        self.max_tokens = kwargs.get('max_tokens', 50)

class MockBeamSearchParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.beam_width = kwargs.get('beam_width', 1)
        self.max_tokens = kwargs.get('max_tokens', 50)

class MockBeamSearchSequence:
    def __init__(self, tokens, logprobs, cum_logprob):
        self.tokens = tokens
        self.logprobs = logprobs
        self.cum_logprob = cum_logprob

class MockBeamSearchOutput:
    def __init__(self, sequences):
        self.sequences = sequences

class MockLoRARequest:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

# Set up the mocks
sys.modules['vllm'].RequestOutput = MockRequestOutput
sys.modules['vllm'].SamplingParams = MockSamplingParams
sys.modules['vllm.sampling_params'].RequestOutputKind = Mock()
sys.modules['vllm.sampling_params'].BeamSearchParams = MockBeamSearchParams
sys.modules['vllm.beam_search'].BeamSearchOutput = MockBeamSearchOutput
sys.modules['vllm.beam_search'].BeamSearchSequence = MockBeamSearchSequence
sys.modules['vllm.lora.request'].LoRARequest = MockLoRARequest
sys.modules['vllm.utils'].random_uuid = Mock(return_value="test_uuid")

# Now import the actual modules
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.vllm_strategy import VllmStrategy
from roll.distributed.executor.worker import Worker


class TestVllmStrategyBeamSearch:
    """Test cases for VllmStrategy beam search functionality."""

    @pytest.fixture
    def mock_worker(self):
        """Create a mock worker for testing."""
        worker = Mock(spec=Worker)
        worker.pipeline_config = Mock()
        worker.pipeline_config.seed = 42
        worker.worker_config = Mock()
        worker.worker_config.strategy_args = Mock()
        worker.worker_config.strategy_args.strategy_config = {}
        worker.worker_config.model_args = Mock()
        worker.worker_config.model_args.model_name_or_path = "test_model"
        worker.worker_config.model_args.dtype = "fp16"
        worker.worker_config.model_args.lora_target = None
        worker.get_free_port = Mock(return_value=12345)
        worker.rank = 0
        worker.world_size = 1
        worker.rank_info = Mock()
        worker.rank_info.dp_rank = 0
        worker.rank_info.dp_size = 1
        return worker

    @pytest.fixture
    def vllm_strategy(self, mock_worker):
        """Create VllmStrategy instance for testing."""
        strategy = VllmStrategy(mock_worker)

        # Mock the model and tokenizer
        strategy.model = Mock()
        strategy.tokenizer = Mock()
        strategy.tokenizer.pad_token_id = 0
        strategy.is_lora = False
        strategy.is_model_in_gpu = True

        return strategy

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        batch_size = 2
        seq_length = 10

        # Create sample input tensors
        input_ids = torch.randint(1, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        batch = DataProto.from_single_dict({
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })

        return batch

    def test_should_use_beam_search_detection(self, vllm_strategy):
        """Test beam search detection logic."""

        # Test with num_beams > 1
        config_with_beam = {"num_beams": 3, "max_new_tokens": 50}
        assert vllm_strategy._should_use_beam_search(config_with_beam) is True

        # Test with use_beam_search flag
        config_with_flag = {"use_beam_search": True, "max_new_tokens": 50}
        assert vllm_strategy._should_use_beam_search(config_with_flag) is True

        # Test without beam search parameters
        config_without_beam = {"max_new_tokens": 50, "temperature": 0.8}
        assert vllm_strategy._should_use_beam_search(config_without_beam) is False

        # Test with num_beams = 1
        config_single_beam = {"num_beams": 1, "max_new_tokens": 50}
        assert vllm_strategy._should_use_beam_search(config_single_beam) is False

    def test_generate_with_beam_search_success(self, vllm_strategy, sample_batch):
        """Test successful beam search generation."""
        generation_config = {"num_beams": 3, "max_new_tokens": 50}

        # Create mock beam search outputs
        beam_width = 3
        batch_size = 2

        beam_search_outputs = []
        for batch_idx in range(batch_size):
            sequences = []
            for beam_idx in range(beam_width):
                # Include prompt + generated tokens
                prompt_length = 10
                generated_tokens = [100 + beam_idx, 200 + beam_idx, 300 + beam_idx]
                full_tokens = list(range(prompt_length)) + generated_tokens

                sequence = MockBeamSearchSequence(
                    tokens=full_tokens,
                    logprobs=[],
                    cum_logprob=-1.0 * beam_idx
                )
                sequences.append(sequence)

            output = MockBeamSearchOutput(sequences=sequences)
            beam_search_outputs.append(output)

        # Mock the beam_search method
        vllm_strategy.model.beam_search = Mock(return_value=beam_search_outputs)

        # Mock breakpoint to avoid actual debugging
        with patch('builtins.breakpoint'):
            result = vllm_strategy.generate(sample_batch, generation_config)

        # Verify beam_search was called
        vllm_strategy.model.beam_search.assert_called_once()

        # Check result shape
        assert result.shape[0] == batch_size * beam_width  # 2 * 3 = 6
        assert result.shape[1] >= 13  # prompt_length + generated_tokens

    def test_generate_with_beam_search_multimodal(self, vllm_strategy):
        """Test beam search generation with multimodal data."""
        generation_config = {"num_beams": 2, "max_new_tokens": 30}

        # Create multimodal batch
        multimodal_data = [
            {
                "prompt_token_ids": [1, 2, 3, 4, 5],
                "multi_modal_data": {"image": "test_image.jpg"}
            },
            {
                "prompt_token_ids": [6, 7, 8, 9, 10],
                "multi_modal_data": {"image": "test_image2.jpg"}
            }
        ]

        # Create a batch with dummy tensors to satisfy DataProto requirements
        batch = DataProto.from_single_dict({
            "input_ids": torch.randint(1, 1000, (2, 5)),
            "attention_mask": torch.ones(2, 5)
        })
        batch.non_tensor_batch["multi_modal_data"] = multimodal_data

        # Create mock beam search outputs
        beam_search_outputs = []
        for batch_idx in range(2):
            sequences = []
            for beam_idx in range(2):
                prompt_length = 5
                generated_tokens = [100 + beam_idx, 200 + beam_idx]
                full_tokens = multimodal_data[batch_idx]["prompt_token_ids"] + generated_tokens

                sequence = MockBeamSearchSequence(
                    tokens=full_tokens,
                    logprobs=[],
                    cum_logprob=-1.0 * beam_idx
                )
                sequences.append(sequence)

            output = MockBeamSearchOutput(sequences=sequences)
            beam_search_outputs.append(output)

        # Mock the beam_search method
        vllm_strategy.model.beam_search = Mock(return_value=beam_search_outputs)

        # Mock breakpoint to avoid actual debugging
        with patch('builtins.breakpoint'):
            result = vllm_strategy.generate(batch, generation_config)

        # Verify beam_search was called with correct prompts
        vllm_strategy.model.beam_search.assert_called_once()
        call_args = vllm_strategy.model.beam_search.call_args
        assert call_args[1]['prompts'] == multimodal_data

        # Check result shape
        assert result.shape[0] == 4  # batch_size * beam_width
