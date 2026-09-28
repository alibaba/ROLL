import asyncio
import importlib
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import torch

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

class MockTokensPrompt(dict):
    pass


from roll.distributed.scheduler.protocol import DataProto


def _install_mock_vllm_modules(monkeypatch):
    mock_vllm = ModuleType("vllm")
    mock_vllm.__path__ = []
    mock_vllm.__version__ = "0.11.0"
    mock_vllm.RequestOutput = MockRequestOutput
    mock_vllm.SamplingParams = MockSamplingParams

    sampling_params = ModuleType("vllm.sampling_params")
    sampling_params.RequestOutputKind = Mock()
    sampling_params.BeamSearchParams = MockBeamSearchParams

    beam_search = ModuleType("vllm.beam_search")
    beam_search.BeamSearchOutput = MockBeamSearchOutput
    beam_search.BeamSearchSequence = MockBeamSearchSequence

    lora = ModuleType("vllm.lora")
    lora.__path__ = []
    lora_request = ModuleType("vllm.lora.request")
    lora_request.LoRARequest = MockLoRARequest

    inputs = ModuleType("vllm.inputs")
    inputs.__path__ = []
    inputs.TokensPrompt = MockTokensPrompt
    inputs_data = ModuleType("vllm.inputs.data")
    inputs_data.TokensPrompt = MockTokensPrompt

    utils = ModuleType("vllm.utils")
    utils.random_uuid = Mock(return_value="test_uuid")

    gdn_patcher = ModuleType("roll.third_party.vllm.gdn_patcher")
    gdn_patcher.patch_gdn_attention = Mock()

    monkeypatch.setitem(sys.modules, "vllm", mock_vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)
    monkeypatch.setitem(sys.modules, "vllm.beam_search", beam_search)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", lora_request)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)
    monkeypatch.setitem(sys.modules, "vllm.inputs.data", inputs_data)
    monkeypatch.setitem(sys.modules, "vllm.utils", utils)
    monkeypatch.setitem(sys.modules, "roll.third_party.vllm", Mock())
    monkeypatch.setitem(sys.modules, "roll.third_party.vllm.gdn_patcher", gdn_patcher)


@pytest.fixture
def vllm_strategy_module(monkeypatch):
    module_name = "roll.distributed.strategy.vllm_strategy"
    original_module = sys.modules.pop(module_name, None)
    _install_mock_vllm_modules(monkeypatch)
    module = importlib.import_module(module_name)
    try:
        yield module
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module


class TestVllmStrategyBeamSearch:
    """Test cases for VllmStrategy beam search functionality."""

    @pytest.fixture
    def mock_worker(self):
        """Create a mock worker for testing."""
        worker = Mock()
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
    def vllm_strategy(self, vllm_strategy_module, mock_worker):
        """Create VllmStrategy instance for testing."""
        strategy = vllm_strategy_module.VllmStrategy(mock_worker)

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
        beam_width = 3
        batch_size = 2

        # Mock beam_search as an async generator that yields RequestOutput-like objects
        # _generate_with_beam_search accesses .outputs[].token_ids, not .sequences[]
        async def mock_beam_search(prompt, request_id, params):
            output = MagicMock()
            output.outputs = []
            for beam_idx in range(beam_width):
                completion = MagicMock()
                completion.token_ids = [100 + beam_idx, 200 + beam_idx, 300 + beam_idx]
                output.outputs.append(completion)
            yield output

        vllm_strategy.model.beam_search = Mock(side_effect=mock_beam_search)

        # Mock breakpoint to avoid actual debugging
        with patch('builtins.breakpoint'):
            result = asyncio.run(
                vllm_strategy.generate(sample_batch, generation_config)
            )

        # beam_search is called once per prompt via asyncio.gather
        assert vllm_strategy.model.beam_search.call_count == batch_size

        # Check result shape: (batch_size * beam_width, prompt_len + max_output_len)
        assert result.shape[0] == batch_size * beam_width  # 2 * 3 = 6
        assert result.shape[1] >= 13  # prompt_length (10) + generated_tokens (3)

    def test_generate_with_beam_search_multimodal(self, vllm_strategy):
        """Test beam search generation with multimodal data."""
        generation_config = {"num_beams": 2, "max_new_tokens": 30}
        beam_width = 2
        batch_size = 2

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

        # Mock beam_search as an async generator that yields RequestOutput-like objects
        async def mock_beam_search(prompt, request_id, params):
            output = MagicMock()
            output.outputs = []
            for beam_idx in range(beam_width):
                completion = MagicMock()
                completion.token_ids = [100 + beam_idx, 200 + beam_idx]
                output.outputs.append(completion)
            yield output

        vllm_strategy.model.beam_search = Mock(side_effect=mock_beam_search)

        # Mock breakpoint to avoid actual debugging
        with patch('builtins.breakpoint'):
            result = asyncio.run(
                vllm_strategy.generate(batch, generation_config)
            )

        # beam_search is called once per prompt via asyncio.gather
        assert vllm_strategy.model.beam_search.call_count == batch_size

        # Verify each multimodal prompt was passed to beam_search
        calls = vllm_strategy.model.beam_search.call_args_list
        actual_prompts = [call[1]['prompt'] for call in calls]
        for prompt in multimodal_data:
            assert prompt in actual_prompts

        # Check result shape: (batch_size * beam_width, ...)
        assert result.shape[0] == batch_size * beam_width  # 2 * 2 = 4

    @pytest.mark.parametrize(
        ("version", "supports_external_dp", "uses_logical_nodes", "enable_expert_parallel"),
        [
            ("0.11.0", True, False, False),
            ("0.11.1", True, True, False),
            ("0.12.0", True, True, False),
            ("0.16.0.dev1+local", True, True, True),
            ("0.16.0rc2", True, True, True),
            ("0.16.0", True, True, True),
            ("0.17.0.dev1+local", True, True, True),
            ("0.17.0rc2", True, True, True),
            ("0.17.0", True, True, True),
        ],
    )
    def test_external_dp_launch_args_follow_vllm_version(
        self,
        monkeypatch,
        vllm_strategy_module,
        mock_worker,
        version,
        supports_external_dp,
        uses_logical_nodes,
        enable_expert_parallel,
    ):
        mock_worker.world_size = 2
        mock_worker.rank_info.dp_size = 2
        mock_worker.cluster_name = "actor-infer"
        mock_worker.master_port = 29500
        mock_worker.get_node_ip = Mock(return_value="127.0.0.1")
        mock_worker.worker_config.strategy_args.strategy_config = {
            "data_parallel_size": 2,
            "enable_expert_parallel": enable_expert_parallel,
        }
        mock_worker.worker_config.resource_placement_groups = [
            {"node_rank": 0, "gpu_rank": 0, "placement_group": "pg-0"}
        ]
        put = Mock()
        put.remote = AsyncMock()
        mock_worker.shared_storage.put = put

        model = Mock()
        model.get_tokenizer = AsyncMock(return_value=Mock())
        create_async_llm = AsyncMock(return_value=model)
        monkeypatch.setattr(vllm_strategy_module, "create_async_llm", create_async_llm)
        vllm_strategy_module.vllm.__version__ = version
        strategy = vllm_strategy_module.VllmStrategy(mock_worker)
        strategy._start_headless_workers = AsyncMock()

        if not supports_external_dp:
            with pytest.raises(RuntimeError, match="external data parallelism"):
                asyncio.run(strategy.initialize(None))
            return

        asyncio.run(strategy.initialize(None))
        kwargs = create_async_llm.await_args.kwargs
        assert kwargs["data_parallel_rank"] == 0
        assert kwargs["data_parallel_address"] == "127.0.0.1"
        assert kwargs["data_parallel_rpc_port"] == 12345
        assert kwargs["enable_expert_parallel"] is enable_expert_parallel
        for name in ("nnodes", "node_rank", "master_addr", "master_port"):
            assert (name in kwargs) is uses_logical_nodes

    @pytest.mark.parametrize(
        "version", ["0.11.0", "0.11.1", "0.11.1rc2", "0.12.0", "0.15.1"]
    )
    def test_ep_requires_vllm_0_16_or_later(
        self, vllm_strategy_module, mock_worker, version
    ):
        mock_worker.worker_config.strategy_args.strategy_config = {
            "enable_expert_parallel": True
        }
        mock_worker.worker_config.resource_placement_groups = [
            {"node_rank": 0, "gpu_rank": 0, "placement_group": "pg-0"}
        ]
        vllm_strategy_module.vllm.__version__ = version

        with pytest.raises(RuntimeError, match="Upgrade vLLM"):
            asyncio.run(vllm_strategy_module.VllmStrategy(mock_worker).initialize(None))

    @pytest.mark.parametrize("version", ["0.11.0"])
    def test_legacy_vllm_cross_node_tp_uses_ray(
        self, monkeypatch, vllm_strategy_module, mock_worker, version
    ):
        mock_worker.cluster_name = "actor-infer"
        mock_worker.master_port = 29500
        mock_worker.get_node_ip = Mock(return_value="127.0.0.1")
        mock_worker.worker_config.strategy_args.strategy_config = {
            "tensor_parallel_size": 2
        }
        mock_worker.worker_config.resource_placement_groups = [
            {"node_rank": 0, "gpu_rank": 0, "placement_group": "pg-0"},
            {"node_rank": 1, "gpu_rank": 0, "placement_group": "pg-1"},
        ]

        model = Mock()
        model.get_tokenizer = AsyncMock(return_value=Mock())
        create_async_llm = AsyncMock(return_value=model)
        monkeypatch.setattr(vllm_strategy_module, "create_async_llm", create_async_llm)
        vllm_strategy_module.vllm.__version__ = version
        strategy = vllm_strategy_module.VllmStrategy(mock_worker)
        strategy._start_headless_workers = AsyncMock()

        asyncio.run(strategy.initialize(None))

        assert create_async_llm.await_args.kwargs["distributed_executor_backend"] == "ray"
        strategy._start_headless_workers.assert_not_awaited()
