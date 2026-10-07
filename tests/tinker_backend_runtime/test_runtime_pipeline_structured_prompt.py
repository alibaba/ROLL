from __future__ import annotations

import asyncio
import threading
from concurrent import futures
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


def _pipeline(module, backend, tmp_path: Path):
    pipeline = module.TinkerBackendRuntimePipeline.__new__(module.TinkerBackendRuntimePipeline)
    pipeline.roll_backend = backend
    pipeline.args = SimpleNamespace(runtime_config={"generator": {"enable_thinking": False}})
    pipeline.checkpoints_base = tmp_path
    return pipeline


def test_backend_sample_tokenizes_structured_prompt_and_returns_prompt_ids(tmp_path):
    from roll.pipeline.tinker_backend_runtime import runtime_pipeline as module

    class FakeTokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            assert kwargs["tokenize"] is True
            assert kwargs["add_generation_prompt"] is True
            assert kwargs["tools"][0]["function"]["name"] == "run"
            assert kwargs["enable_thinking"] is False
            return {"input_ids": [[101, 102, 103]]}

        def decode(self, tokens, skip_special_tokens=False):
            assert tokens == [201, 202]
            return '<tool_call>{"name":"run","arguments":{"cmd":"pytest"}}</tool_call>'

    class FakeBackend:
        def __init__(self):
            self.tokenizer = FakeTokenizer()
            self.prepared = None

        def sample(self, prepared):
            self.prepared = prepared
            request_id = prepared.request_batch_slices[0][0]
            return {
                request_id: module.tinker_types.SampleOutput(
                    sequences=[
                        module.tinker_types.GeneratedSequence(
                            stop_reason="stop",
                            tokens=[201, 202],
                            logprobs=[-0.1, -0.2],
                        )
                    ],
                )
            }

    backend = FakeBackend()
    pipeline = _pipeline(module, backend, tmp_path)
    payload = {
        "prompt": module.openai_chat_completion_request(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "run", "parameters": {"type": "object"}}}],
            }
        ),
        "sampling_params": {"max_tokens": 16, "temperature": 0.3},
    }

    result = pipeline.handle_backend_sample(
        {"action_id": 7, "env_id": "env_test", "payload": payload}
    )

    assert backend.prepared is not None
    assert backend.prepared.all_env_ids == ["env_test"]
    model_input = backend.prepared.all_model_inputs[0]
    assert [chunk.tokens for chunk in model_input.chunks] == [[101, 102, 103]]
    assert result["prompt_token_ids"] == [101, 102, 103]
    sequence = result["sequences"][0]
    assert sequence["tokens"] == [201, 202]
    assert sequence["output_token_ids"] == [201, 202]
    assert sequence["logprobs"] == [-0.1, -0.2]
    assert sequence["raw_text"] == '<tool_call>{"name":"run","arguments":{"cmd":"pytest"}}</tool_call>'
    assert sequence["finish_reason"] == "tool_calls"
    assert sequence["tool_calls"][0]["function"]["name"] == "run"


def test_backend_sample_accepts_tokenized_model_input_and_echoes_prompt_ids(tmp_path):
    from roll.pipeline.tinker_backend_runtime import runtime_pipeline as module

    class FakeTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("tokenized model_input must not call chat template")

    class FakeBackend:
        tokenizer = FakeTokenizer()

        def __init__(self):
            self.prepared = None

        def sample(self, prepared):
            self.prepared = prepared
            request_id = prepared.request_batch_slices[0][0]
            return {
                request_id: module.tinker_types.SampleOutput(
                    sequences=[
                        module.tinker_types.GeneratedSequence(
                            stop_reason="stop",
                            tokens=[301],
                            output_token_ids=[301],
                            raw_text="ok",
                            text="ok",
                            logprobs=[-0.3],
                        )
                    ],
                    prompt_token_ids=[11, 12],
                )
            }

    backend = FakeBackend()
    pipeline = _pipeline(module, backend, tmp_path)
    payload = {
        "model_input": module.model_input_from_token_ids([11, 12]),
        "sampling_params": {"max_tokens": 1},
    }

    result = pipeline.handle_backend_sample({"action_id": 8, "payload": payload})

    assert backend.prepared is not None
    assert [chunk.tokens for chunk in backend.prepared.all_model_inputs[0].chunks] == [[11, 12]]
    assert result["prompt_token_ids"] == [11, 12]
    assert result["sequences"][0]["output_token_ids"] == [301]


def test_prompt_rejects_legacy_model_input_envelope():
    from roll.pipeline.tinker_backend_runtime import runtime_pipeline as module

    with pytest.raises(ValueError, match="prompt.type"):
        module._request_from_payload({"prompt": module.model_input_from_token_ids([123])})


def test_output_decode_does_not_fallback_to_unicode_codepoints(tmp_path):
    from roll.pipeline.tinker_backend_runtime import runtime_pipeline as module

    class FakeBackend:
        tokenizer = None

    pipeline = _pipeline(module, FakeBackend(), tmp_path)
    payload = {"model_input": module.model_input_from_token_ids([1])}
    data = {"sequences": [{"tokens": [65], "logprobs": [0.0], "stop_reason": "stop"}]}

    with pytest.raises(RuntimeError, match="without a tokenizer"):
        pipeline._enrich_sample_response(payload, data, prompt_token_ids=[1])


def test_score_result_echoes_exact_prompt_token_ids():
    from roll.pipeline.tinker_backend_runtime import roll_backend as module

    results = {}
    module.ROLLRuntimeBackend._fill_score_results(
        results,
        [("request_1", [11, 12, 13])],
        [[None, -0.2, -0.3]],
    )

    result = results["request_1"]
    assert result.prompt_token_ids == [11, 12, 13]
    assert result.prompt_logprobs == [None, -0.2, -0.3]


def test_vllm_generator_bypass_is_rejected():
    from roll.pipeline.tinker_backend_runtime import runtime_pipeline as module

    with pytest.raises(ValueError, match="native actor_infer router"):
        module._create_generator({"generator": {"type": "vllm"}}, "unused")


def test_native_router_generation_preserves_sampling_result():
    from roll.distributed.scheduler.protocol import DataProto
    from roll.pipeline.tinker_backend_runtime import roll_backend as module

    class FakeGeneratingArgs:
        @staticmethod
        def to_dict():
            return {
                "max_new_tokens": 128,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 50,
                "num_return_sequences": 1,
                "stop_strings": [],
                "repetition_penalty": 1.0,
            }

    class FakeRouterClient:
        def generate_request_sync(self, data, request_id, uid):
            assert request_id == "action_7"
            assert uid == "env_3"
            assert data.meta_info["generation_config"]["temperature"] == 0.3
            assert data.meta_info["generation_config"]["seed"] == 0
            assert data.meta_info["generation_config"]["num_return_sequences"] == 2
            return DataProto(
                meta_info={
                    "output_token_ids": [[201, 202], [301]],
                    "finish_reasons": ["stop", "length"],
                    "output_logprobs": [[-0.1, -0.2], [-0.3]],
                }
            )

    backend = module.ROLLRuntimeBackend.__new__(module.ROLLRuntimeBackend)
    backend.router_client = FakeRouterClient()
    backend.actor_infer = SimpleNamespace(
        worker_config=SimpleNamespace(generating_args=FakeGeneratingArgs())
    )
    backend.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)

    sequences = backend._generate_one(
        [101, 102],
        module.types.SamplingParams(
            temperature=0.3,
            max_tokens=2,
            seed=0,
            top_k=-1,
            top_p=0.9,
        ),
        2,
        request_id="action_7",
        env_id="env_3",
    )

    assert [sequence.tokens for sequence in sequences] == [[201, 202], [301]]
    assert [sequence.logprobs for sequence in sequences] == [[-0.1, -0.2], [-0.3]]
    assert [sequence.stop_reason for sequence in sequences] == ["length", "length"]


def test_rollout_sample_actions_use_existing_action_executor_concurrently():
    from roll.pipeline.tinker_backend_runtime import runtime_pipeline as module

    pipeline = module.TinkerBackendRuntimePipeline.__new__(module.TinkerBackendRuntimePipeline)
    pipeline.action_executor = futures.ThreadPoolExecutor(max_workers=2)
    entered = []
    barrier = threading.Barrier(2)

    def handle_action(action):
        entered.append(action["action_id"])
        barrier.wait(timeout=2)

    pipeline.handle_action = handle_action
    pipeline.post_action_result = lambda *args, **kwargs: None
    actions = [
        {"action_id": 1, "action_type": "sample", "env_id": "env_1", "payload": {}},
        {"action_id": 2, "action_type": "sample", "env_id": "env_2", "payload": {}},
    ]

    async def run_actions():
        await asyncio.gather(*(pipeline._run_rollout_sample_action(action) for action in actions))

    try:
        asyncio.run(run_actions())
    finally:
        pipeline.action_executor.shutdown(wait=True)

    assert sorted(entered) == [1, 2]
    assert all(pipeline._is_rollout_sample_action(action) for action in actions)
    assert not pipeline._is_rollout_sample_action(
        {"action_id": 3, "action_type": "sample", "payload": {"model_input": {}}}
    )


def test_native_vllm_sampling_params_forward_optional_seed():
    from roll.distributed.strategy.vllm_strategy import create_sampling_params_for_vllm

    params = create_sampling_params_for_vllm(
        {
            "max_new_tokens": 32,
            "temperature": 0.8,
            "top_p": 1.0,
            "top_k": 50,
            "seed": 0,
            "eos_token_id": [2],
            "repetition_penalty": 1.0,
            "num_return_sequences": 1,
            "stop_strings": [],
        }
    )

    assert params["seed"] == 0

    params_without_seed = create_sampling_params_for_vllm(
        {
            "max_new_tokens": 32,
            "temperature": 0.8,
            "top_p": 1.0,
            "top_k": 50,
            "eos_token_id": [2],
            "repetition_penalty": 1.0,
            "num_return_sequences": 1,
            "stop_strings": [],
        }
    )
    # ROLL forwards the unspecified seed explicitly as None to vLLM.
    assert params_without_seed.get("seed") is None


def test_unspecified_tinker_seed_remains_unspecified(tmp_path):
    from roll.pipeline.tinker_backend_runtime import runtime_pipeline as runtime_module
    from roll.pipeline.tinker_backend_runtime import types
    from roll.pipeline.tinker_backend_runtime.train_adapters import prepare_sample_batch

    normalized = runtime_module._normalize_sampling_params(
        {"temperature": 0.8, "max_tokens": 32}
    )
    assert normalized.seed is None

    sample_input = types.SampleInput(
        prompt=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[101])]),
        sampling_params=normalized,
        num_samples=3,
        checkpoint_id="",
        prompt_logprobs=False,
        env_id="env_unspecified_seed",
    )
    prepared = prepare_sample_batch(
        {"action_unspecified": ("", sample_input)},
        checkpoints_base=tmp_path,
    )
    assert [params.seed for params in prepared.all_sampling_params] == [None, None, None]


def test_multi_sample_requests_preserve_each_prepared_seed(tmp_path):
    from roll.pipeline.tinker_backend_runtime import roll_backend as module
    from roll.pipeline.tinker_backend_runtime.train_adapters import prepare_sample_batch

    sample_input = module.types.SampleInput(
        prompt=module.types.ModelInput(
            chunks=[module.types.EncodedTextChunk(tokens=[101, 102])]
        ),
        sampling_params=module.types.SamplingParams(
            temperature=0.8,
            max_tokens=4,
            seed=7,
        ),
        num_samples=3,
        checkpoint_id="",
        prompt_logprobs=False,
        env_id="env_multi",
    )
    prepared = prepare_sample_batch(
        {"action_8": ("", sample_input)},
        checkpoints_base=tmp_path,
    )
    backend = module.ROLLRuntimeBackend.__new__(module.ROLLRuntimeBackend)
    calls = []

    def generate_one(prompt_ids, sampling_params, num_samples, *, request_id, env_id):
        calls.append((prompt_ids, sampling_params.seed, num_samples, request_id, env_id))
        return [
            module.types.GeneratedSequence(
                stop_reason="stop",
                tokens=[sampling_params.seed],
                output_token_ids=[sampling_params.seed],
                logprobs=[0.0],
            )
        ]

    backend._generate_one = generate_one
    result = backend._sample_generate(prepared)

    assert [call[1] for call in calls] == [7, 8, 9]
    assert [call[3] for call in calls] == ["action_8:0", "action_8:1", "action_8:2"]
    assert [sequence.tokens for sequence in result["action_8"].sequences] == [[7], [8], [9]]


def test_publish_uses_base_pipeline_model_update_postprocessing():
    from roll.pipeline.tinker_backend_runtime import roll_backend as module

    backend = module.ROLLRuntimeBackend.__new__(module.ROLLRuntimeBackend)
    backend.model_update_groups = [object()]
    calls = []
    backend.model_update = lambda global_step: calls.append(global_step)

    backend._publish_actor_train_to_infer()

    assert calls == [0]


def test_poll_actions_waits_for_samples_before_close():
    from roll.pipeline.tinker_backend_runtime import runtime_pipeline as module

    pipeline = module.TinkerBackendRuntimePipeline.__new__(module.TinkerBackendRuntimePipeline)
    pipeline.args = SimpleNamespace(
        claim_limit=3,
        action_poll_interval=0.0,
        empty_claim_backoff_seconds=[0.01],
    )
    pipeline.should_stop = False
    pipeline.sample_tasks = set()
    pipeline.action_executor = futures.ThreadPoolExecutor(max_workers=3)
    pipeline.maybe_report_periodic_heartbeat = lambda: None
    completed_samples = []
    barrier = threading.Barrier(2)
    claim_count = 0
    actions = [
        {"action_id": 1, "action_type": "sample", "env_id": "env_1", "payload": {}},
        {"action_id": 2, "action_type": "sample", "env_id": "env_2", "payload": {}},
        {"action_id": 3, "action_type": "close_runtime", "payload": {}},
    ]

    def claim_actions(limit):
        nonlocal claim_count
        claim_count += 1
        assert limit == 3
        return actions

    def handle_action(action):
        if action["action_type"] == "sample":
            barrier.wait(timeout=2)
            completed_samples.append(action["action_id"])
            return
        assert sorted(completed_samples) == [1, 2]
        pipeline.should_stop = True

    pipeline.claim_actions = claim_actions
    pipeline.handle_action = handle_action
    pipeline.post_action_result = lambda *args, **kwargs: None

    try:
        asyncio.run(pipeline._poll_actions())
    finally:
        pipeline.action_executor.shutdown(wait=True)

    assert claim_count == 1


def test_sample_failure_is_isolated_to_its_action_future():
    from roll.pipeline.tinker_backend_runtime import runtime_pipeline as module

    pipeline = module.TinkerBackendRuntimePipeline.__new__(module.TinkerBackendRuntimePipeline)
    failures = []

    def handle_action(action):
        if action["action_id"] == 1:
            raise RuntimeError("sample failed")

    pipeline.handle_action = handle_action
    pipeline.post_action_result = (
        lambda action_id, status, **kwargs: failures.append((action_id, status, kwargs))
    )

    pipeline._execute_action({"action_id": 1, "action_type": "sample"})
    pipeline._execute_action({"action_id": 2, "action_type": "sample"})

    assert failures == [
        (1, "failed", {"error_message": "sample failed"}),
    ]


def test_backend_close_shuts_router_before_ray(monkeypatch):
    from roll.pipeline.tinker_backend_runtime import roll_backend as module

    events = []

    class RemoteShutdown:
        @staticmethod
        def remote():
            events.append("router_shutdown")
            return "shutdown_ref"

    backend = module.ROLLRuntimeBackend.__new__(module.ROLLRuntimeBackend)
    backend.router_manager = SimpleNamespace(shutdown=RemoteShutdown())
    backend.router_client = object()
    monkeypatch.setattr(module.ray, "get", lambda ref: events.append(("ray_get", ref)))
    monkeypatch.setattr(module.ray, "shutdown", lambda: events.append("ray_shutdown"))

    backend.close()

    assert events == ["router_shutdown", ("ray_get", "shutdown_ref"), "ray_shutdown"]


def test_training_engine_uses_native_rollout_workers():
    examples = Path(__file__).resolve().parents[2] / "examples" / "tinker_backend_runtime"
    engine = yaml.safe_load((examples / "public_swe_training_engine.yaml").read_text(encoding="utf-8"))
    assert engine["actor_infer"]["worker_cls"] == "roll.pipeline.tinker_backend_runtime.workers.TinkerInferWorker"
    assert engine["actor_infer"]["device_mapping"] == "list(range(4,8))"
    assert engine["actor_train"]["device_mapping"] == "list(range(0,4))"
    assert engine["actor_infer"]["strategy_args"]["strategy_name"] == "vllm"
    strategy = engine["actor_infer"]["strategy_args"]["strategy_config"]
    assert strategy["tensor_parallel_size"] == 4
    assert strategy["max_model_len"] == 8192
    assert strategy["load_format"] == "auto"


def test_sample_enrichment_preserves_raw_eos_and_action_probabilities(tmp_path):
    from roll.pipeline.tinker_backend_runtime import runtime_pipeline as module
    raw = '<tool_call>{"name":"run","arguments":{"cmd":"echo <|im_end|>"}}</tool_call><|im_end|>'
    tokenizer = SimpleNamespace(decode=lambda tokens, skip_special_tokens=False: raw,
                                eos_token_id=151645, eos_token='<|im_end|>')
    pipeline = _pipeline(module, SimpleNamespace(tokenizer=tokenizer), tmp_path)
    tokens, probabilities = [151657, 1, 151658, 151645], [-.1, -.2, -.3, -.4]
    sequence = {"tokens": tokens[:], "logprobs": probabilities[:], "raw_text": None, "text": None}
    payload = {"prompt": module.openai_chat_completion_request({"messages": [],
        "tools": [{"type": "function", "function": {"name": "run"}}]})}
    output = pipeline._enrich_sample_response(payload, {"sequences": [sequence]}, prompt_token_ids=[1, 2])
    actual = output["sequences"][0]
    assert actual["raw_text"] == raw
    assert actual["text"] == ''
    assert actual["tokens"] == tokens and actual["logprobs"] == probabilities
    assert actual["output_token_ids"] == tokens
    assert 'echo <|im_end|>' in actual["tool_calls"][0]["function"]["arguments"]


@pytest.mark.parametrize('raw,tokens,eos_id,expected', [
    ('answer<|im_end|>\n', [1, 151645], 151645, 'answer\n'),
    ('answer<|im_end|><|im_end|>\n', [1, 151645], 151645, 'answer<|im_end|>\n'),
    ('answer<|im_end|>\n', [1, 2], 151645, 'answer<|im_end|>\n'),
    ('answer<|im_end|>\n', [1, 151645], None, 'answer<|im_end|>\n'),
    ('answer<|im_end|>embedded', [1, 151645], 151645, 'answer<|im_end|>embedded'),
    ('answer', [1, 151645], 151645, 'answer'),
])
def test_sample_enrichment_removes_only_token_verified_terminal_eos(
        tmp_path, raw, tokens, eos_id, expected):
    from roll.pipeline.tinker_backend_runtime import runtime_pipeline as module

    tokenizer = SimpleNamespace(eos_token_id=eos_id, eos_token='<|im_end|>')
    pipeline = _pipeline(module, SimpleNamespace(tokenizer=tokenizer), tmp_path)
    probabilities = [-.1] * len(tokens)
    sequence = {'tokens': tokens[:], 'logprobs': probabilities[:], 'raw_text': raw}
    payload = {'prompt': module.openai_chat_completion_request({'messages': []})}
    actual = pipeline._enrich_sample_response(payload, {'sequences': [sequence]})['sequences'][0]

    assert actual['text'] == expected
    assert actual['raw_text'] == raw
    assert actual['tokens'] == tokens
    assert actual['logprobs'] == probabilities
    assert actual['output_token_ids'] == tokens


def test_roll_runtime_config_and_environment_contract(tmp_path, monkeypatch):
    from roll.pipeline.tinker_backend_runtime import runtime_pipeline as module
    config = tmp_path / "runtime.yaml"
    config.write_text(yaml.safe_dump({"tinker_runtime": {
        "backend_config": {"config_path": "/public/config", "config_name": "engine"},
        "claim_limit": 4, "enable_thinking": True}}))
    monkeypatch.setenv("TINKER_ROLL_CLAIM_LIMIT", "2")
    args = module.parse_args(["--runtime-id", "rt-test", "--backend-base-url",
                              "http://127.0.0.1:19210", "--config-path", str(config)])
    assert args.claim_limit == 2
    assert args.runtime_config["enable_thinking"] is True
    assert args.backend_config == {"config_path": "/public/config", "config_name": "engine"}
