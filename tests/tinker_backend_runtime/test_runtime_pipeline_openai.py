from __future__ import annotations

import importlib

import pytest


def _prompt(module, request):
    return module.openai_chat_completion_request(request)


def test_openai_runtime_pipeline_imports_without_training_stack() -> None:
    module = importlib.import_module("roll.pipeline.tinker_backend_runtime.openai_runtime.runtime_pipeline_openai")

    assert hasattr(module, "OpenAITextGenerator")


def test_openai_generator_parses_qwen_tool_call_text(monkeypatch) -> None:
    module = importlib.import_module("roll.pipeline.tinker_backend_runtime.openai_runtime.runtime_pipeline_openai")

    class FakeMessage:
        content = '<tool_call>{"name":"run","arguments":{"cmd":"pytest"}}</tool_call>'
        tool_calls = None

    class FakeChoice:
        message = FakeMessage()
        finish_reason = "stop"

    class FakeCompletions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return type("FakeCompletion", (), {"choices": [FakeChoice()]})()

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.chat = FakeChat()

    monkeypatch.setattr(module, "OpenAI", FakeClient)
    generator = module.OpenAITextGenerator(
        {
            "base_url": "https://example.invalid/v1",
            "api_key": "test-key",
            "model_name": "test-model",
            "tool_call_parser": "qwen25",
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 128,
        }
    )

    payload = {
        "prompt": _prompt(
            module,
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "run", "parameters": {"type": "object"}}}],
            },
        )
    }
    result = generator.generate(payload=payload, runtime_id="rt_test", env_id="env_test")

    assert result["prompt_token_ids"] == []
    sequence = result["sequences"][0]
    assert sequence["tokens"] == []
    assert sequence["output_token_ids"] == []
    assert sequence["logprobs"] == []
    assert sequence["finish_reason"] == "tool_calls"
    assert sequence["tool_calls"][0]["function"]["name"] == "run"
    assert "pytest" in sequence["tool_calls"][0]["function"]["arguments"]


def test_openai_generator_raises_provider_error_when_choices_are_missing(monkeypatch) -> None:
    module = importlib.import_module("roll.pipeline.tinker_backend_runtime.openai_runtime.runtime_pipeline_openai")

    def fake_post_json_with_headers(*args, **kwargs):
        return {
            "error_code": 514,
            "message": "Engine internal server error!: failed to connect to all addresses",
        }

    monkeypatch.setattr(module, "post_json_with_headers", fake_post_json_with_headers)
    generator = module.OpenAITextGenerator(
        {
            "base_url": "https://example.invalid/v1/chat/completions/",
            "api_key": "test-key",
            "model_name": "test-model",
            "tool_call_parser": "qwen25",
            "max_retries": 1,
        }
    )

    payload = {"prompt": _prompt(module, {"messages": [{"role": "user", "content": "hi"}]})}
    with pytest.raises(RuntimeError, match="error_code=514.*failed to connect"):
        generator.generate(payload=payload, runtime_id="rt_test", env_id="env_test")


def test_openai_generator_ignores_negative_sampling_top_k_and_uses_config(monkeypatch) -> None:
    module = importlib.import_module("roll.pipeline.tinker_backend_runtime.openai_runtime.runtime_pipeline_openai")
    captured = {}

    def fake_post_json_with_headers(url, payload, **kwargs):
        captured["payload"] = payload
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ]
        }

    monkeypatch.setattr(module, "post_json_with_headers", fake_post_json_with_headers)
    generator = module.OpenAITextGenerator(
        {
            "base_url": "https://example.invalid/v1/chat/completions/",
            "api_key": "test-key",
            "model_name": "test-model",
            "top_k": 100,
            "tool_call_parser": "qwen25",
        }
    )

    payload = {
        "prompt": _prompt(module, {"messages": [{"role": "user", "content": "hi"}]}),
        "sampling_params": {"top_k": -1, "max_tokens": 64},
    }
    result = generator.generate(payload=payload, runtime_id="rt_test", env_id="env_test")

    assert result["sequences"][0]["text"] == "ok"
    assert captured["payload"]["top_k"] == 100


def test_openai_generator_reads_api_key_from_environment(monkeypatch) -> None:
    module = importlib.import_module("roll.pipeline.tinker_backend_runtime.openai_runtime.runtime_pipeline_openai")
    captured = {}

    def fake_post_json_with_headers(url, payload, **kwargs):
        captured["headers"] = kwargs.get("headers")
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ]
        }

    monkeypatch.setenv("WHALE_OPENAI_API_KEY", "env-secret")
    monkeypatch.setattr(module, "post_json_with_headers", fake_post_json_with_headers)
    generator = module.OpenAITextGenerator(
        {
            "base_url": "https://example.invalid/v1/chat/completions/",
            "api_key_env": "WHALE_OPENAI_API_KEY",
            "model_name": "test-model",
            "tool_call_parser": "qwen25",
            "max_retries": 1,
        }
    )

    payload = {"prompt": _prompt(module, {"messages": [{"role": "user", "content": "hi"}]})}
    result = generator.generate(payload=payload, runtime_id="rt_test", env_id="env_test")

    assert result["sequences"][0]["text"] == "ok"
    assert captured["headers"] == {"Authorization": "Bearer env-secret"}


def test_openai_generator_reads_base_url_and_model_name_from_environment(monkeypatch) -> None:
    module = importlib.import_module("roll.pipeline.tinker_backend_runtime.openai_runtime.runtime_pipeline_openai")
    captured = {}

    def fake_post_json_with_headers(url, payload, **kwargs):
        captured["url"] = url
        captured["payload"] = payload
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ]
        }

    monkeypatch.setenv("WHALE_OPENAI_BASE_URL", "https://example.invalid/v1/chat/completions/")
    monkeypatch.setenv("WHALE_OPENAI_MODEL_NAME", "env-model")
    monkeypatch.setattr(module, "OpenAI", object)
    monkeypatch.setattr(module, "post_json_with_headers", fake_post_json_with_headers)
    generator = module.OpenAITextGenerator(
        {
            "base_url_env": "WHALE_OPENAI_BASE_URL",
            "api_key": "test-key",
            "model_name_env": "WHALE_OPENAI_MODEL_NAME",
            "tool_call_parser": "qwen25",
            "max_retries": 1,
        }
    )

    payload = {"prompt": _prompt(module, {"messages": [{"role": "user", "content": "hi"}]})}
    result = generator.generate(payload=payload, runtime_id="rt_test", env_id="env_test")

    assert result["sequences"][0]["text"] == "ok"
    assert captured["url"] == "https://example.invalid/v1/chat/completions"
    assert captured["payload"]["model"] == "env-model"


def test_openai_generator_prefers_explicit_base_url_and_model_name_over_environment(monkeypatch) -> None:
    module = importlib.import_module("roll.pipeline.tinker_backend_runtime.openai_runtime.runtime_pipeline_openai")
    captured = {}

    def fake_post_json_with_headers(url, payload, **kwargs):
        captured["url"] = url
        captured["payload"] = payload
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ]
        }

    monkeypatch.setenv("WHALE_OPENAI_BASE_URL", "https://env.invalid/v1/chat/completions/")
    monkeypatch.setenv("WHALE_OPENAI_MODEL_NAME", "env-model")
    monkeypatch.setattr(module, "OpenAI", object)
    monkeypatch.setattr(module, "post_json_with_headers", fake_post_json_with_headers)
    generator = module.OpenAITextGenerator(
        {
            "base_url": "https://explicit.invalid/v1/chat/completions/",
            "base_url_env": "WHALE_OPENAI_BASE_URL",
            "api_key": "test-key",
            "model_name": "explicit-model",
            "model_name_env": "WHALE_OPENAI_MODEL_NAME",
            "tool_call_parser": "qwen25",
            "max_retries": 1,
        }
    )

    payload = {"prompt": _prompt(module, {"messages": [{"role": "user", "content": "hi"}]})}
    result = generator.generate(payload=payload, runtime_id="rt_test", env_id="env_test")

    assert result["sequences"][0]["text"] == "ok"
    assert captured["url"] == "https://explicit.invalid/v1/chat/completions"
    assert captured["payload"]["model"] == "explicit-model"


def test_openai_generator_rejects_tokenized_model_input(monkeypatch) -> None:
    module = importlib.import_module("roll.pipeline.tinker_backend_runtime.openai_runtime.runtime_pipeline_openai")

    monkeypatch.setattr(module, "OpenAI", object)
    generator = module.OpenAITextGenerator(
        {
            "base_url": "https://example.invalid/v1/chat/completions/",
            "api_key": "test-key",
            "model_name": "test-model",
            "tool_call_parser": "qwen25",
            "max_retries": 1,
        }
    )

    with pytest.raises(ValueError, match="tokenized model_input"):
        generator.generate(
            payload={"model_input": module.model_input_from_token_ids([1, 2, 3])},
            runtime_id="rt_test",
            env_id="env_test",
        )


def test_openai_generator_reports_missing_api_key_env(monkeypatch) -> None:
    module = importlib.import_module("roll.pipeline.tinker_backend_runtime.openai_runtime.runtime_pipeline_openai")

    monkeypatch.delenv("WHALE_OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="WHALE_OPENAI_API_KEY"):
        module.OpenAITextGenerator(
            {
                "base_url": "https://example.invalid/v1/chat/completions/",
                "api_key_env": "WHALE_OPENAI_API_KEY",
                "model_name": "test-model",
                "tool_call_parser": "qwen25",
            }
        )
