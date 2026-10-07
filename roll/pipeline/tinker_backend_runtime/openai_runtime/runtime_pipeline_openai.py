from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlparse

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - reported at runtime with a clearer error.
    OpenAI = None  # type: ignore[assignment]


DEFAULT_ACTION_TYPES = ["sample", "close_runtime"]
OPENAI_CHAT_COMPLETION_REQUEST_TYPE = "openai_chat_completion_request"
OPENAI_CHAT_COMPLETION_REQUEST_VERSION = 1


class BackendRequestError(RuntimeError):
    pass


def _load_yaml_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    try:
        import yaml
    except Exception:
        return {}
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _runtime_section(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("tinker_runtime")
    return section if isinstance(section, dict) else {}


def _get_config_value(config: dict[str, Any], key: str, default: Any) -> Any:
    return _runtime_section(config).get(key, default)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_sequence(value: Any, default: list[float]) -> list[float]:
    if value is None:
        return default
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    parsed: list[float] = []
    for item in items:
        try:
            parsed.append(max(0.0, float(item)))
        except (TypeError, ValueError):
            continue
    return parsed or default


def post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise BackendRequestError(f"POST {url} failed with {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise BackendRequestError(f"POST {url} failed: {exc}") from exc
    return json.loads(response_body) if response_body else {}


def post_json_with_headers(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"POST {url} failed: {exc}") from exc
    return json.loads(response_body) if response_body else {}


def log_event(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def model_input_from_token_ids(tokens: list[int]) -> dict[str, Any]:
    return {"chunks": [{"type": "encoded_text", "tokens": tokens}]}


def openai_chat_completion_request(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": OPENAI_CHAT_COMPLETION_REQUEST_TYPE,
        "version": OPENAI_CHAT_COMPLETION_REQUEST_VERSION,
        "request": request,
    }


def _model_input_token_ids_from_payload(payload: dict[str, Any]) -> list[int] | None:
    model_input = payload.get("model_input")
    if model_input is None:
        return None
    if payload.get("prompt") is not None:
        raise ValueError("sample action must include exactly one of prompt or model_input")
    if not isinstance(model_input, dict):
        raise ValueError("sample model_input must be a ModelInput object")
    token_ids: list[int] = []
    for chunk in model_input.get("chunks", []):
        if not isinstance(chunk, dict) or chunk.get("type") != "encoded_text":
            raise ValueError("sample model_input only supports encoded_text chunks")
        token_ids.extend(int(token_id) for token_id in chunk.get("tokens", []))
    return token_ids


def _prompt_envelope_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    prompt = payload.get("prompt")
    if prompt is None:
        return None
    if payload.get("model_input") is not None:
        raise ValueError("sample action must include exactly one of prompt or model_input")
    if not isinstance(prompt, dict):
        raise ValueError("sample prompt must be an openai_chat_completion_request envelope")
    if prompt.get("type") != OPENAI_CHAT_COMPLETION_REQUEST_TYPE:
        raise ValueError(
            "sample prompt.type must be "
            f"{OPENAI_CHAT_COMPLETION_REQUEST_TYPE!r}; tokenized prompts must use model_input"
        )
    if int(prompt.get("version", 0)) != OPENAI_CHAT_COMPLETION_REQUEST_VERSION:
        raise ValueError(
            "sample prompt.version must be "
            f"{OPENAI_CHAT_COMPLETION_REQUEST_VERSION}"
        )
    request = prompt.get("request")
    if not isinstance(request, dict):
        raise ValueError("sample prompt.request must be an OpenAI chat completion request object")
    return prompt


def _request_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    envelope = _prompt_envelope_from_payload(payload)
    if envelope is None:
        if _model_input_token_ids_from_payload(payload) is not None:
            return {}
        raise ValueError("sample action missing prompt or model_input")
    request = envelope["request"]
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise ValueError("sample prompt.request.messages must be a list")
    return request


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _normalize_message(message: Any) -> Any:
    if not isinstance(message, dict):
        return message
    normalized = dict(message)
    if "content" in normalized:
        normalized["content"] = _message_content_to_text(normalized.get("content"))
    return normalized


def _normalize_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return [{"role": "user", "content": _message_content_to_text(messages)}]
    normalized = [_normalize_message(message) for message in messages]
    return [message for message in normalized if isinstance(message, dict)]


def _attr_or_item(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _is_chat_completions_endpoint(url: str) -> bool:
    return urlparse(url).path.rstrip("/").endswith("/chat/completions")


def _positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


class _DictMessage:
    def __init__(self, value: dict[str, Any]):
        self.content = value.get("content")
        self.tool_calls = value.get("tool_calls")


class _DictChoice:
    def __init__(self, value: dict[str, Any]):
        self.message = _DictMessage(value.get("message") or {})
        self.finish_reason = value.get("finish_reason")


class _DictCompletion:
    def __init__(self, value: dict[str, Any]):
        self.choices = [_DictChoice(item) for item in value.get("choices", []) if isinstance(item, dict)]
        self.error_code = value.get("error_code")
        self.message = value.get("message")


def _native_tool_calls(message: Any) -> list[dict[str, Any]] | None:
    raw_calls = _attr_or_item(message, "tool_calls") or []
    if not raw_calls:
        return None
    tool_calls: list[dict[str, Any]] = []
    for raw_call in raw_calls:
        function = _attr_or_item(raw_call, "function") or {}
        name = _attr_or_item(function, "name")
        arguments = _attr_or_item(function, "arguments", "{}")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        tool_calls.append(
            {
                "id": str(_attr_or_item(raw_call, "id", f"call_{uuid.uuid4().hex[:24]}")),
                "type": "function",
                "function": {
                    "name": str(name),
                    "arguments": str(arguments) if arguments else "{}",
                },
            }
        )
    return tool_calls


def _raise_if_no_choices(completion: Any) -> None:
    choices = _attr_or_item(completion, "choices", []) or []
    if choices:
        return
    error_code = _attr_or_item(completion, "error_code")
    message = _attr_or_item(completion, "message")
    if error_code is not None or message:
        raise RuntimeError(f"OpenAI provider returned no choices: error_code={error_code} message={message}")
    raise RuntimeError("OpenAI provider returned no choices")


def _parse_tool_calls_with_sglang(text: str, raw_tools: Any, parser_name: str) -> tuple[str, list[dict[str, Any]] | None, str]:
    if not raw_tools:
        return text, None, "stop"
    normalized_text = text
    if "<tool_call>" in normalized_text:
        normalized_text = re.sub(r"<tool_call>(?!\n)", "<tool_call>\n", normalized_text)
        normalized_text = re.sub(r"(?<!\n)</tool_call>", "\n</tool_call>", normalized_text)

    from sglang.srt.entrypoints.openai.protocol import Tool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    tools = [Tool.model_validate(tool) for tool in raw_tools if isinstance(tool, dict)]
    parser = FunctionCallParser(tools, parser_name)
    if not parser.has_tool_call(normalized_text):
        return text, None, "stop"
    parsed_text, call_info_list = parser.parse_non_stream(normalized_text)
    tool_calls = []
    for call_info in call_info_list:
        args = call_info.parameters
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": call_info.name,
                    "arguments": args if args else "{}",
                },
            }
        )
    return parsed_text, tool_calls, "tool_calls"


class OpenAITextGenerator:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        if OpenAI is None:
            raise RuntimeError("openai package is required for tinker_runtime.generator.type=openai")
        api_key_env = str(config.get("api_key_env") or "")
        api_key = str(config.get("api_key") or os.environ.get(api_key_env or "OPENAI_API_KEY", ""))
        if api_key_env and not api_key:
            raise ValueError(f"OpenAI api key environment variable {api_key_env!r} is not set")
        self.api_key = api_key
        base_url_env = str(config.get("base_url_env") or "")
        base_url = str(config.get("base_url") or os.environ.get(base_url_env, ""))
        if base_url_env and not base_url:
            raise ValueError(f"OpenAI base URL environment variable {base_url_env!r} is not set")
        if not base_url:
            raise ValueError("openai generator requires base_url or base_url_env")
        self.base_url = base_url.rstrip("/")
        self.timeout = float(config.get("timeout", 120))
        self.max_retries = int(config.get("max_retries", 3))
        self.retry_delay = float(config.get("retry_delay", 2))
        model_name_env = str(config.get("model_name_env") or "")
        self.model_name = str(config.get("model_name") or config.get("model") or os.environ.get(model_name_env, ""))
        if model_name_env and not self.model_name:
            raise ValueError(f"OpenAI model name environment variable {model_name_env!r} is not set")
        if not self.model_name:
            raise ValueError("openai generator requires model_name, model, or model_name_env")
        self.use_direct_endpoint = _is_chat_completions_endpoint(self.base_url)
        self.client = None if self.use_direct_endpoint else OpenAI(api_key=api_key, base_url=self.base_url, timeout=self.timeout)

    def preload(self) -> None:
        return None

    def close(self) -> None:
        return None

    def generate(self, *, payload: dict[str, Any], runtime_id: str, env_id: str) -> dict[str, Any]:
        del runtime_id, env_id
        if _model_input_token_ids_from_payload(payload) is not None:
            raise ValueError("OpenAI generator does not support tokenized model_input sample requests")
        request = _request_from_payload(payload)
        sampling_params = payload.get("sampling_params", {}) if isinstance(payload.get("sampling_params"), dict) else {}
        messages = _normalize_messages(request.get("messages"))
        tools = request.get("tools") or []
        max_tokens = int(
            sampling_params.get("max_tokens")
            or sampling_params.get("max_new_tokens")
            or request.get("max_tokens")
            or self.config.get("max_tokens", 256)
        )
        temperature = float(sampling_params.get("temperature", request.get("temperature", self.config.get("temperature", 0.0))))
        top_p = float(sampling_params.get("top_p", request.get("top_p", self.config.get("top_p", 1.0))))
        tool_choice = request.get("tool_choice", self.config.get("tool_choice", "none"))
        extra_body: dict[str, Any] = {}
        top_k = (
            _positive_int(sampling_params.get("top_k"))
            or _positive_int(request.get("top_k"))
            or _positive_int(self.config.get("top_k"))
        )
        if top_k is not None:
            extra_body["top_k"] = top_k
        if not bool(self.config.get("enable_thinking", False)):
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        create_payload = {
            "model": str(request.get("model") or self.model_name),
            "messages": messages,
            "tools": tools or None,
            "tool_choice": tool_choice,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "extra_body": extra_body if extra_body else None,
        }
        completion = self._create_completion(create_payload)
        _raise_if_no_choices(completion)
        choices = _attr_or_item(completion, "choices", []) or []
        choice = choices[0] if choices else None
        message = _attr_or_item(choice, "message", {}) if choice is not None else {}
        raw_text = str(_attr_or_item(message, "content") or "")
        tool_calls = _native_tool_calls(message)
        finish_reason = "tool_calls" if tool_calls else str(_attr_or_item(choice, "finish_reason", "stop") or "stop")
        parsed_text = raw_text
        if not tool_calls and raw_text:
            parsed_text, tool_calls, finish_reason = _parse_tool_calls_with_sglang(
                raw_text,
                tools,
                str(self.config.get("tool_call_parser", "qwen25")),
            )
        return {
            "type": "sample",
            "prompt_token_ids": [],
            "sequences": [
                {
                    "tokens": [],
                    "output_token_ids": [],
                    "text": parsed_text,
                    "raw_text": raw_text,
                    "logprobs": [],
                    "stop_reason": "stop",
                    "finish_reason": finish_reason,
                    "tool_calls": tool_calls or [],
                }
            ],
        }

    def _create_completion(self, payload: dict[str, Any]) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.use_direct_endpoint:
                    headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                    body = {key: value for key, value in payload.items() if value is not None}
                    extra_body = body.pop("extra_body", None)
                    if isinstance(extra_body, dict):
                        body.update(extra_body)
                    return _DictCompletion(post_json_with_headers(self.base_url, body, headers=headers, timeout=self.timeout))
                if self.client is None:
                    raise RuntimeError("OpenAI client is not initialized")
                return self.client.chat.completions.create(**payload)
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_delay + attempt * 0.5)
        raise RuntimeError(f"OpenAI completion failed: {last_error}")


def _create_generator(runtime_config: dict[str, Any]) -> OpenAITextGenerator:
    generator_config = runtime_config.get("generator")
    if not isinstance(generator_config, dict):
        raise ValueError("tinker_runtime.generator must be configured")
    generator_type = str(generator_config.get("type", "")).lower()
    if generator_type != "openai":
        raise ValueError(f"runtime_pipeline_openai only supports generator.type=openai, got {generator_type!r}")
    return OpenAITextGenerator(generator_config)


class OpenAIRuntimePipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.backend_base_url = args.backend_base_url.rstrip("/")
        self.last_heartbeat_at = 0.0
        self.should_stop = False
        self.generator = _create_generator(args.runtime_config)

    @property
    def runtime_id(self) -> str:
        return self.args.runtime_id

    def heartbeat_url(self) -> str:
        return f"{self.backend_base_url}/api/v1/runtimes/{self.runtime_id}/heartbeat"

    def claim_url(self) -> str:
        return f"{self.backend_base_url}/api/v1/runtimes/{self.runtime_id}/actions/claim"

    def action_result_url(self, action_id: int) -> str:
        return f"{self.backend_base_url}/api/v1/runtimes/{self.runtime_id}/actions/{action_id}/result"

    def report_heartbeat(self, status: str = "ready", ready: bool = True, error_message: str | None = None) -> dict[str, Any]:
        payload = {
            "type": "runtime_heartbeat",
            "runtime_id": self.runtime_id,
            "status": status,
            "ready": ready,
            "adapter_base_url": self.args.adapter_base_url,
            "process_pid": os.getpid(),
            "error_message": error_message,
            "metadata": {
                "runtime": "roll-openai",
                "pipeline": "tinker_backend_runtime_openai",
                "config_path": self.args.config_path,
                "pid": os.getpid(),
            },
        }
        response = post_json(self.heartbeat_url(), payload, timeout=self.args.request_timeout)
        self.last_heartbeat_at = time.monotonic()
        return response

    def claim_actions(self) -> list[dict[str, Any]]:
        payload = {"action_types": self.args.action_types, "limit": self.args.claim_limit}
        response = post_json(self.claim_url(), payload, timeout=self.args.request_timeout)
        actions = response.get("actions", [])
        return actions if isinstance(actions, list) else []

    def post_action_result(
        self,
        action_id: int,
        status: str,
        *,
        result_data: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        payload = {"status": status, "result_data": result_data, "error_message": error_message}
        return post_json(self.action_result_url(action_id), payload, timeout=self.args.request_timeout)

    def handle_sample(self, action: dict[str, Any]) -> None:
        payload = action.get("payload", {}) if isinstance(action.get("payload"), dict) else {}
        env_id = action.get("env_id") or payload.get("env_id")
        if not env_id:
            raise RuntimeError("sample action missing env_id")
        sample_response = self.generator.generate(payload=payload, runtime_id=self.runtime_id, env_id=str(env_id))
        self.post_action_result(action["action_id"], "completed", result_data=sample_response)

    def handle_close_runtime(self, action: dict[str, Any]) -> None:
        try:
            self.generator.close()
        finally:
            self.post_action_result(
                action["action_id"],
                "completed",
                result_data={
                    "runtime_id": self.runtime_id,
                    "status": "stopping",
                    "graceful": True,
                    "process_pid": os.getpid(),
                },
            )
            try:
                self.report_heartbeat(status="stopped", ready=False)
            except Exception:
                pass
            self.should_stop = True

    def handle_action(self, action: dict[str, Any]) -> None:
        action_type = action.get("action_type")
        if action_type == "sample":
            self.handle_sample(action)
        elif action_type == "close_runtime":
            self.handle_close_runtime(action)
        else:
            raise RuntimeError(f"unsupported action_type={action_type!r}")

    def report_ready_with_retries(self) -> None:
        last_error: Exception | None = None
        for attempt in range(1, self.args.startup_retries + 1):
            try:
                response = self.report_heartbeat(status="ready", ready=True)
                log_event("runtime_ready", runtime_id=self.runtime_id, response=response)
                return
            except BackendRequestError as exc:
                last_error = exc
                log_event("runtime_ready_failed", attempt=attempt, error=str(exc))
                time.sleep(self.args.startup_retry_interval)
        raise RuntimeError(f"failed to report ready heartbeat: {last_error}")

    def maybe_report_periodic_heartbeat(self) -> None:
        now = time.monotonic()
        if now - self.last_heartbeat_at >= self.args.heartbeat_interval:
            response = self.report_heartbeat(status="ready", ready=True)
            log_event("runtime_heartbeat", runtime_id=self.runtime_id, response=response)

    def run(self) -> int:
        try:
            self.generator.preload()
        except Exception as exc:
            try:
                self.report_heartbeat(status="failed", ready=False, error_message=str(exc))
            except Exception:
                pass
            raise

        self.report_ready_with_retries()
        if self.args.exit_after_ready:
            return 0

        empty_claim_count = 0
        while not self.should_stop:
            try:
                self.maybe_report_periodic_heartbeat()
                actions = self.claim_actions()
                if not actions:
                    backoff = self.args.empty_claim_backoff_seconds[
                        min(empty_claim_count, len(self.args.empty_claim_backoff_seconds) - 1)
                    ]
                    empty_claim_count += 1
                    time.sleep(backoff)
                    continue
                empty_claim_count = 0
                for action in actions:
                    try:
                        self.handle_action(action)
                        log_event("action_completed", action_id=action.get("action_id"), action_type=action.get("action_type"))
                        if self.should_stop:
                            break
                    except Exception as exc:
                        log_event("action_failed", action=action, error=str(exc))
                        try:
                            self.post_action_result(action["action_id"], "failed", error_message=str(exc))
                        except Exception:
                            pass
                if not self.should_stop and self.args.action_poll_interval > 0:
                    time.sleep(self.args.action_poll_interval)
            except BackendRequestError as exc:
                log_event("runtime_loop_error", error=str(exc))
                time.sleep(self.args.empty_claim_backoff_seconds[-1])
        log_event("runtime_closed", runtime_id=self.runtime_id, pid=os.getpid())
        return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROLL Tinker-backend OpenAI runtime pipeline")
    parser.add_argument("--runtime-id", default=os.environ.get("TINKER_RUNTIME_ID"), required=False)
    parser.add_argument("--backend-base-url", default=os.environ.get("TINKER_BACKEND_BASE_URL"), required=False)
    parser.add_argument("--config-path", default=os.environ.get("TINKER_CONFIG_PATH", ""), required=False)
    parser.add_argument("--adapter-base-url", default=os.environ.get("TINKER_ROLL_ADAPTER_BASE_URL"), required=False)
    parser.add_argument("--exit-after-ready", action="store_true", default=False)
    parsed, _remaining = parser.parse_known_args(argv)

    config = _load_yaml_config(parsed.config_path)
    runtime_config = _runtime_section(config)
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=float(os.environ.get("TINKER_ROLL_HEARTBEAT_INTERVAL", _get_config_value(config, "heartbeat_interval", 30))),
    )
    parser.add_argument(
        "--action-poll-interval",
        type=float,
        default=float(os.environ.get("TINKER_ROLL_ACTION_POLL_INTERVAL", _get_config_value(config, "action_poll_interval", 0.5))),
    )
    parser.add_argument(
        "--empty-claim-backoff-seconds",
        nargs="*",
        type=float,
        default=_float_sequence(
            os.environ.get(
                "TINKER_ROLL_EMPTY_CLAIM_BACKOFF_SECONDS",
                _get_config_value(config, "empty_claim_backoff_seconds", [1, 5, 10]),
            ),
            [1, 5, 10],
        ),
    )
    parser.add_argument(
        "--claim-limit",
        type=int,
        default=int(os.environ.get("TINKER_ROLL_CLAIM_LIMIT", _get_config_value(config, "claim_limit", 4))),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=float(os.environ.get("TINKER_ROLL_REQUEST_TIMEOUT", _get_config_value(config, "request_timeout", 120))),
    )
    parser.add_argument(
        "--startup-retries",
        type=int,
        default=int(os.environ.get("TINKER_ROLL_STARTUP_RETRIES", _get_config_value(config, "startup_retries", 600))),
    )
    parser.add_argument(
        "--startup-retry-interval",
        type=float,
        default=float(
            os.environ.get(
                "TINKER_ROLL_STARTUP_RETRY_INTERVAL",
                _get_config_value(config, "startup_retry_interval", 1),
            )
        ),
    )
    parser.add_argument(
        "--action-types",
        nargs="+",
        default=runtime_config.get("action_types") or DEFAULT_ACTION_TYPES,
    )
    args = parser.parse_args(argv)
    args.runtime_config = runtime_config

    if not args.runtime_id:
        parser.error("--runtime-id or TINKER_RUNTIME_ID is required")
    if not args.backend_base_url:
        parser.error("--backend-base-url or TINKER_BACKEND_BASE_URL is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pipeline = OpenAIRuntimePipeline(args)
    try:
        return pipeline.run()
    except Exception as exc:
        log_event("runtime_fatal", error=str(exc), traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
