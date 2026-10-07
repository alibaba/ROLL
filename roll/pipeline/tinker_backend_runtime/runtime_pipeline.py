from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import sys
import time
import traceback
import uuid
from concurrent import futures
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from roll.pipeline.tinker_backend_runtime import types as tinker_types
from roll.pipeline.tinker_backend_runtime.chat_prompt import (
    _message_content_to_text,
    _normalize_tool_calls_for_chat_template,
    _normalize_messages_for_chat_template,
    _messages_to_prompt_text,
    _render_prompt_token_ids,
)
from roll.pipeline.tinker_backend_runtime.data_adapter import _model_input_to_token_ids
from roll.pipeline.tinker_backend_runtime.train_adapters import (
    init_roll_backend,
    prepare_model_pass_batch,
    prepare_sample_batch,
)


DEFAULT_ACTION_TYPES = [
    "init_task_env",
    "sample",
    "create_model",
    "forward",
    "forward_backward",
    "optim_step",
    "save_weights",
    "publish_to_sampler",
    "load_weights",
    "close_runtime",
]


class BackendRequestError(RuntimeError):
    pass


OPENAI_CHAT_COMPLETION_REQUEST_TYPE = "openai_chat_completion_request"
OPENAI_CHAT_COMPLETION_REQUEST_VERSION = 1


def _load_yaml_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    try:
        import yaml
    except Exception:
        # This runtime pipeline deliberately keeps dependencies minimal. The
        # backend already consumed the runtime.launch_* section before starting
        # us, so config parsing here is optional for smoke settings only.
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


def post_json(url: str, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
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


def openai_chat_completion_request_from_text(text: str) -> dict[str, Any]:
    return openai_chat_completion_request({"messages": [{"role": "user", "content": text}]})


def sample_response_from_text(
    text: str,
    *,
    prompt_token_ids: list[int] | None = None,
    output_token_ids: list[int] | None = None,
) -> dict[str, Any]:
    tokens = output_token_ids or []
    return {
        "type": "sample",
        "prompt_token_ids": prompt_token_ids or [],
        "sequences": [
            {
                "tokens": tokens,
                "output_token_ids": tokens,
                "text": text,
                "logprobs": [0.0 for _ in tokens],
                "stop_reason": "stop",
            }
        ],
    }


def _model_input_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    model_input = payload.get("model_input")
    if model_input is None:
        return None
    if payload.get("prompt") is not None:
        raise ValueError("sample action must include exactly one of prompt or model_input")
    if not isinstance(model_input, dict):
        raise ValueError("sample model_input must be a ModelInput object")
    return model_input


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
        if _model_input_from_payload(payload) is not None:
            return {}
        raise ValueError("sample action missing prompt or model_input")
    request = envelope["request"]
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise ValueError("sample prompt.request.messages must be a list")
    return request




def _dump_model(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return value
    return json.loads(json.dumps(value))


def _dense_from_csr(value: dict[str, Any]) -> list[Any]:
    shape = value.get("shape") or []
    if len(shape) != 2:
        return list(value.get("data", []))
    rows, cols = int(shape[0]), int(shape[1])
    dense = [[0 for _ in range(cols)] for _ in range(rows)]
    crow = list(value.get("sparse_crow_indices") or [])
    ccol = list(value.get("sparse_col_indices") or [])
    data = list(value.get("data") or [])
    for row in range(rows):
        start = int(crow[row])
        end = int(crow[row + 1])
        for idx in range(start, end):
            dense[row][int(ccol[idx])] = data[idx]
    return [item for row in dense for item in row]


def _as_tensor_payload(value: Any, *, dtype: str, length: int | None = None) -> dict[str, Any]:
    if isinstance(value, dict):
        data = value.get("data", [])
        if value.get("sparse_crow_indices") is not None and value.get("sparse_col_indices") is not None:
            data = _dense_from_csr(value)
        return {"data": list(data)}
    if isinstance(value, list):
        return {"data": value}
    if value is None:
        return {"data": [0.0 if dtype == "float32" else 0 for _ in range(length or 0)]}
    return {"data": [value]}


def _data_len(tensor_payload: dict[str, Any]) -> int:
    data = tensor_payload.get("data", [])
    return len(data) if isinstance(data, list) else 0


def _normalize_datum_payload(datum: dict[str, Any]) -> dict[str, Any]:
    loss_inputs = dict(datum.get("loss_fn_inputs") or {})
    target = _as_tensor_payload(loss_inputs.get("target_tokens"), dtype="int64")
    n_tokens = _data_len(target)
    weights = _as_tensor_payload(loss_inputs.get("weights"), dtype="float32", length=n_tokens)
    advantages = _as_tensor_payload(loss_inputs.get("advantages"), dtype="float32", length=n_tokens)
    logprobs = _as_tensor_payload(loss_inputs.get("logprobs"), dtype="float32", length=n_tokens)
    return {
        "model_input": datum.get("model_input"),
        "loss_fn_inputs": {
            "target_tokens": target,
            "weights": weights,
            "advantages": advantages,
            "logprobs": logprobs,
        },
    }


def _normalize_forward_input(payload: dict[str, Any], key: str) -> tinker_types.ForwardBackwardInput:
    raw = payload.get(key)
    if not isinstance(raw, dict):
        raw = {
            "data": payload.get("data") or [],
            "loss_fn": payload.get("loss_fn") or "cross_entropy",
            "loss_fn_config": payload.get("loss_fn_config"),
        }
    data = raw.get("data") or []
    normalized = {
        "data": [_normalize_datum_payload(item) for item in data],
        "loss_fn": raw.get("loss_fn") or "cross_entropy",
        "loss_fn_config": raw.get("loss_fn_config"),
    }
    return tinker_types.ForwardBackwardInput.model_validate(normalized)


def _normalize_sampling_params(raw: Any) -> tinker_types.SamplingParams:
    params = raw if isinstance(raw, dict) else {}
    raw_seed = params.get("seed")
    stop = params.get("stop")
    stop_tokens: list[int] | None = None
    stop_strings: list[str] | None = None
    if isinstance(stop, str):
        stop_strings = [stop]
    elif isinstance(stop, list):
        if all(isinstance(item, int) for item in stop):
            stop_tokens = [int(item) for item in stop]
        else:
            stop_strings = [str(item) for item in stop]
    return tinker_types.SamplingParams(
        temperature=float(params.get("temperature", 1.0)),
        max_tokens=int(params.get("max_tokens") or params.get("max_new_tokens") or 1),
        seed=int(raw_seed) if raw_seed is not None else None,
        stop_tokens=stop_tokens,
        stop_strings=stop_strings,
        top_k=int(params.get("top_k", -1)),
        top_p=float(params.get("top_p", 1.0)),
    )


def _parse_qwen_tool_calls_from_text(text: str, raw_tools: Any) -> tuple[str, list[dict[str, Any]] | None, str]:
    valid_names = {
        str(tool.get("function", {}).get("name"))
        for tool in raw_tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }
    tool_calls = []
    normalized_text = text
    if "<tool_call>" in normalized_text:
        normalized_text = re.sub(r"<tool_call>(?!\n)", "<tool_call>\n", normalized_text)
        normalized_text = re.sub(r"(?<!\n)</tool_call>", "\n</tool_call>", normalized_text)
    for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", normalized_text, flags=re.DOTALL):
        raw_call = match.group(1).strip()
        try:
            call = json.loads(raw_call)
        except json.JSONDecodeError:
            continue
        if not isinstance(call, dict):
            continue
        name = call.get("name") or call.get("function", {}).get("name")
        if valid_names and name not in valid_names:
            continue
        args = call.get("arguments", call.get("parameters", {}))
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": str(name),
                    "arguments": str(args) if args else "{}",
                },
            }
        )
    if not tool_calls:
        return text, None, "stop"
    parsed_text = re.sub(r"<tool_call>\s*.*?\s*</tool_call>", "", normalized_text, flags=re.DOTALL).strip()
    return parsed_text, tool_calls, "tool_calls"


class StaticTextGenerator:
    def __init__(self, sample_text: str):
        self.sample_text = sample_text

    def preload(self) -> None:
        return None

    def close(self) -> None:
        return None

    def generate(self, *, payload: dict[str, Any], runtime_id: str, env_id: str) -> dict[str, Any]:
        sampling_params = payload.get("sampling_params", {}) if isinstance(payload.get("sampling_params"), dict) else {}
        prompt_token_ids = []
        model_input = _model_input_from_payload(payload)
        if model_input is not None:
            prompt_token_ids = _model_input_to_token_ids(tinker_types.ModelInput.model_validate(model_input))
        response_text = self.sample_text.format(
            runtime_id=runtime_id,
            env_id=env_id,
            seq_id=payload.get("seq_id", ""),
            max_tokens=sampling_params.get("max_tokens", sampling_params.get("max_new_tokens", "")),
        )
        return sample_response_from_text(response_text, prompt_token_ids=prompt_token_ids)


def _create_generator(runtime_config: dict[str, Any], sample_text: str):
    generator_config = runtime_config.get("generator")
    if not isinstance(generator_config, dict):
        return StaticTextGenerator(sample_text)
    generator_type = str(generator_config.get("type", "static")).lower()
    if generator_type == "static":
        return StaticTextGenerator(str(generator_config.get("sample_text", sample_text)))
    if generator_type == "vllm":
        raise ValueError(
            "generator.type='vllm' is no longer supported; configure "
            "tinker_runtime.backend_config to use ROLL's native actor_infer router"
        )
    raise ValueError(f"unsupported generator type: {generator_type}")


class TinkerBackendRuntimePipeline:
    """ROLL runtime process that polls Tinker backend actions.

    This is intentionally a runtime adapter, not a standalone Tinker backend.
    Tinker backend owns sessions, futures, env ids, and action durability. This
    process only heartbeats ready, claims runtime-scoped actions, and posts
    action results. Real sampling is delegated to ROLL's native rollout
    router; the static generator is retained only for explicit smoke configs.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.backend_base_url = args.backend_base_url.rstrip("/")
        self.env_steps: dict[str, int] = {}
        self.last_heartbeat_at = 0.0
        self.should_stop = False
        self.generator = (
            None
            if args.backend_config is not None
            else _create_generator(args.runtime_config, args.sample_text)
        )
        self.roll_backend = None
        self.sample_tasks: set[asyncio.Task[None]] = set()
        self.action_executor = futures.ThreadPoolExecutor(
            max_workers=max(1, args.claim_limit),
            thread_name_prefix="tinker-roll-sample",
        )
        self.checkpoints_base = Path(args.checkpoints_base)
        self.checkpoints_base.mkdir(parents=True, exist_ok=True)

    @property
    def runtime_id(self) -> str:
        return self.args.runtime_id

    def heartbeat_url(self) -> str:
        return f"{self.backend_base_url}/api/v1/runtimes/{self.runtime_id}/heartbeat"

    def claim_url(self) -> str:
        return f"{self.backend_base_url}/api/v1/runtimes/{self.runtime_id}/actions/claim"

    def action_result_url(self, action_id: int) -> str:
        return f"{self.backend_base_url}/api/v1/runtimes/{self.runtime_id}/actions/{action_id}/result"

    def step_url(self, env_id: str) -> str:
        return f"{self.backend_base_url}/api/v1/runtimes/{self.runtime_id}/envs/{env_id}/steps"

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
                "runtime": "roll",
                "pipeline": "tinker_backend_runtime",
                "config_path": self.args.config_path,
                "pid": os.getpid(),
            },
        }
        response = post_json(self.heartbeat_url(), payload, timeout=self.args.request_timeout)
        self.last_heartbeat_at = time.monotonic()
        return response

    def claim_actions(self, limit: int | None = None) -> list[dict[str, Any]]:
        payload = {
            "action_types": self.args.action_types,
            "limit": self.args.claim_limit if limit is None else limit,
        }
        response = post_json(self.claim_url(), payload, timeout=self.args.request_timeout)
        actions = response.get("actions", [])
        return actions if isinstance(actions, list) else []

    def post_step(
        self,
        env_id: str,
        step_id: int,
        *,
        prompt: dict[str, Any] | None = None,
        finish_reason: str | None = None,
        reward: float | None = None,
    ) -> dict[str, Any]:
        payload = {
            "step_id": step_id,
            "prompt": prompt,
            "finish_reason": finish_reason,
            "reward": reward,
        }
        return post_json(self.step_url(env_id), payload, timeout=self.args.request_timeout)

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

    def handle_init_task_env(self, action: dict[str, Any]) -> None:
        payload = action.get("payload", {}) if isinstance(action.get("payload"), dict) else {}
        env_id = action.get("env_id") or payload.get("env_id")
        if not env_id:
            raise RuntimeError("init_task_env action missing env_id")

        instance_id = str(payload.get("instance_id") or "unknown")
        dataset_name = str(payload.get("dataset_name") or "unknown")
        dataset_type = str(payload.get("dataset_type") or "unknown")
        prompt_text = self.args.prompt_template.format(
            runtime_id=self.runtime_id,
            env_id=env_id,
            instance_id=instance_id,
            dataset_name=dataset_name,
            dataset_type=dataset_type,
        )
        self.post_step(env_id, 0, prompt=openai_chat_completion_request_from_text(prompt_text))
        self.env_steps[env_id] = 0
        self.post_action_result(action["action_id"], "completed", result_data={"env_id": env_id})

    def _init_roll_backend(self) -> None:
        if self.args.backend_config is None:
            return
        log_event("runtime_backend_init_start", runtime_id=self.runtime_id, backend_config=self.args.backend_config)
        self.roll_backend = init_roll_backend(self.args.backend_config)
        log_event("runtime_backend_init_ready", runtime_id=self.runtime_id)

    def _require_backend(self):
        if self.roll_backend is None:
            raise RuntimeError("ROLL backend is not initialized")
        return self.roll_backend

    def _decode_sample_text(self, tokens: list[int]) -> str:
        if not tokens:
            return ""
        backend = self.roll_backend
        tokenizer = getattr(backend, "tokenizer", None) if backend is not None else None
        if tokenizer is not None:
            try:
                return tokenizer.decode(tokens, skip_special_tokens=False)
            except TypeError:
                return tokenizer.decode(tokens)
            except Exception as exc:
                raise RuntimeError(f"failed to decode sample output tokens: {exc}") from exc
        raise RuntimeError("cannot decode sample output tokens without a tokenizer")

    def _enrich_sample_response(
        self,
        payload: dict[str, Any],
        data: dict[str, Any],
        *,
        prompt_token_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        if prompt_token_ids is not None:
            data["prompt_token_ids"] = prompt_token_ids
        else:
            data.setdefault("prompt_token_ids", [])
        request = _request_from_payload(payload)
        raw_tools = request.get("tools") or []
        for sequence in data.get("sequences", []) or []:
            if not isinstance(sequence, dict):
                continue
            tokens = [int(token) for token in (sequence.get("tokens") or [])]
            if sequence.get("output_token_ids") is None:
                sequence["output_token_ids"] = tokens
            raw_text = sequence.get("raw_text") or sequence.get("text") or self._decode_sample_text(tokens)
            parser_name = self.args.runtime_config.get("tool_call_parser")
            if parser_name:
                from roll.pipeline.tinker_backend_runtime.openai_runtime.runtime_pipeline_openai import (
                    _parse_tool_calls_with_sglang,
                )

                parsed_text, tool_calls, finish_reason = _parse_tool_calls_with_sglang(
                    str(raw_text), raw_tools, str(parser_name)
                )
            else:
                parsed_text, tool_calls, finish_reason = _parse_qwen_tool_calls_from_text(str(raw_text), raw_tools)
            if sequence.get("raw_text") is None:
                sequence["raw_text"] = raw_text
            tokenizer = getattr(self.roll_backend, "tokenizer", None)
            eos_id = getattr(tokenizer, "eos_token_id", None)
            eos_marker = getattr(tokenizer, "eos_token", None)
            if (tokens and eos_id is not None and tokens[-1] == eos_id
                    and isinstance(eos_marker, str) and eos_marker):
                end = len(parsed_text.rstrip())
                if parsed_text[:end].endswith(eos_marker):
                    # OpenAI content excludes transport EOS; sampled evidence is unchanged.
                    parsed_text = parsed_text[:end - len(eos_marker)] + parsed_text[end:]
            sequence["text"] = parsed_text
            if tool_calls:
                sequence["tool_calls"] = tool_calls
                sequence["finish_reason"] = finish_reason
            else:
                sequence.setdefault("tool_calls", [])
                sequence.setdefault("finish_reason", sequence.get("stop_reason", "stop"))
        return data

    def handle_sample(self, action: dict[str, Any]) -> None:
        payload = action.get("payload", {}) if isinstance(action.get("payload"), dict) else {}
        env_id = action.get("env_id") or payload.get("env_id")

        if self.roll_backend is not None:
            sample_response = self.handle_backend_sample(action)
        else:
            if not env_id:
                raise RuntimeError("sample action missing env_id")
            sample_response = self.generator.generate(payload=payload, runtime_id=self.runtime_id, env_id=env_id)

        self.post_action_result(action["action_id"], "completed", result_data=sample_response)

        if self.roll_backend is None and self.args.finish_after_sample and env_id:
            next_step = self.env_steps.get(env_id, 0) + 1
            self.env_steps[env_id] = next_step
            self.post_step(env_id, next_step, prompt=None, finish_reason="finish", reward=self.args.reward)

    def handle_create_model(self, action: dict[str, Any]) -> dict[str, Any]:
        backend = self._require_backend()
        payload = action.get("payload") or {}
        model_id = str(payload["model_id"])
        raw_lora = payload.get("lora_config") or {}
        lora_config = tinker_types.LoraConfig(
            rank=int(raw_lora.get("rank", 0)),
            alpha=float(raw_lora.get("alpha", self.args.lora_alpha)),
            seed=int(raw_lora.get("seed") if raw_lora.get("seed") is not None else self.args.lora_seed),
            train_attn=bool(raw_lora.get("train_attn", True)),
            train_mlp=bool(raw_lora.get("train_mlp", True)),
            train_unembed=bool(raw_lora.get("train_unembed", True)),
        )
        backend.create_model(model_id, lora_config)
        return {
            "type": "create_model",
            "model_id": model_id,
            "base_model": payload.get("base_model"),
            "lora_config": _dump_model(lora_config),
        }

    def handle_model_pass(self, action: dict[str, Any], *, forward_only: bool) -> dict[str, Any]:
        backend = self._require_backend()
        payload = action.get("payload") or {}
        model_id = str(payload["model_id"])
        key = "forward_input" if forward_only else "forward_backward_input"
        request_input = _normalize_forward_input(payload, key)
        request_id = str(action["action_id"])
        prepared = prepare_model_pass_batch({request_id: (model_id, request_input)})
        results = backend.forward(prepared) if forward_only else backend.forward_backward(prepared)
        return _dump_model(results[request_id])

    def handle_optim_step(self, action: dict[str, Any]) -> dict[str, Any]:
        backend = self._require_backend()
        payload = action.get("payload") or {}
        adam = payload.get("adam_params") or {}
        optim_input = tinker_types.OptimStepInput(
            adam_params=tinker_types.AdamParams(
                learning_rate=float(adam.get("learning_rate", 1e-4)),
                beta1=float(adam.get("beta1", 0.9)),
                beta2=float(adam.get("beta2", 0.95)),
                eps=float(adam.get("eps", 1e-12)),
                weight_decay=float(adam.get("weight_decay", 0.0)),
            )
        )
        result = backend.optim_step(str(payload["model_id"]), optim_input)
        data = _dump_model(result)
        data.setdefault("type", "optim_step")
        return data

    def _training_checkpoint_path(self, model_id: str, checkpoint_id: str) -> Path:
        return self.checkpoints_base / model_id / "weights" / f"{checkpoint_id}.tar.gz"

    def _sampler_checkpoint_path(self, model_id: str, checkpoint_id: str) -> Path:
        return self.checkpoints_base / model_id / "sampler_weights" / f"{checkpoint_id}.tar.gz"

    def handle_save_weights(self, action: dict[str, Any]) -> dict[str, Any]:
        backend = self._require_backend()
        payload = action.get("payload") or {}
        model_id = str(payload["model_id"])
        checkpoint_id = str(payload.get("checkpoint_id") or payload.get("path"))
        path = self._training_checkpoint_path(model_id, checkpoint_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        backend.save_checkpoint(path, model_id)
        return {"type": "save_weights", "path": f"tinker://{model_id}/weights/{checkpoint_id}"}

    def handle_publish_to_sampler(self, action: dict[str, Any]) -> dict[str, Any]:
        backend = self._require_backend()
        payload = action.get("payload") or {}
        model_id = str(payload["model_id"])
        checkpoint_id = str(payload["checkpoint_id"])
        sampling_session_id = payload.get("sampling_session_id")
        persist = sampling_session_id is None
        path = self._sampler_checkpoint_path(model_id, checkpoint_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        backend.save_sampler_checkpoint(path, model_id, persist=persist)
        return {
            "type": "publish_to_sampler",
            "path": None if sampling_session_id else f"tinker://{model_id}/sampler_weights/{checkpoint_id}",
            "sampling_session_id": sampling_session_id,
        }

    def handle_load_weights(self, action: dict[str, Any]) -> dict[str, Any]:
        backend = self._require_backend()
        payload = action.get("payload") or {}
        model_id = str(payload["model_id"])
        source_model_id = str(payload["source_model_id"])
        checkpoint_id = str(payload["checkpoint_id"])
        path = self._training_checkpoint_path(source_model_id, checkpoint_id)
        backend.load_checkpoint(path, model_id, load_optimizer=bool(payload.get("optimizer", False)))
        return {"type": "load_weights"}

    def handle_backend_sample(self, action: dict[str, Any]) -> dict[str, Any]:
        backend = self._require_backend()
        payload = action.get("payload") or {}
        prompt = payload.get("prompt")
        model_input_payload = _model_input_from_payload(payload)
        if prompt is None and model_input_payload is None:
            raise RuntimeError("sample action missing prompt or model_input")
        if model_input_payload is not None:
            model_input = tinker_types.ModelInput.model_validate(model_input_payload)
            prompt_token_ids = _model_input_to_token_ids(model_input)
        else:
            request = _request_from_payload(payload)
            prompt_token_ids = _render_prompt_token_ids(
                getattr(backend, "tokenizer", None),
                request,
                enable_thinking=bool(self.args.runtime_config.get("enable_thinking", False)),
            )
            model_input = tinker_types.ModelInput.model_validate(model_input_from_token_ids(prompt_token_ids))
        model_id = str(payload.get("model_id") or "")
        checkpoint_id = str(payload.get("checkpoint_id") or "")
        sample_input = tinker_types.SampleInput(
            base_model=payload.get("base_model"),
            prompt=model_input,
            sampling_params=_normalize_sampling_params(payload.get("sampling_params")),
            num_samples=int(payload.get("num_samples") or 1),
            checkpoint_id=checkpoint_id,
            prompt_logprobs=bool(payload.get("prompt_logprobs", False)),
            env_id=action.get("env_id") or payload.get("env_id"),
        )
        request_id = str(action["action_id"])
        prepared = prepare_sample_batch({request_id: (model_id, sample_input)}, self.checkpoints_base)
        results = backend.sample(prepared)
        data = _dump_model(results[request_id])
        data.setdefault("type", "sample")
        return self._enrich_sample_response(payload, data, prompt_token_ids=prompt_token_ids)

    def _close_roll_backend(self) -> None:
        if self.roll_backend is None or getattr(self, "_roll_backend_closed", False):
            return
        close = getattr(self.roll_backend, "close", None)
        if callable(close):
            close()
        self._roll_backend_closed = True

    def handle_close_runtime(self, action: dict[str, Any]) -> None:
        try:
            close = getattr(self.generator, "close", None)
            if callable(close):
                close()
            self._close_roll_backend()
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
        if action_type == "init_task_env":
            self.handle_init_task_env(action)
        elif action_type == "sample":
            self.handle_sample(action)
        elif action_type == "create_model":
            self.post_action_result(action["action_id"], "completed", result_data=self.handle_create_model(action))
        elif action_type == "forward":
            self.post_action_result(action["action_id"], "completed", result_data=self.handle_model_pass(action, forward_only=True))
        elif action_type == "forward_backward":
            self.post_action_result(action["action_id"], "completed", result_data=self.handle_model_pass(action, forward_only=False))
        elif action_type == "optim_step":
            self.post_action_result(action["action_id"], "completed", result_data=self.handle_optim_step(action))
        elif action_type == "save_weights":
            self.post_action_result(action["action_id"], "completed", result_data=self.handle_save_weights(action))
        elif action_type == "publish_to_sampler":
            self.post_action_result(action["action_id"], "completed", result_data=self.handle_publish_to_sampler(action))
        elif action_type == "load_weights":
            self.post_action_result(action["action_id"], "completed", result_data=self.handle_load_weights(action))
        elif action_type == "close_runtime":
            self.handle_close_runtime(action)
        else:
            raise RuntimeError(f"unsupported action_type={action_type!r}")

    @staticmethod
    def _is_rollout_sample_action(action: dict[str, Any]) -> bool:
        if action.get("action_type") != "sample":
            return False
        payload = action.get("payload") if isinstance(action.get("payload"), dict) else {}
        return bool(action.get("env_id") or payload.get("env_id"))

    def _execute_action(self, action: dict[str, Any]) -> None:
        try:
            self.handle_action(action)
            log_event(
                "action_completed",
                action_id=action.get("action_id"),
                action_type=action.get("action_type"),
            )
        except Exception as exc:
            log_event("action_failed", action=action, error=str(exc))
            with contextlib.suppress(Exception):
                self.post_action_result(action["action_id"], "failed", error_message=str(exc))

    async def _run_rollout_sample_action(self, action: dict[str, Any]) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.action_executor, self._execute_action, action)

    async def _wait_for_rollout_samples(self) -> None:
        if self.sample_tasks:
            await asyncio.gather(*tuple(self.sample_tasks))

    async def _poll_actions(self) -> None:
        empty_claim_count = 0
        while not self.should_stop:
            try:
                self.maybe_report_periodic_heartbeat()
                available_slots = max(0, self.args.claim_limit - len(self.sample_tasks))
                if available_slots == 0:
                    await asyncio.sleep(max(self.args.action_poll_interval, 0.05))
                    continue

                actions = self.claim_actions(limit=available_slots)
                if not actions:
                    if self.sample_tasks:
                        await asyncio.sleep(max(self.args.action_poll_interval, 0.05))
                        continue
                    backoff = self.args.empty_claim_backoff_seconds[
                        min(empty_claim_count, len(self.args.empty_claim_backoff_seconds) - 1)
                    ]
                    empty_claim_count += 1
                    await asyncio.sleep(backoff)
                    continue

                empty_claim_count = 0
                for action in actions:
                    if self._is_rollout_sample_action(action):
                        task = asyncio.create_task(self._run_rollout_sample_action(action))
                        self.sample_tasks.add(task)
                        task.add_done_callback(self.sample_tasks.discard)
                        continue

                    await self._wait_for_rollout_samples()
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(self.action_executor, self._execute_action, action)
                    if self.should_stop:
                        break

                if not self.should_stop and self.args.action_poll_interval > 0:
                    await asyncio.sleep(self.args.action_poll_interval)
            except BackendRequestError as exc:
                log_event("runtime_loop_error", error=str(exc))
                await asyncio.sleep(self.args.empty_claim_backoff_seconds[-1])

        await self._wait_for_rollout_samples()

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
            try:
                self._init_roll_backend()
                if self.args.preload_generator and self.generator is not None:
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
            asyncio.run(self._poll_actions())
            log_event("runtime_closed", runtime_id=self.runtime_id, pid=os.getpid())
            return 0
        finally:
            try:
                # Only the explicit local mode owns a Ray instance to tear down
                # on early/error exits. Default cluster cleanup stays action-driven.
                backend_config = getattr(self.args, "backend_config", None)
                if isinstance(backend_config, dict) and backend_config.get("ray_address") == "local":
                    self._close_roll_backend()
            finally:
                self.action_executor.shutdown(wait=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROLL Tinker-backend runtime pipeline")
    parser.add_argument("--runtime-id", default=os.environ.get("TINKER_RUNTIME_ID"), required=False)
    parser.add_argument("--backend-base-url", default=os.environ.get("TINKER_BACKEND_BASE_URL"), required=False)
    parser.add_argument("--config-path", default=os.environ.get("TINKER_CONFIG_PATH", ""), required=False)
    parser.add_argument("--adapter-base-url", default=os.environ.get("TINKER_ROLL_ADAPTER_BASE_URL"), required=False)
    parser.add_argument("--exit-after-ready", action="store_true", default=False)
    parsed, remaining = parser.parse_known_args(argv)

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
        default=int(os.environ.get("TINKER_ROLL_CLAIM_LIMIT", _get_config_value(config, "claim_limit", 10))),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=float(os.environ.get("TINKER_ROLL_REQUEST_TIMEOUT", _get_config_value(config, "request_timeout", 30))),
    )
    parser.add_argument(
        "--startup-retries",
        type=int,
        default=int(os.environ.get("TINKER_ROLL_STARTUP_RETRIES", _get_config_value(config, "startup_retries", 30))),
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
        "--prompt-template",
        default=os.environ.get(
            "TINKER_ROLL_PROMPT_TEMPLATE",
            str(
                _get_config_value(
                    config,
                    "prompt_template",
                    "ROLL Tinker-backend runtime prompt. instance_id={instance_id}; dataset_name={dataset_name}; dataset_type={dataset_type}.",
                )
            ),
        ),
    )
    parser.add_argument(
        "--sample-text",
        default=os.environ.get(
            "TINKER_ROLL_SAMPLE_TEXT",
            str(_get_config_value(config, "sample_text", "ROLL runtime response from {runtime_id} for {env_id}.")),
        ),
    )
    parser.add_argument(
        "--reward",
        type=float,
        default=float(os.environ.get("TINKER_ROLL_REWARD", _get_config_value(config, "reward", 1.0))),
    )
    parser.add_argument(
        "--finish-after-sample",
        action="store_true",
        default=_as_bool(
            os.environ.get("TINKER_ROLL_FINISH_AFTER_SAMPLE"),
            _as_bool(runtime_config.get("finish_after_sample"), True),
        ),
    )
    parser.add_argument(
        "--action-types",
        nargs="+",
        default=runtime_config.get("action_types") or DEFAULT_ACTION_TYPES,
    )
    parser.add_argument(
        "--preload-generator",
        action="store_true",
        default=_as_bool(
            os.environ.get("TINKER_ROLL_PRELOAD_GENERATOR"),
            _as_bool(runtime_config.get("preload_generator"), False),
        ),
    )
    parser.add_argument(
        "--checkpoints-base",
        default=str(_get_config_value(config, "checkpoints_base", "/tmp/tinker-backend-checkpoints")),
    )
    parser.add_argument("--lora-alpha", type=float, default=float(_get_config_value(config, "lora_alpha", 32.0)))
    parser.add_argument("--lora-seed", type=int, default=int(_get_config_value(config, "lora_seed", 0)))
    args = parser.parse_args(argv)
    args.runtime_config = runtime_config
    backend_config = runtime_config.get("backend_config")
    args.backend_config = backend_config if isinstance(backend_config, dict) else None

    if not args.runtime_id:
        parser.error("--runtime-id or TINKER_RUNTIME_ID is required")
    if not args.backend_base_url:
        parser.error("--backend-base-url or TINKER_BACKEND_BASE_URL is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pipeline = TinkerBackendRuntimePipeline(args)
    try:
        return pipeline.run()
    except KeyboardInterrupt:
        log_event("runtime_interrupted", runtime_id=args.runtime_id)
        return 130
    except Exception as exc:
        log_event(
            "runtime_failed",
            runtime_id=args.runtime_id,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
