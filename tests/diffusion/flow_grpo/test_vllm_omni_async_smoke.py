import asyncio
import json
import os
import socket
import uuid
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.utils.checkpoint_manager import download_model
from roll.third_party.vllm_omni import create_async_llm_omni


TEST_CONFIG_PATH = Path("examples/qwen-image-diffusion/flow_grpo_fsdp2.yaml")
QWEN_IMAGE_SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:"
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _normalize_token_ids(tokenized_output):
    token_ids = tokenized_output
    if isinstance(tokenized_output, dict) and "input_ids" in tokenized_output:
        token_ids = tokenized_output["input_ids"]
    elif hasattr(tokenized_output, "input_ids"):
        token_ids = tokenized_output.input_ids

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)
    if isinstance(token_ids, list) and len(token_ids) == 1 and isinstance(token_ids[0], (list, tuple)):
        token_ids = list(token_ids[0])
    return [int(x.item() if hasattr(x, "item") else x) for x in token_ids]


def _chat_template_to_ids(tokenizer, messages, max_length: int):
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        max_length=max_length,
        padding=True,
        truncation=True,
    )
    return _normalize_token_ids(token_ids)


def _normalize_vllm_omni_multi_modal_data(multi_modal_data: Dict[str, Any]) -> Dict[str, Any]:
    from roll.distributed.scheduler.router import RouterClient

    client = RouterClient(
        proxy=None,
        meta={"strategy_name": "vllm_omni", "eos_token_id": 0, "pad_token_id": 0},
    )
    return client._normalize_vllm_omni_multi_modal_data(multi_modal_data)


def _resolve_device_mapping_expr(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    if not isinstance(value, str):
        raise TypeError(f"Unsupported device_mapping type: {type(value).__name__}")
    return [int(v) for v in eval(value, {"__builtins__": {}}, {"list": list, "range": range})]


def _load_actor_infer_config() -> Dict[str, Any]:
    omegaconf = pytest.importorskip("omegaconf")
    cfg = omegaconf.OmegaConf.load(TEST_CONFIG_PATH)
    resolved = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    actor_infer = resolved["actor_infer"]
    strategy_config = dict(actor_infer["strategy_args"]["strategy_config"])
    vllm_omni_cfg = dict(strategy_config.pop("vllm_omni", {}))
    custom_pipeline = vllm_omni_cfg.get("custom_pipeline")
    if custom_pipeline is not None:
        strategy_config["diffusion_load_format"] = "custom_pipeline"
        strategy_config["custom_pipeline_args"] = {"pipeline_class": custom_pipeline}

    return {
        "seed": int(resolved.get("seed", 42)),
        "model_path": actor_infer["model_args"]["model_name_or_path"],
        "dtype": actor_infer["model_args"]["dtype"],
        "device_mapping": _resolve_device_mapping_expr(actor_infer["device_mapping"]),
        "strategy_config": strategy_config,
        "sampling_params": dict(actor_infer["generating_args"]),
        "custom_pipeline": custom_pipeline,
    }


def _build_payload(tokenizer, sampling_params: Dict[str, Any]) -> Dict[str, Any]:
    prompt_messages = [
        {"role": "system", "content": QWEN_IMAGE_SYSTEM_PROMPT},
        {"role": "user", "content": os.getenv("ROLL_TEST_VLLM_OMNI_PROMPT", "A photo of a fluffy cat in sunlight.")},
    ]
    negative_prompt_messages = [
        {"role": "system", "content": QWEN_IMAGE_SYSTEM_PROMPT},
        {"role": "user", "content": os.getenv("ROLL_TEST_VLLM_OMNI_NEGATIVE_PROMPT", " ")},
    ]

    prompt_ids = _chat_template_to_ids(tokenizer, prompt_messages, max_length=1058)
    negative_prompt_ids = _chat_template_to_ids(tokenizer, negative_prompt_messages, max_length=1058)

    multi_modal_data = {
        "prompt_ids": prompt_ids,
        "prompt_mask": [1] * len(prompt_ids),
        "negative_prompt_ids": negative_prompt_ids,
        "negative_prompt_mask": [1] * len(negative_prompt_ids),
    }

    return {
        "rid": f"smoke-{uuid.uuid4().hex}",
        "multi_modal_data": _normalize_vllm_omni_multi_modal_data(multi_modal_data),
        "sampling_params": dict(sampling_params),
    }


def _summarize_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {k: _summarize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "len": len(value),
            "first": _summarize_value(value[0]) if value else None,
        }
    return {"type": type(value).__name__, "value": repr(value)}


def _summarize_final_output(final_output: Any) -> Dict[str, Any]:
    custom_output = getattr(final_output, "custom_output", None)
    request_output = getattr(final_output, "request_output", None)
    outputs = getattr(final_output, "outputs", None)
    return {
        "final_output_type": type(final_output).__name__ if final_output is not None else None,
        "custom_output_type": type(custom_output).__name__ if custom_output is not None else None,
        "custom_output_keys": sorted(custom_output.keys()) if isinstance(custom_output, dict) else None,
        "custom_output_summary": _summarize_value(custom_output) if custom_output is not None else None,
        "request_output_type": type(request_output).__name__ if request_output is not None else None,
        "request_output_keys": sorted(request_output.keys()) if isinstance(request_output, dict) else None,
        "request_output_summary": _summarize_value(request_output) if isinstance(request_output, dict) else None,
        "outputs_len": len(outputs) if hasattr(outputs, "__len__") else None,
    }


def _inspect_object_fields(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {"type": None, "fields": None}

    field_names = []
    try:
        for name in dir(obj):
            if name.startswith("_"):
                continue
            try:
                value = getattr(obj, name)
            except Exception as exc:
                field_names.append(f"{name}<error:{type(exc).__name__}>")
                continue
            if callable(value):
                continue
            field_names.append(name)
    except Exception as exc:
        return {"type": type(obj).__name__, "fields_error": repr(exc)}

    return {
        "type": type(obj).__name__,
        "fields": sorted(field_names),
    }


def _extract_tensor_fields(mapping: Any) -> Dict[str, Any] | None:
    if not isinstance(mapping, dict):
        return None
    return {
        key: _summarize_value(value)
        for key, value in mapping.items()
    }


async def _collect_raw_final_output(async_omni, payload: Dict[str, Any]) -> Any:
    request_id = str(payload.get("rid", uuid.uuid4().hex))
    prompt = payload.get("multi_modal_data", payload.get("input_ids"))
    sampling_params = async_omni._convert_sampling_params(payload.get("sampling_params", {}))

    if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
        prompt = [{"prompt_ids": prompt}]
    elif isinstance(prompt, dict) and "prompt_ids" in prompt:
        prompt = [prompt]

    final_output = None
    async for output in async_omni._engine.generate(
        prompt=prompt,
        request_id=request_id,
        sampling_params_list=sampling_params,
    ):
        final_output = output
    return final_output


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="requires at least one GPU")
def test_vllm_omni_async_smoke_custom_output_visibility():
    pytest.importorskip("vllm_omni")
    transformers = pytest.importorskip("transformers")
    import ray

    started_ray = False
    if not ray.is_initialized():
        # Force a local Ray runtime for this single-node smoke test before any
        # helper (such as download_model) can auto-initialize Ray to a cluster.
        ray.init(address="local", ignore_reinit_error=True, log_to_driver=True)
        started_ray = True

    config = _load_actor_infer_config()
    model_path = download_model(config["model_path"])
    required_gpus = len(config["device_mapping"])

    if torch.cuda.device_count() < required_gpus:
        pytest.skip(
            f"{TEST_CONFIG_PATH} actor_infer.device_mapping requires {required_gpus} GPUs, "
            f"but only found {torch.cuda.device_count()}"
        )

    resource_manager = None
    placement_groups = None
    async_omni = None

    try:
        print(f"[smoke] loaded actor_infer config from {TEST_CONFIG_PATH}")
        print(f"[smoke] model_path={model_path}")
        print(f"[smoke] required_gpus={required_gpus} visible_gpus={torch.cuda.device_count()}")

        tokenizer_path = os.path.join(model_path, "tokenizer")
        tokenizer_source = tokenizer_path if os.path.exists(tokenizer_path) else model_path
        print(f"[smoke] before tokenizer load: tokenizer_source={tokenizer_source}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True, use_fast=False)
        print("[smoke] after tokenizer load")

        payload = _build_payload(tokenizer, config["sampling_params"])
        print("[smoke] after payload build")

        device_mapping = config["device_mapping"]
        print(f"[smoke] before ResourceManager: device_mapping={device_mapping}")
        resource_manager = ResourceManager(num_gpus_per_node=len(device_mapping), num_nodes=1)
        print("[smoke] after ResourceManager init")
        placement_groups = resource_manager.allocate_placement_group(world_size=1, device_mapping=device_mapping)
        print("[smoke] after placement group allocation")

        async def _run():
            nonlocal async_omni
            init_kwargs = dict(config["strategy_config"])
            init_kwargs.update(
                {
                    "model": model_path,
                    "dtype": "bfloat16" if config["dtype"] == "bf16" else config["dtype"],
                    "seed": config["seed"],
                    "master_port": _find_free_port(),
                }
            )
            print("[smoke] before create_async_llm_omni")
            print(json.dumps(_summarize_value(init_kwargs), ensure_ascii=False, indent=2, default=str))
            async_omni = await create_async_llm_omni(
                resource_placement_groups=placement_groups[0],
                **init_kwargs,
            )
            print("[smoke] after create_async_llm_omni")

            print("[smoke] before raw final output collection")
            raw_final_output = await _collect_raw_final_output(async_omni, payload)
            print("[smoke] after raw final output collection")
            raw_summary = _summarize_final_output(raw_final_output)
            raw_fields = _inspect_object_fields(raw_final_output)
            raw_custom_output = getattr(raw_final_output, "custom_output", None)
            raw_request_output = getattr(raw_final_output, "request_output", None)
            print("\n[smoke] raw final output summary:")
            print(json.dumps(raw_summary, ensure_ascii=False, indent=2, default=str))
            print("\n[smoke] raw final output fields:")
            print(json.dumps(raw_fields, ensure_ascii=False, indent=2, default=str))
            print("\n[smoke] raw custom_output details:")
            print(json.dumps(_extract_tensor_fields(raw_custom_output), ensure_ascii=False, indent=2, default=str))
            print("\n[smoke] raw request_output fields:")
            print(json.dumps(_inspect_object_fields(raw_request_output), ensure_ascii=False, indent=2, default=str))
            if isinstance(raw_request_output, dict):
                print("\n[smoke] raw request_output details:")
                print(json.dumps(_extract_tensor_fields(raw_request_output), ensure_ascii=False, indent=2, default=str))
            print("\n[smoke] init kwargs summary:")
            print(json.dumps(_summarize_value(init_kwargs), ensure_ascii=False, indent=2, default=str))
            print("\n[smoke] payload summary:")
            print(json.dumps(_summarize_value(payload), ensure_ascii=False, indent=2, default=str))

            print("[smoke] before adapted generate_request")
            adapted_output = await async_omni.generate_request(payload)
            print("[smoke] after adapted generate_request")
            adapted_summary = _summarize_value(adapted_output)
            print("\n[smoke] adapted generate_request output summary:")
            print(json.dumps(adapted_summary, ensure_ascii=False, indent=2, default=str))
            if isinstance(adapted_output, dict):
                print("\n[smoke] adapted generate_request output details:")
                print(json.dumps(_extract_tensor_fields(adapted_output), ensure_ascii=False, indent=2, default=str))

            return raw_summary, adapted_output

        raw_summary, adapted_output = asyncio.run(_run())

        assert raw_summary["final_output_type"] is not None
        assert isinstance(adapted_output, dict)
        assert "custom_output_keys" in raw_summary

        # adapted_output must carry diffusion tensors, not text-fallback format.
        required_keys = {
            "responses",
            "all_latents",
            "all_timesteps",
            "rollout_log_probs",
            "prompt_embeds",
            "prompt_embeds_mask",
        }
        missing = required_keys - set(adapted_output.keys())
        assert not missing, f"adapted_output missing diffusion keys: {missing}"

        all_latents = torch.as_tensor(adapted_output["all_latents"])
        all_timesteps = torch.as_tensor(adapted_output["all_timesteps"])
        rollout_log_probs = torch.as_tensor(adapted_output["rollout_log_probs"])

        assert all_latents.ndim >= 2, f"all_latents ndim={all_latents.ndim}"
        assert all_timesteps.ndim >= 2, f"all_timesteps ndim={all_timesteps.ndim}"
        assert rollout_log_probs.ndim >= 2, f"rollout_log_probs ndim={rollout_log_probs.ndim}"

        k = rollout_log_probs.shape[1]
        assert all_timesteps.shape[1] == k, f"all_timesteps.shape[1]={all_timesteps.shape[1]} != K={k}"
        assert all_latents.shape[1] == k + 1, f"all_latents.shape[1]={all_latents.shape[1]} != K+1={k + 1}"
    finally:
        if async_omni is not None:
            try:
                asyncio.run(async_omni.sleep(level=1))
            except Exception:
                pass
        if resource_manager is not None:
            try:
                resource_manager.destroy_placement_group()
            except Exception:
                pass
        if started_ray:
            ray.shutdown()
