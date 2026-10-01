import os
import sys
import json
import asyncio
import time
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
from PIL import Image

from roll.configs import ModelArguments
from roll.configs.worker_config import StrategyArguments, WorkerConfig
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.vllm_omni_strategy import VllmOmniStrategy
from roll.pipeline.base_worker import InferWorker
from roll.third_party.vllm_omni import assert_vllm_omni_version
from roll.third_party.vllm_omni.async_omni import CustomAsyncOmni
from roll.third_party.vllm_omni.worker import VllmOmniColocateWorkerExtension
from roll.utils.constants import RAY_NAMESPACE


VLLM_OMNI_TEST_MODEL_PATH = "/mnt/vdb/lrq/models/Qwen/Qwen-Image"
CUSTOM_PIPELINE_QUALNAME = "roll.pipeline.diffusion.models.qwen_image.vllm_omni_qwen_image_adapter.QwenImagePipelineWithLogProb"
_TOKENIZER_MAX_LENGTH = 1024


def _compute_encode_start_idx(tokenizer, system_prompt):
    """Compute the number of prefix tokens (system + user header) before user content."""
    from roll.datasets.chat_template import get_chat_template
    template_func = get_chat_template("native", tokenizer)
    text_a = template_func([{"role": "system", "content": system_prompt}, {"role": "user", "content": "A"}])
    text_b = template_func([{"role": "system", "content": system_prompt}, {"role": "user", "content": "B"}])
    ids_a = tokenizer([text_a])["input_ids"][0]
    ids_b = tokenizer([text_b])["input_ids"][0]
    min_len = min(len(ids_a), len(ids_b))
    for i in range(min_len):
        if ids_a[i] != ids_b[i]:
            return i
    return min_len


_REQUIRED_OUTPUT_KEYS = {
    "responses",
    "rollout_log_probs",
    "all_timesteps",
    "all_latents",
    "prompt_embeds",
    "prompt_embeds_mask",
    "negative_prompt_embeds",
    "negative_prompt_embeds_mask",
}
ROLLOUT_RESULTS_DIR = Path("/tmp/roll_results_cmp")


def _chat_template_to_ids_kwargs(tokenizer, messages, kwargs):
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        **kwargs,
    )
    token_ids = _normalize_token_ids(token_ids)
    return token_ids, None


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


def _tensor_to_json_dict(v):
    t = torch.as_tensor(v).detach().cpu()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "values": t.tolist(),
    }


def _save_response_images(responses, output_dir: Path):
    t = torch.as_tensor(responses).detach().cpu()
    if t.ndim == 3:
        t = t.unsqueeze(0)
    if t.ndim != 4:
        return []

    paths = []
    for i in range(t.shape[0]):
        img = t[i]
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0)

        img = img.float()
        if img.numel() == 0:
            continue
        min_v = float(img.min())
        max_v = float(img.max())
        if max_v <= 1.0 and min_v >= 0.0:
            img = img * 255.0
        elif max_v > min_v:
            img = (img - min_v) / (max_v - min_v) * 255.0
        else:
            img = torch.zeros_like(img)

        img_np = img.clamp(0, 255).to(torch.uint8).numpy()
        if img_np.ndim == 3 and img_np.shape[-1] == 1:
            img_np = img_np[..., 0]
        if img_np.ndim == 3 and img_np.shape[-1] > 3:
            img_np = img_np[..., :3]

        img_path = output_dir / f"response_{i}.png"
        Image.fromarray(img_np).save(img_path)
        paths.append(str(img_path))
    return paths


def _save_rollout_output(output: dict, output_dir: Path, request_meta: dict, perfs: dict | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    result_json = {"request": request_meta, "outputs": {}, "perfs": perfs or {}, "images": []}

    for k, v in output.items():
        if k == "responses":
            result_json["images"] = _save_response_images(v, output_dir)
        else:
            result_json["outputs"][k] = _tensor_to_json_dict(v)

    with open(output_dir / "rollout_output.json", "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False)


def _pad_rollout_output_for_verl_parity(output: dict, target_prompt_len: int) -> dict:
    padded = dict(output)

    if "prompt_embeds" in padded:
        pe = torch.as_tensor(padded["prompt_embeds"]).detach().cpu()
        if pe.ndim == 2:
            pe = pe.unsqueeze(0)
        pad_len = max(target_prompt_len - pe.shape[1], 0)
        if pad_len > 0:
            pe = torch.nn.functional.pad(pe, (0, 0, 0, pad_len), value=0)
        padded["prompt_embeds"] = pe

    if "negative_prompt_embeds" in padded and padded["negative_prompt_embeds"] is not None:
        npe = torch.as_tensor(padded["negative_prompt_embeds"]).detach().cpu()
        if npe.ndim == 2:
            npe = npe.unsqueeze(0)
        pad_len = max(target_prompt_len - npe.shape[1], 0)
        if pad_len > 0:
            npe = torch.nn.functional.pad(npe, (0, 0, 0, pad_len), value=0)
        padded["negative_prompt_embeds"] = npe

    if "prompt_embeds_mask" in padded:
        pem = torch.as_tensor(padded["prompt_embeds_mask"]).detach().cpu()
        if pem.ndim == 1:
            pem = pem.unsqueeze(0)
        pad_len = max(target_prompt_len - pem.shape[1], 0)
        if pad_len > 0:
            pem = torch.nn.functional.pad(pem, (0, pad_len), value=0)
        padded["prompt_embeds_mask"] = pem

    if "negative_prompt_embeds_mask" in padded and padded["negative_prompt_embeds_mask"] is not None:
        npem = torch.as_tensor(padded["negative_prompt_embeds_mask"]).detach().cpu()
        if npem.ndim == 1:
            npem = npem.unsqueeze(0)
        pad_len = max(target_prompt_len - npem.shape[1], 0)
        if pad_len > 0:
            npem = torch.nn.functional.pad(npem, (0, pad_len), value=0)
        padded["negative_prompt_embeds_mask"] = npem

    return padded


def test_strategy_factory_returns_vllm_omni_strategy():
    worker_cfg = WorkerConfig(strategy_args=StrategyArguments(strategy_name="vllm_omni", strategy_config={}))
    worker = SimpleNamespace(worker_config=worker_cfg)
    strategy = create_strategy(worker=worker)
    assert isinstance(strategy, VllmOmniStrategy)


def test_vllm_omni_version_gate(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_omni", SimpleNamespace(__version__="0.17.0rc1"))
    assert assert_vllm_omni_version() == "0.17.0rc1"

    monkeypatch.setitem(sys.modules, "vllm_omni", SimpleNamespace(__version__="0.17.0"))
    with pytest.raises(RuntimeError):
        assert_vllm_omni_version()


def test_vllm_omni_output_contract_validation():
    strategy = VllmOmniStrategy(worker=SimpleNamespace(worker_config=WorkerConfig(), pipeline_config=SimpleNamespace(seed=1)))

    ok_output = {
        "responses": torch.rand(2, 3, 32, 32),
        "rollout_log_probs": torch.rand(2, 4),
        "all_timesteps": torch.rand(2, 4),
        "all_latents": torch.rand(2, 5, 8),
        "prompt_embeds": torch.rand(2, 4, 16),
        "prompt_embeds_mask": torch.ones(2, 4),
        "negative_prompt_embeds": torch.rand(2, 4, 16),
        "negative_prompt_embeds_mask": torch.ones(2, 4),
    }
    strategy._normalize_output(ok_output)

    bad_output = dict(ok_output)
    bad_output["all_latents"] = torch.rand(2, 4, 8)
    with pytest.raises(ValueError):
        strategy._normalize_output(bad_output)


def test_vllm_omni_generate_request_injects_seed_when_missing():
    class DummyModel:
        def __init__(self):
            self.last_payload = None

        async def generate_request(self, payload):
            self.last_payload = payload
            return {
                "responses": torch.rand(1, 3, 32, 32),
                "rollout_log_probs": torch.rand(1, 2),
                "all_timesteps": torch.rand(1, 2),
                "all_latents": torch.rand(1, 3, 8),
                "prompt_embeds": torch.rand(1, 2, 16),
                "prompt_embeds_mask": torch.ones(1, 2),
                "negative_prompt_embeds": torch.rand(1, 2, 16),
                "negative_prompt_embeds_mask": torch.ones(1, 2),
            }

    worker = SimpleNamespace(
        worker_config=WorkerConfig(),
        pipeline_config=SimpleNamespace(seed=11),
        rank=2,
    )
    strategy = VllmOmniStrategy(worker=worker)
    strategy.model = DummyModel()

    payload = {"rid": "r0", "sampling_params": {"num_inference_steps": 10}}
    _ = asyncio.run(strategy.generate_request(payload))

    assert "seed" in strategy.model.last_payload["sampling_params"]
    assert strategy.model.last_payload["sampling_params"]["seed"] == 13
    # input payload should not be mutated in-place
    assert "seed" not in payload["sampling_params"]


def test_custom_async_omni_prefers_local_control_path_for_inline_runtime():
    class DummyLocalRuntime:
        def __init__(self):
            self.calls = []

        def sleep(self, level):
            self.calls.append(("sleep", level))
            return True

        def update_parameter_in_bucket(self, serialized_named_tensors, is_lora=False):
            self.calls.append(("update_parameter_in_bucket", serialized_named_tensors, is_lora))
            return True

    class DummyEngine:
        def __init__(self):
            self.inline_runtime = DummyLocalRuntime()
            self.collective_calls = []

        async def sleep(self, level=1):
            self.collective_calls.append(("sleep", level))
            raise AssertionError("engine sleep path should not be used in inline runtime")

        async def collective_rpc(self, method, args=(), kwargs=None):
            self.collective_calls.append((method, args, kwargs))
            raise AssertionError("collective_rpc path should not be used in inline runtime")

    model = CustomAsyncOmni(DummyEngine())
    asyncio.run(model.offload_states(level=2))
    asyncio.run(model.sleep(level=2))
    asyncio.run(model.update_parameter_in_bucket(["bucket"], is_lora=True))

    assert model._engine.inline_runtime.calls == [
        ("sleep", 2),
        ("sleep", 2),
        ("update_parameter_in_bucket", ["bucket"], True),
    ]
    assert model._engine.collective_calls == []


def test_vllm_omni_strategy_offload_uses_offload_states_before_sleep():
    class DummyModel:
        def __init__(self):
            self.calls = []

        async def offload_states(self, level):
            self.calls.append(("offload_states", level))

        async def sleep(self, level):
            self.calls.append(("sleep", level))

    worker = SimpleNamespace(
        worker_config=WorkerConfig(),
        pipeline_config=SimpleNamespace(seed=1, is_actor_infer_colocated=True),
    )
    strategy = VllmOmniStrategy(worker=worker)
    strategy.model = DummyModel()
    strategy.is_model_in_gpu = True
    strategy.sleep_level = 2

    asyncio.run(strategy.offload_states())

    assert strategy.model.calls == [("offload_states", 2)]
    assert strategy.is_model_in_gpu is False


def test_vllm_omni_worker_extension_wakes_weights_before_loading():
    class FakeDiffusionWorkerBase:
        def __init__(self):
            self.calls = []

        def sleep(self, level=1):
            self.calls.append(("sleep", level))
            return True

        def wake_up(self, tags=None):
            self.calls.append(("wake_up", tags))
            return True

        def load_weights(self, weights):
            materialized = list(weights)
            self.calls.append(("load_weights", materialized))
            return {"loaded": len(materialized)}

    class FakeWorker(VllmOmniColocateWorkerExtension, FakeDiffusionWorkerBase):
        pass

    worker = FakeWorker()
    assert worker.sleep(level=1) is True

    weights = [("weight", torch.ones(2))]
    result = worker.load_weights(weights)

    assert result == {"loaded": 1}
    assert worker.calls[0] == ("sleep", 1)
    assert worker.calls[1] == ("wake_up", ["weights"])
    assert worker.calls[2][0] == "load_weights"
    assert worker.calls[2][1][0][0] == "weight"
    assert torch.equal(worker.calls[2][1][0][1], torch.ones(2))


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="requires at least one GPU")
def test_roll_real_vllm_omni_start_and_dummy_request_output_contract():
    pytest.importorskip("vllm_omni")

    if not os.path.exists(VLLM_OMNI_TEST_MODEL_PATH):
        pytest.skip(f"model path not found: {VLLM_OMNI_TEST_MODEL_PATH}")

    import ray
    from transformers import AutoTokenizer

    started_ray = False
    resource_manager = None
    if not ray.is_initialized():
        ray.init(namespace=RAY_NAMESPACE, ignore_reinit_error=True, log_to_driver=True)
        started_ray = True

    try:
        worker_config = WorkerConfig(
            name="actor_infer",
            worker_cls="roll.pipeline.base_worker.InferWorker",
            model_args=ModelArguments(
                model_name_or_path=VLLM_OMNI_TEST_MODEL_PATH,
                dtype="bf16",
            ),
            strategy_args=StrategyArguments(
                strategy_name="vllm_omni",
                strategy_config={
                    "tensor_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "trust_remote_code": True,
                    "compilation_config": {
                        "cudagraph_mode": "FULL_AND_PIECEWISE",
                    },
                    "vllm_omni": {
                        "custom_pipeline": CUSTOM_PIPELINE_QUALNAME,
                    },
                },
            ),
            device_mapping="[0]",
        )

        resource_manager = ResourceManager(num_gpus_per_node=1, num_nodes=1)
        cluster = Cluster(
            name="test_vllm_omni_infer",
            worker_cls=InferWorker,
            resource_manager=resource_manager,
            worker_config=worker_config,
        )

        pipeline_config = SimpleNamespace(
            seed=42,
            resume_from_checkpoint=False,
            is_actor_infer_colocated=False,
        )
        cluster.initialize(pipeline_config=pipeline_config, blocking=True)

        tokenizer_path = os.path.join(VLLM_OMNI_TEST_MODEL_PATH, "tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)
        system_prompt = (
            "Describe the image by detailing the color, shape, size, texture, quantity, text, "
            "spatial relationships of the objects and background:"
        )
        encode_start_idx = _compute_encode_start_idx(tokenizer, system_prompt)
        apply_chat_template_kwargs = {
            "max_length": _TOKENIZER_MAX_LENGTH + encode_start_idx,
            "padding": True,
            "truncation": True,
        }
        sampling_params = {
            "guidance_scale": 4.0,
            "height": 512,
            "width": 512,
            "num_inference_steps": 10,
            "max_sequence_length": _TOKENIZER_MAX_LENGTH + encode_start_idx,
            "extra_args": {
                "logprobs": True,
                "noise_level": 1.0,
                "sde_type": "sde",
                "sde_window_size": 2,
                "sde_window_range": [0, 5],
            },
        }
        user_prompts = [
            "A photo of cute cat with long fur and big eyes.",
            "A photo of cute dog with short hair.",
            "A photo of beautiful girl dancing",
            "A photo of handsome man working",
            "Generate whatevery you like",
        ]
        for i, user_prompt in enumerate(user_prompts):
            raw_prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            raw_negative_prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": " "},
            ]
            prompt_ids, prompt_mask = _chat_template_to_ids_kwargs(tokenizer, raw_prompt, apply_chat_template_kwargs)
            negative_prompt_ids, negative_prompt_mask = _chat_template_to_ids_kwargs(tokenizer, raw_negative_prompt, apply_chat_template_kwargs)

            payload = {
                "rid": f"dummy_vllm_omni_req_{i}",
                "multi_modal_data": {
                    "prompt_ids": prompt_ids,
                    "prompt_mask": prompt_mask,
                    "negative_prompt_ids": negative_prompt_ids,
                    "negative_prompt_mask": negative_prompt_mask,
                    "encode_start_idx": encode_start_idx,
                },
                "sampling_params": sampling_params,
            }
            req_start = time.time()
            output = ray.get(cluster.workers[0].generate_request.remote(payload))
            e2e_latency = time.time() - req_start

            assert isinstance(output, dict)
            assert _REQUIRED_OUTPUT_KEYS.issubset(output.keys())

            rollout_log_probs = torch.as_tensor(output["rollout_log_probs"])
            all_timesteps = torch.as_tensor(output["all_timesteps"])
            all_latents = torch.as_tensor(output["all_latents"])

            assert rollout_log_probs.ndim == 2
            assert all_timesteps.shape[1] == rollout_log_probs.shape[1]
            assert all_latents.shape[1] == rollout_log_probs.shape[1] + 1

            request_meta = {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "sampling_params": sampling_params,
            }
            perfs = {
                "e2e_latency": e2e_latency,
            }
            output_to_save = _pad_rollout_output_for_verl_parity(
                output=output,
                target_prompt_len=apply_chat_template_kwargs["max_length"],
            )
            output_to_save["prompt_ids"] = [prompt_ids]
            output_to_save["negative_prompt_ids"] = [negative_prompt_ids]
            _save_rollout_output(
                output=output_to_save,
                output_dir=ROLLOUT_RESULTS_DIR / f"req_{i}",
                request_meta=request_meta,
                perfs=perfs,
            )
    finally:
        if resource_manager is not None:
            resource_manager.destroy_placement_group()
        if started_ray:
            ray.shutdown()
