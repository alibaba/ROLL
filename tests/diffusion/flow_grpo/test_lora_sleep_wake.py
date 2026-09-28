"""LoRA sleep/wake image generation test.

Verifies that after sleep(level=2), the next model_update restores the
transformer base weights via the base weight stream (train side re-broadcasts
the frozen base in LoRA mode; no disk reload). Generates images at three
phases and saves them for visual inspection:

1. phase1_base.png   — base model before LoRA adapter is loaded
2. phase2_after_lora.png — after LoRA adapter update + custom_add_lora
3. phase3_after_wake.png — after sleep(level=2) + model_update + load_states

If phase3 is a garbage/noise image, the model_update base weight stream failed
to restore the transformer base weights.
"""

import os
import time
from pathlib import Path

import pytest
import torch
from PIL import Image

from roll.configs import ModelArguments
from roll.configs.generating_args import GeneratingArguments
from roll.configs.worker_config import StrategyArguments, WorkerConfig
from roll.distributed.executor.cluster import Cluster
from roll.distributed.executor.model_update_group import ModelUpdateGroup
from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.pipeline.base_worker import InferWorker
from roll.pipeline.diffusion.diffusion_config import DiffusionConfig
from roll.utils.constants import RAY_NAMESPACE

VLLM_OMNI_TEST_MODEL_PATH = os.environ.get(
    "VLLM_OMNI_TEST_MODEL_PATH",
    "Qwen/Qwen-Image",
)
CUSTOM_PIPELINE_QUALNAME = (
    "roll.pipeline.diffusion.models.qwen_image.vllm_omni_qwen_image_adapter.QwenImagePipelineWithLogProb"
)
TMP_DIR = Path("/tmp/test_lora_sleep_wake")
FINAL_DIR = TMP_DIR / "final"

LORA_TARGET = (
    "to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,"
    "img_mlp.net.0.proj,img_mlp.net.2,txt_mlp.net.0.proj,txt_mlp.net.2"
)
LORA_RANK = 64
LORA_ALPHA = 128

SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:"
)
USER_PROMPT = (
    'A close-up of a medicine bottle with a clear, red warning label that reads '
    '"Take With Food" prominently displayed, set against a neutral background.'
)
NEGATIVE_PROMPT = " "

_TOKENIZER_MAX_LENGTH = 1024


# ---------------------------------------------------------------------------
# Helper functions (adapted from test_vllm_omni.py)
# ---------------------------------------------------------------------------

def _compute_encode_start_idx(tokenizer, system_prompt):
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


def _chat_template_to_ids(tokenizer, messages, kwargs):
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        **kwargs,
    )
    return _normalize_token_ids(token_ids), None


def _save_response_images(responses, output_path: Path):
    t = torch.as_tensor(responses).detach().cpu()
    if t.ndim == 3:
        t = t.unsqueeze(0)
    if t.ndim != 4:
        return
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
        # OSS mount does not support direct PIL writes; save to /tmp first then copy.
        tmp_path = TMP_DIR / output_path.name
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img_np).save(str(tmp_path))
        import shutil
        shutil.copy(str(tmp_path), str(output_path))


def _build_payload(tokenizer, encode_start_idx, sampling_params):
    max_length = _TOKENIZER_MAX_LENGTH + encode_start_idx
    apply_chat_template_kwargs = {
        "max_length": max_length,
        "padding": True,
        "truncation": True,
    }
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    negative_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": NEGATIVE_PROMPT},
    ]
    prompt_ids, _ = _chat_template_to_ids(tokenizer, prompt_messages, apply_chat_template_kwargs)
    negative_prompt_ids, _ = _chat_template_to_ids(tokenizer, negative_messages, apply_chat_template_kwargs)

    return {
        "rid": f"lora-sleep-wake-{int(time.time())}",
        "multi_modal_data": {
            "prompt_ids": prompt_ids,
            "prompt_mask": None,
            "negative_prompt_ids": negative_prompt_ids,
            "negative_prompt_mask": None,
            "encode_start_idx": encode_start_idx,
        },
        "sampling_params": dict(sampling_params),
    }


def _generate_and_save(actor_infer, payload, output_path, label):
    import ray

    print(f"[test] === {label} === generating image ...")
    t0 = time.time()
    output = ray.get(actor_infer.workers[0].generate_request.remote(payload))
    latency = time.time() - t0
    print(f"[test] {label} done in {latency:.1f}s, keys={list(output.keys())}")

    responses = output.get("responses")
    assert responses is not None, f"{label}: no 'responses' in output"
    _save_response_images(responses, output_path)
    print(f"[test] {label} saved image to {output_path}")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="requires at least one GPU")
def test_lora_sleep_wake_generate():
    pytest.importorskip("vllm_omni")

    if not os.path.exists(VLLM_OMNI_TEST_MODEL_PATH):
        pytest.skip(f"model path not found: {VLLM_OMNI_TEST_MODEL_PATH}")

    import ray
    from transformers import AutoTokenizer

    started_ray = False
    resource_manager = None
    actor_train = None
    actor_infer = None

    if not ray.is_initialized():
        ray.init(namespace=RAY_NAMESPACE, ignore_reinit_error=True, log_to_driver=True)
        started_ray = True

    try:
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        FINAL_DIR.mkdir(parents=True, exist_ok=True)

        # ---- tokenizer ----
        tokenizer_path = os.path.join(VLLM_OMNI_TEST_MODEL_PATH, "tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)
        encode_start_idx = _compute_encode_start_idx(tokenizer, SYSTEM_PROMPT)
        print(f"[test] encode_start_idx={encode_start_idx}")

        sampling_params = {
            "num_return_sequences": 1,
            "guidance_scale": 1.0,
            "height": 512,
            "width": 512,
            "num_inference_steps": 10,
            "max_sequence_length": _TOKENIZER_MAX_LENGTH + encode_start_idx,
            "max_new_tokens": 1,
            "extra_args": {
                "logprobs": False,
                "noise_level": 0.0,
                "sde_type": "ode",
            },
        }
        payload = _build_payload(tokenizer, encode_start_idx, sampling_params)

        # ---- WorkerConfig objects (created before pipeline_config) ----
        generating_args = GeneratingArguments(
            num_return_sequences=1,
            temperature=0,
            top_p=1.0,
            top_k=50,
            max_length=1058,
            max_sequence_length=1058,
            max_new_tokens=1,
            height=512,
            width=512,
            num_inference_steps=10,
            guidance_scale=1.0,
            extra_args={
                "logprobs": False,
                "noise_level": 0.0,
                "sde_type": "ode",
            },
        )

        infer_worker_config = WorkerConfig(
            name="actor_infer",
            worker_cls="roll.pipeline.base_worker.InferWorker",
            model_args=ModelArguments(
                model_name_or_path=VLLM_OMNI_TEST_MODEL_PATH,
                model_type="diffusion_model",
                dtype="bf16",
            ),
            generating_args=generating_args,
            strategy_args=StrategyArguments(
                strategy_name="vllm_omni",
                strategy_config={
                    "tensor_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "trust_remote_code": True,
                    "enforce_eager": True,
                    "load_format": "safetensors",
                    "sleep_level": 2,
                    "vllm_omni": {
                        "custom_pipeline": CUSTOM_PIPELINE_QUALNAME,
                    },
                },
            ),
            device_mapping="list(range(0,8))",
        )

        train_worker_config = WorkerConfig(
            name="actor_train",
            worker_cls="roll.pipeline.diffusion.actor_nft_worker.ActorNFTWorker",
            model_args=ModelArguments(
                model_name_or_path=VLLM_OMNI_TEST_MODEL_PATH,
                model_type="diffusion_model",
                dtype="bf16",
                lora_target=LORA_TARGET,
                lora_rank=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=0.0,
            ),
            strategy_args=StrategyArguments(
                strategy_name="fsdp2_diffusion_train",
                strategy_config={
                    "fsdp_size": 8,
                    "reshard_after_forward": True,
                    "offload_policy": False,
                },
            ),
            model_update_frequency=1,
            device_mapping="list(range(0,8))",
        )

        reference_worker_config = WorkerConfig(name="reference")

        # ---- pipeline_config (DiffusionConfig) ----
        pipeline_config = DiffusionConfig(
            seed=42,
            pretrain=VLLM_OMNI_TEST_MODEL_PATH,
            resume_from_checkpoint=False,
            adv_estimator="diffnft",
            prompt_length=1024,
            sequence_length=1058,
            num_return_sequences_in_group=1,
            is_num_return_sequences_expand=True,
            nft_beta=0.1,
            adv_clip_max=5.0,
            ref_kl_coef=0.0,
            use_adaptive_weight=True,
            training_timestep_fraction=1.0,
            shuffle_train_timesteps=True,
            ema_range=[0, 75],
            ema_decays=[0.001, 0.99],
            min_train_timestep=None,
            max_train_timestep=None,
            track_with="stdout",
            actor_train=train_worker_config,
            actor_infer=infer_worker_config,
            reference=reference_worker_config,
        )

        # ---- actor_infer (vLLM omni) ----
        print("[test] creating actor_infer cluster ...")
        resource_manager = ResourceManager(num_gpus_per_node=8, num_nodes=1)
        actor_infer = Cluster(
            name="test_lora_infer",
            worker_cls=InferWorker,
            resource_manager=resource_manager,
            worker_config=pipeline_config.actor_infer,
        )
        actor_infer.initialize(pipeline_config=pipeline_config, blocking=True)
        print("[test] actor_infer initialized")

        # actor_infer.load_states(blocking=True)
        # ---- Phase 1: base model generate ----
        # _generate_and_save(
        #     actor_infer, payload,
        #     FINAL_DIR / "phase1_base.png",
        #     "phase1_base",
        # )
        # actor_infer.offload_states(blocking=True)

        # ---- actor_train (FSDP2 diffusion + LoRA) ----
        print("[test] creating actor_train cluster ...")
        actor_train = Cluster(
            name="test_lora_train",
            worker_cls="roll.pipeline.diffusion.actor_nft_worker.ActorNFTWorker",
            resource_manager=resource_manager,
            worker_config=pipeline_config.actor_train,
        )
        actor_train.initialize(pipeline_config=pipeline_config, blocking=True)
        print("[test] actor_train initialized")

        # ---- DiffNFT preflight: seed EMA shadow adapter ----
        print("[test] running preflight_nft_ops ...")
        ray.get(actor_train.workers[0].preflight_nft_ops.remote())
        print("[test] preflight_nft_ops done")

        # ---- model_update pair ----
        print("[test] setting up model_update pair ...")
        model_update_group = ModelUpdateGroup(
            src_cluster=actor_train,
            tgt_cluster=actor_infer,
            frequency=1,
            pipeline_config=pipeline_config,
        )

        # ---- Phase 2: model_update (sends LoRA adapter) + generate ----
        print("[test] === model_update (LoRA adapter) ===")
        # In real pipeline, actor_train.offload_states happens before model_update.
        # Here we skip it since we never ran a training step (weights are fresh init).
        update_metrics = model_update_group.model_update(step=0)
        actor_infer.process_weights_after_loading()
        print(f"[test] model_update done, metrics={update_metrics}")

        _generate_and_save(
            actor_infer, payload,
            FINAL_DIR / "phase2_after_lora.png",
            "phase2_after_lora",
        )

        # ---- Phase 3: sleep(level=2) + model_update + load_states + generate ----
        # is_actor_infer_colocated=True is required for offload_states to trigger sleep.
        # Mirrors the real pipeline order: after a level-2 sleep the transformer
        # base is garbage until the next model_update streams it back.
        print("[test] === sleep(level=2) ===")
        actor_infer.offload_states(blocking=True)
        print("[test] offload_states done (sleep level=2)")

        print("[test] === model_update after sleep (restores base + LoRA) ===")
        update_metrics = model_update_group.model_update(step=1)
        actor_infer.process_weights_after_loading()
        print(f"[test] model_update done, metrics={update_metrics}")

        print("[test] === load_states (wake_up) ===")
        actor_infer.load_states(blocking=True)
        print("[test] load_states done (wake_up)")

        _generate_and_save(
            actor_infer, payload,
            FINAL_DIR / "phase3_after_wake.png",
            "phase3_after_wake",
        )

        # ---- summary ----
        print(f"\n[test] === All images saved to {FINAL_DIR} ===")
        for name in ("phase1_base.png", "phase2_after_lora.png", "phase3_after_wake.png"):
            p = FINAL_DIR / name
            if p.exists():
                print(f"  {name}: {p.stat().st_size} bytes")
            else:
                print(f"  {name}: MISSING!")

    finally:
        print("[test] cleanup ...")
        if resource_manager is not None:
            try:
                resource_manager.destroy_placement_group()
            except Exception:
                pass
        if started_ray:
            ray.shutdown()
