from types import SimpleNamespace
import sys
import types

if "ray" not in sys.modules:
    ray_module = types.ModuleType("ray")
    ray_module.ObjectRef = object
    ray_module.get = lambda value, **kwargs: value
    ray_module.remote = lambda cls=None, **kwargs: cls
    ray_module.actor = types.SimpleNamespace(ActorHandle=object)
    sys.modules["ray"] = ray_module

import numpy as np
import pytest
import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.diffusion.schema import (
    DataProtoSchemaError,
    require_batch_keys,
    validate_diffusion_rollout_batch,
)
from roll.pipeline.diffusion.utils import attach_group_ids, prepare_training_batch


def _flowgrpo_batch(*, include_negative: bool = True) -> DataProto:
    batch_size = 4
    num_steps = 3
    prompt_len = 5
    tensors = {
        "rollout_log_probs": torch.zeros(batch_size, num_steps),
        "prompt_embeds": torch.randn(batch_size, prompt_len, 8),
        "prompt_embeds_mask": torch.ones(batch_size, prompt_len, dtype=torch.long),
        "scores": torch.tensor([0.1, 0.4, 0.2, 0.8], dtype=torch.float32),
        "token_level_rewards": torch.ones(batch_size, num_steps),
        "all_latents": torch.randn(batch_size, num_steps + 1, 2, 2),
        "all_timesteps": torch.arange(num_steps).repeat(batch_size, 1),
    }
    if include_negative:
        tensors["negative_prompt_embeds"] = torch.randn(batch_size, prompt_len, 8)
        tensors["negative_prompt_embeds_mask"] = torch.ones(batch_size, prompt_len, dtype=torch.long)

    return DataProto.from_dict(
        tensors=tensors,
        non_tensors={
            "id": ["p0", "p0", "p1", "p1"],
            "domain": ["ocr", "ocr", "ocr", "ocr"],
            "ground_truth": ["a", "a", "b", "b"],
        },
        meta_info={"generation_config": {"guidance_scale": 1.0}},
    )


def test_schema_error_reports_missing_keys_and_available_context():
    data = _flowgrpo_batch()
    del data.batch["scores"]

    with pytest.raises(DataProtoSchemaError) as exc_info:
        require_batch_keys(data, ["scores"], "unit schema check")

    message = str(exc_info.value)
    assert "unit schema check" in message
    assert "missing batch keys=['scores']" in message
    assert "available batch keys=" in message
    assert "available non_tensor keys=" in message


def test_flowgrpo_validate_rollout_batch_rejects_latent_timestep_shape_mismatch():
    data = _flowgrpo_batch()
    data.batch["all_latents"] = torch.randn(4, 3, 2, 2)

    with pytest.raises(ValueError, match="latents"):
        validate_diffusion_rollout_batch(data, "flowgrpo", SimpleNamespace(actor_infer=SimpleNamespace(generating_args=None)))


def test_flowgrpo_validate_rollout_batch_requires_negative_prompt_for_cfg():
    data = _flowgrpo_batch(include_negative=False)
    data.meta_info["generation_config"]["guidance_scale"] = 4.0

    with pytest.raises(DataProtoSchemaError) as exc_info:
        validate_diffusion_rollout_batch(data, "flowgrpo", SimpleNamespace(actor_infer=SimpleNamespace(generating_args=None)))

    assert "negative_prompt_embeds" in str(exc_info.value)


def test_flowgrpo_attach_group_ids_uses_prompt_id_groups():
    data = _flowgrpo_batch()
    attach_group_ids(data)

    assert data.non_tensor_batch["group_id"].dtype == np.dtype("O")
    assert data.non_tensor_batch["group_id"].tolist() == [0, 0, 1, 1]


def _fake_pipeline(algorithm: str) -> SimpleNamespace:
    config = SimpleNamespace(
        algorithm=algorithm,
        is_offload_states=True,
        is_offload_optimizer_states_in_train_step=True,
        actor_infer=SimpleNamespace(generating_args=None),
    )
    return SimpleNamespace(pipeline_config=config)


def test_flowgrpo_prepare_training_batch_attaches_old_and_reference_log_probs():
    data = _flowgrpo_batch()
    attach_group_ids(data)

    prepared, metrics = prepare_training_batch(batch=data, pipeline=_fake_pipeline("flowgrpo"), global_step=7)

    assert prepared.meta_info["global_step"] == 7
    assert prepared.meta_info["is_offload_states"] is True
    assert prepared.meta_info["loss_mask_keys"] == ["flow_loss_mask"]
    assert torch.equal(prepared.batch["flow_loss_mask"], torch.ones_like(prepared.batch["rollout_log_probs"]))
    assert torch.allclose(prepared.batch["old_log_probs"], prepared.batch["rollout_log_probs"])
    assert torch.allclose(prepared.batch["ref_log_probs"], prepared.batch["rollout_log_probs"])
    assert prepared.batch["advantages"].shape == prepared.batch["rollout_log_probs"].shape
    assert "grpo_group/advantage/group_0/mean" in metrics


def _diffnft_batch() -> DataProto:
    batch_size = 4
    num_steps = 3
    prompt_len = 5
    return DataProto.from_dict(
        tensors={
            # ODE mode: only the final latent x0 is kept (trajectory length 1)
            "all_latents": torch.randn(batch_size, 1, 2, 2),
            "all_timesteps": torch.arange(num_steps).repeat(batch_size, 1),
            "rollout_log_probs": torch.zeros(batch_size, num_steps),
            "prompt_embeds": torch.randn(batch_size, prompt_len, 8),
            "prompt_embeds_mask": torch.ones(batch_size, prompt_len, dtype=torch.long),
            "scores": torch.tensor([0.1, 0.4, 0.2, 0.8], dtype=torch.float32),
        },
        non_tensors={
            "id": ["p0", "p0", "p1", "p1"],
            "domain": ["ocr", "ocr", "ocr", "ocr"],
            "ground_truth": ["a", "a", "b", "b"],
        },
        meta_info={"generation_config": {"guidance_scale": 1.0}},
    )


def test_diffnft_validate_rollout_batch_accepts_ode_format():
    data = _diffnft_batch()
    validate_diffusion_rollout_batch(data, "diffnft", SimpleNamespace(actor_infer=SimpleNamespace(generating_args=None)))


def test_diffnft_validate_rollout_batch_rejects_full_trajectory():
    data = _diffnft_batch()
    data.batch["all_latents"] = torch.randn(4, 3, 2, 2)

    with pytest.raises(DataProtoSchemaError, match="shape\\[1\\] must be 1"):
        validate_diffusion_rollout_batch(data, "diffnft", SimpleNamespace(actor_infer=SimpleNamespace(generating_args=None)))


def test_diffnft_prepare_training_batch_writes_optimality_probability():
    data = _diffnft_batch()
    attach_group_ids(data)

    prepared, metrics = prepare_training_batch(batch=data, pipeline=_fake_pipeline("diffnft"), global_step=7)

    # advantages carries the optimality probability r in [0, 1], one value per sample
    r = prepared.batch["advantages"]
    assert r.shape == (4, 1)
    assert float(r.min()) >= 0.0 and float(r.max()) <= 1.0
    # DiffNFT does not attach FlowGRPO-only fields
    assert prepared.meta_info["loss_mask_keys"] == []
    assert "flow_loss_mask" not in prepared.batch.keys()
    assert "old_log_probs" not in prepared.batch.keys()
    assert "ref_log_probs" not in prepared.batch.keys()
    assert "grpo_group/advantage/group_0/mean" in metrics
