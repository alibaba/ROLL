import os
import pickle
import sys
import types
import contextlib
from pathlib import Path
from types import SimpleNamespace

if "ray" not in sys.modules:
    ray_module = types.ModuleType("ray")
    ray_module.ObjectRef = object
    ray_module.get = lambda value: value
    ray_module.remote = lambda cls=None, **kwargs: cls
    ray_module.actor = types.SimpleNamespace(ActorHandle=object)
    sys.modules["ray"] = ray_module

    util_module = types.ModuleType("ray.util")
    sys.modules["ray.util"] = util_module

    placement_group_module = types.ModuleType("ray.util.placement_group")
    placement_group_module.PlacementGroup = object
    sys.modules["ray.util.placement_group"] = placement_group_module

    scheduling_module = types.ModuleType("ray.util.scheduling_strategies")
    scheduling_module.PlacementGroupSchedulingStrategy = object
    scheduling_module.NodeAffinitySchedulingStrategy = object
    sys.modules["ray.util.scheduling_strategies"] = scheduling_module

    runtime_env_module = types.ModuleType("ray.runtime_env")
    runtime_env_module.RuntimeEnv = object
    sys.modules["ray.runtime_env"] = runtime_env_module

    private_module = types.ModuleType("ray._private")
    sys.modules["ray._private"] = private_module

    async_compat_module = types.ModuleType("ray._private.async_compat")
    async_compat_module.has_async_methods = lambda cls: False
    sys.modules["ray._private.async_compat"] = async_compat_module

    worker_module = types.ModuleType("ray._private.worker")

    class _RemoteFunctionNoArgs:
        @classmethod
        def __class_getitem__(cls, item):
            return cls

    worker_module.RemoteFunctionNoArgs = _RemoteFunctionNoArgs
    sys.modules["ray._private.worker"] = worker_module

if "more_itertools" not in sys.modules:
    more_itertools_module = types.ModuleType("more_itertools")
    more_itertools_module.chunked = lambda seq, n: [seq[i:i + n] for i in range(0, len(seq), n)]
    sys.modules["more_itertools"] = more_itertools_module

import pytest
import torch

from roll.configs.base_config import RolloutMockConfig
from roll.configs.model_args import ModelArguments
from roll.configs.training_args import TrainingArguments
from roll.configs.worker_config import StrategyArguments, WorkerConfig
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.diffusion.utils import compute_grpo_outcome_advantage
from roll.pipeline.diffusion.diffusion_config import DiffusionConfig


def _build_mock_batch(step: int) -> DataProto:
    batch_size = 4
    num_steps = 3
    seq_len = 5
    channels = 4
    prompt_len = 6

    all_latents = torch.randn(batch_size, num_steps + 1, seq_len, channels)
    rollout_log_probs = torch.randn(batch_size, num_steps)
    all_timesteps = torch.arange(num_steps).repeat(batch_size, 1)
    prompt_embeds = torch.randn(batch_size, prompt_len, 8)
    prompt_embeds_mask = torch.ones(batch_size, prompt_len, dtype=torch.long)
    negative_prompt_embeds = torch.randn(batch_size, prompt_len, 8)
    negative_prompt_embeds_mask = torch.ones(batch_size, prompt_len, dtype=torch.long)

    return DataProto.from_dict(
        tensors={
            "responses": torch.randn(batch_size, 3, 8, 8),
            "rollout_log_probs": rollout_log_probs,
            "all_timesteps": all_timesteps,
            "all_latents": all_latents,
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
            "response_mask": torch.tensor(
                [
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                ],
                dtype=torch.long,
            ),
        },
        non_tensors={"group_id": [0, 0, 1, 1]},
        meta_info={"mock_step": step},
    )


def _build_mock_reward(step: int) -> DataProto:
    scores = torch.tensor([0.1 + step, 0.4 + step, 0.2 + step, 0.8 + step], dtype=torch.float32)
    token_level_rewards = torch.tensor(
        [
            [0.1 + step, 0.2 + step, 0.3 + step],
            [0.4 + step, 0.5 + step, 0.6 + step],
            [0.2 + step, 0.3 + step, 0.4 + step],
            [0.8 + step, 0.9 + step, 1.0 + step],
        ],
        dtype=torch.float32,
    )
    return DataProto.from_dict(
        tensors={
            "scores": scores,
            "token_level_rewards": token_level_rewards,
        }
    )


def _dump_dataproto(base_dir: Path, step: int, data: DataProto):
    base_dir.mkdir(parents=True, exist_ok=True)
    with open(base_dir / f"step_{step:06d}.pkl", "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def test_flowgrpo_config_enables_num_return_sequences_expand_for_group_sampling():
    cfg = DiffusionConfig(
        num_return_sequences_in_group=4,
        is_num_return_sequences_expand=False,
    )

    assert cfg.is_num_return_sequences_expand is True
    assert cfg.actor_infer.generating_args.num_return_sequences == 4


class _FakeCluster:
    created_names = []
    instances = {}

    def __init__(self, name, worker_cls, resource_manager, worker_config):
        self.name = name
        self.worker_cls = worker_cls
        self.resource_manager = resource_manager
        self.worker_config = worker_config
        self.initialized = False
        self.train_calls = 0
        self.weight = 0.0
        _FakeCluster.created_names.append(name)
        _FakeCluster.instances[name] = self

    def initialize(self, pipeline_config, blocking=True):
        self.initialized = True
        return []

    def compute_log_probs(self, data, blocking=True):
        source = data.batch["rollout_log_probs"]
        offset = 0.0 if self.name == "reference" else self.weight
        return DataProto.from_dict(tensors={"log_probs": source + offset, "entropy": torch.zeros_like(source)})

    def train_step(self, data, blocking=False):
        self.train_calls += 1
        self.weight += 0.1
        return [DataProto(meta_info={"metrics": {"actor/total_loss@sum": 1.0 + self.weight}})]

    def do_checkpoint(self, global_step, is_last_step=None, blocking=False):
        return [DataProto(meta_info={"metrics": {"checkpoint/save_secs": 0.0}})]


def test_qwen_image_tokenizer_provider_strict(monkeypatch, tmp_path):
    from roll.models import model_providers

    captured = {}

    def _fake_download(path):
        return path

    def _fake_from_pretrained(path, **kwargs):
        captured["path"] = path
        return object()

    monkeypatch.setattr(model_providers, "download_model", _fake_download)
    monkeypatch.setattr(model_providers.AutoTokenizer, "from_pretrained", _fake_from_pretrained)

    model_root = tmp_path / "Qwen-Image"
    tokenizer_dir = model_root / "tokenizer"
    tokenizer_dir.mkdir(parents=True)

    model_providers.default_tokenizer_provider(
        ModelArguments(model_name_or_path=str(model_root), model_type="diffusion_model")
    )
    assert captured["path"] == str(tokenizer_dir)

    missing_root = tmp_path / "Missing-Qwen-Image"
    missing_root.mkdir(parents=True)
    try:
        model_providers.default_tokenizer_provider(
            ModelArguments(model_name_or_path=str(missing_root), model_type="diffusion_model")
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected strict tokenizer loading to fail when tokenizer/ is missing.")


def test_qwen_image_diffusion_provider_uses_explicit_transformer_class(monkeypatch, tmp_path):
    from roll.models import model_providers

    model_root = tmp_path / "Qwen-Image"
    transformer_dir = model_root / "transformer"
    transformer_dir.mkdir(parents=True)

    captured = {}

    class _FakeTransformer:
        def train(self):
            captured["train_called"] = True

        def eval(self):
            captured["eval_called"] = True

        def requires_grad_(self, value):
            captured["requires_grad"] = value

    class _FakeQwenImageTransformer2DModel:
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            captured["path"] = pretrained_model_name_or_path
            captured["kwargs"] = kwargs
            return _FakeTransformer()

    monkeypatch.setattr(model_providers, "download_model", lambda path: path)
    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(QwenImageTransformer2DModel=_FakeQwenImageTransformer2DModel))

    model_args = ModelArguments(
        model_name_or_path=str(model_root),
        model_type="diffusion_model",
        disable_gradient_checkpointing=True,
    )
    model_args.compute_dtype = torch.bfloat16
    model = model_providers.default_diffusion_model_provider(
        tokenizer=None,
        model_args=model_args,
        is_trainable=False,
    )

    assert model is not None
    assert captured["path"] == str(model_root)
    assert captured["kwargs"]["subfolder"] == "transformer"
    assert captured["kwargs"]["local_files_only"] is True
    assert captured["eval_called"] is True
    assert captured["requires_grad"] is False


def test_fsdp2_prepare_qwen_image_skips_root_hf_autoconfig(monkeypatch):
    from roll.distributed.strategy.fsdp2_strategy import FSDP2TrainStrategy
    import roll.distributed.strategy.fsdp2_strategy as fsdp2_module

    worker_config = WorkerConfig(
        model_args=ModelArguments(model_name_or_path="/tmp/Qwen-Image", model_type="diffusion_model"),
        training_args=TrainingArguments(per_device_train_batch_size=1, gradient_accumulation_steps=1),
        strategy_args=StrategyArguments(strategy_name="fsdp2_train", strategy_config={"fsdp_size": 4}),
        device_mapping="list(range(0,4))",
    )
    worker = SimpleNamespace(
        worker_config=worker_config,
        pipeline_config=SimpleNamespace(seed=42),
        rank_info=SimpleNamespace(dp_rank=None, dp_size=None, cp_rank=None, cp_size=None),
    )
    strategy = FSDP2TrainStrategy(worker)

    monkeypatch.setattr(fsdp2_module, "download_model", lambda path: path)
    monkeypatch.setattr(fsdp2_module, "default_tokenizer_provider", lambda model_args: object())
    monkeypatch.setattr(fsdp2_module, "default_processor_provider", lambda model_args: None)
    monkeypatch.setattr(fsdp2_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(fsdp2_module.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(fsdp2_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(fsdp2_module, "create_device_mesh_with_ulysses", lambda world_size, fsdp_size: object())
    monkeypatch.setattr(fsdp2_module, "get_init_weight_context_manager", lambda **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(fsdp2_module, "set_fsdp2_init_context", lambda context: None)
    monkeypatch.setattr(fsdp2_module, "clear_fsdp2_init_context", lambda: None)

    def _unexpected_autoconfig(*args, **kwargs):
        raise AssertionError("Qwen-Image should not read AutoConfig from the checkpoint root in FSDP2 prepare.")

    monkeypatch.setattr(fsdp2_module.AutoConfig, "from_pretrained", _unexpected_autoconfig)

    captured = {}

    def _provider(*, tokenizer, model_args, is_trainable):
        captured["tokenizer"] = tokenizer
        captured["model_type"] = model_args.model_type
        captured["is_trainable"] = is_trainable
        return object()

    model, torch_dtype, cp_size = strategy._prepare_fsdp2_model(
        _provider,
        is_trainable=True,
        default_model_dtype=torch.float32,
    )

    assert model is not None
    assert torch_dtype == torch.float32
    assert cp_size == 1
    assert captured["tokenizer"] is not None
    assert captured["model_type"] == "diffusion_model"
    assert captured["is_trainable"] is True


def test_flow_grpo_algorithms_output_shapes():
    scores = torch.tensor([1.0, 3.0, 2.0, 4.0])
    advantages, returns = compute_grpo_outcome_advantage(
        scores=scores,
        group_ids=[0, 0, 1, 1],
        num_steps=3,
    )
    assert advantages.shape == (4, 3)
    assert returns.shape == (4, 3)

    loss, metrics = compute_policy_loss_flow_grpo(
        log_probs=torch.zeros(4, 3),
        old_log_probs=torch.zeros(4, 3),
        ref_log_probs=torch.zeros(4, 3),
        advantages=torch.ones(4, 3),
        loss_mask=torch.ones(4, 3),
        pg_clip=0.2,
        advantage_clip=None,
        use_kl_loss=True,
        kl_loss_coef=0.1,
    )
    assert loss.ndim == 0
    assert "actor/total_loss@sum" in metrics


def test_flow_grpo_policy_loss_applies_advantage_clip():
    loss, _ = compute_policy_loss_flow_grpo(
        log_probs=torch.zeros(1, 1),
        old_log_probs=torch.zeros(1, 1),
        ref_log_probs=torch.zeros(1, 1),
        advantages=torch.tensor([[10.0]]),
        loss_mask=torch.ones(1, 1),
        pg_clip=0.0001,
        advantage_clip=5.0,
        use_kl_loss=False,
        kl_loss_coef=0.0,
    )

    assert torch.isclose(loss, torch.tensor(-5.0))


def test_flowgrpo_train_pipeline_uses_only_actor_train_and_reference(monkeypatch, tmp_path):
    """Integration test for DiffusionPipeline with FlowGRPO config (rollout mock mode).

    NOTE: This test previously tested the now-removed FlowGRPOTrainPipeline.
    DiffusionPipeline requires additional adapters and validation that makes
    a simple mock-only integration test impractical without a full Ray cluster.
    The algorithm-level unit tests above validate the core FlowGRPO math.
    """
    pytest.skip(
        "FlowGRPOTrainPipeline has been removed in favor of DiffusionPipeline. "
        "DiffusionPipeline integration tests require a Ray cluster and are in test_vllm_omni_scheduler_e2e.py."
    )
