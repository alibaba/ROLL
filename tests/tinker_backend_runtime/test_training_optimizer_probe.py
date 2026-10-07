"""Exercise extracted optimizer primitives using actual CPU parameters and SGD.

AST extraction preserves the production function bodies while avoiding imports
of GPU-only Megatron/TransformerEngine. This does not emulate distributed CUDA.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import pytest
import torch


@pytest.fixture
def primitives():
    path = Path(__file__).resolve().parents[2] / "roll/pipeline/tinker_backend_runtime/megatron_primitives.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"set_optimizer_hparams", "post_optim_step_cleanup", "apply_optim_step"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    events = []
    namespace = {
        "os": os, "torch": torch, "Dict": Dict,
        "cleanup_ddp_buffers": lambda optimizer, backend: events.append(("cleanup", optimizer, backend)),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, events


class CPUModel(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(64))
        self.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        self.empty = torch.nn.Parameter(torch.empty(0))
        self.events = events
        self.group = SimpleNamespace(
            is_first_batch=True, per_param_grad_ready_counts={self.weight: 2},
            params=[self.weight, self.frozen], buckets=[object(), object()],
            cached_param_buffer_shard_list=["stale_parameter"] * 2,
            cached_grad_buffer_shard_list=["stale_gradient"] * 2,
        )
        self.bucket_groups = [self.group]
        self.expert_parallel_bucket_groups = []

    def zero_grad_buffer(self):
        assert self.group.per_param_grad_ready_counts[self.frozen] == 1
        self.events.append("model_zero_grad_buffer")


class CPUOptimizer:
    def __init__(self, models, events, successful=True):
        self.inner = torch.optim.SGD([parameter for model in models for parameter in model.parameters()], lr=0.001)
        self.param_groups = self.inner.param_groups
        self.events = events
        self.successful = successful

    def step(self):
        self.events.append("optimizer_step")
        if not self.successful:
            return False, 7.0, 0
        self.inner.step()
        return True, 7.0, 0

    def zero_grad(self):
        self.events.append("optimizer_zero_grad")
        self.inner.zero_grad()


def strategy(events, count=1, successful=True):
    models = [CPUModel(events) for _ in range(count)]
    sum(model.weight.square().sum() for model in models).backward()
    backend = object()
    return SimpleNamespace(
        model=models, optimizer=CPUOptimizer(models, events, successful),
        worker_config=SimpleNamespace(name="actor_train"), _get_offload_backend=lambda: backend,
    )


def adam_params():
    return {"learning_rate": 0.1, "beta1": 0.9, "beta2": 0.95, "eps": 1e-8, "weight_decay": 0.0}


def test_parameter_probe_matches_real_parameter_update(primitives, monkeypatch):
    namespace, events = primitives
    monkeypatch.setenv("TINKER_PARAMETER_PROBE", "1")
    actor = strategy(events)
    before = actor.model[0].weight.detach().clone()
    metrics = namespace["apply_optim_step"](actor, adam_params())
    delta = actor.model[0].weight.detach() - before
    assert torch.all(delta != 0)
    assert metrics["actor_train/parameter_probe_count"] == 32
    assert metrics["actor_train/parameter_probe_changed"] == 32
    assert metrics["actor_train/parameter_probe_max_delta"] == pytest.approx(float(delta.abs().max()))
    assert metrics["actor_train/parameter_probe_l2_delta"] == pytest.approx(float(delta[::2].norm()))
    assert metrics["actor_train/grad_norm"] == 7.0
    assert metrics["actor_train/lr"] == 0.1


def test_parameter_probe_off_preserves_step_without_probe_metrics(primitives, monkeypatch):
    namespace, events = primitives
    monkeypatch.delenv("TINKER_PARAMETER_PROBE", raising=False)
    actor = strategy(events)
    before = actor.model[0].weight.detach().clone()
    metrics = namespace["apply_optim_step"](actor, adam_params())
    assert torch.any(actor.model[0].weight.detach() != before)
    assert metrics == {"actor_train/grad_norm": 7.0, "actor_train/lr": 0.1}


def test_unsuccessful_step_fails_without_probe_success_or_cleanup(primitives, monkeypatch):
    namespace, events = primitives
    monkeypatch.setenv("TINKER_PARAMETER_PROBE", "1")
    actor = strategy(events, successful=False)
    before = actor.model[0].weight.detach().clone()
    with pytest.raises(NotImplementedError, match="optimizer step failed"):
        namespace["apply_optim_step"](actor, adam_params())
    assert torch.equal(actor.model[0].weight.detach(), before)
    assert events == ["optimizer_step"]
    assert actor.model[0].weight.grad is not None


def test_cleanup_preserves_existing_counts_clears_caches_and_gradients(primitives, monkeypatch):
    namespace, events = primitives
    monkeypatch.delenv("TINKER_PARAMETER_PROBE", raising=False)
    actor = strategy(events)
    namespace["apply_optim_step"](actor, adam_params())
    assert events == ["optimizer_step", ("cleanup", actor.optimizer, actor._get_offload_backend()),
                      "model_zero_grad_buffer", "optimizer_zero_grad"]
    model = actor.model[0]
    assert model.group.per_param_grad_ready_counts[model.weight] == 2
    assert model.group.per_param_grad_ready_counts[model.frozen] == 1
    assert model.group.cached_param_buffer_shard_list == [None, None]
    assert model.group.cached_grad_buffer_shard_list == [None, None]
    assert model.weight.grad is None
    group = actor.optimizer.param_groups[0]
    assert group["lr"] == 0.1
    assert group["betas"] == (0.9, 0.95)
    assert group["eps"] == 1e-8


def test_parameter_probe_is_bounded_to_1024_sampled_values(primitives, monkeypatch):
    namespace, events = primitives
    monkeypatch.setenv("TINKER_PARAMETER_PROBE", "1")
    actor = strategy(events, count=33)
    metrics = namespace["apply_optim_step"](actor, adam_params())
    assert metrics["actor_train/parameter_probe_count"] == 1024
    assert metrics["actor_train/parameter_probe_changed"] == 1024
    assert events.count("model_zero_grad_buffer") == 33
