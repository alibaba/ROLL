"""Run the checkpoint strategy boundary with real Gloo, file IO, and CPU Adam.

The production method is compiled unchanged from its AST because importing the
strategy eagerly requires Ray, CUDA Megatron, and PEFT. Only model assembly,
PEFT state extraction, and MCA filename selection use small CPU doubles; the
collectives, validation, state restoration, and failures execute production code.
Real PEFT/MCA restoration is covered by test_strategy_checkpoint_resume.py.
"""
import ast
import copy
from datetime import timedelta
from contextlib import nullcontext
import logging
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class _Adapters(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapters = torch.nn.ModuleDict({
            name: torch.nn.Linear(2, 2, bias=False) for name in ("default", "secondary")
        })
        self.peft_config = {name: object() for name in self.adapters}

    def state_dict_for_save_checkpoint(self):
        return self.state_dict()


class _VirtualModels:
    def __init__(self, models):
        self.models = models

    def load_external_assets(self, load_dir, **kwargs):
        # These ordinary CPU adapters have no external frozen assets.
        pass

    def load_state_dict(self, state):
        for name, adapter_state in state.items():
            self.models[0].adapters[name].load_state_dict({
                "weight": adapter_state["model"]["lora_A.weight"]
            })


def _load_adapter_checkpoint(directory):
    checkpoint = Path(directory) / f"rank-{dist.get_rank()}.pt"
    return torch.load(checkpoint, weights_only=True) if checkpoint.exists() else None


def _strategy_class():
    source = Path(__file__).parents[3] / "roll/distributed/strategy/megatron_strategy.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    strategy = next(node for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainStrategy")
    method = next(node for node in strategy.body
                  if isinstance(node, ast.FunctionDef) and node.name == "load_checkpoint")
    tracker = SimpleNamespace(get_states=lambda: {"cpu": torch.get_rng_state()}, set_states=lambda states: None)
    namespace = dict(
        os=os, random=random, np=np, torch=torch, dist=dist, nullcontext=nullcontext,
        logger=logging.getLogger(__name__), RNG_STATE_DIR="rng_state",
        DIST_OPTIMIZER_DIR="dist_optimizer", OPTIMIZER_NAME="optimizer.pt", SCHEDULER_NAME="scheduler.pt",
        PeftModel=_Adapters, is_peft_available=lambda: True,
        get_peft_model_state_dict=lambda model, state, name: {
            "lora_A.weight": state[f"adapters.{name}.weight"]
        },
        load_state_dict_from_checkpoint=_load_adapter_checkpoint,
        get_checkpoint_dir=lambda directory, **kwargs: directory,
        current_platform=SimpleNamespace(set_rng_state=lambda state: None),
        tensor_parallel=SimpleNamespace(get_cuda_rng_tracker=lambda: tracker),
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return type("MegatronTrainStrategy", (), {"load_checkpoint": namespace["load_checkpoint"]})


def _advance(strategy):
    for parameter in strategy.models_unwrapped[0].parameters():
        parameter.grad = torch.full_like(parameter, 0.125)
    strategy.optimizer.step()
    strategy.optimizer.zero_grad(set_to_none=True)
    strategy.scheduler.step()


def _worker(rank, directory, corruption):
    directory = Path(directory)
    dist.init_process_group("gloo", init_method=(directory / "rendezvous").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        # Separate local views let only nonzero rank 1 see a damaged adapter.
        checkpoint = directory / f"checkpoint-{rank}"
        checkpoint.mkdir()
        torch.manual_seed(713)
        model = _Adapters()
        strategy = _strategy_class()()
        strategy.models_unwrapped = [model]
        strategy.models_wrapped = [SimpleNamespace(module=model)]
        strategy.model = _VirtualModels(strategy.models_wrapped)
        strategy.optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        strategy.scheduler = torch.optim.lr_scheduler.StepLR(strategy.optimizer, step_size=1, gamma=0.5)
        # The production Megatron scheduler exposes this restore counter;
        # native counter behavior is covered by the real strategy GPU test.
        strategy.scheduler.num_steps = 0
        strategy.megatron_train_args = SimpleNamespace(
            use_distributed_optimizer=False, process_index=rank, device="cpu"
        )
        _advance(strategy)
        for name, adapter in model.adapters.items():
            (checkpoint / name).mkdir()
            state = {"model": {"lora_A.weight": adapter.weight.detach().clone()}}
            torch.save(state, checkpoint / name / f"rank-{rank}.pt")
        saved_model = copy.deepcopy(model.state_dict())
        saved_optimizer = copy.deepcopy(strategy.optimizer.state_dict())
        saved_scheduler = copy.deepcopy(strategy.scheduler.state_dict())
        torch.save(saved_optimizer, checkpoint / "optimizer.pt")
        torch.save(saved_scheduler, checkpoint / "scheduler.pt")
        (checkpoint / "rng_state").mkdir()
        torch.save({
            "random_rng_state": random.getstate(), "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.get_rng_state(),
            "rng_tracker_states": {"cpu": torch.get_rng_state()},
        }, checkpoint / "rng_state" / f"rng_state_{rank}.pth")
        _advance(strategy)
        before_model = copy.deepcopy(model.state_dict())
        before_optimizer = copy.deepcopy(strategy.optimizer.state_dict())
        before_scheduler = copy.deepcopy(strategy.scheduler.state_dict())
        before_rng = torch.get_rng_state().clone()

        if rank == 1:
            path = checkpoint / "secondary" / "rank-1.pt"
            original = path.read_bytes()
            damaged = torch.load(path, weights_only=True)
            if corruption == "missing_file":
                path.unlink()
            elif corruption == "unreadable":
                path.write_bytes(b"not a torch checkpoint")
            else:
                if corruption == "missing_tensor":
                    damaged["model"].clear()
                elif corruption == "unknown_tensor":
                    damaged["model"]["unknown.lora_A.weight"] = torch.ones(2, 2)
                elif corruption == "shape":
                    damaged["model"]["lora_A.weight"] = torch.ones(1)
                elif corruption == "non_tensor":
                    damaged["model"]["lora_A.weight"] = None
                torch.save(damaged, path)
        error = None
        try:
            strategy.load_checkpoint(str(checkpoint))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        # Both ranks must reach the same final collective without hanging,
        # and every live training state must retain its newer pre-load value.
        outcomes = [None, None]
        dist.all_gather_object(outcomes, error)
        assert all(outcomes), f"every rank must reject rank 1's {corruption}: {outcomes}"
        assert "adapter" in outcomes[0].lower() and "rank 1" in outcomes[0].lower(), outcomes
        assert strategy.model.models is strategy.models_wrapped
        torch.testing.assert_close(model.state_dict(), before_model, atol=0, rtol=0)
        torch.testing.assert_close(strategy.optimizer.state_dict(), before_optimizer, atol=0, rtol=0)
        assert strategy.scheduler.state_dict() == before_scheduler
        torch.testing.assert_close(torch.get_rng_state(), before_rng, atol=0, rtol=0)
        dist.barrier()

        # Repairing the bad rank must allow both ranks to restore normally;
        # an implementation that rejects every checkpoint cannot pass.
        if rank == 1:
            path.write_bytes(original)
        strategy.load_checkpoint(str(checkpoint))
        assert strategy.model.models is strategy.models_wrapped
        torch.testing.assert_close(model.state_dict(), saved_model, atol=0, rtol=0)
        torch.testing.assert_close(strategy.optimizer.state_dict(), saved_optimizer, atol=0, rtol=0)
        assert strategy.scheduler.state_dict() == saved_scheduler
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires CPU Gloo")
@pytest.mark.parametrize("corruption", [
    "missing_file", "unreadable", "missing_tensor", "unknown_tensor", "shape", "non_tensor"
])
def test_nonzero_rank_adapter_failure_rejects_before_training_state_changes(tmp_path, corruption):
    mp.spawn(_worker, args=(str(tmp_path), corruption), nprocs=2, join=True)
