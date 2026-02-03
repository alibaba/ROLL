import os
import random
import socket
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy import fsdp2_strategy
from roll.distributed.strategy.fsdp2_strategy import (
    FSDP2InferStrategy, FSDP2StrategyBase, FSDP2TrainStrategy,
    create_device_mesh_with_ulysses)
from roll.platforms import current_platform
from roll.utils.fsdp_utils import (apply_fsdp2, fsdp2_load_full_state_dict,
                                   get_shard_placement_fn)
from roll.utils.offload_states import OffloadStateType


class _PlatformStub:
    def __init__(self, device_type="cpu", backend=None):
        self.device_type = device_type
        if backend is None:
            backend = "nccl" if device_type == "cuda" else "gloo"
        self.communication_backend = backend

    def current_device(self):
        if self.device_type == "cuda":
            current = (
                torch.cuda.current_device()
                if torch.cuda.is_available()
                else 0
            )
            return torch.device("cuda", current)
        return "cpu"

    def apply_ulysses_patch(self):
        return None


class DummyTrainingArgs:
    def __init__(self):
        self.per_device_train_batch_size = 2
        self.gradient_accumulation_steps = 1
        self.learning_rate = 3e-4
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.weight_decay = 0.01
        self.lr_scheduler_type = "linear"
        self.max_steps = 10

    def get_warmup_steps(self, max_steps):
        return 1


class DummyModelArgs:
    def __init__(self, ulysses_size=1):
        self.ulysses_size = ulysses_size
        self.model_name_or_path = "dummy-model"
        self.model_config_kwargs = {}
        self.lora_target = None


def make_worker(
    strategy_config=None, use_remove_padding=False, ulysses_size=1
):
    worker_config = SimpleNamespace(
        name="dummy_worker",
        training_args=DummyTrainingArgs(),
        model_args=DummyModelArgs(ulysses_size=ulysses_size),
        strategy_args=SimpleNamespace(
            strategy_config=strategy_config or {}
        ),
        use_remove_padding=use_remove_padding,
        checkpoint_config=None,
        offload_nccl=False,
    )
    worker = SimpleNamespace(
        worker_config=worker_config,
        pipeline_config=SimpleNamespace(seed=0, max_grad_norm=1.0),
        rank_info=SimpleNamespace(
            dp_rank=0,
            dp_size=1,
            cp_rank=0,
            cp_size=1,
            tp_rank=0,
            pp_rank=0,
        ),
        world_size=1,
        rank=0,
    )
    return worker


@pytest.fixture
def worker_factory():
    def _factory(
        strategy_config=None, use_remove_padding=False, ulysses_size=1
    ):
        return make_worker(
            strategy_config=strategy_config,
            use_remove_padding=use_remove_padding,
            ulysses_size=ulysses_size,
        )

    return _factory


@pytest.fixture
def strategy_factory(worker_factory):
    strategies = []

    def _factory(strategy_cls, **worker_kwargs):
        worker = worker_factory(**worker_kwargs)
        strategy = strategy_cls(worker)
        strategies.append(strategy)
        return strategy

    yield _factory

    for strategy in strategies:
        strategy.thread_executor.shutdown(wait=True)


@pytest.fixture
def platform_stub():
    return _PlatformStub()


@pytest.fixture(autouse=True)
def _patch_platform(monkeypatch, platform_stub):
    monkeypatch.setattr(fsdp2_strategy, "current_platform", platform_stub)


class DummyCheckpointManager:
    def __init__(self, checkpoint_config=None):
        self.checkpoint_config = checkpoint_config
        self.upload_calls = []

    def upload(self, *args, **kwargs):
        self.upload_calls.append((args, kwargs))


@pytest.fixture(autouse=True)
def patch_checkpoint_manager(monkeypatch):
    monkeypatch.setattr(
        fsdp2_strategy, "CheckpointManager", DummyCheckpointManager
    )


class DummyForwardModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.kwargs = None
        self._ret = SimpleNamespace(logits=logits)

    def forward(self, **kwargs):
        self.kwargs = kwargs
        return self._ret


class MockModel:
    def __init__(self):
        self.to_calls = []
        self.cpu_called = False

    def to(self, device, non_blocking=False):
        self.to_calls.append((device, non_blocking))
        return self

    def cpu(self):
        self.cpu_called = True
        return self


def test_create_device_mesh_with_ulysses_global_mesh(
    monkeypatch, platform_stub
):
    """1D global mesh"""
    captured = {}

    def fake_init(device_type, mesh_shape, mesh_dim_names):
        captured["device_type"] = device_type
        captured["mesh_shape"] = mesh_shape
        captured["mesh_dim_names"] = mesh_dim_names
        return "mesh"

    monkeypatch.setattr(fsdp2_strategy, "init_device_mesh", fake_init)

    mesh = create_device_mesh_with_ulysses(world_size=4, fsdp_size=1)

    assert mesh == "mesh"
    assert captured["device_type"] == platform_stub.device_type
    assert captured["mesh_shape"] == (4,)
    assert captured["mesh_dim_names"] == ["fsdp"]


def test_create_device_mesh_with_ulysses_hsdp_mesh(monkeypatch):
    """2D HSDP mesh"""
    captured = {}

    def fake_init(device_type, mesh_shape, mesh_dim_names):
        captured["mesh_shape"] = mesh_shape
        captured["mesh_dim_names"] = mesh_dim_names
        return "mesh"

    monkeypatch.setattr(fsdp2_strategy, "init_device_mesh", fake_init)

    mesh = create_device_mesh_with_ulysses(world_size=8, fsdp_size=4)

    assert mesh == "mesh"
    assert captured["mesh_shape"] == (2, 4)
    assert captured["mesh_dim_names"] == ["ddp", "fsdp"]


def test_build_checkpoint_paths_uses_rank_and_world(strategy_factory):
    """Test that the checkpoint paths are built correctly"""
    strategy = strategy_factory(FSDP2StrategyBase)
    model_path, optim_path, extra_path = strategy._build_checkpoint_paths(
        "/tmp/ckpts", world_size=2, dp_rank=1
    )
    assert model_path.endswith("model_world_size_2_rank_1.pt")
    assert optim_path.endswith("optim_world_size_2_rank_1.pt")
    assert extra_path.endswith("extra_state_world_size_2_rank_1.pt")


def test_copy_weight_to_param(strategy_factory):
    """Test that the weight is copied to the parameter correctly"""
    strategy = strategy_factory(FSDP2StrategyBase)
    param = torch.nn.Parameter(torch.zeros(3))
    weight = torch.arange(3).float()

    strategy._copy_weight_to_param(param, weight)

    assert torch.allclose(param.detach(), weight)


def test_gather_full_tensor_returns_clone(strategy_factory):
    strategy = strategy_factory(FSDP2StrategyBase)
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    gathered = strategy._gather_full_tensor(param)
    assert torch.allclose(gathered, param.detach())

    # _gather_full_tensor needs to return a detached clone of the parameter;
    gathered += 1
    assert torch.allclose(param.detach(), torch.tensor([1.0, 2.0]))


def test_move_optimizer_states_respects_target_device(
    strategy_factory, monkeypatch
):
    """
    Make sure that the optimizer states are moved to the correct device after load/offload.
    """
    strategy = strategy_factory(FSDP2StrategyBase)

    class FakeTensor:
        def __init__(self):
            self.device = "cpu"

        def to(self, device, non_blocking=False):
            self.device = device
            return self

    fake_tensor = FakeTensor()
    strategy.optimizer = SimpleNamespace(
        state={"p": {"momentum": fake_tensor}}
    )

    orig_is_tensor = fsdp2_strategy.torch.is_tensor
    monkeypatch.setattr(
        fsdp2_strategy.torch,
        "is_tensor",
        lambda obj: isinstance(obj, FakeTensor) or orig_is_tensor(obj),
    )

    strategy._move_optimizer_states("meta")

    assert fake_tensor.device == "meta"


def test_get_broadcast_tensor_returns_cpu_view(strategy_factory):
    strategy = strategy_factory(FSDP2StrategyBase)
    weight_cpu = torch.ones(5)

    result = strategy._get_broadcast_tensor(weight_cpu)

    assert result is weight_cpu


def test_get_feature_on_cp_rank_slices_correct_window(strategy_factory):
    strategy = strategy_factory(FSDP2InferStrategy)
    strategy.worker.rank_info.cp_size = 2
    strategy.worker.rank_info.cp_rank = 1

    input_ids = torch.arange(8).view(1, 8)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(16).view(2, 1, 8)

    features = strategy.get_feature_on_cp_rank(
        input_ids, attention_mask, position_ids
    )

    expected_ids = torch.arange(4, 8).view(1, 4)
    assert torch.equal(features["input_ids"], expected_ids)
    assert torch.equal(
        features["attention_mask"], torch.ones_like(expected_ids)
    )
    assert torch.equal(
        features["position_ids"],
        torch.tensor(
            [[[4, 5, 6, 7]], [[12, 13, 14, 15]]], dtype=position_ids.dtype
        ),
    )


def test_op_compute_log_probs_matches_manual(strategy_factory):
    strategy = strategy_factory(FSDP2InferStrategy)
    logits = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [0.5, -0.5]]])
    input_ids = torch.tensor([[0, 1, 0]])
    attention_mask = torch.tensor([[1, 1, 0]])

    result = strategy.op_compute_log_probs(
        logits, input_ids, attention_mask
    )

    labels = input_ids[:, 1:].clone()
    labels[attention_mask[:, 1:] == 0] = 0
    labels = torch.cat([labels, torch.zeros_like(labels[:, :1])], dim=1)
    log_probs = (
        torch.nn.functional.log_softmax(logits.float(), dim=-1)
        .gather(dim=-1, index=labels.unsqueeze(-1))
        .squeeze(-1)
    )
    expected = log_probs[:, :-1] * attention_mask[:, 1:]

    assert torch.allclose(result, expected)


def test_op_compute_entropy_masks_prompt(strategy_factory):
    strategy = strategy_factory(FSDP2InferStrategy)
    logits = torch.tensor(
        [[[0.0, 1.0], [1.5, 0.5], [0.3, 0.7], [1.2, 0.2]]]
    )
    attention_mask = torch.tensor([[1, 1, 1, 0]])

    result = strategy.op_compute_entropy(logits, attention_mask)

    probs = torch.softmax(logits.float(), dim=-1)
    manual_entropy = torch.logsumexp(logits.float(), dim=-1) - (
        probs * logits
    ).sum(dim=-1)
    expected = manual_entropy[:, :-1] * attention_mask[:, 1:]

    assert torch.allclose(result, expected)


def test_setup_fsdp2_configuration_respects_strategy_config(
    strategy_factory,
):
    strategy_config = {
        "param_dtype": torch.float16,
        "reduce_dtype": torch.float32,
        "reshard_after_forward": False,
        "offload_policy": True,
        "fsdp_size": 2,
    }
    strategy = strategy_factory(
        FSDP2InferStrategy, strategy_config=strategy_config
    )
    strategy.device_mesh = "mesh-handle"

    strategy.setup_fsdp2_configuration()

    cfg = strategy.fsdp_config
    assert cfg["mesh"] == "mesh-handle"
    assert cfg["reshard_after_forward"] is False
    assert cfg["offload_policy"] is not False
    assert cfg["mp_policy"].param_dtype == torch.float16
    assert callable(cfg["shard_placement_fn"])


def test_clip_grad_norm_cpu_offload_uses_dummy_helper(
    strategy_factory, monkeypatch
):
    strategy = strategy_factory(FSDP2TrainStrategy)
    strategy.model = torch.nn.Linear(2, 2, bias=False)
    expected_params = list(strategy.model.parameters())

    for param in expected_params:
        param.grad = torch.ones_like(param)

    strategy.cpu_offload_enabled = True

    recorded = {}

    def fake_get_total_norm(grads, norm_type, error_if_nonfinite, foreach):
        recorded["total_norm_args"] = (
            list(grads),
            norm_type,
            error_if_nonfinite,
            foreach,
        )
        return torch.tensor(2.0)

    def fake_clip_grads_with_norm_(parameters, max_norm, total_norm, foreach):
        recorded["clip_args"] = (
            list(parameters),
            max_norm,
            total_norm.clone(),
            foreach,
        )

    monkeypatch.setattr(
        fsdp2_strategy, "_get_total_norm", fake_get_total_norm
    )
    monkeypatch.setattr(
        fsdp2_strategy, "_clip_grads_with_norm_", fake_clip_grads_with_norm_
    )

    returned_norm = strategy._clip_grad_norm(max_norm=1.0)

    assert "total_norm_args" in recorded
    grads_arg, norm_type, err_flag, foreach_flag = recorded["total_norm_args"]
    assert grads_arg == [param.grad for param in expected_params]
    assert norm_type == 2.0
    assert err_flag is False
    assert foreach_flag is None

    assert "clip_args" in recorded
    clip_params, clip_max_norm, clip_total_norm, clip_foreach = recorded[
        "clip_args"
    ]
    assert clip_params == expected_params
    assert clip_max_norm == 1.0
    assert clip_foreach is None
    assert clip_total_norm.item() == pytest.approx(2.0)

    assert returned_norm.item() == pytest.approx(2.0)


def _fsdp2_cpu_offload_grad_clip_worker(rank, world_size, port):
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    fsdp2_strategy.current_platform = _PlatformStub(
        device_type="cuda" if use_cuda else "cpu",
        backend=backend,
    )
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )
    try:
        if use_cuda:
            torch.cuda.set_device(rank % torch.cuda.device_count())
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")

        model = _TinyMLP(input_dim=4, hidden_dim=4, output_dim=2).to(device)
        mesh = create_device_mesh_with_ulysses(
            world_size=world_size, fsdp_size=world_size
        )
        mp_policy = (
            MixedPrecisionPolicy(
                param_dtype=torch.float16,
                reduce_dtype=torch.float32,
                cast_forward_inputs=True,
            )
            if use_cuda
            else None
        )
        offload_policy = (
            CPUOffloadPolicy(pin_memory=True) if use_cuda else False
        )
        fsdp_kwargs = {
            "mesh": mesh,
            "reshard_after_forward": True,
            "mp_policy": mp_policy,
            "offload_policy": offload_policy,
            "shard_placement_fn": get_shard_placement_fn(world_size),
        }
        full_state = model.state_dict()
        apply_fsdp2(model, fsdp_kwargs, {"fsdp_size": world_size})
        fsdp2_load_full_state_dict(model, full_state, mesh, offload_policy)

        features = torch.randn(2, 4, device=device, requires_grad=False)
        targets = torch.randn(2, 2, device=device, requires_grad=False)
        loss = model(features, targets)
        loss.backward()

        strategy = FSDP2TrainStrategy.__new__(FSDP2TrainStrategy)
        strategy.model = model
        strategy.cpu_offload_enabled = True

        total_norm = strategy._clip_grad_norm(max_norm=0.5)
        scalar_norm = (
            total_norm.to_local() if hasattr(total_norm, "to_local") else total_norm
        )
        scalar_norm = float(scalar_norm.detach().cpu().item())
        gathered = [0.0 for _ in range(world_size)]
        dist.all_gather_object(gathered, scalar_norm)

        if rank == 0:
            baseline = gathered[0]
            print(f"Gathered norms: {gathered}")
            for idx, other in enumerate(gathered[1:], start=1):
                print(f"Rank 0 norm: {baseline}, Rank {idx} norm: {other}, diff: {abs(baseline - other)}")
                assert other > 0, f"Rank {idx} returned zero/negative norm"
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    fsdp2_strategy.MixedPrecisionPolicy is None,
    reason="FSDP2 requires torch>=2.4",
)
@pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed is not available",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CPU-offload grad clip test requires CUDA",
)
def test_fsdp2_cpu_offload_grad_clip_distributed():
    world_size = min(2, torch.cuda.device_count())
    port = _find_free_port()
    mp.spawn(
        _fsdp2_cpu_offload_grad_clip_worker,
        args=(world_size, port),
        nprocs=world_size,
        join=True,
    )


def test_fsdp2_forward_without_remove_padding(strategy_factory):
    strategy = strategy_factory(
        FSDP2TrainStrategy, use_remove_padding=False
    )
    strategy.worker.rank_info.cp_size = 1
    logits = torch.randn(1, 2, 4)
    strategy.model = DummyForwardModel(logits=logits)

    input_ids = torch.ones(1, 2, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.zeros_like(input_ids)

    output = strategy._fsdp2_forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        forward_args={"foo": torch.tensor(1)},
    )

    assert torch.equal(output, logits)
    assert strategy.model.kwargs["input_ids"] is input_ids
    assert strategy.model.kwargs["attention_mask"] is attention_mask
    assert strategy.model.kwargs["position_ids"] is position_ids


def test_fsdp2_forward_slices_cp_inputs(strategy_factory):
    strategy = strategy_factory(
        FSDP2TrainStrategy, use_remove_padding=False
    )
    strategy.worker.rank_info.cp_size = 2
    strategy.worker.rank_info.cp_rank = 1
    logits = torch.randn(1, 2, 4)
    strategy.model = DummyForwardModel(logits=logits)
    strategy.param_dtype = torch.float32

    input_ids = torch.arange(0, 4).view(1, 4).long()
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.zeros_like(input_ids)

    strategy._fsdp2_forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        forward_args={},
    )

    expected_slice = input_ids[:, 2:]
    assert torch.equal(strategy.model.kwargs["input_ids"], expected_slice)
    assert torch.equal(
        strategy.model.kwargs["attention_mask"], attention_mask[:, 2:]
    )
    assert torch.equal(
        strategy.model.kwargs["position_ids"], position_ids[:, 2:]
    )


def test_forward_step_uses_cp_slice(strategy_factory):
    strategy = strategy_factory(
        FSDP2InferStrategy, use_remove_padding=False
    )
    strategy.worker.rank_info.cp_size = 2
    strategy.worker.rank_info.cp_rank = 1
    logits = torch.zeros(1, 2, 3)
    strategy.model = DummyForwardModel(logits=logits)
    strategy.param_dtype = torch.float32

    seq_len = 4
    batch = TensorDict(
        {
            "input_ids": torch.arange(seq_len).view(1, seq_len),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
            "position_ids": torch.zeros(1, seq_len, dtype=torch.long),
            "response_mask": torch.ones(1, seq_len, dtype=torch.long),
        },
        batch_size=[1],
    )
    data = DataProto(batch=batch, meta_info={"micro_batch_size": 1})

    def dummy_forward_func(local_data, output_tensor):
        zeros = torch.zeros_like(local_data.batch["input_ids"]).float()
        return output_tensor.sum(), {"log_probs": zeros, "entropy": zeros}

    results = strategy.forward_step(
        batch=data,
        forward_func=dummy_forward_func,
    )

    assert "log_probs" in results and "entropy" in results
    expected_slice = torch.arange(seq_len).view(1, seq_len)[:, seq_len // 2 :]
    assert torch.equal(strategy.model.kwargs["input_ids"], expected_slice)


def test_load_states_moves_model_and_optimizer(
    strategy_factory, monkeypatch
):
    strategy = strategy_factory(FSDP2StrategyBase)
    strategy.model = MockModel()

    captured = {}

    def fake_move(self, device, non_blocking=False):
        captured["device"] = device
        captured["non_blocking"] = non_blocking

    monkeypatch.setattr(
        FSDP2StrategyBase, "_move_optimizer_states", fake_move
    )

    strategy.load_states(
        include=[
            OffloadStateType.model_params,
            OffloadStateType.optimizer_states,
        ],
        non_blocking=True,
    )

    assert strategy.model.to_calls == [("cpu", True)]
    assert captured["device"] == "cpu"
    assert captured["non_blocking"] is True


def test_offload_states_moves_to_cpu_and_clears_cuda_cache(
    strategy_factory, monkeypatch, platform_stub
):
    strategy = strategy_factory(FSDP2StrategyBase)
    strategy.model = MockModel()
    platform_stub.device_type = "cuda"

    captured = {}

    def fake_move(self, device, non_blocking=False):
        captured["device"] = device
        captured["non_blocking"] = non_blocking

    monkeypatch.setattr(
        FSDP2StrategyBase, "_move_optimizer_states", fake_move
    )

    cache_cleared = {"flag": False}
    monkeypatch.setattr(
        fsdp2_strategy.torch.cuda,
        "empty_cache",
        lambda: cache_cleared.__setitem__("flag", True),
    )

    strategy.offload_states(
        include=[
            OffloadStateType.model_params,
            OffloadStateType.optimizer_states,
        ],
        non_blocking=True,
    )

    assert strategy.model.cpu_called
    assert isinstance(captured["device"], torch.device)
    assert captured["device"].type == "cpu"
    assert captured["non_blocking"] is True
    assert cache_cleared["flag"] is True


def test_rng_state_roundtrip(monkeypatch):
    cpu_state = torch.arange(4, dtype=torch.uint8)
    cuda_state = torch.arange(5, dtype=torch.uint8)
    numpy_state = ("MT19937", np.arange(624, dtype=np.uint32), 0, 0, 0.0)
    random_state = (3, (1, 2, 3), None)

    monkeypatch.setattr(torch, "get_rng_state", lambda: cpu_state.clone())
    monkeypatch.setattr(
        torch.cuda, "get_rng_state", lambda: cuda_state.clone()
    )

    captured = {}
    monkeypatch.setattr(
        torch,
        "set_rng_state",
        lambda state: captured.__setitem__("cpu", state.clone()),
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda state: captured.__setitem__("cuda", state.clone()),
    )
    monkeypatch.setattr(np.random, "get_state", lambda: numpy_state)
    monkeypatch.setattr(
        np.random,
        "set_state",
        lambda state: captured.__setitem__("numpy", state),
    )
    monkeypatch.setattr(random, "getstate", lambda: random_state)
    monkeypatch.setattr(
        random,
        "setstate",
        lambda state: captured.__setitem__("random", state),
    )

    rng_state = FSDP2StrategyBase.get_rng_state()
    FSDP2StrategyBase.load_rng_state(rng_state)

    assert torch.equal(rng_state["cpu"], cpu_state)
    assert torch.equal(captured["cpu"], cpu_state)
    assert torch.equal(rng_state["cuda"], cuda_state)
    assert torch.equal(captured["cuda"], cuda_state)
    assert rng_state["numpy"] == numpy_state
    assert captured["numpy"] == numpy_state
    assert rng_state["random"] == random_state
    assert captured["random"] == random_state


class _TinyMLP(torch.nn.Module):
    _no_split_modules = ["Linear"]

    def __init__(self, input_dim=8, hidden_dim=16, output_dim=2):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, inputs, targets):
        return self.loss_fn(self.layers(inputs), targets)


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _generate_synthetic_batches(steps, batch_size, input_dim, output_dim):
    generator = torch.Generator().manual_seed(2024)
    features = torch.randn(
        steps, batch_size, input_dim, generator=generator
    )
    targets = torch.randn(
        steps, batch_size, output_dim, generator=generator
    )
    return features, targets


def _collect_full_state(model):
    state = {}
    for name, param in model.named_parameters():
        tensor = param.detach()
        if DTensor is not None and isinstance(tensor, DTensor):
            if tensor.device.type == "cpu" and torch.cuda.is_available():
                tensor = tensor.to("cuda")
            tensor = tensor.full_tensor()
        state[name] = tensor.cpu().numpy()
    return state


def _fsdp2_training_worker(rank, world_size, port, steps):
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    fsdp2_strategy.current_platform = _PlatformStub(
        device_type="cuda" if use_cuda else "cpu", backend=backend
    )
    current_platform = _PlatformStub(
        device_type="cuda" if use_cuda else "cpu", backend=backend
    )
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(
        backend=backend, rank=rank, world_size=world_size
    )
    try:
        if use_cuda:
            torch.cuda.set_device(rank)
            device = torch.device("cuda", rank)
        else:
            device = torch.device("cpu")
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        model = _TinyMLP()
        model.train()
        mesh = create_device_mesh_with_ulysses(
            world_size=world_size, fsdp_size=world_size
        )
        mp_policy = (
            MixedPrecisionPolicy(
                param_dtype=torch.float16,
                reduce_dtype=torch.float32,
                cast_forward_inputs=True,
            )
            if use_cuda
            else None
        )
        offload_policy = (
            CPUOffloadPolicy(pin_memory=True) if use_cuda else False
        )
        fsdp_kwargs = {
            "mesh": mesh,
            "reshard_after_forward": True,
            "mp_policy": mp_policy,
            "offload_policy": offload_policy,
            "shard_placement_fn": get_shard_placement_fn(world_size),
        }
        strategy_config = {
            "fsdp_size": world_size,
        }
        full_state = model.state_dict()
        apply_fsdp2(model, fsdp_kwargs, strategy_config)
        fsdp2_load_full_state_dict(model, full_state, mesh, offload_policy)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        inputs, targets = _generate_synthetic_batches(
            steps, batch_size=4, input_dim=8, output_dim=2
        )

        for step in range(steps):
            optimizer.zero_grad()
            batch_inputs = inputs[step].to(device)
            batch_targets = targets[step].to(device)
            loss = model(batch_inputs, batch_targets)
            print("Output Device:", loss.device)
            print("Target Device:", batch_targets.device)
            print("Output Dtype:", loss.dtype)
            print("Target Dtype:", batch_targets.dtype)
            print("Output Shape:", loss.shape)
            print("Target Shape:", batch_targets.shape)
            loss.backward()
            optimizer.step()

        dist.barrier()
        local_state = _collect_full_state(model)
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(local_state, gathered, dst=0)
        if rank == 0:
            baseline = gathered[0]
            for idx, other in enumerate(gathered[1:], start=1):
                for key in baseline.keys():
                    np.testing.assert_allclose(
                        baseline[key],
                        other[key],
                        atol=1e-6,
                        err_msg=f"Parameter {key} mismatch between ranks 0 and {idx}",
                    )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    fsdp2_strategy.MixedPrecisionPolicy is None,
    reason="FSDP2 requires torch>=2.4",
)
@pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed is not available",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FSDP2 distributed training sync test requires CUDA",
)
def test_fsdp2_distributed_training_keeps_states_in_sync():
    world_size = 2
    port = _find_free_port()
    mp.spawn(
        _fsdp2_training_worker,
        args=(world_size, port, 3),
        nprocs=world_size,
        join=True,
    )
