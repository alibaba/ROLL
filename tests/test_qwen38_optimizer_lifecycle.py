import copy
import types
from dataclasses import dataclass

import pytest
import torch

from roll.third_party.megatron.optimizer_config import build_optimizer_config
from roll.third_party.megatron.optimizer_lifecycle import (
    bind_cpu_optimizer_state_lifecycle,
    cpu_optimizer_offload_states,
    cpu_optimizer_reload_states,
)


@dataclass
class _Config:
    lr: float
    optimizer_cpu_offload: bool = False
    optimizer_offload_fraction: float = 0.0
    use_torch_optimizer_for_cpu_offload: bool = False
    overlap_cpu_optimizer_d2h_h2d: bool = False
    pin_cpu_grads: bool = True
    pin_cpu_params: bool = True
    offload_optimizer_states: bool = False


class _Args:
    optimizer_cpu_offload = True
    optimizer_offload_fraction = 1.0
    use_torch_optimizer_for_cpu_offload = True
    overlap_cpu_optimizer_d2h_h2d = True
    pin_cpu_grads = False
    pin_cpu_params = False
    offload_optimizer_states = True


def test_optimizer_config_forwards_cpu_mode_and_state_lifecycle_fields():
    config = build_optimizer_config(_Config, {"lr": 0.01}, _Args())

    assert config.optimizer_cpu_offload is True
    assert config.optimizer_offload_fraction == 1.0
    assert config.use_torch_optimizer_for_cpu_offload is True
    assert config.overlap_cpu_optimizer_d2h_h2d is True
    assert config.pin_cpu_grads is False
    assert config.pin_cpu_params is False
    assert config.offload_optimizer_states is True


def test_requested_cpu_optimizer_mode_is_rejected_when_upstream_lacks_field():
    @dataclass
    class OldConfig:
        lr: float

    with pytest.raises(ValueError, match="optimizer_cpu_offload"):
        build_optimizer_config(OldConfig, {"lr": 0.01}, _Args())


def test_cpu_optimizer_steps_preserve_cpu_state_and_resume_exactly(tmp_path):
    initial = torch.tensor([1.0, -2.0])
    uninterrupted_param = torch.nn.Parameter(initial.clone())
    resumed_param = torch.nn.Parameter(initial.clone())
    uninterrupted = torch.optim.AdamW([uninterrupted_param], lr=0.1)
    resumed = torch.optim.AdamW([resumed_param], lr=0.1)
    bind_cpu_optimizer_state_lifecycle(resumed)

    gradients = [torch.tensor([0.5, -0.25]), torch.tensor([0.25, -0.5]), torch.tensor([-0.125, 0.75])]
    for grad in gradients[:2]:
        uninterrupted_param.grad = grad.clone()
        resumed_param.grad = grad.clone()
        uninterrupted.step()
        resumed.step()

    checkpoint = {
        "param": resumed_param.detach().clone(),
        "optimizer": copy.deepcopy(resumed.state_dict()),
    }
    resumed.offload_states()
    assert all(value.device.type == "cpu" for state in resumed.state.values() for value in state.values() if isinstance(value, torch.Tensor))
    resumed.reload_states()

    restored_param = torch.nn.Parameter(checkpoint["param"].clone())
    restored = torch.optim.AdamW([restored_param], lr=0.1)
    restored.load_state_dict(checkpoint["optimizer"])
    restored_param.grad = gradients[2].clone()
    restored.step()

    uninterrupted_param.grad = gradients[2].clone()
    uninterrupted.step()

    torch.testing.assert_close(restored_param, uninterrupted_param, rtol=0, atol=0)
    torch.testing.assert_close(restored.state_dict()["state"], uninterrupted.state_dict()["state"], rtol=0, atol=0)
