"""Real CUDA/CPU HybridDeviceOptimizer staging, AdamW, and resume regressions.

Run with RUN_CPU_OPTIMIZER_TESTS=1 on the patched Megatron dependency.
"""

import copy
import os

import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CPU_OPTIMIZER_TESTS") != "1",
    reason="requires installed Megatron and CUDA",
)


class CPUAdamW(torch.optim.AdamW):
    """Distinct CPU class, as required by Hybrid's checkpoint device routing."""


def _parameters():
    return [
        torch.nn.Parameter(torch.linspace(-0.8, 1.2, size, device="cuda", dtype=torch.bfloat16))
        for size in (3, 5, 7, 11)
    ]


def _optimizer(parameters, *, bounded=True, fraction=1.0, overlap=True, fused=True):
    from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer

    return HybridDeviceOptimizer(
        [{"params": parameters[:2], "weight_decay": 0.03},
         {"params": parameters[2:], "weight_decay": 0.1}],
        offload_fraction=fraction,
        cpu_optimizer_cls=CPUAdamW,
        gpu_optimizer_cls=torch.optim.AdamW,
        param_update_in_fp32=True,
        overlap_cpu_optimizer_d2h_h2d=overlap,
        bounded_cpu_grad_staging=bounded,
        lr=0.013, betas=(0.7, 0.91), eps=1e-6, foreach=False, fused=fused,
    )


def _cpu_parameters(optimizer):
    return [p for sub in optimizer.cpu_optimizers for group in sub.param_groups for p in group["params"]]


def _gradient_storage_bytes(optimizer):
    values = list(optimizer.cpu_copy_map_grad.values())
    values.extend(p.grad for p in _cpu_parameters(optimizer))
    values.append(getattr(optimizer, "_cpu_grad_staging_buffer", None))
    storages = {value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
                for value in values if isinstance(value, torch.Tensor)}
    return sum(storages.values())


def _set_grads(parameters, step, *, decoupled=False, missing=None):
    for index, parameter in enumerate(parameters):
        grad = torch.arange(parameter.numel(), device="cuda", dtype=torch.float32).add_(1)
        grad.mul_((index + 1) * (-0.2 if step % 2 else 0.3) / (step + 1))
        grad = None if index == missing else grad
        if decoupled:
            parameter.decoupled_grad = grad
        else:
            parameter.grad = None if grad is None else grad.to(parameter.dtype)


def _assert_hybrid_equal(actual, expected):
    for actual_group, expected_group in zip(actual.param_groups, expected.param_groups):
        for left, right in zip(actual_group["params"], expected_group["params"]):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
            left_state, right_state = actual.state[left], expected.state[right]
            assert left_state.keys() == right_state.keys()
            for key in left_state:
                # Hybrid migrates the scalar step to CUDA for a GPU optimizer
                # on reload; torch AdamW originally keeps that scalar on CPU.
                left_value = left_state[key].cpu() if key == "step" else left_state[key]
                right_value = right_state[key].cpu() if key == "step" else right_state[key]
                torch.testing.assert_close(left_value, right_value, atol=0, rtol=0)
            for key in ("master_param", "exp_avg", "exp_avg_sq"):
                assert left_state[key].dtype == torch.float32
            if left in actual.gpu_params_map_cpu_copy:
                assert all(left_state[key].device.type == "cpu"
                           for key in ("master_param", "exp_avg", "exp_avg_sq"))


@pytest.mark.parametrize("fused", [False, True])
def test_cpu_gradient_storage_is_bounded_and_released_between_updates(fused):
    """Eager allocation or retaining each consumed gradient exceeds one tensor."""
    parameters = _parameters()
    optimizer = _optimizer(parameters, bounded=False, fused=fused)
    # ROLL's Megatron wrapper does not forward this opt-in keyword. Enabling
    # the installed Hybrid instance before first step must have the same bound.
    optimizer.bounded_cpu_grad_staging = True
    largest = 11 * 4
    observed = []

    def inspect_before_update(sub, args, kwargs):
        observed.append(_gradient_storage_bytes(optimizer))
        assert observed[-1] <= largest

    for sub in optimizer.cpu_optimizers:
        sub.register_step_pre_hook(inspect_before_update)
    for step in range(3):
        _set_grads(parameters, step, decoupled=True)
        optimizer.step()
        torch.cuda.synchronize()
        assert all(p.grad is None for p in _cpu_parameters(optimizer))
        assert not optimizer.cpu_copy_map_grad
        assert _gradient_storage_bytes(optimizer) <= largest
    assert len(observed) == 12
    assert max(observed) == largest


@pytest.mark.parametrize("fraction", [1.0, 0.5])
@pytest.mark.parametrize("decoupled", [False, True])
def test_bounded_updates_match_eager_after_global_clip_and_resume(fraction, decoupled):
    """Buffer reuse must preserve masters, moments, group hyperparameters, and resume."""
    parameters, reference_parameters = _parameters(), _parameters()
    optimizer = _optimizer(parameters, fraction=fraction)
    reference = _optimizer(reference_parameters, bounded=False, fraction=fraction)
    for step in range(5):
        for params in (parameters, reference_parameters):
            _set_grads(params, step, decoupled=decoupled)
            if decoupled:
                grads = [p.decoupled_grad for p in params]
                norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
                scale = (0.2 / (norm + 1e-6)).clamp(max=1.0)
                for grad in grads:
                    grad.mul_(scale)
            else:
                torch.nn.utils.clip_grad_norm_(params, 0.2)
        optimizer.param_groups[0]["lr"] = reference.param_groups[0]["lr"] = 0.013 / (step + 1)
        optimizer.step()
        reference.step()
        torch.cuda.synchronize()
        _assert_hybrid_equal(optimizer, reference)
        if step == 1:
            from roll.third_party.megatron.offload_states_patch import offload_adam_states, reload_adam_states

            pointers = [(p.data_ptr(), optimizer.state[gpu]["exp_avg"].data_ptr())
                        for gpu, p in optimizer.gpu_params_map_cpu_copy.items()]
            offload_adam_states(optimizer, torch.device("cpu"))
            reload_adam_states(optimizer, torch.device("cuda", torch.cuda.current_device()))
            assert pointers == [(p.data_ptr(), optimizer.state[gpu]["exp_avg"].data_ptr())
                                for gpu, p in optimizer.gpu_params_map_cpu_copy.items()]
        if step == 2:
            checkpoint = copy.deepcopy(optimizer.state_dict())
            parameters = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
            optimizer = _optimizer(parameters, fraction=fraction)
            optimizer.load_state_dict(checkpoint)
            assert _gradient_storage_bytes(optimizer) <= 11 * 4


def test_missing_decoupled_gradient_skips_adamw_and_weight_decay():
    """A reused buffer must not apply a previous step's gradient to an unused tensor."""
    parameters = _parameters()
    optimizer = _optimizer(parameters)
    masters = [torch.nn.Parameter(p.detach().float().cpu()) for p in parameters]
    reference = torch.optim.AdamW(
        [{"params": masters[:2], "weight_decay": 0.03},
         {"params": masters[2:], "weight_decay": 0.1}],
        lr=0.013, betas=(0.7, 0.91), eps=1e-6, foreach=False, fused=True,
    )
    for step, missing in enumerate((None, 1, 3, None)):
        _set_grads(parameters, step, decoupled=True, missing=missing)
        for param, master in zip(parameters, masters):
            master.grad = None if param.decoupled_grad is None else param.decoupled_grad.cpu()
        optimizer.step()
        reference.step()
        torch.cuda.synchronize()
        for param, master in zip(parameters, masters):
            torch.testing.assert_close(param, master.to("cuda", dtype=torch.bfloat16), atol=0, rtol=0)
            for key in ("exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(optimizer.state[param][key], reference.state[master][key], atol=0, rtol=0)


def test_bounded_staging_rejects_unbounded_suboptimizer_group():
    """Fail before allocation if overlap=False would stage the whole model at once."""
    with pytest.raises(ValueError, match="overlap_cpu_optimizer_d2h_h2d"):
        _optimizer(_parameters(), overlap=False)


def test_checkpoint_state_initialization_avoids_cuda_gradient_copy():
    """Cold restore must not allocate another full CUDA gradient image."""
    parameters = [torch.nn.Parameter(torch.full((262144,), 0.25, device="cuda", dtype=torch.bfloat16))
                  for _ in range(32)]
    optimizer = _optimizer(parameters)
    before = [p.detach().clone() for p in parameters]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    optimizer.dummy_step()
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - allocated
    print(f"CHECKPOINT_INIT_EXTRA_CUDA_BYTES={extra}", flush=True)
    assert extra < 2 * 1024**2, "initializing CPU Adam state allocated model-sized CUDA gradients"
    torch.testing.assert_close(parameters, before, atol=0, rtol=0)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    assert all(p.grad is None for p in parameters + _cpu_parameters(optimizer))
    for parameter in parameters:
        state = optimizer.state[parameter]
        assert {"master_param", "step", "exp_avg", "exp_avg_sq"} <= state.keys()
        for key in ("master_param", "exp_avg", "exp_avg_sq"):
            assert state[key].device.type == "cpu" and state[key].dtype == torch.float32


@pytest.mark.parametrize("fraction", [1.0, 0.5])
def test_checkpoint_state_initialization_preserves_loaded_next_update(fraction):
    """Lazy state setup must retain group options and exact restored Adam updates."""
    original = _parameters()
    trained = _optimizer(original, fraction=fraction)
    _set_grads(original, 0, decoupled=True)
    trained.step()
    torch.cuda.synchronize()
    checkpoint = copy.deepcopy(trained.state_dict())
    restored_parameters = [torch.nn.Parameter(p.detach().clone()) for p in original]
    restored = _optimizer(restored_parameters, fraction=fraction)
    # The distributed optimizer restores into state initialized by dummy_step.
    # Preserve an explicit None decoupled_grad, as on its actual model shards.
    for p in restored_parameters:
        p.decoupled_grad = None
    options = [(g["lr"], g["weight_decay"]) for sub in restored.sub_optimizers for g in sub.param_groups]
    restored.dummy_step()
    assert [(g["lr"], g["weight_decay"]) for sub in restored.sub_optimizers for g in sub.param_groups] == options
    assert all(restored.state[p] for p in restored_parameters)
    torch.testing.assert_close(restored_parameters, original, atol=0, rtol=0)
    restored.load_state_dict(checkpoint)
    for params in (original, restored_parameters):
        _set_grads(params, 1, decoupled=True)
    trained.step()
    restored.step()
    torch.cuda.synchronize()
    _assert_hybrid_equal(restored, trained)


@pytest.mark.parametrize("fraction", [1.0, 0.5])
def test_bounded_resume_reuses_existing_master_storage(fraction):
    """Rebuilding all master copies during cold restore exceeds host capacity."""
    parameters = _parameters()
    optimizer = _optimizer(parameters, fraction=fraction)
    _set_grads(parameters, 0, decoupled=True)
    optimizer.step()
    torch.cuda.synchronize()
    checkpoint = copy.deepcopy(optimizer.state_dict())
    resumed_parameters = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    resumed = _optimizer(resumed_parameters, fraction=fraction)
    resumed.dummy_step()
    inner_parameters = dict(resumed.param_to_inner_param)
    resumed.load_state_dict(checkpoint)
    assert all(resumed.param_to_inner_param[p] is inner for p, inner in inner_parameters.items()), (
        "loading a same-layout checkpoint allocated new master parameter storage"
    )
    assert all(resumed.state[p]["master_param"] is inner for p, inner in inner_parameters.items())
    for params in (parameters, resumed_parameters):
        _set_grads(params, 1, decoupled=True)
    optimizer.step()
    resumed.step()
    torch.cuda.synchronize()
    _assert_hybrid_equal(resumed, optimizer)
