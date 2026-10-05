"""Initialize pristine all-CPU AdamW checkpoints without taking an Adam step."""
import torch


def initialize_cpu_adamw_state(optimizer):
    """Ask AdamW to allocate its lazy state with at most one temporary gradient.

    Calling ``step`` with a zero learning rate advances the bias correction
    counter. Use AdamW's initialization routine so a checkpoint before the
    first update retains exactly the same subsequent training trajectory.
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError('initial CPU checkpoint state requires torch.optim.AdamW')
    for group in optimizer.param_groups:
        for parameter in group['params']:
            if parameter.device.type != 'cpu':
                raise ValueError('initial CPU checkpoint state requires CPU parameters')
    for group in optimizer.param_groups:
        for parameter in group['params']:
            if optimizer.state.get(parameter):
                continue
            previous_gradient = parameter.grad
            try:
                if previous_gradient is None:
                    parameter.grad = torch.zeros_like(parameter)
                # The native initializer owns dtype, fused/capturable step
                # placement and optional AMSGrad state. It never updates p.
                optimizer._init_group(dict(group, params=[parameter]), *[[] for _ in range(6)])
            except Exception:
                # A failed allocation may have installed only step/exp_avg.
                # Leave this previously empty entry retryable.
                optimizer.state.pop(parameter, None)
                raise
            finally:
                parameter.grad = previous_gradient


def prepare_cpu_adam_for_checkpoint(optimizer):
    """Synchronize native all-CPU Hybrid AdamW state before distributed save."""
    from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer

    if hasattr(optimizer, 'chained_optimizers'):
        for child in optimizer.chained_optimizers:
            prepare_cpu_adam_for_checkpoint(child)
    elif isinstance(optimizer, HybridDeviceOptimizer):
        # Mixed CPU/GPU optimizers keep their existing initialization contract.
        if optimizer.gpu_optimizer is not None or not optimizer.cpu_optimizers:
            return
        if not all(isinstance(child, torch.optim.AdamW) for child in optimizer.cpu_optimizers):
            return
        for child in optimizer.cpu_optimizers:
            initialize_cpu_adamw_state(child)
        optimizer._sync_sub_optimizers_state_to_hdo()
    elif getattr(optimizer, 'optimizer', None) is not None:
        prepare_cpu_adam_for_checkpoint(optimizer.optimizer)
