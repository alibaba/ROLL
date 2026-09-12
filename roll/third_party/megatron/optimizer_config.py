"""Version-aware construction helpers for Megatron optimizer configuration."""

import inspect


CPU_OPTIMIZER_FIELDS = (
    "optimizer_cpu_offload",
    "optimizer_offload_fraction",
    "use_torch_optimizer_for_cpu_offload",
    "overlap_cpu_optimizer_d2h_h2d",
    "pin_cpu_grads",
    "pin_cpu_params",
    "offload_optimizer_states",
)


def build_optimizer_config(config_cls, base_kwargs, train_args):
    """Build an upstream optimizer config without silently dropping CPU mode.

    MCore has added CPU optimizer fields over time. Forward fields supported by
    the installed config class and fail explicitly when a requested non-default
    mode cannot be represented by that class.
    """
    parameters = inspect.signature(config_cls).parameters
    accepted = set(parameters)
    kwargs = dict(base_kwargs)
    unsupported_base = sorted(set(kwargs) - accepted)
    if unsupported_base:
        raise TypeError(
            f"installed OptimizerConfig does not support required fields: {', '.join(unsupported_base)}"
        )

    for name in CPU_OPTIMIZER_FIELDS:
        if not hasattr(train_args, name):
            continue
        value = getattr(train_args, name)
        default = parameters[name].default if name in parameters else False
        requested = value is not None and value != default
        if name not in accepted:
            if requested:
                raise ValueError(
                    f"requested optimizer CPU mode field {name!r} is unsupported by installed OptimizerConfig"
                )
            continue
        kwargs[name] = value

    return config_cls(**kwargs)
