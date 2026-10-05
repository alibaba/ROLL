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
    "use_precision_aware_optimizer",
)

# Defaults of the ROLL options, also used when an older MCore lacks a field.
# Pinning is enabled by default and does not itself request CPU offload.
_OPTION_DEFAULTS = {name: name in {"pin_cpu_grads", "pin_cpu_params"} for name in CPU_OPTIMIZER_FIELDS}


def validate_bounded_cpu_grad_staging(config):
    """Validate ROLL's opt-in without importing optional Megatron extensions."""
    enabled = getattr(config, "bounded_cpu_grad_staging", False)
    if enabled:
        for name in ("optimizer_cpu_offload", "overlap_cpu_optimizer_d2h_h2d"):
            if not getattr(config, name, False):
                raise ValueError(f"bounded_cpu_grad_staging requires {name}=True")
    return enabled


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
        default = parameters[name].default if name in parameters else _OPTION_DEFAULTS[name]
        requested = value is not None and value != default
        if name not in accepted:
            if requested:
                raise ValueError(
                    f"requested optimizer CPU mode field {name!r} is unsupported by installed OptimizerConfig"
                )
            continue
        kwargs[name] = value

    config = config_cls(**kwargs)
    # This is a ROLL option: upstream OptimizerConfig does not accept it.
    config.bounded_cpu_grad_staging = getattr(train_args, "bounded_cpu_grad_staging", False)
    validate_bounded_cpu_grad_staging(config)
    config.offload_model_from_cpu_master = getattr(train_args, "offload_model_from_cpu_master", False)
    if config.offload_model_from_cpu_master:
        for name, expected in (("optimizer_cpu_offload", True),
                               ("optimizer_offload_fraction", 1.0),
                               ("use_precision_aware_optimizer", True)):
            if getattr(config, name, None) != expected:
                raise ValueError(f"offload_model_from_cpu_master requires {name}={expected}")
    return config
