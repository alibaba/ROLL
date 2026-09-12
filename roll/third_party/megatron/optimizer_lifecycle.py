"""Lifecycle hooks for optimizers whose update state is owned by CPU."""

import types

import torch


def _includes_optimizer_states(include):
    if include is None:
        return True
    return any(getattr(item, "value", item) == "optimizer_states" for item in include)


def cpu_optimizer_offload_states(self, include=None, pin_memory=True, non_blocking=False):
    """Keep CPU optimizer state on CPU while recording the offload phase."""
    if _includes_optimizer_states(include):
        for state in self.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                    state[key] = value.to("cpu", non_blocking=non_blocking)
        self.offloaded_states = getattr(self, "offloaded_states", set())
        self.offloaded_states.add("optimizer_states")


def cpu_optimizer_reload_states(self, include=None, non_blocking=False, **kwargs):
    """End an optimizer-state offload phase without moving CPU-owned state."""
    if _includes_optimizer_states(include):
        self.offloaded_states = getattr(self, "offloaded_states", set())
        self.offloaded_states.discard("optimizer_states")


def bind_cpu_optimizer_state_lifecycle(optimizer):
    """Attach ROLL's phase API to a CPU-owned optimizer instance."""
    optimizer.offload_states = types.MethodType(cpu_optimizer_offload_states, optimizer)
    optimizer.reload_states = types.MethodType(cpu_optimizer_reload_states, optimizer)
    return optimizer
