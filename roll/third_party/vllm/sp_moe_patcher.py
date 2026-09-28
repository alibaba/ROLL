# -*- encoding: utf-8 -*-
import importlib

_INSTALLED_FLAG = "_roll_sp_shared_expert_patched"

# True once at least one class has been patched.
INSTALLED = False


def _log(msg):
    # Use print to stay independent of logging config at import time.
    print("[roll-vllm-patch] %s" % msg, flush=True)


def _rebuild_mlp_linears(shared_expert):
    """Rebuild a Qwen2MoeMLP's two linears with disable_tp=True.

    Returns False when the MLP is already fixed.
    """
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        RowParallelLinear,
    )

    gate_up = shared_expert.gate_up_proj
    down = shared_expert.down_proj
    if getattr(gate_up, "disable_tp", False) and getattr(down, "disable_tp", False):
        return False

    shared_expert.gate_up_proj = MergedColumnParallelLinear(
        input_size=gate_up.input_size,
        output_sizes=list(gate_up.output_sizes),
        bias=gate_up.bias is not None,
        gather_output=getattr(gate_up, "gather_output", False),
        skip_bias_add=gate_up.skip_bias_add,
        params_dtype=gate_up.params_dtype,
        quant_config=gate_up.quant_config,
        prefix=gate_up.prefix,
        return_bias=gate_up.return_bias,
        disable_tp=True,
    )
    shared_expert.down_proj = RowParallelLinear(
        input_size=down.input_size,
        output_size=down.output_size,
        bias=down.bias is not None,
        input_is_parallel=down.input_is_parallel,
        skip_bias_add=down.skip_bias_add,
        params_dtype=down.params_dtype,
        reduce_results=down.reduce_results,
        quant_config=down.quant_config,
        prefix=down.prefix,
        return_bias=down.return_bias,
        disable_tp=True,
    )
    return True


def _patch_qwen2_moe_mlp(cls):
    """Mirror 0.21: Qwen2MoeMLP.__init__ accepts is_sequence_parallel."""
    orig = cls.__dict__.get("__init__")
    if orig is None or getattr(orig, _INSTALLED_FLAG, False):
        return False

    def __init__(self, *args, is_sequence_parallel=False, **kwargs):
        orig(self, *args, **kwargs)
        if is_sequence_parallel:
            _rebuild_mlp_linears(self)

    setattr(__init__, _INSTALLED_FLAG, True)
    cls.__init__ = __init__
    return True


def _patch_sparse_moe_block(cls):
    """Mirror 0.21: the SP path builds the shared expert with TP disabled."""
    orig = cls.__dict__.get("__init__")
    if orig is None or getattr(orig, _INSTALLED_FLAG, False):
        return False

    def __init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        if not getattr(self, "is_sequence_parallel", False):
            return
        shared = getattr(self, "shared_expert", None)
        if shared is None:
            return
        try:
            _rebuild_mlp_linears(shared)
        except Exception as exc:  # noqa: BLE001
            _log(
                "SP-MoE shared expert disable_tp rebuild FAILED on %s: %r -- "
                "DP>1 output will be garbled" % (type(self).__name__, exc)
            )

    setattr(__init__, _INSTALLED_FLAG, True)
    cls.__init__ = __init__
    return True


def apply_sp_moe_shared_expert_fix():
    """Install the 0.21-style patches. Idempotent; safe on other vllm versions."""
    global INSTALLED
    patched = []

    try:
        qwen2_moe = importlib.import_module("vllm.model_executor.models.qwen2_moe")
        if _patch_qwen2_moe_mlp(qwen2_moe.Qwen2MoeMLP):
            patched.append("Qwen2MoeMLP")
    except Exception:  # noqa: BLE001
        pass

    try:
        qwen3_next = importlib.import_module("vllm.model_executor.models.qwen3_next")
        # Qwen3_5SparseMoeBlock subclasses this without its own __init__.
        if _patch_sparse_moe_block(qwen3_next.Qwen3NextSparseMoeBlock):
            patched.append("Qwen3NextSparseMoeBlock")
    except Exception:  # noqa: BLE001
        pass

    if patched:
        INSTALLED = True
        _log("SP-MoE shared expert fix installed (vllm 0.21-style) on: %s" % ", ".join(patched))
    return INSTALLED
