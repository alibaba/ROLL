"""M7 (integration): a TransformerLayer whose residual stream is a hyperconnection.

The plain layer does, twice per layer:

    residual = hidden_states
    hidden_states = block(norm(hidden_states)) + residual

Qwen3.8-Flash-Next instead keeps ``hc_count`` parallel streams and, per block:

    xn           = grouped_gemma_rmsnorm(stream)      # per-stream, Gemma (1+w)
    block_input  = mix down xn to one hidden vector   # gated, averaged by 1/hc
    block_output = block(block_input)
    stream       = stream + block_output * 2*sigmoid(inj/hc)   # broadcast back

So the hyperconnection replaces both the input layernorm and the residual add.
That is why this is a layer subclass and not a wrapper module.

**One deliberate difference from the inference implementation.** vLLM defers each
combine into the *next* block so it can fuse combine+norm into one kernel, which
makes its layer signature carry ``(hidden_states, prev_block_output,
prev_injection)`` across layer boundaries. Doing that here would mean changing
what every layer passes to the next -- and TransformerBlock, pipeline parallelism
and the cuda-graph paths all assume a single hidden-state tensor.

We combine immediately instead. P17 verified these are numerically identical
(``combine_and_mix`` == ``combine`` then ``mix``, difference 0.000e+00), so the
only cost is losing a kernel fusion -- irrelevant here, since the training-side
implementation is unfused anyway. The gain is that each layer stays
"one tensor in, one tensor out" and the surrounding machinery is untouched.

The stream state is widened once before the stack and mixed down once after; see
``expand_to_streams`` and ``HyperConnectionMixer`` in ``hyper_connection.py``.
"""

from __future__ import annotations

import torch
from megatron.core.transformer.transformer_layer import TransformerLayer

from .hyperconnection import HyperConnection


class HyperConnectionTransformerLayer(TransformerLayer):
    """TransformerLayer with hyperconnection residuals around attention and MLP.

    Reads ``hc_count`` / ``hc_lowrank`` off the config. If ``hc_count`` is unset or
    1, this behaves exactly like the base layer, so the same spec can be used for
    models with and without hyperconnection.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        cfg = self.config
        hc_count = getattr(cfg, "hc_count", None) or 1
        self.hc_count = hc_count
        self.hc_enabled = hc_count > 1

        if not self.hc_enabled:
            self.attn_hyper_connection = None
            self.mlp_hyper_connection = None
            return

        lowrank = getattr(cfg, "hc_lowrank", None)
        if lowrank is None:
            raise ValueError(
                "hc_count > 1 but hc_lowrank is not set; both come from the "
                "checkpoint config (real model: hc_count=4, hc_lowrank=320)"
            )

        common = {
            "hidden_size": cfg.hidden_size,
            "hc_count": hc_count,
            "lowrank": lowrank,
            "eps": cfg.layernorm_epsilon,
            "dtype": cfg.params_dtype,
        }
        # Names match the checkpoint (layers.N.attn_hyper_connection.*), so weight
        # conversion is a rename rather than a remap.
        self.attn_hyper_connection = HyperConnection(**common)
        self.mlp_hyper_connection = HyperConnection(**common)

        # The hyperconnection subsumes both layernorms. Leaving them in would apply
        # a second normalisation to an already-normalised tensor -- silent, and it
        # would also add parameters the checkpoint has no weights for.
        self._disable_subsumed_layernorms()

    def _disable_subsumed_layernorms(self) -> None:
        from megatron.core.transformer.identity_op import IdentityOp

        for name in ("input_layernorm", "pre_mlp_layernorm"):
            mod = getattr(self, name, None)
            if mod is not None and not isinstance(mod, IdentityOp):
                setattr(self, name, IdentityOp())

    # ------------------------------------------------------------------ forward
    def forward(self, *args, **kwargs):
        if not self.hc_enabled:
            return super().forward(*args, **kwargs)

        kwargs.pop("dynamic_inference_decode_only", None)

        # `hidden_states` is the multi-stream state, [*, hidden*hc_count].
        if args:
            stream, rest = args[0], args[1:]
        else:
            stream, rest = kwargs.pop("hidden_states"), ()

        expected = self.config.hidden_size * self.hc_count
        if stream.shape[-1] != expected:
            raise ValueError(
                f"hyperconnection layer expected a stream of width {expected} "
                f"(hidden_size {self.config.hidden_size} x hc_count {self.hc_count}) "
                f"but got {stream.shape[-1]}. Widen the embedding output with "
                "expand_to_streams() before the first layer."
            )

        # ---- attention block ------------------------------------------------
        stream, attn_input, attn_inj = self.attn_hyper_connection.mix(stream)
        attn_out, context = self._forward_attention(attn_input, *rest, **kwargs)
        stream = self.attn_hyper_connection.combine(stream, attn_out, attn_inj)

        # ---- mlp block ------------------------------------------------------
        stream, mlp_input, mlp_inj = self.mlp_hyper_connection.mix(stream)
        mlp_out = self._forward_mlp(
            mlp_input,
            kwargs.get("inference_context", None),
            padding_mask=kwargs.get("padding_mask", None),
        )
        # _forward_mlp may return (output, bias); the bias is folded in by the base
        # layer's bda, which we are replacing, so unpack and add it here.
        if isinstance(mlp_out, tuple):
            mlp_out, mlp_bias = mlp_out[0], (mlp_out[1] if len(mlp_out) > 1 else None)
            if mlp_bias is not None:
                mlp_out = mlp_out + mlp_bias
        stream = self.mlp_hyper_connection.combine(stream, mlp_out, mlp_inj)

        return stream, context


def make_hc_layer_spec(base_spec):
    """Swap a layer spec's module for the hyperconnection layer, keeping submodules.

    Used on the specs produced by mcore's hybrid-attention builder, so GDN vs
    full-attention placement (M3) is preserved.
    """
    import copy

    spec = copy.deepcopy(base_spec)
    if hasattr(spec, "layer_specs"):
        for layer_spec in spec.layer_specs:
            layer_spec.module = HyperConnectionTransformerLayer
    else:
        spec.module = HyperConnectionTransformerLayer
    return spec


__all__ = ["HyperConnectionTransformerLayer", "make_hc_layer_spec"]
