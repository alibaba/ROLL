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

    def __init__(self, config, submodules, *args, **kwargs):
        hc_count = getattr(config, "hc_count", None) or 1
        if hc_count > 1:
            # Remove fused norms before their parameters and recomputation hooks
            # are constructed. Replacing only standalone norms is insufficient.
            submodules = _without_subsumed_norms(submodules)
        super().__init__(config, submodules, *args, **kwargs)
        self.hc_count = hc_count
        self.hc_enabled = hc_count > 1
        if not self.hc_enabled:
            self.attn_hyper_connection = None
            self.mlp_hyper_connection = None
            return

        from megatron.core.transformer.identity_op import IdentityOp

        if not isinstance(self.cross_attention, IdentityOp):
            raise ValueError("GR layers support decoder self-attention only")
        lowrank = getattr(config, "hc_lowrank", None)
        if lowrank is None:
            raise ValueError("hc_count > 1 requires hc_lowrank")
        common = {
            "hidden_size": config.hidden_size,
            "hc_count": hc_count,
            "lowrank": lowrank,
            "eps": config.layernorm_epsilon,
            "dtype": config.params_dtype,
            "device": "cpu" if config.use_cpu_initialization else torch.cuda.current_device(),
            "sequence_parallel": config.sequence_parallel,
        }
        self.attn_hyper_connection = HyperConnection(**common)
        self.mlp_hyper_connection = HyperConnection(**common)

    def forward(
        self, hidden_states, attention_mask=None, context=None, context_mask=None,
        rotary_pos_emb=None, rotary_pos_cos=None, rotary_pos_sin=None,
        rotary_pos_cos_sin=None, attention_bias=None, inference_context=None,
        packed_seq_params=None, sequence_len_offset=None, padding_mask=None,
        *, inference_params=None, dynamic_inference_decode_only=None,
    ):
        if not self.hc_enabled:
            return super().forward(
                hidden_states, attention_mask, context, context_mask,
                rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, rotary_pos_cos_sin,
                attention_bias, inference_context, packed_seq_params,
                sequence_len_offset, padding_mask, inference_params=inference_params,
            )

        from megatron.core.utils import deprecate_inference_params, make_viewless_tensor

        inference_context = deprecate_inference_params(inference_context, inference_params)
        expected = self.config.hidden_size * self.hc_count
        if hidden_states.shape[-1] != expected:
            raise ValueError(
                f"hyperconnection layer expected a stream of width {expected}, "
                f"got {hidden_states.shape[-1]}; use expand_to_streams() first"
            )
        # Base _forward_attention/_forward_mlp apply ordinary residual BDA.
        # GR must call the raw blocks, fold bias/dropout once, and write once.
        stream, attn_input, attn_inj = self.attn_hyper_connection.mix(hidden_states)
        attn_out = self.self_attention(
            attn_input, attention_mask=attention_mask,
            inference_context=inference_context, rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos, rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin, attention_bias=attention_bias,
            packed_seq_params=packed_seq_params, sequence_len_offset=sequence_len_offset,
        )
        stream = self.attn_hyper_connection.combine(
            stream, self._block_output(attn_out), attn_inj
        )
        stream, mlp_input, mlp_inj = self.mlp_hyper_connection.mix(stream)
        if self.recompute_mlp:
            from functools import partial
            from megatron.core import tensor_parallel

            if self.config.fp8 or self.config.fp4:
                from megatron.core.extensions.transformer_engine import te_checkpoint

                mlp_out = te_checkpoint(
                    self.mlp, False, tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp, mlp_input, padding_mask=padding_mask,
                )
            else:
                mlp_out = tensor_parallel.checkpoint(
                    partial(self.mlp, padding_mask=padding_mask), False, mlp_input,
                )
        else:
            mlp_out = self.mlp(mlp_input, padding_mask=padding_mask)
        stream = self.mlp_hyper_connection.combine(
            stream, self._block_output(mlp_out), mlp_inj
        )
        return make_viewless_tensor(
            inp=stream, requires_grad=stream.requires_grad, keep_graph=True
        ), context

    def _block_output(self, output_with_bias):
        output, bias = output_with_bias
        if bias is not None:
            output = output + bias
        return torch.nn.functional.dropout(output, p=self.hidden_dropout, training=self.training)

    def _forward_attention(self, *args, **kwargs):
        if self.hc_enabled:
            raise NotImplementedError(
                "Use unified GR forward; split attention CUDA graph/overlap paths are unsupported"
            )
        return super()._forward_attention(*args, **kwargs)

    def _forward_mlp(self, *args, **kwargs):
        if self.hc_enabled:
            raise NotImplementedError(
                "Use unified GR forward; split MLP CUDA graph/overlap paths are unsupported"
            )
        return super()._forward_mlp(*args, **kwargs)


def _without_subsumed_norms(submodules):
    """Copy a spec and remove only the norms that GR itself performs.

    Q/K norms and GDN's output norm remain part of the attention operation.
    """
    import copy
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear, TELayerNormColumnParallelLinear,
    )
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.spec_utils import ModuleSpec

    submodules = copy.deepcopy(submodules)
    submodules.input_layernorm = IdentityOp
    submodules.pre_mlp_layernorm = IdentityOp
    for block, fields in ((submodules.self_attention, ("linear_qkv", "in_proj")),
                          (submodules.mlp, ("linear_fc1",))):
        nested = getattr(block, "submodules", None)
        if nested is None:
            continue
        for field in fields:
            linear = getattr(nested, field, None)
            if linear is TELayerNormColumnParallelLinear:
                setattr(nested, field, TEColumnParallelLinear)
            elif isinstance(linear, ModuleSpec) and linear.module is TELayerNormColumnParallelLinear:
                linear.module = TEColumnParallelLinear
    return submodules


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
            layer_spec.submodules = _without_subsumed_norms(layer_spec.submodules)
    else:
        spec.module = HyperConnectionTransformerLayer
        spec.submodules = _without_subsumed_norms(spec.submodules)
    return spec


__all__ = ["HyperConnectionTransformerLayer", "make_hc_layer_spec"]
