"""Qwen4 GDN keeps convolution activation separate from the output gate."""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from megatron.core.ssm.gated_delta_net import GatedDeltaNet


class Qwen4ExpGatedDeltaNet(GatedDeltaNet):
    def __init__(self, *args, **kwargs):
        if not hasattr(GatedDeltaNet, "_apply_causal_conv1d"):
            raise RuntimeError(
                "Qwen3.8 requires the Megatron GDN convolution hook; "
                "apply scripts/qwen38/patch_megatron_gdn_convolution.py"
            )
        super().__init__(*args, **kwargs)

    @staticmethod
    def _split_input_projection(module, hidden_states, qkvz_rows):
        """Run the native Qwen3.8 two-GEMM input projection.

        The checkpoint intentionally keeps one Megatron ``in_proj`` parameter
        (q, k, v, gate, beta, alpha) for DCP and LoRA compatibility.  Native
        Qwen3.8 computes ``qkvz`` and ``ba`` with separate linear layers.  A
        single BF16 GEMM over the concatenated rows changes the accumulation
        rounding of beta/alpha, and the small difference is amplified by the
        recurrent GDN update.  Splitting the existing local TP weight at
        runtime preserves the checkpoint layout while matching the native
        arithmetic.

        ``module`` may be a plain Transformer Engine column linear or the
        adapter's ``LoraColumnParallelLinear`` wrapper.  The base projection is
        computed from the local TP weight. Preserve column-parallel input
        communication here; the parent GDN then performs CP movement.
        """
        if getattr(module, "disable_adapters", False) and getattr(module, "merged", False):
            module.unmerge()
        base = module.get_base_layer() if hasattr(module, "get_base_layer") else module
        weight = getattr(base, "weight", None)
        if weight is None or weight.ndim != 2:
            raise TypeError(
                "Qwen4 native GDN projection requires a 2-D local weight; "
                f"got {type(base).__name__}"
            )
        if not 0 < qkvz_rows < weight.shape[0]:
            raise ValueError(
                f"invalid Qwen4 GDN projection split {qkvz_rows} for {tuple(weight.shape)}"
            )
        bias = getattr(base, "bias", None)
        # Transformer Engine represents ``bias=False`` as an empty parameter
        # on some releases.  Passing that zero-length tensor to F.linear is
        # interpreted as a real bias and fails only when the first GDN layer
        # executes, so normalize it to the native no-bias path here.
        if bias is not None and bias.numel() == 0:
            bias = None
        qkvz_bias = None if bias is None else bias[:qkvz_rows]
        ba_bias = None if bias is None else bias[qkvz_rows:]
        projection_input = hidden_states
        tp_group = getattr(base, "tp_group", None)
        if tp_group is not None and tp_group.size() > 1:
            from megatron.core.tensor_parallel.mappings import (
                copy_to_tensor_model_parallel_region,
                gather_from_sequence_parallel_region,
            )

            if getattr(base, "sequence_parallel", False):
                projection_input = gather_from_sequence_parallel_region(
                    hidden_states, tensor_parallel_output_grad=True, group=tp_group,
                )
            else:
                projection_input = copy_to_tensor_model_parallel_region(hidden_states, group=tp_group)
        qkvz = F.linear(projection_input, weight[:qkvz_rows], qkvz_bias)
        ba = F.linear(projection_input, weight[qkvz_rows:], ba_bias)
        result = torch.cat((qkvz, ba), dim=-1)

        # Preserve the existing adapter semantics.  LoRA-B has the same fused
        # output layout as the base weight, so splitting its result at the same
        # boundary is exact while leaving adapter state/checkpoint keys intact.
        if (
            hasattr(module, "active_adapters")
            and not getattr(module, "disable_adapters", False)
            and not getattr(module, "merged", False)
        ):
            if getattr(module, "sequence_parallel", False) and getattr(base, "parallel_mode", None) == "column":
                from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

                # LoRA-B is a TE column linear and already sums its input
                # gradient over TP. Scatter that replicated gradient back to
                # sequence shards without a second sum.
                lora_input = gather_from_sequence_parallel_region(
                    hidden_states, tensor_parallel_output_grad=False, group=tp_group,
                )
            else:
                lora_input = hidden_states
            for adapter_name in module.active_adapters:
                if adapter_name not in module.lora_A:
                    continue
                lora_a = module.lora_A[adapter_name]
                lora_b = module.lora_B[adapter_name]
                dropout = module.lora_dropout[adapter_name]
                delta = lora_a(dropout(lora_input.to(lora_a.weight.dtype)))
                if isinstance(delta, tuple):
                    delta = delta[0]
                delta = lora_b(delta)
                if isinstance(delta, tuple):
                    delta = delta[0]
                scaling = module.scaling[adapter_name]
                if scaling != 1.0:
                    delta = delta * scaling
                result = result + delta
        # Match the shared LoRA wrapper's promoted accumulation, including
        # FP32 adapters on a BF16 base and multiple active adapters.
        return result.to(hidden_states.dtype), None

    @contextmanager
    def _native_input_projection(self):
        """Temporarily replace only ``in_proj.forward`` for parent GDN code."""
        module = self.in_proj
        # Qwen4's GR layer spec removes the fused input layernorm.  If a caller
        # constructs a non-GR layer with the norm-bearing TE module, retain the
        # original path rather than silently dropping that norm.
        if hasattr(module, "layer_norm_weight"):
            yield
            return
        original = module.forward
        # ``in_proj`` is column-parallel, so the runtime weight contains only
        # the local TP rows.  The native boundary is split after q/k/v/z on
        # every rank, before the local beta/alpha rows.
        qkvz_rows = (2 * self.qk_dim + 2 * self.v_dim) // self.tp_size

        def split_forward(_module, hidden_states, *args, **kwargs):
            if args or kwargs:
                raise TypeError("Qwen4 native GDN projection does not accept extra linear arguments")
            return self._split_input_projection(_module, hidden_states, qkvz_rows)

        module.forward = split_forward.__get__(module, type(module))
        try:
            yield
        finally:
            module.forward = original

    def forward(self, *args, **kwargs):
        # Keep Megatron's tested convolution, CP and recurrence implementation;
        # retain projection precision across the TP communication boundary.
        with self._native_input_projection(), self._fp32_output_projection():
            return super().forward(*args, **kwargs)

    @contextmanager
    def _fp32_output_projection(self):
        """Cast GDN row-projection results only after the FP32 TP sum.

        Rounding each rank's GEMM before reduction changes the forward result
        with TP size. PLE's signed square-root gate amplifies this difference
        in backward. Keep the checkpoint parameters and any LoRA wrapper;
        replace only the base row linear for the duration of this forward.
        """
        from .precision_linear import linear_with_fp32_output

        module = self.out_proj
        base = module.get_base_layer() if hasattr(module, "get_base_layer") else module
        original = base.forward

        def forward(_module, hidden_states, *args, **kwargs):
            if args or kwargs:
                raise TypeError("Qwen4 GDN row projection does not accept extra linear arguments")
            result = linear_with_fp32_output(hidden_states, _module.weight)
            group = getattr(_module, "tp_group", None)
            if group is not None and group.size() > 1:
                from megatron.core.tensor_parallel.mappings import (
                    reduce_from_tensor_model_parallel_region,
                    reduce_scatter_to_sequence_parallel_region,
                )

                if getattr(_module, "sequence_parallel", False):
                    result = reduce_scatter_to_sequence_parallel_region(result, group=group)
                else:
                    result = reduce_from_tensor_model_parallel_region(result, group=group)
            result = result.to(hidden_states.dtype)
            bias = getattr(_module, "bias", None)
            if bias is not None and bias.numel() == 0:
                bias = None
            if bias is not None and not getattr(_module, "skip_bias_add", False):
                result = result + bias
                bias = None
            return result, bias

        base.forward = forward.__get__(base, type(base))
        try:
            yield
        finally:
            base.forward = original

    def _apply_causal_conv1d(self, x, weight, bias):
        """Match native product rounding across the complete causal window."""
        from .causal_convolution import causal_conv1d

        return causal_conv1d(x, weight.squeeze(1), bias)

    def _apply_gated_norm(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Round after normalization, affine scaling and the output gate.

        Keep the existing norm parameter for checkpoint and optimizer compatibility.
        Calling the norm module first would round its BF16 output before gating.
        """
        x = x.reshape(-1, x.shape[-1])
        gate = gate.reshape(-1, gate.shape[-1])
        weight = self.out_norm.weight.float()
        if self.out_norm.zero_centered_gamma:
            weight = weight + 1
        activation = self.config.gdn_output_gate_type
        if x.is_cuda:
            if activation == "sigmoid":
                from .gated_norm import sigmoid_gated_rms_norm

                return sigmoid_gated_rms_norm(x, gate, weight, self.out_norm.eps)

            from fla.modules.fused_norm_gate import rms_norm_gated

            return rms_norm_gated(x, gate, weight, None, activation=activation, eps=self.out_norm.eps)

        # CPU reference avoids allocating CUDA state for metadata/tests.
        xf = x.float()
        normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.out_norm.eps)
        output_gate = torch.sigmoid(gate.float()) if activation == "sigmoid" else self.act_fn(gate.float())
        return (normalized * weight * output_gate).to(x.dtype)
