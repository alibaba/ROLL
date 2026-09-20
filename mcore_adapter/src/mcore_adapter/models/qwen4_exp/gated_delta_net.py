"""Qwen4 GDN keeps convolution activation separate from the output gate."""
import torch
from megatron.core.ssm.gated_delta_net import GatedDeltaNet


class Qwen4ExpGatedDeltaNet(GatedDeltaNet):
    def __init__(self, *args, **kwargs):
        if not hasattr(GatedDeltaNet, "_apply_causal_conv1d"):
            raise RuntimeError(
                "Qwen3.8 requires the Megatron GDN convolution hook; "
                "apply scripts/qwen38/patch_megatron_gdn_convolution.py"
            )
        super().__init__(*args, **kwargs)

    def _apply_causal_conv1d(self, x, weight, bias):
        """Keep the checkpoint's convolution output dtype before SiLU.

        Fusing activation inside FLA retains the accumulator in FP32 through
        SiLU. The native model applies SiLU to the rounded convolution output.
        Use FLA's differentiable convolution and retain that boundary.
        """
        from fla.modules.convolution import causal_conv1d

        convolved, _ = causal_conv1d(
            x=x, weight=weight.squeeze(1), bias=bias, activation=None,
            initial_state=None, output_final_state=False,
        )
        return self.act_fn(convolved)

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
            from fla.modules.fused_norm_gate import rms_norm_gated

            return rms_norm_gated(x, gate, weight, None, activation=activation, eps=self.out_norm.eps)

        # CPU reference avoids allocating CUDA state for metadata/tests.
        xf = x.float()
        normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.out_norm.eps)
        output_gate = torch.sigmoid(gate.float()) if activation == "sigmoid" else self.act_fn(gate.float())
        return (normalized * weight * output_gate).to(x.dtype)
