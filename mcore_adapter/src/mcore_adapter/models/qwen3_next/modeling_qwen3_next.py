from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from torch.nn import functional as F

from ...platforms import current_platform
from ..auto.modeling_auto import register_model
from ..model_factory import McaGPTModel
from .config_qwen3_next import Qwen3NextConfig


# based on qwen3next code in transformers
class Qwen3NextRMSNorm(nn.Module):
    def __init__(self, config: "Qwen3NextConfig", hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        device = current_platform.current_device() if not config.use_cpu_initialization else None
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=config.params_dtype, device=device))
        self.variance_epsilon = config.layernorm_epsilon

        # set sequence parallelism flag
        setattr(self.weight, "sequence_parallel", config.sequence_parallel)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x).contiguous()

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# based on qwen3next code in transformers
class Qwen3NextGatedDeltaNet(MegatronModule):
    def __init__(
        self,
        config: Qwen3NextConfig,
        submodules,
        layer_number: int,
        **kwargs,
    ):
        try:
            from fla.modules import FusedRMSNormGated
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        except ImportError:
            raise ImportError("Please install flash-linear-attention to use Qwen3NextGatedDeltaNet")

        self.chunk_gated_delta_rule = chunk_gated_delta_rule
        super().__init__(config=config)
        device = current_platform.current_device() if not config.use_cpu_initialization else None
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_number = layer_number
        self.layer_norm_epsilon = config.layernorm_epsilon

        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        self.in_proj_qkvz = nn.Linear(
            self.hidden_size, projection_size_qkvz, bias=False, device=device, dtype=config.params_dtype
        )

        projection_size_ba = self.num_v_heads * 2
        self.in_proj_ba = nn.Linear(
            self.hidden_size, projection_size_ba, bias=False, device=device, dtype=config.params_dtype
        )

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            device=device,
            dtype=config.params_dtype,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads, device=device, dtype=config.params_dtype))
        A = torch.empty(self.num_v_heads, device=device, dtype=config.params_dtype).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = FusedRMSNormGated(
            self.head_v_dim, eps=self.layer_norm_epsilon, device=device, dtype=config.params_dtype
        )
        self.out_proj = nn.Linear(
            self.value_dim, self.hidden_size, bias=False, device=device, dtype=config.params_dtype
        )

    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvz` and `mixed_ba`.
        """

        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads,
            2 * self.head_k_dim + 2 * self.head_v_dim * self.num_v_heads // self.num_k_heads,
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (self.num_k_heads, 2 * self.num_v_heads // self.num_k_heads)

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)
        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [self.num_v_heads // self.num_k_heads, self.num_v_heads // self.num_k_heads]
        query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=3)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim=3)
        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), value.size(1), -1, self.head_v_dim)
        z = z.reshape(z.size(0), z.size(1), -1, self.head_v_dim)
        b = b.reshape(b.size(0), b.size(1), self.num_v_heads)
        a = a.reshape(a.size(0), a.size(1), self.num_v_heads)
        return query, key, value, z, b, a

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = hidden_states.transpose(0, 1)  # [b, s, h]

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        mixed_qkv = torch.cat((query, key, value), dim=-1)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)

        output = self.out_proj(core_attn_out)
        output = output.transpose(0, 1)  # [s, b, h]
        return output, None


class Qwen3NextSelfAttention(SelfAttention):
    def __init__(
        self,
        config: Qwen3NextConfig,
        submodules,
        *args,
        **kwargs,
    ):
        config.num_attention_heads *= 2
        # double size of query weight
        super().__init__(
            config,
            submodules,
            *args,
            **kwargs,
        )
        config.num_attention_heads //= 2

        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size // 2,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="proj",
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        # add gate based on megatron attention forward impl
        assert rotary_pos_cos is None and rotary_pos_sin is None

        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # from get_query_key_value_tensors
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        try:
            import transformer_engine  # noqa: F401
            from megatron.core.extensions.transformer_engine import SplitAlongDim
        except ImportError:
            SplitAlongDim = None

        if SplitAlongDim is not None:
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head * 2)
        query, gate = torch.chunk(query, 2, dim=-1)

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)
        # end get_query_key_value_tensors

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        if rotary_pos_emb is not None and not self.config.flash_decode:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            if q_pos_emb is not None:
                query = apply_rotary_pos_emb(query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q)
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv)

        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=self.attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=self.attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        core_attn_out = core_attn_out * torch.sigmoid(gate.reshape(core_attn_out.shape))
        output, bias = self.linear_proj(core_attn_out)
        return output, bias


@register_model("qwen3_next")
class Qwen3NextModel(McaGPTModel):
    config_class = Qwen3NextConfig

    def _get_transformer_layer_spec(self, config: Optional[Qwen3NextConfig] = None):
        config = config or self.config
        transformer_block_spec = super()._get_transformer_layer_spec(config)
        assert isinstance(transformer_block_spec, TransformerBlockSubmodules), (
            f"Invalid transformer_block_spec: {transformer_block_spec}"
        )
        linear_layer_specs = deepcopy(transformer_block_spec.layer_specs[0])
        linear_layer_specs.submodules.self_attention.module = Qwen3NextGatedDeltaNet
        linear_layer_specs.submodules.input_layernorm = TENorm
        offset = get_transformer_layer_offset(config, vp_stage=self.vp_stage)

        for i in range(len(transformer_block_spec.layer_specs)):
            layer_idx = i + offset
            if config.layer_types[layer_idx] == "linear_attention":
                transformer_block_spec.layer_specs[i] = linear_layer_specs
            else:
                transformer_block_spec.layer_specs[i].submodules.self_attention.module = Qwen3NextSelfAttention
        return transformer_block_spec
