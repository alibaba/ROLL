"""Compatibility layers for Qwen3.5 hybrid attention."""

import copy

import torch
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.transformer.attention import SelfAttention

from .config_qwen3_5 import Qwen3_5Config


class Qwen3_5SelfAttention(SelfAttention):
    """Keep output gates aligned with query heads when KV heads are replicated."""

    def get_query_key_value_tensors(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        output_gate: bool = False,
        split_qkv: bool = True,
    ) -> tuple:
        tensors = super().get_query_key_value_tensors(
            hidden_states, key_value_states, output_gate=output_gate, split_qkv=split_qkv
        )
        if not output_gate or self.config.num_query_groups >= self.world_size:
            return tensors

        query, key, value, gate = tensors
        # Megatron versions that already slice the gate require no adjustment.
        if gate.size(2) != query.size(2):
            replicas = self.world_size // self.config.num_query_groups
            replica_rank = self.pg_collection.tp.rank() % replicas
            gate = gate.narrow(2, replica_rank * query.size(2), query.size(2))
        return query, key, value, gate


class Qwen3_5TorchGatedDeltaNet(GatedDeltaNet):
    """Use Megatron's torch conv/delta-rule paths without changing other layers."""

    def __init__(self, config: Qwen3_5Config, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        # Set this after construction so projection/norm submodules retain the
        # original config. Megatron's forward already supplies both fallbacks.
        self.config = copy.copy(self.config)
        self.config.deterministic_mode = True
