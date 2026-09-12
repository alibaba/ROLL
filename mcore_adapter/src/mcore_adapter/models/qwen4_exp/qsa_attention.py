"""Megatron QSA projection, sparse core and indexer-distillation integration."""
import torch
from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

from .qsa import QSAIndexer, flex_qsa_attention, indexer_distillation_loss


class Qwen4ExpQSAAttention(SelfAttention):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if config.indexer_kv_heads != 1:
            raise ValueError("Qwen4 QSA indexer requires one replicated KV head")
        device = "cpu" if config.use_cpu_initialization else torch.cuda.current_device()
        self.indexer = QSAIndexer(config.hidden_size, config.indexer_n_heads,
                                  config.indexer_head_dim, config.indexer_compress_ratio,
                                  config.indexer_budget, config.layernorm_epsilon,
                                  dtype=config.params_dtype, device=device)
        for param in self.indexer.parameters():
            param.sequence_parallel = config.sequence_parallel
            param.tensor_model_parallel = False
        self.last_indexer_loss = None

    def forward(self, hidden_states, attention_mask=None, inference_context=None,
                rotary_pos_emb=None, rotary_pos_cos=None, rotary_pos_sin=None,
                rotary_pos_cos_sin=None, attention_bias=None, packed_seq_params=None,
                sequence_len_offset=None, qsa_valid_mask=None, qsa_loss_mask=None, **kwargs):
        if inference_context is not None or packed_seq_params is not None or attention_bias is not None:
            raise NotImplementedError("QSA training currently supports unpacked full-sequence inputs")
        if rotary_pos_emb is None:
            raise ValueError("QSA requires explicit rotary embeddings")
        query, key, value, gate = self.get_query_key_value_tensors(hidden_states, output_gate=True)
        seq, batch = query.shape[:2]
        if isinstance(rotary_pos_emb, tuple):
            q_rope, k_rope = rotary_pos_emb
        else:
            q_rope = k_rope = rotary_pos_emb
        query = apply_rotary_pos_emb(query, q_rope, config=self.config, cp_group=self.pg_collection.cp)
        key = apply_rotary_pos_emb(key, k_rope, config=self.config, cp_group=self.pg_collection.cp)
        valid = qsa_valid_mask
        if valid is None:
            if attention_mask is not None:
                raise ValueError("QSA requires explicit token validity alongside an attention mask")
            valid = torch.ones(batch, seq, device=query.device, dtype=torch.bool)
        if valid.shape != (batch, seq):
            raise ValueError("QSA validity must cover the full sequence")
        index_hidden = hidden_states
        if self.config.sequence_parallel:
            index_hidden = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=True, group=self.pg_collection.tp)
        angles = q_rope[:seq, 0, 0].unsqueeze(0).expand(batch, -1, -1)
        selection, index_q, index_k = self.indexer(index_hidden.transpose(0, 1), angles, valid)
        q, k, v = [x.permute(1, 2, 0, 3).contiguous() for x in (query, key, value)]
        context, lse = flex_qsa_attention(q, k, v, selection)
        if self.training and self.config.qsa_indexer_kl_coef > 0:
            aux = indexer_distillation_loss(index_q, index_k, selection, q, k, lse,
                                           temperature=self.config.qsa_indexer_temperature,
                                           loss_mask=qsa_loss_mask, tp_group=self.pg_collection.tp)
            # Each TP rank computes identical indexer loss from globally summed
            # teacher heads. SP reduction sums replicated parameter gradients.
            tp_size = torch.distributed.get_world_size(self.pg_collection.tp)
            context = MoEAuxLossAutoScaler.apply(
                context, aux * self.config.qsa_indexer_kl_coef / tp_size)
            self.last_indexer_loss = aux.detach()
        output = context.permute(2, 0, 1, 3).contiguous() * gate.sigmoid()
        return self.linear_proj(output.flatten(-2))
