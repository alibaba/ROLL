"""M2: HF <-> megatron-core weight conversion template for ``qwen4_exp``.

Modelled on the bundled ``qwen3_next`` template (same GDN-plus-MoE family), with
the differences this checkpoint actually has. The ones that matter:

1. **Nested config.** Everything lives under ``text_config`` (there is also a
   ``vision_config``), so the mapping uses dotted keys -- ``Template``'s
   ``get_hf_config_value`` walks them. Measured: a flat ``hidden_size`` lookup
   returns None, ``text_config.hidden_size`` returns 2560.

2. **Weight prefix.** 1293 of 1658 tensors are ``model.language_model.*``, not
   ``model.*`` -- this was blocker B09.

3. **GDN input projection is split four ways**, not two::

       in_proj_qkv (10240, 2560)  = q(16*128) | k(16*128) | v(48*128)
       in_proj_z   ( 6144, 2560)  = gate, 48*128
       in_proj_b   (   48, 2560)  = beta
       in_proj_a   (   48, 2560)  = alpha

   qwen3_next instead has ``in_proj_qkvz`` + ``in_proj_ba`` and interleaves them
   per key-head, so ``NextGDNConverOp`` cannot be reused.

   P13 measured mcore's side: ``in_proj_dim = 16480``, exactly the sum above, and
   ``forward`` splits it as ``qkv | gate | beta | alpha``. So the conversion is a
   plain ordered concatenation -- no interleaving. Verified, not assumed.

4. **Experts are pre-stacked.** ``mlp.experts.gate_up_proj`` is
   ``(512, 1280, 2560)`` and ``down_proj`` is ``(512, 2560, 640)``: one tensor for
   all 512 experts, whereas qwen3_next has per-expert tensors. Needs slicing per
   expert, and ``gate_up_proj`` is already fused so no gate/up stacking is needed.

5. **Not yet supported, dropped deliberately** (see M6/M7/M8):
   PLE n-gram table, QSA indexer, MTP, vision tower.
   (Hyperconnection is no longer dropped -- M7 maps it; see below.)
   Each is listed explicitly rather than pattern-swallowed, so that a tensor we
   have not thought about raises instead of vanishing.
"""

import re
from dataclasses import dataclass, field

import torch

from ..converter.dist_converter import (
    DistParallelConfig,
    default_dist_config,
    gdn_dist_config,
    register_dist_config,
    shared_moe_dist_config,
)
from ..converter.dist_converter import StackedTensors
from ..converter.template import (
    ConverOp,
    GatedQKVConverOp,
    GDNConv1dConverOp,
    RenameConverOp,
    StackConverOp,
    Template,
    register_template,
)


@dataclass
class DropConverOp(ConverOp):
    """Explicitly discard an auxiliary tensor outside the training graph."""

    def _hf_to_mca(self, weights):
        return []

    def _mca_to_hf(self, weights):
        return []


@dataclass
class ZeroCenteredRMSNormConverOp(ConverOp):
    """Convert HF one-centered norm weights to Megatron zero-centered weights."""

    def _hf_to_mca(self, weights):
        return weights[0].clone() - 1

    def _mca_to_hf(self, weights):
        return weights[0].clone() + 1


@dataclass
class Qwen4ExpGDNInProjConverOp(ConverOp):
    """Concatenate the four HF GDN input projections into mcore's fused in_proj.

    Order is ``qkv | gate | beta | alpha``, taken from mcore's own forward-pass
    split (P13), not inferred. Row counts must sum to ``in_proj_dim``; we assert
    that instead of trusting it, because a wrong layout here loads cleanly and
    corrupts training silently.
    """

    def __post_init__(self):
        super().__post_init__()
        assert len(self.hf_names) == 4, f"expected 4 hf names, got {self.hf_names}"
        assert len(self.mca_names) == 1, f"expected 1 mca name, got {self.mca_names}"

    def _expected_rows(self):
        c = self.mca_config
        qk = c.linear_num_key_heads * c.linear_key_head_dim
        v = c.linear_num_value_heads * c.linear_value_head_dim
        return qk, qk, v, v, c.linear_num_value_heads, c.linear_num_value_heads

    def _hf_to_mca(self, weights):
        qkv, z, b, a = weights
        q_rows, k_rows, v_rows, z_rows, b_rows, a_rows = self._expected_rows()

        assert qkv.shape[0] == q_rows + k_rows + v_rows, (
            f"in_proj_qkv has {qkv.shape[0]} rows, expected {q_rows + k_rows + v_rows} "
            f"(q={q_rows} k={k_rows} v={v_rows})"
        )
        assert z.shape[0] == z_rows, f"in_proj_z has {z.shape[0]} rows, expected {z_rows}"
        assert b.shape[0] == b_rows, f"in_proj_b has {b.shape[0]} rows, expected {b_rows}"
        assert a.shape[0] == a_rows, f"in_proj_a has {a.shape[0]} rows, expected {a_rows}"

        # Return StackedTensors, not a single concatenated tensor.
        #
        # dist_converter._convert_gdn asserts isinstance(weight, StackedTensors) and
        # then column-splits EACH component separately -- it has to, because q/k/v/
        # gate/beta/alpha have different head counts and a naive split of the fused
        # tensor would cut through the middle of a component under TP>1. Returning a
        # plain tensor trips that assert:
        #     AssertionError: weight: tensor([[...]]) swiglu: True
        # (found on the real run; P14 bypasses dist_convert entirely.)
        #
        # Order is qkv | gate | beta | alpha, per mcore's own forward split (P13).
        # q and k are kept separate here so each is split by head correctly.
        q_rows, k_rows, v_rows, _, _, _ = self._expected_rows()
        # .clone() on each slice: these are views into `qkv`, and torch.save writes
        # a tensor's whole backing storage, so views inflate the checkpoint (see the
        # note on the expert slices -- that cost 4.76x before it was caught).
        q = qkv[:q_rows].clone()
        k = qkv[q_rows : q_rows + k_rows].clone()
        v = qkv[q_rows + k_rows :].clone()
        assert v.shape[0] == v_rows, (
            f"v slice has {v.shape[0]} rows, expected {v_rows}"
        )
        return StackedTensors([q, k, v, z, b, a], dim=0)

    def _mca_to_hf(self, weights):
        """Split mca's in_proj back into the checkpoint's four tensors.

        Accepts either a StackedTensors (what dist_converter hands back, with the
        six components already separated) or a single fused tensor (the plain
        non-distributed path).
        """
        assert len(weights) == 1
        w = weights[0]
        q_rows, k_rows, v_rows, z_rows, b_rows, a_rows = self._expected_rows()

        if isinstance(w, StackedTensors):
            assert len(w.tensors) == 6, (
                f"expected 6 stacked components (q,k,v,gate,beta,alpha), "
                f"got {len(w.tensors)}"
            )
            q, k, v, z, b, a = w.tensors
            qkv = torch.cat([q, k, v], dim=0)
            return [qkv, z, b, a]

        qkv_rows = q_rows + k_rows + v_rows
        assert w.shape[0] == qkv_rows + z_rows + b_rows + a_rows, (
            f"in_proj has {w.shape[0]} rows, expected "
            f"{qkv_rows + z_rows + b_rows + a_rows}"
        )
        qkv = w[:qkv_rows]
        z = w[qkv_rows : qkv_rows + z_rows]
        b = w[qkv_rows + z_rows : qkv_rows + z_rows + b_rows]
        a = w[qkv_rows + z_rows + b_rows :]
        return [qkv, z, b, a]


# Hyperconnection (M7) and PLE (M6) weights need dist-parallel rules, or
# dist_convert raises "name: decoder.layers.0.mlp_hyper_connection..." partway
# through a conversion -- every mca weight must match some rule.
#
# All of these are REPLICATED, not tensor-parallel split:
#   * hyperconnection operates on the full hc-wide stream and its lowrank
#     projections are tiny (320); splitting them would add communication for no
#     memory saving.
#   * the PLE projections are per-layer and small; the 95 GiB table is not here at
#     all (it stays a frozen buffer outside the parameter tree -- M6).
_qwen4_exp_extra_dist_config = DistParallelConfig(
    duplicated_weights=[
        ".attn_hyper_connection.hc_norm",
        ".attn_hyper_connection.input_mix_weight_down.weight",
        ".attn_hyper_connection.input_mix_weight_up.weight",
        ".attn_hyper_connection.block_inject_weight.weight",
        ".mlp_hyper_connection.hc_norm",
        ".mlp_hyper_connection.input_mix_weight_down.weight",
        ".mlp_hyper_connection.input_mix_weight_up.weight",
        ".mlp_hyper_connection.block_inject_weight.weight",
        ".ple.key_proj.weight",
        ".ple.value_proj.weight",
        ".ple.conv1d.weight",
        ".ple.norm_key",
        ".ple.norm_query",
        ".ple.norm_conv",
        # full names: the mixer sits outside the layer stack (see the note below)
        "decoder.hyper_connection_mixer.hc.hc_norm",
        "decoder.hyper_connection_mixer.hc.input_mix_weight_down.weight",
        "decoder.hyper_connection_mixer.hc.input_mix_weight_up.weight",
    ],
    # The final mixer is not layer-scoped, so remove_mca_weight_prefix leaves its
    # name intact and dist_convert matches the FULL name. It must therefore be
    # listed in duplicated_weights (which dist_convert consults) rather than
    # post_process_weights -- the latter is used for deciding pipeline placement,
    # not for the parallel-strategy lookup, so listing it there alone gives:
    #   ValueError: name: decoder.hyper_connection_mixer.hc.hc_norm,
    #               pure_name: decoder.hyper_connection_mixer.hc.hc_norm
    post_process_weights=[
        "decoder.hyper_connection_mixer.hc.hc_norm",
        "decoder.hyper_connection_mixer.hc.input_mix_weight_down.weight",
        "decoder.hyper_connection_mixer.hc.input_mix_weight_up.weight",
    ],
)

register_dist_config(
    "qwen4_exp",
    default_dist_config.merge_configs(shared_moe_dist_config)
    .merge_configs(gdn_dist_config)
    .merge_configs(_qwen4_exp_extra_dist_config),
)


@dataclass
class Qwen4ExpTemplate(Template):
    """Template handling this checkpoint's nested config and prefixed weights."""

    # accumulates per-expert weights during MCA -> HF until a layer is complete
    _expert_buffer: dict = field(default_factory=dict)

    # HF ships all experts in ONE tensor; mca wants one name per expert.
    #
    # Naming: use the `local_experts.{e}.linear_fc{1,2}.weight` form, which is what
    # the bundled qwen3_next template emits and what dist_converter's
    # grouped_column_map / grouped_row_map match on (they key off the pure name
    # `.linear_fc1.weight`). The built model's parameters are actually called
    # `linear_fc1.weight{e}` because grouped GEMM fuses them -- mca translates
    # between the two internally.
    #
    # Emitting the model's `weight{e}` form directly does NOT work: it reaches
    # dist_converter with a pure name of `.mlp.experts.linear_fc2.weight0`, which
    # matches no dist rule, and the conversion dies with
    #     ValueError: name: decoder.layers.44.mlp.experts.linear_fc2.weight0 ...
    # (found on the real run; P14's synthetic test never went through dist_convert).
    #
    # Orientation matches HF on both: fc1 is (2*moe_inter, hidden), fc2 is
    # (hidden, moe_inter), so no transpose.
    _STACKED_EXPERTS = {
        "gate_up_proj": "linear_fc1",
        "down_proj": "linear_fc2",
    }

    def _stacked_expert_match(self, name):
        m = re.match(
            r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$",
            name,
        )
        return (int(m.group(1)), m.group(2)) if m else (None, None)

    def add_hf_weight(self, name, weight):
        layer_idx, which = self._stacked_expert_match(name)
        if layer_idx is not None:
            fc = self._STACKED_EXPERTS[which]
            n_experts = weight.shape[0]
            assert n_experts == self.mca_config.num_moe_experts, (
                f"{name} holds {n_experts} experts but config says "
                f"{self.mca_config.num_moe_experts}"
            )
            out = {}
            for e in range(n_experts):
                key = (
                    f"decoder.layers.{layer_idx}.mlp.experts."
                    f"local_experts.{e}.{fc}.weight"
                )
                # .clone() is essential, not tidiness. weight[e] is a VIEW into the
                # full 512-expert HF tensor, and torch.save writes each tensor's
                # entire backing storage -- so every 1.6 MiB expert slice would drag
                # its 1600 MiB parent into the file. Measured on the first TP=2 run:
                # tensors summed to 19.3 GiB but the shard file was 92 GiB (4.76x),
                # with the worst tensors at 1024x their own size. At 16 shards that
                # is ~1.5 TiB instead of ~300 GiB.
                w = weight[e].clone()
                if fc == "linear_fc1":
                    # `.linear_fc1.weight` is a swiglu weight, and
                    # dist_converter._convert_swiglu requires StackedTensors with
                    # EXACTLY 2 components so it can column-split gate and up
                    # separately (a naive split of the fused tensor would cut
                    # through the boundary under TP>1). HF ships gate_up_proj
                    # already fused as (2*moe_inter, hidden), so split it back:
                    #   rows [0 : moe_inter)          -> gate
                    #   rows [moe_inter : 2*moe_inter) -> up
                    # Passing the fused tensor trips:
                    #   AssertionError: weight: tensor([[...]]) swiglu: True
                    # (found on the real run; P14 never goes through dist_convert.)
                    half = w.shape[0] // 2
                    assert w.shape[0] == 2 * half, (
                        f"gate_up_proj for expert {e} has an odd row count "
                        f"{w.shape[0]}; expected 2 x moe_ffn_hidden_size"
                    )
                    expected = self.mca_config.moe_ffn_hidden_size
                    assert half == expected, (
                        f"gate_up_proj half is {half} rows but "
                        f"moe_ffn_hidden_size is {expected}"
                    )
                    # clone the halves too -- same view-vs-storage issue
                    out[key] = StackedTensors(
                        [w[:half].clone(), w[half:].clone()], dim=0
                    )
                else:
                    out[key] = w
            return out

        # GDN layers keep their input layernorm fused into in_proj, exactly as in
        # qwen3_next; full-attention layers use the ordinary qkv layernorm.
        m = re.match(r"^model\.language_model\.layers\.(\d+)\.input_layernorm\.weight$", name)
        if m:
            idx = int(m.group(1))
            if self.mca_config.layer_types[idx] == "linear_attention":
                return {
                    f"decoder.layers.{idx}.self_attention.in_proj.layer_norm_weight": weight
                }
        return super().add_hf_weight(name, weight)

    def hf_name_to_mca_names(self, hf_name):
        """Name-level lookup, used by the conversion driver before reading tensors.

        The stacked expert tensors are handled in add_hf_weight (value level), but
        model_converter also asks, name by name, "which mca tensors does this HF
        tensor become?" -- without this override that lookup finds no ConverOp and
        raises mid-conversion:

            ValueError: can not find conver op for .mlp.experts.down_proj

        P14 exercised only the value path, which is why this surfaced first on the
        real 336 GiB run.
        """
        layer_idx, which = self._stacked_expert_match(hf_name)
        if layer_idx is not None:
            fc = self._STACKED_EXPERTS[which]
            n = self.mca_config.num_moe_experts
            return [
                f"decoder.layers.{layer_idx}.mlp.experts.local_experts.{e}.{fc}.weight"
                for e in range(n)
            ]
        return super().hf_name_to_mca_names(hf_name)

    def add_mca_weight(self, name, weight, **kwargs):
        # Re-stack per-expert weights. Buffer until every expert of a layer has
        # arrived, then emit the single HF tensor.
        em = re.match(
            r"^decoder\.layers\.(\d+)\.mlp\.experts\.local_experts\.(\d+)\."
            r"(linear_fc1|linear_fc2)\.weight$",
            name,
        )
        if em:
            layer_idx, e, fc = int(em.group(1)), int(em.group(2)), em.group(3)
            hf_which = {"linear_fc1": "gate_up_proj", "linear_fc2": "down_proj"}[fc]
            # linear_fc1 arrives as StackedTensors([gate, up]) on the way back;
            # flatten to the fused (2*moe_inter, hidden) form the HF tensor uses.
            if hasattr(weight, "tensors"):
                weight = torch.cat(list(weight.tensors), dim=weight.dim)
            buf = self._expert_buffer.setdefault((layer_idx, hf_which), {})
            buf[e] = weight
            if len(buf) < self.mca_config.num_moe_experts:
                return {}
            stacked = torch.stack([buf[i] for i in sorted(buf)], dim=0)
            del self._expert_buffer[(layer_idx, hf_which)]
            return {
                f"model.language_model.layers.{layer_idx}.mlp.experts.{hf_which}": stacked
            }

        m = re.match(
            r"^decoder\.layers\.(\d+)\.self_attention\.in_proj\.layer_norm_weight$", name
        )
        if m:
            idx = int(m.group(1))
            return {
                f"model.language_model.layers.{idx}.input_layernorm.weight": weight
            }
        return super().add_mca_weight(name, weight, **kwargs)


register_template(
    "qwen4_exp",
    template_class=Qwen4ExpTemplate,
    # NOTE the language_model prefix -- this was blocker B09
    hf_layer_prefix="model.language_model.layers.",
    config_hf_to_mca={
        "text_config.max_position_embeddings": "max_sequence_length",
        "text_config.hidden_size": "hidden_size",
        "text_config.attention_bias": "add_qkv_bias",
        "text_config.head_dim": "kv_channels",
        "text_config.num_attention_heads": "num_attention_heads",
        "text_config.num_key_value_heads": "num_query_groups",
        "text_config.num_hidden_layers": "num_layers",
        "text_config.rms_norm_eps": "layernorm_epsilon",
        "text_config.vocab_size": "padded_vocab_size",
        "text_config.attention_dropout": "attention_dropout",
        "text_config.rope_parameters": "rope_parameters",
        "tie_word_embeddings": "tie_embeddings_and_output_weights",
        "text_config.partial_rotary_factor": "rotary_percent",
        # MoE
        "text_config.moe_intermediate_size": "moe_ffn_hidden_size",
        "text_config.num_experts": "num_moe_experts",
        "text_config.num_experts_per_tok": "moe_router_topk",
        # GDN linear attention
        "text_config.linear_conv_kernel_dim": "linear_conv_kernel_dim",
        "text_config.linear_key_head_dim": "linear_key_head_dim",
        "text_config.linear_value_head_dim": "linear_value_head_dim",
        "text_config.linear_num_key_heads": "linear_num_key_heads",
        "text_config.linear_num_value_heads": "linear_num_value_heads",
        "text_config.full_attention_interval": "linear_attention_freq",
        "text_config.layer_types": "layer_types",
        # carried as metadata for M6/M7/M8 (not consumed by the spec builder yet)
        "text_config.hc_count": "hc_count",
        "text_config.hc_lowrank": "hc_lowrank",
        "text_config.indexer_n_heads": "indexer_n_heads",
        "text_config.indexer_kv_heads": "indexer_kv_heads",
        "text_config.indexer_head_dim": "indexer_head_dim",
        "text_config.indexer_budget": "indexer_budget",
        "text_config.indexer_compress_ratio": "indexer_compress_ratio",
        "text_config.ple_layer_ids": "ple_layer_ids",
        "text_config.ple_embed_dim": "ple_embed_dim",
        "text_config.ple_conv_kernel_size": "ple_conv_kernel_size",
        "text_config.ngram_size": "ngram_size",
        "text_config.heads_per_ngram": "heads_per_ngram",
        "text_config.ngram_vocab_size_base": "ngram_vocab_size_base",
        "text_config.make_ngram_vocab_size_divisible_by": "make_ngram_vocab_size_divisible_by",
    },
    constant_mca_config={
        "swiglu": True,
        "position_embedding_type": "rope",
        "normalization": "RMSNorm",
        "add_bias_linear": False,
        "hidden_dropout": 0.0,
        "moe_router_load_balancing_type": "aux_loss",
        "moe_router_pre_softmax": False,
        "qk_layernorm": True,
        "moe_shared_expert_gate": True,
        "layernorm_zero_centered_gamma": True,
        "hetereogenous_dist_checkpoint": True,
        "attention_output_gate": True,
        "linear_attention_type": "gated_delta_net",
        "experimental_attention_variant": "gated_delta_net",
        "transformer_impl": "transformer_engine",
    },
    weight_converters=[
        # --- embeddings / head ---------------------------------------------
        RenameConverOp(hf_names="lm_head.weight", mca_names="output_layer.weight"),
        RenameConverOp(
            hf_names="model.language_model.embed_tokens.weight",
            mca_names="embedding.word_embeddings.weight",
        ),
        RenameConverOp(
            hf_names="model.language_model.norm.weight",
            mca_names="decoder.final_layernorm.weight",
        ),
        # --- per-layer norms ----------------------------------------------
        RenameConverOp(
            hf_names=".input_layernorm.weight",
            mca_names=".self_attention.linear_qkv.layer_norm_weight",
        ),
        RenameConverOp(
            hf_names=".post_attention_layernorm.weight",
            mca_names=".pre_mlp_layernorm.weight",
        ),
        # --- MoE ------------------------------------------------------------
        # NOTE: the stacked expert tensors (.mlp.experts.gate_up_proj /
        # .mlp.experts.down_proj) are NOT handled by a ConverOp. mcore names each
        # expert separately (mlp.experts.linear_fc1.weight{N}), which a single
        # ConverOp cannot express, so Qwen4ExpTemplate intercepts them directly.
        RenameConverOp(hf_names=".mlp.gate.weight", mca_names=".mlp.router.weight"),
        # --- shared expert -------------------------------------------------
        RenameConverOp(
            hf_names=".mlp.shared_expert.down_proj.weight",
            mca_names=".mlp.shared_experts.linear_fc2.weight",
        ),
        StackConverOp(
            hf_names=[
                ".mlp.shared_expert.gate_proj.weight",
                ".mlp.shared_expert.up_proj.weight",
            ],
            mca_names=".mlp.shared_experts.linear_fc1.weight",
            dim=0,
        ),
        RenameConverOp(
            hf_names=".mlp.shared_expert_gate.weight",
            mca_names=".mlp.shared_experts.gate_weight",
        ),
        # --- full-attention layers ----------------------------------------
        GatedQKVConverOp(
            hf_names=[
                ".self_attn.q_proj.weight",
                ".self_attn.k_proj.weight",
                ".self_attn.v_proj.weight",
            ],
            mca_names=".self_attention.linear_qkv.weight",
        ),
        RenameConverOp(
            hf_names=".self_attn.o_proj.weight", mca_names=".self_attention.linear_proj.weight"
        ),
        RenameConverOp(
            hf_names=".self_attn.q_norm.weight", mca_names=".self_attention.q_layernorm.weight"
        ),
        RenameConverOp(
            hf_names=".self_attn.k_norm.weight", mca_names=".self_attention.k_layernorm.weight"
        ),
        # --- GDN layers ----------------------------------------------------
        Qwen4ExpGDNInProjConverOp(
            hf_names=[
                ".linear_attn.in_proj_qkv.weight",
                ".linear_attn.in_proj_z.weight",
                ".linear_attn.in_proj_b.weight",
                ".linear_attn.in_proj_a.weight",
            ],
            mca_names=".self_attention.in_proj.weight",
        ),
        GDNConv1dConverOp(
            hf_names=".linear_attn.conv1d.weight", mca_names=".self_attention.conv1d.weight"
        ),
        RenameConverOp(hf_names=".linear_attn.dt_bias", mca_names=".self_attention.dt_bias"),
        RenameConverOp(hf_names=".linear_attn.A_log", mca_names=".self_attention.A_log"),
        ZeroCenteredRMSNormConverOp(
            hf_names=".linear_attn.norm.weight", mca_names=".self_attention.out_norm.weight"
        ),
        RenameConverOp(
            hf_names=".linear_attn.out_proj.weight", mca_names=".self_attention.out_proj.weight"
        ),
        # --- deliberately dropped (each has an owning milestone) -----------
        # NOTE on the patterns: get_conver_op matches with re.match, i.e. anchored
        # at the start of the name, and turns "{}" into a wildcard group. So these
        # need a leading ".*" to match mid-name segments -- a bare ".ple." would
        # never fire and the tensors would fall through to a ValueError.
        #
        # M6: PLE. The trainable projections/norms are mapped; the 95 GiB n-gram
        # table is NOT -- it stays a frozen buffer loaded straight from the
        # checkpoint shards, deliberately outside the Megatron parameter tree
        # (P21: as a Parameter it would add ~191 GiB of Adam state).
        RenameConverOp(hf_names=".ple.key_proj.weight", mca_names=".ple.key_proj.weight"),
        RenameConverOp(
            hf_names=".ple.value_proj.weight", mca_names=".ple.value_proj.weight"
        ),
        RenameConverOp(hf_names=".ple.conv1d.weight", mca_names=".ple.conv1d.weight"),
        RenameConverOp(hf_names=".ple.norm_key.weight", mca_names=".ple.norm_key"),
        RenameConverOp(hf_names=".ple.norm_query.weight", mca_names=".ple.norm_query"),
        RenameConverOp(hf_names=".ple.norm_conv.weight", mca_names=".ple.norm_conv"),
        # The table itself and its hash buffers: loaded outside the converter.
        DropConverOp(hf_names=".*\\.ple\\.ple_embedding\\..*", mca_names=[]),
        # M7: hyperconnection. Names were chosen to match the checkpoint, so these
        # are plain renames (verified in P17: load_state_dict reports no missing or
        # unexpected keys). Two groups per layer: attn_ and mlp_.
        RenameConverOp(
            hf_names=".attn_hyper_connection.hc_norm.weight",
            mca_names=".attn_hyper_connection.hc_norm",
        ),
        RenameConverOp(
            hf_names=".attn_hyper_connection.input_mix_weight_down.weight",
            mca_names=".attn_hyper_connection.input_mix_weight_down.weight",
        ),
        RenameConverOp(
            hf_names=".attn_hyper_connection.input_mix_weight_up.weight",
            mca_names=".attn_hyper_connection.input_mix_weight_up.weight",
        ),
        RenameConverOp(
            hf_names=".attn_hyper_connection.block_inject_weight.weight",
            mca_names=".attn_hyper_connection.block_inject_weight.weight",
        ),
        RenameConverOp(
            hf_names=".mlp_hyper_connection.hc_norm.weight",
            mca_names=".mlp_hyper_connection.hc_norm",
        ),
        RenameConverOp(
            hf_names=".mlp_hyper_connection.input_mix_weight_down.weight",
            mca_names=".mlp_hyper_connection.input_mix_weight_down.weight",
        ),
        RenameConverOp(
            hf_names=".mlp_hyper_connection.input_mix_weight_up.weight",
            mca_names=".mlp_hyper_connection.input_mix_weight_up.weight",
        ),
        RenameConverOp(
            hf_names=".mlp_hyper_connection.block_inject_weight.weight",
            mca_names=".mlp_hyper_connection.block_inject_weight.weight",
        ),
        # The final mix-down sits outside the layer stack (top-level in the HF
        # checkpoint), so it needs its own full-path renames.
        RenameConverOp(
            hf_names="model.language_model.hyper_connection_mixer.hc_norm.weight",
            mca_names="decoder.hyper_connection_mixer.hc.hc_norm",
        ),
        RenameConverOp(
            hf_names="model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
            mca_names="decoder.hyper_connection_mixer.hc.input_mix_weight_down.weight",
        ),
        RenameConverOp(
            hf_names="model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
            mca_names="decoder.hyper_connection_mixer.hc.input_mix_weight_up.weight",
        ),
        # M8: QSA sparse indexer -- mcore's dsa has an incompatible weight layout.
        DropConverOp(hf_names=".*\\.self_attn\\.indexer\\..*", mca_names=[]),
        # MTP: not supported by mca yet (same as qwen3_next).
        DropConverOp(hf_names=".*mtp\\..*", mca_names=[]),
        # Vision tower: this is language_model_only training.
        DropConverOp(hf_names="model\\.visual\\..*", mca_names=[]),
    ],
)
