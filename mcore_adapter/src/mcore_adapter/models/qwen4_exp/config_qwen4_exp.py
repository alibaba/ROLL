from dataclasses import dataclass
from typing import Optional

from ...utils import get_logger
from ..auto.config_auto import register_config
from ..model_config import McaModelConfig

logger = get_logger(__name__)


@register_config("qwen4_exp")
@dataclass
class Qwen4ExpConfig(McaModelConfig):
    """Config for Qwen3.8-Flash-Next (HF model_type ``qwen4_exp``).

    Modelled on Qwen3NextConfig -- both are GDN linear attention interleaved with
    full attention, on top of a large MoE. The differences that matter here:

    * The HF config nests everything under ``text_config`` (there is also a
      ``vision_config``); the template's ``config_hf_to_mca`` handles that, this
      class only needs the flattened fields.
    * Full-attention layers use the adapter's QSA attention and indexer modules,
      and the transformer block wraps attention and MLP branches in the GR
      hyperconnection path. Their fields control construction and conversion;
      the capability checks define which training combinations are enabled.
    * Layer 1 (0-based) carries the PLE n-gram table. That table is frozen and
      handled outside the Megatron parameter tree, so no field is needed for the
      table itself, only for locating the layer.
    """

    # --- hybrid attention pattern -------------------------------------------
    # mcore derives GDN-vs-full placement from linear_attention_freq, but the HF
    # config also ships an explicit layer_types list. Keep both and cross-check
    # in __post_init__ so a mismatch surfaces at construction instead of after a
    # multi-hour conversion.
    layer_types: Optional[list[str]] = None

    # TE combines top-k expert outputs with a deterministic FP32 reduction.
    # Megatron's unfused BF16 scatter_add rounds after each atomic update;
    # top-10 routing amplifies rank-dependent accumulation error across 48 layers.
    moe_permute_fusion: bool = True
    gdn_output_gate_type: Optional[str] = None

    # Bound vocabulary logits/softmax memory in SFT and token-logprob training.
    # Zero selects Megatron's original output-layer/loss path.
    vocab_loss_chunk_size: int = 256
    # ROLL RL forwards normally omit labels. This explicit opt-in replaces the
    # full-vocabulary output with [selected logprob, entropy] token statistics.
    bounded_rl_token_statistics: bool = False
    # Full recomputation at separate GR attention/MLP, PLE and final-mixer
    # boundaries. Only checkpoint inputs are saved on pinned CPU memory;
    # recomputed branch activations and parameter storage stay on GPU.
    checkpoint_cpu_offload: bool = False

    # --- QSA (Qwen Sparse Attention) on full-attention layers ----------------
    # mcore has a `dsa` variant, but its weight layout does not match QSA (dsa
    # keeps q/k projections separate and has a learned per-head weights_proj;
    # QSA fuses q/k and has both q and k layernorms). The adapter supplies the
    # matching attention/indexer implementation and auxiliary-loss contract.
    indexer_n_heads: Optional[int] = None
    indexer_kv_heads: Optional[int] = None
    indexer_head_dim: Optional[int] = None
    indexer_budget: Optional[int] = None
    indexer_compress_ratio: Optional[int] = None
    qsa_indexer_kl_coef: float = 0.0
    qsa_indexer_temperature: float = 1.0

    # --- hyperconnection (replaces the ordinary residual stream) -------------
    # 398 checkpoint tensors; residual width is hidden_size * hc_count.
    # Consumed by the adapter's GR hyperconnection modules.
    hc_count: Optional[int] = None
    hc_lowrank: Optional[int] = None

    # --- PLE n-gram table (frozen, lives outside the parameter tree) ---------
    ple_layer_ids: Optional[list[int]] = None  # 1-based in the HF config
    ple_embed_dim: Optional[int] = None
    ple_conv_kernel_size: Optional[int] = None
    ngram_size: Optional[int] = None
    heads_per_ngram: Optional[int] = None
    ngram_vocab_size_base: Optional[int] = None
    make_ngram_vocab_size_divisible_by: Optional[int] = None
    eos_token_id: int = 0

    def __post_init__(self):
        router_dtype = self.moe_router_dtype
        super().__post_init__()
        # The checkpoint projects router logits in the model dtype, then runs
        # softmax in FP32. The generic large-MoE default changes that projection
        # to FP32 and can change top-k expert selection. Keep explicit overrides.
        self.moe_router_dtype = router_dtype
        if self.checkpoint_cpu_offload and self.recompute_granularity != "full":
            raise ValueError("checkpoint_cpu_offload requires full recomputation")
        if (isinstance(self.vocab_loss_chunk_size, bool)
                or not isinstance(self.vocab_loss_chunk_size, int)
                or self.vocab_loss_chunk_size < 0):
            raise ValueError("vocab_loss_chunk_size must be a nonnegative integer (0 disables chunking)")
        from .capabilities import validate_bounded_rl_token_statistics

        validate_bounded_rl_token_statistics(
            enabled=self.bounded_rl_token_statistics,
            chunk_size=self.vocab_loss_chunk_size,
            context_parallel_size=self.context_parallel_size,
            mtp_num_layers=self.mtp_num_layers,
        )
        if self.gdn_output_gate_type not in (None, "silu", "sigmoid"):
            raise ValueError("Qwen4 GDN output gate must be silu or sigmoid")
        if self.pipeline_model_parallel_size > 1 or self.virtual_pipeline_model_parallel_size:
            raise ValueError("Qwen4Exp training currently requires PP=1 and VPP disabled")
        if self.context_parallel_size > 1:
            raise ValueError("Qwen4Exp training currently requires CP=1")
        if self.fp8 or self.fp4 or self.mtp_num_layers:
            raise ValueError("Qwen4Exp text training currently requires BF16/FP32 without MTP")
        if self.cpu_offloading or self.fine_grained_activation_offloading:
            raise ValueError("Qwen4Exp activation offloading is not yet validated")
        if self.qsa_indexer_kl_coef < 0 or self.qsa_indexer_temperature <= 0:
            raise ValueError("QSA indexer KL coefficient must be nonnegative and temperature positive")

        # GDN asserts activation in {silu, swish} with a bare assert and no
        # message (gated_delta_net.py). TransformerConfig defaults to gelu, so
        # without this the failure surfaces deep in the forward pass.
        if self.experimental_attention_variant == "gated_delta_net":
            import torch

            if self.activation_func is not torch.nn.functional.silu:
                self.activation_func = torch.nn.functional.silu

        # mcore refuses MoE with TP>1 unless sequence parallelism is on
        # (moe_layer.py raises ValueError, worded as "performance may degrade"
        # but it is a hard failure). Every layer here is MoE.
        if self.num_moe_experts and self.tensor_model_parallel_size > 1:
            if not self.sequence_parallel:
                self.sequence_parallel = True

        derived = self._derive_layer_types()
        if self.layer_types is None:
            self.layer_types = derived
        elif list(self.layer_types) != derived:
            # A mismatch means linear_attention_freq and the explicit list
            # disagree; silently trusting either one risks building the wrong
            # architecture, so surface it now.
            raise ValueError(
                "layer_types from the checkpoint disagrees with the pattern derived "
                f"from linear_attention_freq={self.linear_attention_freq}.\n"
                f"  checkpoint: {list(self.layer_types)[:8]}...\n"
                f"  derived:    {derived[:8]}...\n"
                "Fix linear_attention_freq (or drop layer_types) before converting."
            )

    def _derive_layer_types(self) -> list[str]:
        """Same rule as Qwen3NextConfig: every freq-th layer is full attention."""
        freq = self.linear_attention_freq
        if not freq:
            return ["full_attention"] * self.num_layers
        return [
            "linear_attention" if bool((i + 1) % freq) else "full_attention"
            for i in range(self.num_layers)
        ]

    @property
    def ple_layer_indices(self) -> list[int]:
        """PLE layer ids as 0-based indices (the HF config stores them 1-based)."""
        if not self.ple_layer_ids:
            return []
        return [i - 1 for i in self.ple_layer_ids]


# ---------------------------------------------------------------------------
# Tolerate kwargs that are not declared fields.
#
# mca builds the config as
#     template.convert_hf_to_mca_config(hf_config, bf16=..., **dist_args.get_config_dict())
# so the kwargs are a superset of this dataclass's fields, and which extras appear
# depends on the installed mca version -- `moe_enable_routing_replay` lives in
# training_args.py but in no architecture config, including the bundled qwen3_next.
# A dataclass rejects unknown keys, and it does so midway through a 336 GiB
# conversion:
#     TypeError: Qwen4ExpConfig.__init__() got an unexpected keyword argument ...
#
# The generated __init__ is wrapped rather than overridden: the fields come from
# three base classes (McaModelConfig / TransformerConfig / PretrainedConfig), so
# hand-forwarding to super().__init__ drops the subclass's own fields, and __new__
# cannot filter what the generated __init__ receives.
_generated_init = Qwen4ExpConfig.__init__


def _tolerant_init(self, **kwargs):
    import dataclasses

    rl_temperature = kwargs.pop("rl_token_statistics_temperature", 1.0)
    if (
        isinstance(rl_temperature, bool)
        or not isinstance(rl_temperature, (int, float))
        or float(rl_temperature) != 1.0
    ):
        raise ValueError("bounded RL token statistics use raw logits and require temperature=1")
    known = {f.name for f in dataclasses.fields(self)}
    extra = {k: kwargs.pop(k) for k in list(kwargs) if k not in known}
    _generated_init(self, **kwargs)  # runs __post_init__
    # Set after construction: these are carried for downstream code, not consumed by
    # __post_init__'s derivations.
    for k, v in extra.items():
        setattr(self, k, v)
    if extra:
        logger.info(
            f"Qwen4ExpConfig: carried {len(extra)} kwarg(s) that are not declared "
            f"fields: {sorted(extra)}"
        )


Qwen4ExpConfig.__init__ = _tolerant_init
