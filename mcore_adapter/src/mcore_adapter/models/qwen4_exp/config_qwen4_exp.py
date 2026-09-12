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
    * Full-attention layers use QSA (a sparse indexer), and every layer is
      wrapped in a hyperconnection. Neither has a megatron-core counterpart yet,
      so the corresponding fields are carried here as plain metadata and are NOT
      yet consumed by the spec builder. They are recorded so that conversion does
      not silently drop them and so M7/M8 have somewhere to read them from.
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

    # --- QSA (Qwen Sparse Attention) on full-attention layers ----------------
    # mcore has a `dsa` variant, but its weight layout does not match QSA (dsa
    # keeps q/k projections separate and has a learned per-head weights_proj;
    # QSA fuses q/k and has both q and k layernorms). Carried as metadata for M8.
    indexer_n_heads: Optional[int] = None
    indexer_kv_heads: Optional[int] = None
    indexer_head_dim: Optional[int] = None
    indexer_budget: Optional[int] = None
    indexer_compress_ratio: Optional[int] = None
    qsa_indexer_kl_coef: float = 0.0
    qsa_indexer_temperature: float = 1.0

    # --- hyperconnection (replaces the ordinary residual stream) -------------
    # 398 checkpoint tensors; residual width is hidden_size * hc_count.
    # No mcore counterpart -> metadata only, consumed by M7 when implemented.
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
        super().__post_init__()
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
