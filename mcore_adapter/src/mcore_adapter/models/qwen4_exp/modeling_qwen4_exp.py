from typing import Optional

from ..auto.modeling_auto import register_model
from ..model_factory import McaGPTModel
from .config_qwen4_exp import Qwen4ExpConfig
from .capabilities import validate_training_capabilities
from .hyperconnection_layer import make_hc_layer_spec


@register_model("qwen4_exp")
class Qwen4ExpModel(McaGPTModel):
    """Qwen3.8-Flash-Next on megatron-core.

    This is M3: route model construction to mcore's hybrid-attention spec builder
    instead of the default (which would build every layer as ordinary attention).

    Why this matters more than its size suggests: if this routing is missing, the
    default path still builds a 48-layer model, weights still load (the shapes
    line up), training still runs and the loss still falls -- but every GDN layer
    has silently become a full-attention layer, so the model is not
    Qwen3.8-Flash-Next. There is no error to catch. The assertion in
    ``_get_transformer_layer_spec`` below exists to make that failure loud.
    """

    config_class = Qwen4ExpConfig

    @staticmethod
    def validate_training_capabilities(sequence_length: int, qsa_training_kernel: bool = False) -> dict[str, object]:
        """Validate the QSA mode before constructing a training worker."""
        return validate_training_capabilities(
            sequence_length=sequence_length,
            qsa_training_kernel=qsa_training_kernel,
        )

    def _get_transformer_layer_spec(self, config: Optional[Qwen4ExpConfig] = None):
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_transformer_block_with_experimental_attention_variant_spec,
        )

        config = config or self.config

        # The hybrid spec builder is TE-only upstream.
        assert config.transformer_impl == "transformer_engine", (
            "Qwen4ExpModel requires transformer_impl='transformer_engine'; "
            f"got {config.transformer_impl!r}. The hybrid attention spec builder "
            "has no local (non-TE) implementation."
        )

        if config.experimental_attention_variant is None:
            # Refuse rather than fall back: the fallback produces a plausible but
            # wrong model (see class docstring).
            raise ValueError(
                "experimental_attention_variant is None, which would build all "
                f"{config.num_layers} layers as full attention and silently produce "
                "the wrong architecture. Set it to 'gated_delta_net' (the mca "
                "template should do this via constant_mca_config)."
            )

        block_spec = get_transformer_block_with_experimental_attention_variant_spec(
            config=config, vp_stage=self.vp_stage
        )

        self._assert_layer_pattern(config, block_spec)
        if getattr(config, "hc_count", None) and config.hc_count > 1:
            block_spec = make_hc_layer_spec(block_spec)
        return block_spec

    @staticmethod
    def _assert_layer_pattern(config: Qwen4ExpConfig, block_spec) -> None:
        """Check the built layer pattern against the checkpoint's layer_types.

        Cheap, and it catches the one failure mode that otherwise stays silent:
        a wrong linear_attention_freq (or a spec-builder change upstream) yielding
        GDN and full-attention layers in the wrong positions.
        """
        expected = config.layer_types or config._derive_layer_types()
        if len(block_spec.layer_specs) != len(expected):
            raise ValueError(
                f"spec built {len(block_spec.layer_specs)} layers but config expects "
                f"{len(expected)}"
            )

        mismatches = []
        for i, (spec, want) in enumerate(zip(block_spec.layer_specs, expected)):
            module = spec.submodules.self_attention.module
            name = getattr(module, "__name__", str(module))
            is_linear = "GatedDeltaNet" in name
            got = "linear_attention" if is_linear else "full_attention"
            if got != want:
                mismatches.append((i, want, name))

        if mismatches:
            head = ", ".join(f"layer {i}: want {w}, built {n}" for i, w, n in mismatches[:6])
            raise ValueError(
                f"hybrid layer pattern mismatch ({len(mismatches)} layer(s)): {head}"
                + ("..." if len(mismatches) > 6 else "")
                + f"\nlinear_attention_freq={config.linear_attention_freq}. "
                "Building the wrong pattern does not raise on its own -- it yields a "
                "model that trains but is not this architecture."
            )
