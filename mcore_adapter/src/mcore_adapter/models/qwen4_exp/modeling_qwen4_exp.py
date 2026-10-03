from typing import Optional
import torch

from ..auto.modeling_auto import register_model
from ..model_factory import McaGPTModel
from .config_qwen4_exp import Qwen4ExpConfig
from .capabilities import validate_bounded_rl_token_statistics, validate_training_capabilities
from .hyperconnection_layer import make_hc_layer_spec
from .transformer_block import Qwen4ExpTransformerBlock
from .qsa_attention import Qwen4ExpQSAAttention
from .gated_delta_net import Qwen4ExpGatedDeltaNet
from .chunked_loss import (
    chunked_vocab_parallel_cross_entropy,
    chunked_vocab_parallel_logprobs_and_entropy,
)


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
    transformer_block_class = Qwen4ExpTransformerBlock
    bounded_rl_token_statistics_capability = "qwen4_exp_v1"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        if not isinstance(self.decoder, Qwen4ExpTransformerBlock):
            raise RuntimeError("Megatron GPTModel needs the Qwen4 decoder factory patch; run scripts/qwen38/patch_megatron_block_factory.py on the isolated dependency checkout")

    def forward(self, input_ids, position_ids, attention_mask, *args,
                extra_block_kwargs=None, padding_mask=None, loss_mask=None, **kwargs):
        if input_ids is None:
            raise ValueError("Qwen4 training requires original input_ids for lexical embeddings")
        if kwargs.get("packed_seq_params") is not None:
            raise NotImplementedError("Qwen4 cross-sample packing is not supported")
        valid = self._resolve_valid_mask(input_ids, attention_mask, padding_mask)
        ids = torch.where(valid, input_ids, self.config.eos_token_id)
        effective_padding_mask = ~valid
        block_kwargs = dict(extra_block_kwargs or {})
        block_kwargs.update(ple_input_ids=ids, qsa_valid_mask=valid, qsa_loss_mask=loss_mask)
        return super().forward(input_ids, position_ids, attention_mask, *args,
                               extra_block_kwargs=block_kwargs, padding_mask=effective_padding_mask,
                               loss_mask=loss_mask, **kwargs)

    def compute_output_layer_and_language_model_loss(
        self, hidden, labels, weight=None, sequence_parallel_enabled=False,
        column_parallel_linear=None, col_linear_kwargs=None, reduction="none", ignore_index=-100,
    ):
        """Compute token losses without retaining full-sequence vocabulary logits."""
        kwargs = dict(col_linear_kwargs or {})
        if self.config.vocab_loss_chunk_size == 0:
            return super().compute_output_layer_and_language_model_loss(
                hidden, labels, weight=weight, sequence_parallel_enabled=sequence_parallel_enabled,
                column_parallel_linear=column_parallel_linear, col_linear_kwargs=kwargs,
                reduction=reduction, ignore_index=ignore_index,
            )
        if labels is None:
            raise ValueError("Qwen4 chunked vocabulary loss requires labels")
        if reduction not in ("none", "sum", "mean"):
            raise ValueError("Qwen4 vocabulary loss reduction must be none, sum, or mean")

        from megatron.core.tensor_parallel.layers import ColumnParallelLinear

        head = column_parallel_linear
        # Calling a wrapper's base weight directly would silently drop adapters
        # or other projection behavior. Accept only the plain Megatron head.
        unsupported = None
        if type(head) is not ColumnParallelLinear:
            unsupported = "an adapted or nonstandard output head"
        elif head.bias is not None:
            unsupported = "an output-head bias"
        elif head.config.defer_embedding_wgrad_compute:
            unsupported = "deferred output-head weight-gradient computation"
        elif head.explicit_expert_comm or head.disable_grad_reduce:
            unsupported = "custom output-head gradient communication"
        elif set(kwargs) - {"weight", "runtime_gather_output"}:
            unsupported = "additional output-head keyword arguments"
        if unsupported is not None:
            raise NotImplementedError(
                f"Qwen4 chunked vocabulary loss does not support {unsupported}; "
                "set vocab_loss_chunk_size=0 to use the original output-layer path"
            )
        if bool(sequence_parallel_enabled) != bool(head.sequence_parallel):
            raise ValueError("Qwen4 vocabulary loss sequence parallelism must match its output head")

        # GPTModel also passes shared_embedding_or_output_weight() as `weight`
        # for an untied model. ColumnParallelLinear's own explicit weight kwarg
        # and parameter determine the projection actually used in that case.
        head_weight = kwargs.get("weight")
        if head_weight is None:
            head_weight = head.weight
        if head_weight is None:
            raise ValueError("Qwen4 vocabulary output head requires its explicit shared weight")
        losses = chunked_vocab_parallel_cross_entropy(
            hidden, head_weight, labels.transpose(0, 1).contiguous(),
            chunk_size=self.config.vocab_loss_chunk_size, tp_group=head.tp_group,
            sequence_parallel=sequence_parallel_enabled, ignore_index=ignore_index,
        ).transpose(0, 1).contiguous()
        if reduction == "sum":
            return losses.sum()
        if reduction == "mean":
            return losses.sum() / (labels != ignore_index).sum()
        return losses

    def _postprocess(
        self,
        hidden_states,
        input_ids,
        position_ids,
        labels,
        rotary_pos_emb,
        rotary_pos_cos,
        rotary_pos_sin,
        **kwargs,
    ):
        if labels is not None and self.config.vocab_loss_chunk_size:
            # ROLL replaces GPTModel._postprocess for MTP and older Megatron
            # versions materialize full logits there. Qwen4 does not support
            # MTP; keep its bounded loss independent of that global patch in
            # both training and held-out validation.
            if not self.post_process:
                return hidden_states
            if kwargs.get("inference_context") is not None or kwargs.get("inference_params") is not None:
                raise NotImplementedError("Qwen4 chunked labeled loss does not support an inference context")
            output_weight = self.shared_embedding_or_output_weight() if self.share_embeddings_and_output_weights else None
            return self.compute_output_layer_and_language_model_loss(
                hidden_states,
                labels=labels,
                weight=output_weight,
                sequence_parallel_enabled=self.output_layer.sequence_parallel,
                column_parallel_linear=self.output_layer,
                col_linear_kwargs={"weight": output_weight, "runtime_gather_output": kwargs.get("runtime_gather_output")},
            )
        if labels is not None or not self.config.bounded_rl_token_statistics:
            return super()._postprocess(
                hidden_states=hidden_states,
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                **kwargs,
            )

        loss_mask = kwargs.get("loss_mask")
        packed_seq_params = kwargs.get("packed_seq_params")
        validate_bounded_rl_token_statistics(
            enabled=True,
            chunk_size=self.config.vocab_loss_chunk_size,
            context_parallel_size=self.config.context_parallel_size,
            sequence_packing=packed_seq_params is not None,
            mtp_num_layers=self.config.mtp_num_layers,
            output_head_adapter=not self._is_plain_output_head(),
        )
        if not self.post_process:
            return hidden_states
        if input_ids is None:
            raise ValueError("bounded RL token statistics require input_ids")
        if loss_mask is None:
            loss_mask = self._resolve_valid_mask(input_ids, kwargs.get("attention_mask"))
        elif loss_mask.ndim != 2 or loss_mask.shape != input_ids.shape:
            raise ValueError("bounded RL token statistics require a [batch, sequence] loss mask")
        elif loss_mask.dtype != torch.bool and not bool(((loss_mask == 0) | (loss_mask == 1)).all()):
            raise ValueError("bounded RL token-statistics loss mask must contain only 0 and 1")

        block_kwargs = kwargs.get("extra_block_kwargs") or {}
        token_labels = block_kwargs.get("rl_token_labels")
        if token_labels is None:
            # Direct model callers use loss_mask; ROLL supplies labels built
            # from response_mask because its auxiliary mask may be narrower.
            token_labels = input_ids[:, 1:].clone()
            token_labels[~loss_mask[:, 1:].bool()] = 0
            token_labels = torch.cat((token_labels, torch.zeros_like(token_labels[:, :1])), dim=1)
        elif (
            token_labels.dtype != torch.long
            or token_labels.device != input_ids.device
            or token_labels.shape != input_ids.shape
        ):
            raise ValueError("bounded RL token labels must be int64 [batch, sequence] on the input device")
        output_weight = self.shared_embedding_or_output_weight() if self.share_embeddings_and_output_weights else None
        head_weight = output_weight if output_weight is not None else self.output_layer.weight
        statistics = chunked_vocab_parallel_logprobs_and_entropy(
            hidden_states,
            head_weight,
            token_labels.transpose(0, 1).contiguous(),
            chunk_size=self.config.vocab_loss_chunk_size,
            tp_group=self.output_layer.tp_group,
            sequence_parallel=self.output_layer.sequence_parallel,
        )
        return statistics.transpose(0, 1).contiguous()

    def _is_plain_output_head(self):
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear

        head = self.output_layer
        return (
            type(head) is ColumnParallelLinear
            and head.bias is None
            and not head.config.defer_embedding_wgrad_compute
            and not head.explicit_expert_comm
            and not head.disable_grad_reduce
        )

    @staticmethod
    def _resolve_valid_mask(input_ids, attention_mask=None, padding_mask=None):
        """Normalize ROLL's token mask for QSA, PLE, and masked GDN inputs."""
        valid = None
        for name, mask in (("attention_mask", attention_mask), ("padding_mask", padding_mask)):
            if mask is None:
                continue
            if mask.ndim != 2 or mask.shape != input_ids.shape:
                raise ValueError(f"Qwen4 requires a [batch, sequence] token {name} matching input_ids")
            mask = mask.to(device=input_ids.device)
            if mask.dtype != torch.bool and not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError(f"Qwen4 {name} must contain only 0 and 1")
            candidate = mask.bool() if name == "attention_mask" else ~mask.bool()
            if valid is not None and not torch.equal(valid, candidate):
                raise ValueError("Qwen4 attention_mask and padding_mask disagree on valid tokens")
            valid = candidate
        return valid if valid is not None else torch.ones_like(input_ids, dtype=torch.bool)

    def attach_ngram_assets(self, checkpoint, manifests=None):
        loaded = {}
        for layer in self.decoder.layers:
            if hasattr(layer, "ple"):
                index = layer.layer_number - 1
                loaded[str(index)] = layer.ple.ple_embedding.attach_checkpoint(
                    checkpoint, index, expected_manifest=(manifests or {}).get(str(index)))
        return loaded

    def load_external_assets(self, model_name_or_path, external_asset_path=None):
        from .asset_lifecycle import restore_ngram_assets

        return restore_ngram_assets(
            self,
            model_name_or_path,
            external_asset_path=external_asset_path,
        )

    def save_external_assets(self, save_directory):
        from .asset_lifecycle import persist_ngram_assets

        return persist_ngram_assets(self, save_directory)

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

        local_layer_types = self._assert_layer_pattern(config, block_spec, vp_stage=self.vp_stage)
        if getattr(config, "hc_count", None) and config.hc_count > 1:
            block_spec = make_hc_layer_spec(block_spec)
        for spec, kind in zip(block_spec.layer_specs, local_layer_types):
            if kind == "full_attention":
                spec.submodules.self_attention.module = Qwen4ExpQSAAttention
            else:
                spec.submodules.self_attention.module = Qwen4ExpGatedDeltaNet
        return block_spec

    @staticmethod
    def _assert_layer_pattern(config: Qwen4ExpConfig, block_spec, vp_stage=None) -> list[str]:
        """Check the built layer pattern against the checkpoint's layer_types.

        Cheap, and it catches the one failure mode that otherwise stays silent:
        a wrong linear_attention_freq (or a spec-builder change upstream) yielding
        GDN and full-attention layers in the wrong positions.
        """
        from megatron.core.transformer.transformer_block import get_num_layers_to_build
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        count = get_num_layers_to_build(config, vp_stage=vp_stage)
        global_types = config.layer_types or config._derive_layer_types()
        expected = global_types[offset:offset + count]
        if len(expected) != count or len(block_spec.layer_specs) != count:
            raise ValueError(
                f"spec built {len(block_spec.layer_specs)} layers but config expects "
                f"{count} local layers at global offset {offset} "
                f"({len(expected)} configured layer types available)"
            )

        mismatches = []
        for i, (spec, want) in enumerate(zip(block_spec.layer_specs, expected)):
            module = spec.submodules.self_attention.module
            name = getattr(module, "__name__", str(module))
            is_linear = "GatedDeltaNet" in name
            got = "linear_attention" if is_linear else "full_attention"
            if got != want:
                mismatches.append((offset + i, want, name))

        if mismatches:
            head = ", ".join(f"layer {i}: want {w}, built {n}" for i, w, n in mismatches[:6])
            raise ValueError(
                f"hybrid layer pattern mismatch ({len(mismatches)} layer(s)): {head}"
                + ("..." if len(mismatches) > 6 else "")
                + f"\nlinear_attention_freq={config.linear_attention_freq}. "
                "Building the wrong pattern does not raise on its own -- it yields a "
                "model that trains but is not this architecture."
            )
        return expected
