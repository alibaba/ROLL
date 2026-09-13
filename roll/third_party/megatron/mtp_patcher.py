"""
Monkey patches for Megatron Multi-Token Prediction (MTP) training.

This module provides patches for MTP-related functions in Megatron to add
extra operations during MTP training.

MTP training mode is read from `self.config.mtp_training_mode`:
- 'disabled': MTP is loaded but not trained (default)
- 'standalone': MTP is trained independently with truncated gradients
- 'joint': MTP participates in main model updates with full gradient flow
"""

from typing import TYPE_CHECKING, Callable, Optional

import torch

if TYPE_CHECKING:
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

def patch_mtp_functions():
    """
    Apply monkey patches for MTP training.

    This function patches the following methods:
    1. GPTModel.forward - for forward pass modifications
    2. GPTModel._postprocess - for postprocess modifications
    3. MultiTokenPredictionLayer._get_embeddings - for embedding modifications
    """
    from collections import OrderedDict

    from megatron.core import parallel_state
    from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
    from megatron.core.transformer.multi_token_prediction import (
        MTPLossAutoScaler,
        MTPLossLoggingHelper,
        MultiTokenPredictionLayer,
        roll_tensor,
    )

    # Save original methods for potential use in patched versions
    original_gpt_postprocess = GPTModel._postprocess
    original_mtp_get_embeddings = MultiTokenPredictionLayer._get_embeddings

    # ============================================================================
    # GPTModel._postprocess
    # ============================================================================
    def patched_gpt_postprocess(
        self: "GPTModel",
        hidden_states,
        input_ids,
        position_ids,
        labels,
        rotary_pos_emb,
        rotary_pos_cos,
        rotary_pos_sin,
        mtp_in_postprocess=None,
        loss_mask=None,
        decoder_input=None,
        attention_mask=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        runtime_gather_output=None,
        extra_block_kwargs=None,
        inference_context=None,
        **kwargs,
    ):
        """
        Patched _postprocess method for GPTModel with MTP support.
        Reads mtp_training_mode from self.config.
        """
        # signature of mtp related methods differs between megatron_core 0.16/0.17 and dev.
        # currently, dev is needed for GDN cp and packing which includes padding_mask,
        # and use kwargs to be compatible
        if "padding_mask" in kwargs:
            if extra_block_kwargs:
                extra_block_kwargs["padding_mask"] = kwargs["padding_mask"]
            else:
                extra_block_kwargs = {"padding_mask": kwargs["padding_mask"]}
        mtp_training_mode = getattr(self.config, "mtp_training_mode", "disabled")

        in_inference_mode = inference_context is not None and not self.training
        if in_inference_mode:
            assert runtime_gather_output, "Inference must always gather TP logits"

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        if mtp_in_postprocess and mtp_training_mode != "disabled":
            hidden_states = self.mtp(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                embedding=self.embedding,
                **(extra_block_kwargs or {}),
            )

        if not self.post_process:
            return hidden_states

        if self.config.mtp_num_layers is not None and mtp_training_mode != "disabled":
            mtp_labels, _ = roll_tensor(
                input_ids,
                shifts=-1,
                dims=-1,
                cp_group=self.cp_group,
                packed_seq_params=packed_seq_params,
            )

            hidden_states_list = torch.chunk(hidden_states, 1 + self.config.mtp_num_layers, dim=0)
            hidden_states = hidden_states_list[0]
            if loss_mask is None:
                # if loss_mask is not provided, use all ones as loss_mask
                loss_mask = torch.ones_like(mtp_labels)

            for mtp_layer_number in range(self.config.mtp_num_layers):
                # output
                mtp_logits, _ = self.output_layer(
                    hidden_states_list[mtp_layer_number + 1],
                    weight=output_weight.detach() if mtp_training_mode == "standalone" and output_weight is not None else output_weight,
                    runtime_gather_output=runtime_gather_output,
                )
                # Calc loss for the current Multi-Token Prediction (MTP) layers.
                mtp_labels, _ = roll_tensor(
                    mtp_labels,
                    shifts=-1,
                    dims=-1,
                    cp_group=self.cp_group,
                    packed_seq_params=packed_seq_params,
                )
                loss_mask, num_tokens = roll_tensor(
                    loss_mask,
                    shifts=-1,
                    dims=-1,
                    cp_group=self.cp_group,
                    packed_seq_params=packed_seq_params,
                )

                mtp_loss = self.compute_language_model_loss(mtp_labels, mtp_logits)
                mtp_loss = loss_mask * mtp_loss

                num_tokens = max(num_tokens, 1)

                if self.training:
                    MTPLossLoggingHelper.save_loss_to_tracker(
                        torch.sum(mtp_loss) / num_tokens,
                        mtp_layer_number,
                        self.config.mtp_num_layers,
                        avg_group=parallel_state.get_data_parallel_group(
                            with_context_parallel=True
                        ),
                    )
                mtp_loss_scale = self.config.mtp_loss_scaling_factor / self.config.mtp_num_layers
                if self.config.calculate_per_token_loss:
                    hidden_states = MTPLossAutoScaler.apply(
                        hidden_states, mtp_loss_scale * mtp_loss
                    )
                else:
                    hidden_states = MTPLossAutoScaler.apply(
                        hidden_states, mtp_loss_scale * mtp_loss / num_tokens
                    )
        sequence_parallel_override = False

        if in_inference_mode and inference_context.materialize_only_last_token_logits:
            if inference_context.is_static_batching():
                hidden_states = hidden_states[-1:, :, :]
            else:
                if self.output_layer.sequence_parallel:
                    hidden_states = gather_from_sequence_parallel_region(
                        hidden_states, group=self.pg_collection.tp
                    )
                    self.output_layer.sequence_parallel = False
                    sequence_parallel_override = True

                hidden_states = inference_context.last_token_logits(
                    hidden_states.squeeze(1).unsqueeze(0)
                ).unsqueeze(1)

        logits, _ = self.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )

        # Restore sequence parallel execution to the output layer if necessary.
        if sequence_parallel_override:
            assert (
                in_inference_mode
                and inference_context.is_dynamic_batching()
                and inference_context.materialize_only_last_token_logits
            )
            self.output_layer.sequence_parallel = True

        if has_config_logger_enabled(self.config):
            payload = OrderedDict(
                {
                    'input_ids': input_ids,
                    'position_ids': position_ids,
                    'attention_mask': attention_mask,
                    'decoder_input': decoder_input,
                    'logits': logits,
                }
            )
            log_config_to_disk(self.config, payload, prefix='input_and_logits')

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels, logits)
        return loss

    # ============================================================================
    # MultiTokenPredictionLayer._get_embeddings
    # ============================================================================
    def patched_mtp_get_embeddings(
        self: "MultiTokenPredictionLayer",
        input_ids: "Tensor",
        position_ids: "Tensor",
        embedding: Callable,
        hidden_states: "Tensor",
        packed_seq_params=None,
        **kwargs,
    ):
        """
        Patched _get_embeddings method for MultiTokenPredictionLayer.
        Reads mtp_training_mode from self.config.
        """
        from megatron.core.utils import make_viewless_tensor

        # signature of mtp related methods differs between megatron_core 0.16/0.17 and dev.
        # currently, dev is needed for GDN cp and packing which includes padding_mask,
        # and use kwargs to be compatible
        padding_mask = kwargs.get("padding_mask", None)
        mtp_training_mode = getattr(self.config, "mtp_training_mode", "disabled")

        # Calc logits for the current Multi-Token Prediction (MTP) layers.
        input_ids, _ = roll_tensor(
            input_ids,
            shifts=-1,
            dims=-1,
            cp_group=self.cp_group,
            packed_seq_params=packed_seq_params,
        )
        position_ids, _ = roll_tensor(
            position_ids,
            shifts=-1,
            dims=-1,
            cp_group=self.cp_group,
            packed_seq_params=packed_seq_params,
        )
        if padding_mask is not None:
            padding_mask, _ = roll_tensor(
                padding_mask,
                shifts=-1,
                dims=-1,
                cp_group=self.cp_group,
                packed_seq_params=packed_seq_params,
            )
        # embedding
        decoder_input = embedding(input_ids=input_ids, position_ids=position_ids)

        #
        if mtp_training_mode == "standalone":
            decoder_input = decoder_input.detach()
            hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=False)
        else:
            hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if "padding_mask" in kwargs:
            result = input_ids, position_ids, padding_mask, decoder_input, hidden_states
        else:
            result = input_ids, position_ids, decoder_input, hidden_states
        return result

    # ============================================================================
    # Apply patches
    # ============================================================================
    GPTModel._postprocess = patched_gpt_postprocess
    MultiTokenPredictionLayer._get_embeddings = patched_mtp_get_embeddings

    # Store original methods for potential restoration
    GPTModel._original_postprocess = original_gpt_postprocess
    MultiTokenPredictionLayer._original_get_embeddings = original_mtp_get_embeddings
