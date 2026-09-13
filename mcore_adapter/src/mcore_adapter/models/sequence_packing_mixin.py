"""
Multimodal Embedding Mixin.

Provides ``prepare_packing_state`` and ``build_multimodal_embeddings`` for
multimodal models (Qwen3VL, Qwen3OmniMoe, Qwen3_5, Qwen3OmniNext).
Handles both packing and non-packing paths with a unified two-phase API.

The two-phase API separates lightweight input preparation (needed by
**all** PP stages) from heavy multimodal embedding construction (only needed
on the ``pre_process`` stage when ``decoder_input is None``):

Phase 1 – ``prepare_packing_state`` (all PP stages):
  Build CP batch & pack position_ids (when packing).

Phase 2 – ``build_multimodal_embeddings`` (pre_process only):
  1. Compute text embeddings (full-sequence or CP-sliced depending on mode)
  2. Replace media tokens with encoder outputs via each model's own
     ``construct_multimodal_inputs`` hook
  3. Pack ``inputs_embeds`` via CP-zigzag + SP scatter (when packing)
  4. Optionally pack ``visual_pos_masks`` / ``deepstack_visual_embeds``
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams

from .model_factory import McaGPTModel


@dataclass
class PackingState:
    """Lightweight result of :meth:`MultimodalEmbeddingMixin.prepare_packing_state`.

    Contains only the bookkeeping data needed by every PP stage:
    packed ``position_ids`` and the CP-sliced batch dict.
    """

    position_ids: torch.Tensor
    cp_batch: dict
    is_packing: bool


@dataclass
class MultimodalEmbeddingResult:
    """Result of :meth:`MultimodalEmbeddingMixin.build_multimodal_embeddings`."""

    inputs_embeds: torch.Tensor
    position_ids: torch.Tensor
    cp_batch: dict
    visual_pos_masks: Optional[torch.Tensor] = None
    deepstack_visual_embeds: Optional[list] = None
    extra_kwargs: dict = field(default_factory=dict)


class MultimodalEmbeddingMixin:
    """
    Mixin that unifies the multimodal embedding construction pipeline for
    VLM / omni models, handling both sequence-packing and non-packing paths.

    Models using this mixin **must** provide:

    - ``self.config``  (``sequence_parallel``, ``context_parallel_size``, …)
    - ``self.embedding`` (with ``scatter_to_sequence_parallel``,
      ``reduce_scatter_embeddings``)
    - ``self.rotary_pos_emb`` (with ``is_thd_format``)
    - ``self.pre_process`` (bool)
    - ``self.mtp_process`` (bool)
    - ``get_batch_on_this_cp_rank(batch, dim3_keys)`` (from ``McaGPTModel``)

    Each concrete model must also implement
    :meth:`construct_multimodal_inputs` (see docstring below).
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_sequence_packing(self, packed_seq_params, attention_mask):
        """Determine if we are in sequence packing mode."""
        return packed_seq_params is not None and attention_mask is not None

    @staticmethod
    def build_attention_mask_from_packed_seq_params(
        packed_seq_params: PackedSeqParams, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Construct neat-packing attention_mask from cu_seqlens_q.

        Each sub-sequence is labeled 1, 2, 3, ...; positions beyond the last
        sub-sequence boundary are left as 0 (padding).

        Args:
            packed_seq_params: contains cu_seqlens_q with shape [num_seqs + 1].
            seq_len: total sequence length of input_ids.
            device: device to create the tensor on.

        Returns:
            attention_mask of shape [batch_size=num_seqs, seq_len_per_sample]
            when cu_seqlens indicates per-sample layout, or [1, seq_len] when
            cu_seqlens indicates flat packed layout.
        """
        cu_seqlens = packed_seq_params.cu_seqlens_q
        num_seqs = len(cu_seqlens) - 1
        total_tokens = cu_seqlens[-1].item()

        if total_tokens == seq_len:
            # Flat packed layout: all sub-seqs concatenated in one row
            attention_mask = torch.zeros((1, seq_len), dtype=torch.long, device=device)
            for seq_id in range(num_seqs):
                start = cu_seqlens[seq_id].item()
                end = cu_seqlens[seq_id + 1].item()
                attention_mask[0, start:end] = seq_id + 1
        else:
            # Per-sample layout: [num_seqs, max_seq_len]
            max_seq_len = seq_len
            attention_mask = torch.zeros(
                (num_seqs, max_seq_len), dtype=torch.long, device=device
            )
            for i in range(num_seqs):
                valid_len = (cu_seqlens[i + 1] - cu_seqlens[i]).item()
                attention_mask[i, :valid_len] = 1
        return attention_mask

    def get_input_ranges(self, total_seqlen: int) -> list[list[int]]:
        """Compute the local sequence ranges for current SP/CP rank.

        In non-packing mode, each rank processes a slice of the full
        sequence determined by its tensor-parallel and context-parallel
        position.  Returns a list of ``[start, end)`` ranges.
        """
        slice_rank, slice_size = 0, 1
        if self.config.sequence_parallel:
            slice_rank = mpu.get_tensor_model_parallel_rank()
            slice_size = mpu.get_tensor_model_parallel_world_size()

        def get_sequence_range(start, end, rank, size):
            return start + (end - start) * rank // size, start + (end - start) * (rank + 1) // size

        if self.config.context_parallel_size <= 1:
            return [list(get_sequence_range(0, total_seqlen, slice_rank, slice_size))]

        cp_rank = mpu.get_context_parallel_rank()
        cp_size = mpu.get_context_parallel_world_size()
        left_start = (total_seqlen // cp_size // 2) * cp_rank
        left_end = (total_seqlen // cp_size // 2) * (cp_rank + 1)
        right_start = total_seqlen - left_end
        right_end = total_seqlen - left_start
        slice_len = (left_end - left_start + right_end - right_start) // slice_size
        start = left_start + slice_len * slice_rank
        end = start + slice_len
        if start >= left_end:
            start = start - left_end + right_start
            end = start + slice_len
            return [[start, end]]
        if end <= left_end:
            return [[start, end]]
        end = end - left_end + right_start
        return [[start, left_end], [right_start, end]]

    def prepare_packing_state(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        packed_seq_params: Optional[PackedSeqParams],
    ) -> PackingState:
        """Phase 1: lightweight packing bookkeeping (all PP stages).

        Builds the CP-sliced batch dict and packs ``position_ids`` when in
        sequence-packing mode.  Also patches Megatron's
        ``apply_rotary_pos_emb`` so that the THD path does NOT perform
        CP-based frequency slicing (freqs are generated from packed,
        CP-sliced position_ids).

        This is cheap and must run on **every** PP stage so that
        non-``pre_process`` stages get correct ``position_ids`` and
        ``cp_batch``.

        Args:
            input_ids: ``[b, s]``
            position_ids: ``[C, b, s]`` (already computed by the caller)
            attention_mask: ``[b, s]`` (``None`` when not packing)
            packed_seq_params: packing parameters
        """
        if attention_mask is None and packed_seq_params is not None:
            attention_mask = self.build_attention_mask_from_packed_seq_params(
                packed_seq_params, input_ids.shape[1], input_ids.device
            )
        is_packing = self.is_sequence_packing(packed_seq_params, attention_mask)

        if is_packing:
            self._patch_rotary_pos_emb()

        cp_batch = self._build_cp_batch(input_ids, attention_mask, is_packing, packed_seq_params)
        if is_packing:
            position_ids = self._pack_position_ids(
                position_ids, attention_mask, packed_seq_params
            )

        return PackingState(
            position_ids=position_ids,
            cp_batch=cp_batch,
            is_packing=is_packing,
        )

    def build_multimodal_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        packed_seq_params: Optional[PackedSeqParams],
        packing_state: PackingState,
        multimodal_kwargs: dict,
    ) -> MultimodalEmbeddingResult:
        """Phase 2: heavy multimodal embedding construction (pre_process only).

        Should only be called when ``self.pre_process`` is ``True`` **and**
        ``decoder_input is None``.  Computes text embeddings, replaces
        media tokens with encoder outputs, and packs the result.

        Args:
            input_ids: ``[b, s]``
            attention_mask: ``[b, s]`` (``None`` when not packing)
            packed_seq_params: packing parameters
            packing_state: the :class:`PackingState` returned by
                :meth:`prepare_packing_state`
            multimodal_kwargs: model-specific kwargs forwarded to
                :meth:`construct_multimodal_inputs`.

        Returns:
            :class:`MultimodalEmbeddingResult`
        """
        is_packing = packing_state.is_packing
        cp_batch = packing_state.cp_batch
        if attention_mask is None:
            attention_mask = cp_batch.get("attention_mask")

        # 1. Compute text embeddings
        if is_packing:
            inputs_embeds = self._compute_full_sequence_embedding(input_ids)
            inputs_ranges = None
        else:
            inputs_ranges = self.get_input_ranges(input_ids.shape[1])
            inputs_embeds = self.embedding(
                input_ids=cp_batch["input_ids"], position_ids=None
            )

        # 2. Model-specific multimodal token replacement
        mm_result = self.construct_multimodal_inputs(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            inputs_ranges=inputs_ranges,
            **multimodal_kwargs,
        )
        # Normalise to a tuple of (embeds, visual_pos_masks, deepstack)
        if isinstance(mm_result, torch.Tensor):
            inputs_embeds = mm_result
            visual_pos_masks = None
            deepstack_visual_embeds = None
        else:
            inputs_embeds, visual_pos_masks, deepstack_visual_embeds = mm_result

        # 3. Pack inputs_embeds (CP zigzag → SP scatter)
        if is_packing:
            inputs_embeds = self._pack_inputs_embeds(
                inputs_embeds, attention_mask, packed_seq_params
            )
            visual_pos_masks, deepstack_visual_embeds = (
                self._pack_deepstack_visual_embeds(
                    visual_pos_masks,
                    deepstack_visual_embeds,
                    attention_mask,
                    packed_seq_params,
                )
            )

        return MultimodalEmbeddingResult(
            inputs_embeds=inputs_embeds,
            position_ids=packing_state.position_ids,
            cp_batch=cp_batch,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

    def construct_multimodal_inputs(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        inputs_ranges: Optional[list],
        **kwargs,
    ):
        """Hook for model-specific vision / audio embedding replacement.

        Must be overridden by each concrete model class.

        TODO: unified method for vl/omni model can be used actually

        Returns:
            Either ``inputs_embeds`` (a single tensor), or a tuple of
            ``(inputs_embeds, visual_pos_masks, deepstack_visual_embeds)``
            for models that use deepstack.
        """
        raise NotImplementedError(
            "Subclasses must implement construct_multimodal_inputs"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _patch_rotary_pos_emb():
        """Patch Megatron's ``apply_rotary_pos_emb`` for THD packing.

        The standard ``_apply_rotary_pos_emb_thd`` divides seqlens by
        ``cp_size`` and applies zigzag freq selection, which is wrong
        when freqs are generated from packed, CP-sliced position_ids.
        ``apply_rotary_pos_emb_absolute`` correctly handles both THD
        and BSHD formats without CP-based freq slicing.

        We must patch both the source module (``rope_utils``) and the
        consumer module (``attention``), because ``from ... import``
        creates independent name bindings in each module's namespace.
        """
        from .qwen3_vl.rope_utils import apply_rotary_pos_emb_absolute

        import megatron.core.models.common.embeddings.rope_utils as _mcore_rope_utils
        import megatron.core.transformer.attention as _mcore_attn

        # NOTE: currently, only patch method instead of attention module replacement
        # which is different from mbridge. And we have to make sure split_qkv=True
        # to use apply_rotary_pos_emb, which is true when packed_seq_params is not None
        _mcore_rope_utils.apply_rotary_pos_emb = apply_rotary_pos_emb_absolute
        _mcore_attn.apply_rotary_pos_emb = apply_rotary_pos_emb_absolute

    @staticmethod
    def _pack_to_thd(
        data: torch.Tensor,
        attention_mask: torch.Tensor,
        packed_seq_params: PackedSeqParams,
        apply_cp_split: bool = True,
    ) -> torch.Tensor:
        """Pack data from ``[b, s, ...]`` to THD ``[packed_len, 1, ...]``.

        Follows the same packing logic as mbridge's ``preprocess_packed_seqs``:
        for each sample, extract valid tokens (via attention_mask), then
        optionally apply CP zigzag splitting on the **padded valid length**
        (not the original sequence length).

        Args:
            data: ``[b, s, ...]`` format data. The sequence dimension ``s``
                may already be SP-scattered but must NOT be CP-sliced yet.
            attention_mask: ``[b, s]`` format attention mask (1=valid,
                0=padding), must match data's sequence dimension.
            packed_seq_params: ``PackedSeqParams`` containing
                ``cu_seqlens_q_padded`` (based on full, un-split padded
                valid lengths).
            apply_cp_split: If ``True`` (default), apply CP zigzag splitting
                so the output length is ``total_padded_len // cp_size``.
                If ``False``, output the full packed sequence without CP
                splitting (useful for tensors like ``input_ids`` where
                downstream consumers handle CP splitting themselves).

        Returns:
            Packed data of shape ``[packed_len, 1, ...]`` where
            ``packed_len`` is CP-local when *apply_cp_split* is True,
            or full length otherwise.
        """
        cu_seqlens_padded = packed_seq_params.cu_seqlens_q_padded
        cp_size = mpu.get_context_parallel_world_size() if apply_cp_split else 1
        cp_rank = mpu.get_context_parallel_rank() if apply_cp_split else 0

        batch_size = data.shape[0]
        extra_dims = data.shape[2:]

        cu_seqlens_padded_cpu = cu_seqlens_padded.tolist()
        local_packed_len = cu_seqlens_padded_cpu[-1] // cp_size

        output = torch.zeros(
            [local_packed_len] + list(extra_dims),
            dtype=data.dtype,
            device=data.device,
        )

        for i in range(batch_size):
            sample_mask = attention_mask[i].bool()
            valid_tokens = data[i][sample_mask]

            padded_len_i = cu_seqlens_padded_cpu[i + 1] - cu_seqlens_padded_cpu[i]
            start_offset = cu_seqlens_padded_cpu[i] // cp_size

            if cp_size <= 1:
                num_valid = valid_tokens.shape[0]
                output[start_offset : start_offset + num_valid] = valid_tokens
            else:
                chunk_len = padded_len_i // cp_size
                half_chunk = chunk_len // 2

                front_start = half_chunk * cp_rank
                front_end = half_chunk * (cp_rank + 1)
                front_len = min(front_end, valid_tokens.shape[0]) - front_start
                if front_len > 0:
                    output[start_offset : start_offset + front_len] = (
                        valid_tokens[front_start : front_start + front_len]
                    )

                back_start = padded_len_i - half_chunk * (cp_rank + 1)
                back_end = padded_len_i - half_chunk * cp_rank
                back_end = min(back_end, valid_tokens.shape[0])
                back_len = back_end - back_start
                if back_len > 0:
                    output[
                        start_offset + half_chunk : start_offset + half_chunk + back_len
                    ] = valid_tokens[back_start:back_end]

        return output.unsqueeze(1)

    def _build_cp_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        is_packing: bool,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> dict:
        if is_packing:
            if self.mtp_process:
                # Pack input_ids to [1, local_packed_len] with CP splitting.
                # roll_tensor's _roll_tensor_packed_seq expects tensor length
                # = cu_seqlens[-1] // cp_size (CP-local).
                packed_ids = self._pack_to_thd(
                    input_ids.unsqueeze(-1), attention_mask, packed_seq_params,
                ).squeeze(-1).squeeze(-1).unsqueeze(0)  # [local_packed_len,1,1] → [1, local_packed_len]
            else:
                packed_ids = input_ids
            return {"input_ids": packed_ids, "attention_mask": attention_mask}

        cp_batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.config.context_parallel_size > 1:
            cp_batch = {
                k: v.clone() if v is not None else None
                for k, v in cp_batch.items()
            }
            # Call McaGPTModel's base implementation directly.
            # VLM models override get_batch_on_this_cp_rank to skip
            # CP-slicing input_ids, but here we need the base version
            # that slices everything.  We cannot use super() because
            # MultimodalEmbeddingMixin sits at the end of the MRO
            # (before object), so super() would resolve to object.
            cp_batch = McaGPTModel.get_batch_on_this_cp_rank(
                self,
                cp_batch,
                dim3_keys=[]
                if (cp_batch["attention_mask"] is None or cp_batch["attention_mask"].dim() == 2)
                else ["attention_mask"],
            )
        return cp_batch

    def _pack_position_ids(
        self,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        packed_seq_params: PackedSeqParams,
    ) -> torch.Tensor:
        pos_ids_bsc = position_ids.permute(1, 2, 0)  # [C, b, s] → [b, s, C]
        pos_ids_packed = self._pack_to_thd(
            pos_ids_bsc, attention_mask, packed_seq_params
        )
        self.rotary_pos_emb.is_thd_format = True
        return pos_ids_packed.permute(2, 1, 0)  # → [C, 1, packed_len]

    def _compute_full_sequence_embedding(
        self, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute embeddings on full sequence (SP scatter temporarily off)."""
        orig_scatter = self.embedding.scatter_to_sequence_parallel
        orig_reduce = self.embedding.reduce_scatter_embeddings
        orig_word_reduce = None
        if hasattr(self.embedding, "word_embeddings"):
            orig_word_reduce = (
                self.embedding.word_embeddings.reduce_scatter_embeddings
            )

        self.embedding.scatter_to_sequence_parallel = False
        self.embedding.reduce_scatter_embeddings = False
        if hasattr(self.embedding, "word_embeddings"):
            self.embedding.word_embeddings.reduce_scatter_embeddings = False

        inputs_embeds = self.embedding(input_ids=input_ids, position_ids=None)

        self.embedding.scatter_to_sequence_parallel = orig_scatter
        self.embedding.reduce_scatter_embeddings = orig_reduce
        if hasattr(self.embedding, "word_embeddings"):
            self.embedding.word_embeddings.reduce_scatter_embeddings = (
                orig_word_reduce
            )
        return inputs_embeds

    def _pack_inputs_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        packed_seq_params: PackedSeqParams,
    ) -> torch.Tensor:
        """Pack ``[s, b, h]`` → THD, then SP scatter."""
        inputs_embeds_bsh = inputs_embeds.transpose(0, 1)
        inputs_embeds = self._pack_to_thd(
            inputs_embeds_bsh, attention_mask, packed_seq_params
        )
        if self.config.sequence_parallel:
            inputs_embeds = (
                tensor_parallel.scatter_to_sequence_parallel_region(
                    inputs_embeds
                )
            )
        return inputs_embeds

    def _pack_deepstack_visual_embeds(
        self,
        visual_pos_masks: Optional[torch.Tensor],
        deepstack_visual_embeds: Optional[list],
        attention_mask: torch.Tensor,
        packed_seq_params: PackedSeqParams,
    ) -> tuple[Optional[torch.Tensor], Optional[list]]:
        """Pack visual_pos_masks + re-index deepstack for CP+SP."""
        if visual_pos_masks is None:
            return None, None

        packed_masks = (
            self._pack_to_thd(
                visual_pos_masks.unsqueeze(-1).float(),
                attention_mask,
                packed_seq_params,
            )
            .squeeze(-1)
            .squeeze(-1)
            .bool()
        )

        tp_size = mpu.get_tensor_model_parallel_world_size()
        if self.config.sequence_parallel and tp_size > 1:
            tp_rank = mpu.get_tensor_model_parallel_rank()
            packed_masks = packed_masks.chunk(tp_size)[tp_rank]

        if deepstack_visual_embeds is not None:
            idx_full = torch.full_like(
                visual_pos_masks, -1, dtype=torch.long
            )
            idx_full[visual_pos_masks] = torch.arange(
                visual_pos_masks.sum(), device=visual_pos_masks.device
            )
            idx_packed = (
                self._pack_to_thd(
                    idx_full.unsqueeze(-1).float(),
                    attention_mask,
                    packed_seq_params,
                )
                .squeeze(-1)
                .squeeze(-1)
                .long()
            )
            if self.config.sequence_parallel and tp_size > 1:
                idx_packed = idx_packed.chunk(tp_size)[tp_rank]

            final_idx = idx_packed[packed_masks]
            deepstack_visual_embeds = [
                embed[final_idx] for embed in deepstack_visual_embeds
            ]

        # [sp_packed_len] → [1, sp_packed_len]
        visual_pos_masks = packed_masks.unsqueeze(0)
        return visual_pos_masks, deepstack_visual_embeds
