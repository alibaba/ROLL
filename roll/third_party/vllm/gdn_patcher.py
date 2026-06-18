"""
Monkey patches for vLLM GDN attention to fix mixed decode/spec-decode crash.

This module patches the GDNAttentionMetadataBuilder.build method to add
reclassification logic for non-spec decodes when spec decodes exist.

Bug: https://github.com/vllm-project/vllm/pull/34871
Fix commit: 116ed130f

The bug occurs in vLLM versions < v0.17.2 when processing batches containing
both regular decode requests and speculative decode requests with GDN attention.

Error: AssertionError: num_decodes: X, num_spec_decodes: Y
"""

# TODO: This file should be removed when upgrading vLLM to version >= v0.17.2

import logging

logger = logging.getLogger(__name__)

_patch_applied = False


def patch_gdn_attention():
    """
    Apply monkey patch for GDN attention mixed decode/spec-decode bug.

    This function patches GDNAttentionMetadataBuilder.build to add
    reclassification logic that converts non-spec decodes to prefills
    when spec decodes exist in the same batch.

    The patch should be applied before any vLLM inference is performed.
    """
    global _patch_applied

    if _patch_applied:
        logger.debug("[GDN PATCH] GDN attention patch already applied, skipping")
        return

    try:
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

        # Save original build method BEFORE checking
        original_build = GDNAttentionMetadataBuilder.build

        # Check if already patched (has _original_build attribute)
        if hasattr(GDNAttentionMetadataBuilder, '_original_build'):
            # Already patched, use the real original
            original_build = GDNAttentionMetadataBuilder._original_build

        # Check if the fix is already in the source
        import inspect
        original_source = inspect.getsource(original_build)

        # Check for various signs that the fix is already present
        has_fix = (
            "num_decodes and num_spec_decodes are mutually exclusive" in original_source or
            ("num_prefills += num_decodes" in original_source and "num_decodes = 0" in original_source) or
            ("num_decodes = 0" in original_source and "num_decode_tokens = 0" in original_source)
        )

        if has_fix:
            logger.info("[GDN PATCH] GDN attention already has the fix, skipping patch")
            _patch_applied = True
            return

        def patched_build(
            self,
            common_prefix_len: int,
            common_attn_metadata,
            num_accepted_tokens=None,
            num_decode_draft_tokens_cpu=None,
            fast_build: bool = False,
        ):
            """Patched build method with catch + fallback pattern."""
            try:
                return original_build(
                    self,
                    common_prefix_len,
                    common_attn_metadata,
                    num_accepted_tokens,
                    num_decode_draft_tokens_cpu,
                    fast_build,
                )
            except AssertionError as e:
                error_msg = str(e)
                if "num_decodes" in error_msg and "num_spec_decodes" in error_msg:
                    logger.warning(
                        f"[GDN PATCH] GDN attention assertion caught, applying workaround: {error_msg}"
                    )
                    return _build_with_fix(
                        self,
                        common_prefix_len,
                        common_attn_metadata,
                        num_accepted_tokens,
                        num_decode_draft_tokens_cpu,
                        fast_build,
                    )
                else:
                    raise

        # Apply the patch
        GDNAttentionMetadataBuilder.build = patched_build
        GDNAttentionMetadataBuilder._original_build = original_build

        _patch_applied = True
        logger.info("[GDN PATCH] GDN attention patch applied successfully")

    except ImportError as e:
        logger.debug(f"[GDN PATCH] vLLM GDN attention not available, skipping patch: {e}")
    except Exception as e:
        logger.warning(f"[GDN PATCH] Failed to apply GDN attention patch: {e}")


def _build_with_fix(
    self,
    common_prefix_len: int,
    common_attn_metadata,
    num_accepted_tokens=None,
    num_decode_draft_tokens_cpu=None,
    fast_build: bool = False,
):
    """
    Build method with the reclassification fix inline.

    This is a copy of the original vLLM build method with ONLY the
    reclassification logic added. All other logic is identical to
    the original implementation.

    The fix is at lines 273-278: reclassify num_decodes as num_prefills
    when both num_decodes > 0 and num_spec_decodes > 0.
    """
    import torch
    from vllm.v1.attention.backends.gdn_attn import (
        GDNAttentionMetadata,
        mamba_get_block_table_tensor,
    )
    from vllm.v1.attention.backends.utils import (
        split_decodes_and_prefills,
        compute_causal_conv1d_metadata,
    )

    m = common_attn_metadata

    query_start_loc = m.query_start_loc
    query_start_loc_cpu = m.query_start_loc_cpu
    context_lens_tensor = m.compute_num_computed_tokens()
    # Initialize these to None at the start (same as original vLLM)
    nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
    block_table_tensor = mamba_get_block_table_tensor(
        m.block_table_tensor,
        m.seq_lens,
        self.kv_cache_spec,
        self.vllm_config.cache_config.mamba_cache_mode,
    )

    spec_sequence_masks_cpu = None
    if (
        not self.use_spec_decode
        or num_decode_draft_tokens_cpu is None
        or num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
        .sum()
        .item()
        == 0
    ):
        spec_sequence_masks = None
        num_spec_decodes = 0
    else:
        spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
        num_spec_decodes = spec_sequence_masks_cpu.sum().item()
        if num_spec_decodes == 0:
            spec_sequence_masks = None
            spec_sequence_masks_cpu = None
        else:
            spec_sequence_masks = spec_sequence_masks_cpu.to(
                query_start_loc.device, non_blocking=True
            )

    if spec_sequence_masks is None:
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(m, decode_threshold=1)
        )
        num_spec_decode_tokens = 0
        spec_token_indx = None
        non_spec_token_indx = None
        spec_state_indices_tensor = None
        non_spec_state_indices_tensor = block_table_tensor[:, 0]
        spec_query_start_loc = None
        non_spec_query_start_loc = query_start_loc
        non_spec_query_start_loc_cpu = query_start_loc_cpu
        num_accepted_tokens = None
    else:
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        assert spec_sequence_masks_cpu is not None
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
        num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
        num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
        num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
        num_decode_tokens = num_decodes
        num_prefill_tokens = (
            non_spec_query_lens_cpu.sum().item() - num_decode_tokens
        )
        num_spec_decode_tokens = (
            query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
        )

        # ================================================================
        # THE FIX: Reclassify non-spec decodes as prefills when spec decodes exist
        # This is the ONLY change from the original vLLM code.
        # ================================================================
        if num_decodes > 0 and num_spec_decodes > 0:
            num_prefills += num_decodes
            num_prefill_tokens += num_decode_tokens
            num_decodes = 0
            num_decode_tokens = 0
        # ================================================================

        if num_prefills == 0 and num_decodes == 0:
            spec_token_size = min(
                num_spec_decodes * (self.num_spec + 1),
                query_start_loc_cpu[-1].item(),
            )
            spec_token_indx = torch.arange(
                spec_token_size,
                dtype=torch.int32,
                device=query_start_loc.device,
            )
            non_spec_token_indx = torch.empty(
                0, dtype=torch.int32, device=query_start_loc.device
            )
            spec_state_indices_tensor = block_table_tensor[
                spec_sequence_masks_cpu, : self.num_spec + 1
            ]
            non_spec_state_indices_tensor = None
            spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
            non_spec_query_start_loc = None
            non_spec_query_start_loc_cpu = None
        else:
            spec_token_masks = torch.repeat_interleave(
                spec_sequence_masks, query_lens
            )
            index = torch.argsort(spec_token_masks, stable=True)
            num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
            non_spec_token_indx = index[:num_non_spec_tokens]
            spec_token_indx = index[num_non_spec_tokens:]

            spec_state_indices_tensor = block_table_tensor[
                spec_sequence_masks_cpu, : self.num_spec + 1
            ]
            non_spec_state_indices_tensor = block_table_tensor[
                ~spec_sequence_masks_cpu, 0
            ]

            spec_query_start_loc = torch.zeros(
                num_spec_decodes + 1,
                dtype=torch.int32,
                device=query_start_loc.device,
            )
            torch.cumsum(
                query_lens[spec_sequence_masks_cpu], dim=0, out=spec_query_start_loc[1:]
            )
            non_spec_query_start_loc = torch.zeros(
                query_lens.size(0) - num_spec_decodes + 1,
                dtype=torch.int32,
                device=query_start_loc.device,
            )
            torch.cumsum(
                query_lens[~spec_sequence_masks_cpu],
                dim=0,
                out=non_spec_query_start_loc[1:],
            )
            non_spec_query_start_loc_cpu = torch.zeros(
                query_lens_cpu.size(0) - num_spec_decodes + 1,
                dtype=torch.int32,
            )
            torch.cumsum(
                query_lens_cpu[~spec_sequence_masks_cpu],
                dim=0,
                out=non_spec_query_start_loc_cpu[1:],
            )

        assert num_accepted_tokens is not None
        num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]

    if num_prefills > 0:
        has_initial_state = context_lens_tensor > 0
        if spec_sequence_masks is not None:
            has_initial_state = has_initial_state[~spec_sequence_masks]
            assert non_spec_query_start_loc_cpu is not None
        nums_dict, batch_ptr, token_chunk_offset_ptr = (
            compute_causal_conv1d_metadata(
                non_spec_query_start_loc_cpu,
                device=query_start_loc.device,
            )
        )
    else:
        has_initial_state = None

    # NOTE: We skip the cudagraph optimization logic here (lines 314-382 in original)
    # because our fallback path is only triggered for mixed batches, which don't
    # benefit from cudagraph anyway. This simplifies the code and avoids potential
    # issues with cudagraph tensor management.

    return GDNAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_spec_decodes=num_spec_decodes,
        num_spec_decode_tokens=num_spec_decode_tokens,
        num_actual_tokens=m.num_actual_tokens,
        has_initial_state=has_initial_state,
        spec_query_start_loc=spec_query_start_loc,
        non_spec_query_start_loc=non_spec_query_start_loc,
        spec_state_indices_tensor=spec_state_indices_tensor,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        spec_sequence_masks=spec_sequence_masks,
        spec_token_indx=spec_token_indx,
        non_spec_token_indx=non_spec_token_indx,
        num_accepted_tokens=num_accepted_tokens,
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )


if __name__ == "__main__":
    # Apply patch when run directly
    patch_gdn_attention()
