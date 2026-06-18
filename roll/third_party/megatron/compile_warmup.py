import random
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.distributed as dist
from megatron.core import mpu, tensor_parallel

from mcore_adapter.trainer.utils import get_ltor_masks_and_position_ids
from roll.platforms import current_platform
from roll.utils.constants import IGNORE_INDEX
from roll.utils.logging import get_logger


if TYPE_CHECKING:
    from roll.distributed.strategy.megatron_strategy import MegatronTrainStrategy


logger = get_logger()


@contextmanager
def _preserve_rng_states():
    """Save and restore all RNG states to prevent warmup from perturbing training randomness."""
    saved = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": current_platform.get_rng_state(),
        "tracker": tensor_parallel.get_cuda_rng_tracker().get_states(),
    }
    try:
        yield
    finally:
        random.setstate(saved["random"])
        np.random.set_state(saved["numpy"])
        torch.set_rng_state(saved["torch"])
        current_platform.set_rng_state(saved["cuda"])
        tensor_parallel.get_cuda_rng_tracker().set_states(saved["tracker"])


def _build_warmup_inputs(strategy: "MegatronTrainStrategy", seq_length: int) -> dict[str, Any]:
    """Build synthetic inputs that mirror the shape/structure produced by _prepare_train_inputs."""
    batch_size = strategy.worker_config.training_args.per_device_train_batch_size
    input_ids = torch.arange(seq_length, device=strategy.megatron_train_args.device, dtype=torch.long).repeat(batch_size, 1)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    if strategy.worker_config.use_sequence_packing:
        input_ids, packed_seq_params, cu_seqlens, cu_seqlens_padded = strategy._pack_sequences(
                input_ids, attention_mask,
            )
        labels, _, _, _ = strategy._pack_sequences(labels, attention_mask, pad_val=IGNORE_INDEX)
        attention_mask = None
        position_ids = None
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "packed_seq_params": packed_seq_params}
    else:
        attention_mask, position_ids = get_ltor_masks_and_position_ids(
            input_ids,
            build_attention_mask=strategy.model.config.transformer_impl != "transformer_engine",
            attn_mask_1D=attention_mask,
        )
        if not strategy.model.config.num_moe_experts and strategy.model.config.transformer_impl == "transformer_engine":
            attention_mask = None
        inputs["attention_mask"] = attention_mask

    inputs["position_ids"] = position_ids if strategy.model.config.mtp_num_layers else None
    return strategy.models_unwrapped[0].get_batch_on_this_cp_rank(inputs, dim3_keys=[])


def _build_decoder_input(
    strategy: "MegatronTrainStrategy", inputs: dict[str, Any], module: torch.nn.Module
) -> torch.Tensor:
    """Build a zero-filled hidden-state tensor for non-first pipeline stages."""
    config = strategy.model.config
    cp_size = strategy.worker.rank_info.cp_size
    seq_length = inputs["input_ids"].shape[1] // cp_size
    if config.sequence_parallel:
        seq_length = max(1, seq_length // mpu.get_tensor_model_parallel_world_size())
    batch_size = inputs["input_ids"].shape[0]
    dtype = (
        getattr(config, "pipeline_dtype", None)
        or getattr(config, "params_dtype", None)
        or next(module.parameters()).dtype
    )
    return torch.zeros((seq_length, batch_size, config.hidden_size), device=strategy.megatron_train_args.device, dtype=dtype)


@torch.no_grad()
def compile_warmup_pipeline_stages(strategy: "MegatronTrainStrategy"):
    """Run a local forward pass on each pipeline chunk to trigger torch.compile ahead of real training.

    Without this, the first training step serializes compilation across pipeline stages
    because P2P communication blocks until upstream stages finish compiling.
    """
    if mpu.get_pipeline_model_parallel_world_size() <= 1:
        return

    seq_length = strategy.seq_length or 8192
    cp_size = strategy.worker.rank_info.cp_size
    tp_size = strategy.worker.rank_info.tp_size
    chunk = 128
    if cp_size > 1:
        # _get_batch_on_this_cp_rank will split by cp_size*2; ensure seq_length is divisible.
        chunk *= cp_size
    if tp_size > 1 and strategy.megatron_train_args.sequence_parallel:
        chunk *= tp_size
    seq_length = ((seq_length + chunk - 1) // chunk) * chunk

    logger.info(
        "Running local pipeline compile warmup with batch_size=%s, seq_length=%s.",
        strategy.worker_config.training_args.per_device_train_batch_size,
        seq_length,
    )

    original_vp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
    with _preserve_rng_states():
        try:
            for model in strategy.models_wrapped:
                model.train()
                model.zero_grad_buffer()
            strategy.optimizer.zero_grad()

            for model in strategy.models_wrapped:
                module = getattr(model, "module", model)
                vp_stage = getattr(module, "vp_stage", None)
                if vp_stage is not None:
                    mpu.set_virtual_pipeline_model_parallel_rank(vp_stage)

                inputs = _build_warmup_inputs(strategy, seq_length)
                if not getattr(module, "pre_process", True):
                    decoder_input = _build_decoder_input(strategy, inputs, module)
                    module.set_input_tensor(decoder_input)
                    inputs["decoder_input"] = decoder_input

                model(**inputs)
        finally:
            if original_vp_rank is not None:
                mpu.set_virtual_pipeline_model_parallel_rank(original_vp_rank)
            for model in strategy.models_wrapped:
                model.zero_grad_buffer()
            strategy.optimizer.zero_grad()
            if dist.is_initialized():
                dist.barrier()
