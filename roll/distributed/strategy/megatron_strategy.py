import math
import os
import random
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import ray
import ray.actor
import torch
import torch.distributed as dist
from codetiming import Timer
from megatron.core import DistributedDataParallel, dist_checkpointing, mpu, tensor_parallel
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy
from megatron.core.distributed import DistributedDataParallelConfig, finalize_model_grads
from megatron.core.models.common.embeddings import RotaryEmbedding
from megatron.core.optimizer import MegatronOptimizer, OptimizerConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from megatron.core.transformer.moe.moe_utils import (
    clear_aux_losses_tracker,
    get_moe_layer_wise_logging_tracker,
    reduce_aux_losses_tracker_across_ranks,
    save_to_aux_losses_tracker,
)
from megatron.core.transformer.moe.router_replay import (
    RouterReplay,
    RouterReplayAction,
)
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from transformers.utils import is_peft_available

from mcore_adapter import TrainingArguments
from mcore_adapter.checkpointing import generate_model_state_dict, get_checkpoint_dir, load_state_dict_from_checkpoint
from mcore_adapter.checkpoint_read_order import patch_mcore_checkpoint_read_order
from mcore_adapter.checkpoint_write import streaming_save_strategy
from mcore_adapter.models.qwen4_exp.capabilities import validate_bounded_rl_token_statistics
from mcore_adapter.parallel_functions import context_parallel_gather, vocab_parallel_logprobs
from mcore_adapter.patcher import (
    patch_apply_aux_loss,
    patch_hybrid_optimizer,
    patch_megatron_preload_tensors_non_blocking,
    patch_torch_find_nd_overlapping_shards,
    patch_torch_validate_global_plan,
)
from mcore_adapter.trainer.utils import build_sharded_state_dict_metadata, get_megatron_lr_scheduler
from roll.datasets.collator import collate_fn_to_dict_list
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_processor_provider, default_tokenizer_provider, freeze_except_mtp
from roll.platforms import current_platform
from roll.third_party.megatron.compile_warmup import compile_warmup_pipeline_stages
from roll.third_party.megatron.model_update import MegatronWeightUpdater
from roll.third_party.megatron.mtp_patcher import patch_mtp_functions
from roll.third_party.megatron.offload_states_patch import (
    MegatronOffloadStateType,
    bind_megatron_offload_states_func,
    checkpoint_grad_buffer_offload,
    cleanup_ddp_buffers,
    is_model_params_offloaded,
    offload_megatron_no_grad_module,
    reload_megatron_no_grad_module,
)
from roll.third_party.megatron.optimizer import get_megatron_optimizer
from roll.third_party.megatron.optimizer_config import build_optimizer_config
from roll.third_party.megatron.router_replay_utils import (
    collect_r2_router_indices,
    finalize_r2_routed_experts,
    set_router_replay_data,
    RouterReplayHelper
)
from roll.third_party.megatron.tensor_parallel import vocab_parallel_entropy
from roll.third_party.megatron.util import unwrap_model
from roll.utils.constants import (
    DIST_OPTIMIZER_DIR,
    IGNORE_INDEX,
    OPTIMIZER_NAME,
    RNG_STATE_DIR,
    SCHEDULER_NAME,
)
from roll.utils.context_managers import disable_gradients
from roll.utils.dynamic_batching import make_micro_batch_iter_for_dynamic_batching
from roll.utils.functionals import (
    adjust_sequence_length,
    append_to_dict,
    reduce_metrics,
)
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType, clear_memory
from roll.utils.sequence_packing import make_micro_batch_iter_for_sequence_packing, restore_results_order


if TYPE_CHECKING:
    from mcore_adapter.models.model_factory import VirtualModels


if is_peft_available():
    from peft import PeftModel, get_peft_model_state_dict


logger = get_logger()


_INCLUDE_MAP = {
    OffloadStateType.model_params: MegatronOffloadStateType.model_params,
    OffloadStateType.optimizer_states: MegatronOffloadStateType.optimizer_states,
    OffloadStateType.other_params: MegatronOffloadStateType.other_params,
}


def _to_megatron_include(include):
    """Map OffloadStateType include list to MegatronOffloadStateType list."""
    if include is None:
        return list(MegatronOffloadStateType)
    return [_INCLUDE_MAP[s] for s in include if s in _INCLUDE_MAP]


class MegatronInferStrategy(InferenceStrategy):
    strategy_name = "megatron_infer"

    def __init__(self, worker: Worker):
        # Apply MTP patches BEFORE model instantiation
        patch_mtp_functions()
        #TODO remove the patches when the latest pytorch version > v2.9.1
        patch_torch_find_nd_overlapping_shards()
        patch_torch_validate_global_plan()
        patch_mcore_checkpoint_read_order()
        super().__init__(worker)
        config_dict = self.worker_config.training_args.to_dict()
        config_dict.update(self.worker_config.strategy_args.strategy_config)
        # maybe put max_grad_norm into training_args as transformers do, rather
        # than in pipeline_config (PPOConfig)
        config_dict.update({"max_grad_norm": self.worker.pipeline_config.max_grad_norm})
        config_dict.setdefault("lr_scheduler_kwargs", {})
        logger.info(f"training_args: {config_dict}")
        self.megatron_train_args = TrainingArguments(**config_dict)
        patch_megatron_preload_tensors_non_blocking(non_blocking=self.megatron_train_args.ckpt_d2h_non_blocking)
        patch_hybrid_optimizer()
        patch_apply_aux_loss()
        self.model = None
        self.forward_backward_func = None
        self.seq_length = None
        self.use_sequence_packing = self.worker_config.use_sequence_packing
        self._bounded_rl_token_statistics_request = self._validate_bounded_rl_token_statistics_request()
        self._bounded_rl_token_statistics_enabled = False
        # hard to impl with offload states
        assert not self.megatron_train_args.overlap_param_gather, "overlap_param_gather is not supported"

        # Router Replay config
        self.router_replay_config = self.worker_config.router_replay
        self.enable_router_replay = (self.router_replay_config.mode in ["R2", "R3"])
        self.router_replay_mode = self.router_replay_config.mode

        # Force enable moe_enable_routing_replay when router replay is enabled,
        # so that RouterReplay instances are created in Megatron MoE layers.
        if self.enable_router_replay:
            self.megatron_train_args.moe_enable_routing_replay = True
            # moe_router_fusion's fused TE topk bypasses the router_replay hook, disabling
            # record/replay. Force it off via additional_configs (merged last in get_config_dict).
            additional_configs = self.megatron_train_args.additional_configs or {}
            if additional_configs.get("moe_router_fusion"):
                logger.warning(
                    "[RouterReplay] moe_router_fusion=True disables router replay "
                    "(fused TE topk bypasses the replay hook); forcing it off."
                )
            additional_configs["moe_router_fusion"] = False
            self.megatron_train_args.additional_configs = additional_configs

        # store router replay data
        if self.enable_router_replay and self.router_replay_mode == "R2":
            self.router_topk_indices_list = []
            logger.info("Router Replay R2 mode: RECORD enabled in MegatronInferStrategy")

        # Offload backend (selected by config)
        self._offload_backend = None

    def initialize(self, model_provider):
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.processor = default_processor_provider(model_args=self.worker_config.model_args)
        self.model: "VirtualModels" = model_provider(
            tokenizer=self.tokenizer,
            model_args=self.worker_config.model_args,
            training_args=self.megatron_train_args,
            is_trainable=False,
        )
        self.model.config.finalize_model_grads_func = finalize_model_grads

        # Inject mtp_training_mode from WorkerConfig to model config
        if hasattr(self.worker_config, "mtp_training_mode"):
            self.model.config.mtp_training_mode = self.worker_config.mtp_training_mode

        self.models_unwrapped = self.model.get_models()
        self._configure_bounded_rl_token_statistics()
        self.forward_backward_func = get_forward_backward_func()
        self.is_multimodal = self.processor is not None
        self._validate_vlm_packing_support()

        self.seq_length = self.worker.pipeline_config.sequence_length

        self.worker.rank_info.dp_rank = mpu.get_data_parallel_rank(with_context_parallel=False)
        self.worker.rank_info.dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
        self.worker.rank_info.tp_rank = mpu.get_tensor_model_parallel_rank()
        self.worker.rank_info.tp_size = mpu.get_tensor_model_parallel_world_size()
        self.worker.rank_info.pp_rank = mpu.get_pipeline_model_parallel_rank()
        self.worker.rank_info.pp_size = mpu.get_pipeline_model_parallel_world_size()
        self.worker.rank_info.cp_size = mpu.get_context_parallel_world_size()
        self.worker.rank_info.cp_rank = mpu.get_context_parallel_rank()

        if (self.worker_config.use_dynamic_batching_in_infer or self.worker_config.use_sequence_packing) and self.worker.rank_info.pp_size > 1:
            self.model.config.variable_seq_lengths = True
            logger.info("Set variable_seq_lengths to True when use dynamic batching and pipeline parallel.")

        if self.enable_router_replay and self.router_replay_mode == "R2":
            # R2 mode: init router_replay_action=RouterReplayAction.RECORD
            RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)

        logger.info(f"{self.model.get_models()}")
        self._warmup_p2p_comms()
        dist.barrier()
        clear_memory(clear_host_memory=True)

    def _warmup_p2p_comms(self):
        # Pre-create lazy 2-rank p2p communicators at init to avoid deadlock
        # when they are first built inside the staggered pipeline schedule.
        groups = [mpu.get_pipeline_model_parallel_group()]
        if self.worker.rank_info.cp_size > 1:
            groups.append(mpu.get_context_parallel_group())
        device = current_platform.current_device()
        for group in groups:
            size = dist.get_world_size(group)
            if size < 2:
                continue
            rank = dist.get_rank(group)
            t = torch.zeros(1, device=device)
            next_peer = dist.get_global_rank(group, (rank + 1) % size)
            prev_peer = dist.get_global_rank(group, (rank - 1) % size)
            ops = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, t, next_peer, group),
                    dist.P2POp(dist.irecv, t, prev_peer, group),
                ]
            )
            for op in ops:
                op.wait()
        logger.info("[P2P_WARMUP] done")

    def _validate_vlm_packing_support(self):
        """Check if the VLM model supports sequence packing via MultimodalEmbeddingMixin.

        If use_sequence_packing is enabled but the multimodal model does not
        inherit MultimodalEmbeddingMixin, automatically disable packing and
        log a warning.
        """
        if not (self.use_sequence_packing and self.is_multimodal):
            return
        from mcore_adapter.models.sequence_packing_mixin import MultimodalEmbeddingMixin
        model_supports_packing = isinstance(self.models_unwrapped[0], MultimodalEmbeddingMixin)
        if not model_supports_packing:
            logger.warning(
                "use_sequence_packing is enabled but the multimodal model does not "
                "inherit MultimodalEmbeddingMixin. Disabling sequence packing for "
                "this model. To enable packing, the model must inherit "
                "MultimodalEmbeddingMixin."
            )
            self.use_sequence_packing = False
            self.worker_config.use_sequence_packing = False

    # TODO: drop this once deepstack patches TransformerBlock.forward instead of rebuilding the decoder.
    def _rebuild_router_replay_instances(self):
        """Re-register RouterReplay.global_router_replay_instances from the live model's routers: VL
        replaces self.decoder after super().__init__(), orphaning stale instances that break R2's slice.
        """
        if not self.enable_router_replay:
            return
        instances = RouterReplay.global_router_replay_instances
        active = [
            m.router_replay
            for chunk in self.model.get_models()
            for m in unwrap_model(chunk).modules()
            if getattr(m, "router_replay", None) is not None
        ]
        logger.info(f"[RouterReplay] rebuilt global instances: {len(instances)} -> {len(active)}")
        instances[:] = active

    def forward_step(
        self,
        batch: DataProto,
        forward_func: Callable[[DataProto, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()

        if self.enable_router_replay:
            if self.router_replay_mode == "R3":
                if "routed_experts" in batch.batch:
                    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
                else:
                    RouterReplay.clear_global_router_replay_action()
            elif self.router_replay_mode == "R2":
                RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
                self.router_topk_indices_list = []

        batch.meta_info['batch_num_tokens'] = self._get_batch_num_tokens(batch, dp_group=mpu.get_data_parallel_group())
        batch.meta_info['global_valid_samples'] = self._get_global_valid_samples(batch, dp_group=mpu.get_data_parallel_group())

        output_on_all_tp_cp_ranks = batch.meta_info.get("output_on_all_tp_cp_ranks", False)
        if self.worker_config.use_dynamic_batching_in_infer:
            micro_batches_list = list(make_micro_batch_iter_for_dynamic_batching(batch))
            num_microbatches = batch.meta_info["num_micro_batchs"]
            micro_batch_size = 1
        elif self.use_sequence_packing:
            vp_size = self.worker_config.strategy_args.strategy_config['virtual_pipeline_model_parallel_size'] \
                if 'virtual_pipeline_model_parallel_size' in self.worker_config.strategy_args.strategy_config else 1
            # Flatten [B,S,L,K] -> [B,S*L*K] (zero-copy) for fast per-microbatch row
            # gather in the packer; restored in inner_forward_step.
            if "routed_experts" in batch.batch and batch.batch["routed_experts"].dim() >= 3:
                _re = batch.batch["routed_experts"]
                batch.meta_info["_routed_experts_tail_shape"] = list(_re.shape[1:])
                batch.batch["routed_experts"] = _re.reshape(_re.shape[0], -1)
            micro_batches_list = list(
                make_micro_batch_iter_for_sequence_packing(batch, tp_size=self.worker.rank_info.tp_size,
                                                           cp_size=self.worker.rank_info.cp_size,
                                                           vp_size=vp_size, is_train=False,
                                                           dp_group=mpu.get_data_parallel_group(with_context_parallel=True),
                                                           micro_batch_size=batch.meta_info["micro_batch_size"],
                                                           config=self.worker_config.sequence_packing_args,
                                                           pp_size=self.worker.rank_info.pp_size))
            self._precompute_pack_layouts(micro_batches_list)
            num_microbatches = micro_batches_list[0].meta_info["num_micro_batchs"]
            micro_batch_size = 1
            seq_packing_metrics = batch.meta_info.pop('sequence_packing_metrics', {})
            if seq_packing_metrics:
                sp_prefix = f"sequence_packing/{self.worker_config.name}"
                batch.meta_info['sequence_packing_metrics'] = {f"{sp_prefix}/{k}": v for k, v in seq_packing_metrics.items()}
        else:
            batch_size = batch.batch.batch_size[0]
            micro_batch_size = batch.meta_info["micro_batch_size"]
            num_microbatches = max(batch_size // micro_batch_size, 1)
            micro_batches_list = batch.chunk(chunks=num_microbatches)

        disable_adapter = batch.meta_info.get("disable_adapter", False)
        adapter_context = self.models_unwrapped[0].disable_adapter() if disable_adapter else nullcontext()

        for micro_batch in micro_batches_list:
            micro_batch.meta_info['loss_scale'] = num_microbatches * mpu.get_data_parallel_world_size()
            micro_batch.meta_info['micro_batch_size'] = micro_batch.batch.batch_size[0]

        data_iterator = [iter(micro_batches_list) for _ in range(len(self.model))]
        with disable_gradients(models=self.model.get_models()), adapter_context:
            # List 是每个 micro-batch 构成的
            losses_reduced: List[Dict[str, torch.Tensor]] = self.forward_backward_func(
                forward_step_func=partial(self.inner_forward_step, forward_func),
                data_iterator=data_iterator,
                model=self.model.get_models(),
                num_microbatches=num_microbatches,
                seq_length=self.seq_length,
                micro_batch_size=micro_batch_size,
                forward_only=True,
            )
        if self.worker_config.use_dynamic_batching_in_infer:
            for data in losses_reduced:
                for k, v in data.items():
                    data[k] = torch.nn.functional.pad(v, (0, self.seq_length - data[k].size(-1) - 1), "constant", 0)
        results = collate_fn_to_dict_list(losses_reduced)

        if self.enable_router_replay and self.router_replay_mode == "R2":
            torch.cuda.current_stream().synchronize()  # wait for pending non_blocking D2H copies
            results["routed_experts"] = finalize_r2_routed_experts(
                self.router_topk_indices_list, self.model.config, num_microbatches
            )
            self.router_topk_indices_list = []

        if self.use_sequence_packing:
            results = restore_results_order(results, micro_batches_list[0].meta_info['partition_indices_list'],
                                  self.worker_config.sequence_packing_args)

        if self.enable_router_replay:
            RouterReplay.clear_global_router_replay_action()
            RouterReplay.clear_global_indices()

        if not (
                ((self.worker.rank_info.tp_rank == 0
                and self.worker.rank_info.cp_rank == 0) or output_on_all_tp_cp_ranks)
                and self.worker.rank_info.is_pipeline_last_stage
        ):
            return None
        return results

    def _get_feature_on_this_cp_rank(self, feature: torch.Tensor, feature_name: str = "input_ids") -> torch.Tensor:
        return self.models_unwrapped[0].get_batch_on_this_cp_rank({feature_name: feature}, dim3_keys=[])[feature_name]

    def _get_unpad_seqlen(self, attention_mask: torch.Tensor, pad_to_multiple_of: int = 256) -> int:
        max_seqlen = attention_mask.sum(dim=1).max().item()

        cp_size = mpu.get_context_parallel_world_size()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        pad_factor = 2 * cp_size * tp_size if cp_size > 1 else tp_size
        pad_factor = math.lcm(pad_factor, pad_to_multiple_of)

        padded_max_seqlen = (max_seqlen + pad_factor - 1) // pad_factor * pad_factor

        return padded_max_seqlen

    def _get_pad_factor(self):
        # caculate pad_factor in sequence packing
        cp_size = mpu.get_context_parallel_world_size()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        pad_factor = cp_size * 2 * tp_size if cp_size > 1 else tp_size
        pad_factor = math.lcm(16, pad_factor)
        return pad_factor

    def _compute_pack_layout(self, attention_mask: torch.Tensor, pad_factor: int) -> Dict:
        seq_lens = attention_mask.sum(dim=-1).tolist()
        use_padded = pad_factor > 1
        cu_seqlens = [0]
        cu_seqlens_padded = [0] if use_padded else None
        max_seqlen = max(seq_lens) if seq_lens else 0
        for seq_len in seq_lens:
            cu_seqlens.append(cu_seqlens[-1] + seq_len)
            if use_padded:
                padded_seq_len = ((seq_len + pad_factor - 1) // pad_factor) * pad_factor
                cu_seqlens_padded.append(cu_seqlens_padded[-1] + padded_seq_len)
                max_seqlen = max(max_seqlen, padded_seq_len)
        device = current_platform.device_type
        return {
            "mask": attention_mask,
            "seq_lens": seq_lens,
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
            "cu_seqlens_padded": torch.tensor(cu_seqlens_padded, dtype=torch.int32, device=device)
            if use_padded else None,
            "max_seqlen": max_seqlen,
        }

    def _precompute_pack_layouts(self, micro_batches_list: List[DataProto]) -> None:
        # The layout computation does a GPU->CPU sync (.tolist()). Doing it inside the
        # pipeline schedule can deadlock interleaved VP + EP all-to-all (a rank stalled
        # on the sync misses its PP p2p send, peers wait, cycle forms). Precompute all
        # layouts before entering forward_backward_func, where the sync is safe.
        self._precomputed_pack_layouts = {}
        pad_factor = self._get_pad_factor()
        for micro_batch in micro_batches_list:
            mask = micro_batch.batch["attention_mask"]
            if mask not in self._precomputed_pack_layouts:
                self._precomputed_pack_layouts[mask] = self._compute_pack_layout(mask, pad_factor)

    def _get_precomputed_layout(self, attention_mask: torch.Tensor):
        layouts = getattr(self, "_precomputed_pack_layouts", None)
        if layouts is None:
            return None
        return layouts.get(attention_mask)

    def _pack_sequences(
        self, input_tensor, attention_mask, pad_val=0, distinct_topk_padding=False,
    ):
        """Pack padded sequences into one contiguous sequence for varlen attention.

        Removes per-sample padding, aligns each sequence to pad_factor, then slices
        to the current CP rank. The layout (seq_lens/cu_seqlens/max_seqlen) is cached
        per attention_mask, so packing input_ids/labels/loss_mask/routed_experts of
        one microbatch shares a single layout computation.

        Returns:
            (packed_input_tensor, packed_seq_params, cu_seqlens, cu_seqlens_padded)
        """

        batch_size = input_tensor.shape[0]
        pad_factor = self._get_pad_factor()

        layout = self._get_precomputed_layout(attention_mask)
        if layout is None:
            layout = getattr(self, "_pack_layout_cache", None)
            if layout is None or layout["mask"] is not attention_mask:
                layout = self._compute_pack_layout(attention_mask, pad_factor)
                self._pack_layout_cache = layout
        seq_lens = layout["seq_lens"]
        cu_seqlens = layout["cu_seqlens"]
        cu_seqlens_padded = layout["cu_seqlens_padded"]
        max_seqlen = layout["max_seqlen"]

        # Remove padding from each sequence
        # Note: attention_mask is not needed in sequence packing mode
        input_tensor_unpadded = [input_tensor[b][:seq_lens[b]] for b in range(batch_size)]

        cp_size = mpu.get_context_parallel_world_size()

        padded_tokens = []
        _topk_pad_rows = None
        for b in range(batch_size):
            seq_len = seq_lens[b]
            # Align to pad_factor boundary
            padded_seq_len = ((seq_len + pad_factor - 1) // pad_factor) * pad_factor

            seq_tokens = input_tensor_unpadded[b]

            # Pad sequence if needed (along the first dim, i.e. seq_len dim)
            if padded_seq_len > seq_len:
                # F.pad pad argument maps from last dim to first dim.
                # For [seq_len, *] tensors, we only pad the first dim (seq_len),
                # so we prepend 2*(ndim-1) zeros for all trailing dimensions.
                pad_tuple = [0] * (2 * (seq_tokens.ndim - 1)) + [0, padded_seq_len - seq_len]
                seq_tokens = torch.nn.functional.pad(
                    seq_tokens, pad_tuple, value=pad_val
                )
                if distinct_topk_padding:
                    # Padding tokens still go through MoE; a constant pad value would dedupe
                    # to one expert per token and desync a2a split sizes, so give each slot
                    # a distinct expert id.
                    if _topk_pad_rows is None:
                        _topk_pad_rows = torch.arange(
                            seq_tokens.shape[-1], dtype=seq_tokens.dtype, device=seq_tokens.device
                        )
                    seq_tokens[seq_len:, :, :] = _topk_pad_rows

            if cp_size > 1:
                # Handle Context Parallel distribution
                # Add batch dimension for processing
                seq_tokens_with_batch = seq_tokens.unsqueeze(0)  # [1, seq_len]
                seq_tokens_with_batch = self._get_feature_on_this_cp_rank(
                    seq_tokens_with_batch, "seq_tokens"
                )
                seq_tokens = seq_tokens_with_batch.squeeze(0)  # Remove batch dimension

            padded_tokens.append(seq_tokens)

        # Concatenate all sequences
        packed_input_tensor = torch.cat(padded_tokens, dim=0).unsqueeze(0)

        if cu_seqlens_padded is None:
            cu_seqlens_padded = cu_seqlens.clone()

        # Create packed sequence parameters for attention computation
        # Only use padded cumulative sequence lengths
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens_padded,
            cu_seqlens_kv=cu_seqlens_padded,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            # Individual sequence length
            max_seqlen_q=int(max_seqlen),
            max_seqlen_kv=int(max_seqlen),
            qkv_format="thd",
        )

        return (
            # Packed input tensor for current rank (especially CP rank) computation
            # Contains all tokens from the batch with individual sample padding/alignment preserved
            packed_input_tensor.contiguous(),

            # Parameters required for sequence packing
            packed_seq_params,

            # Cumulative sequence lengths of original unpadded data
            cu_seqlens,

            # Cumulative sequence lengths after padding/alignment
            cu_seqlens_padded,
        )

    def _unpack_sequences(self, packed_tensor, cu_seqlens_padded, cp_gather=False):
        """Inverse of _pack_sequences: yield (chunk, full_seq_len) per sample.

        chunk is the CP-local slice of each sample; with cp_gather=True it is
        gathered back to the full sequence. Empty samples yield a zero-filled mock.
        """
        cp_size = mpu.get_context_parallel_world_size()
        seq_starts = cu_seqlens_padded[:-1] // cp_size
        seq_ends = cu_seqlens_padded[1:] // cp_size

        for seq_start, seq_end in zip(seq_starts, seq_ends):
            single_chunk = packed_tensor[:, seq_start:seq_end]
            local_seq_len = single_chunk.size(1)
            full_seq_len = local_seq_len * cp_size

            if full_seq_len == 0:
                full_seq_len = self._get_pad_factor()
                local_seq_len = max(1, full_seq_len // cp_size)
                new_shape = (1, local_seq_len) + packed_tensor.shape[2:]
                single_chunk = torch.zeros(
                    new_shape, dtype=packed_tensor.dtype, device=packed_tensor.device,
                )

            if cp_gather and cp_size > 1:
                single_chunk = context_parallel_gather(single_chunk, parallel_dim=1)
                full_seq_len = single_chunk.size(1)

            yield single_chunk, full_seq_len

    def _collect_r2_router_indices(self, vp_rank: int, cu_seqlens_padded=None, input_seq_len: Optional[int] = None):
        """Collect router top-k indices recorded during the R2 forward.

        Stacks and SP-gathers via collect_r2_router_indices, then unpacks per sample
        (with CP gather), pads to seq_length, and copies async to pinned CPU memory.
        Result layout: [mbs, seq_length, moe_layers_in_vp, topk].

        ``input_seq_len`` is the actual per-CP-rank input width of the recorded
        forward; under dynamic batching it is smaller than seq_length.
        """
        with torch.no_grad():
            layers_topk_idx = collect_r2_router_indices(self.model.config, vp_rank)
            if layers_topk_idx is None:
                return

            cp_size = mpu.get_context_parallel_world_size()

            if self.use_sequence_packing and cu_seqlens_padded is not None:
                # Packing path: extract per-sample with CP gather, then pad to seq_length.
                unpacked_list = []
                for single_topk, full_seq_len in self._unpack_sequences(
                    layers_topk_idx, cu_seqlens_padded, cp_gather=True
                ):
                    # Pad to self.seq_length: [1, seq_length, moe_layers_in_vp, topk]
                    single_topk = adjust_sequence_length(
                        single_topk, self.seq_length, full_seq_len, pad_value=0
                    )
                    unpacked_list.append(single_topk)
                # [mbs, seq_length, moe_layers_in_vp, topk]
                layers_topk_idx = torch.cat(unpacked_list, dim=0)
            else:
                # Non-packing path:
                # Data layout after SP gather is [S/cp_size, B] row-major (sequence-major),
                # NOT [B, S/cp_size] batch-major. Must reshape and permute correctly.
                # Under dynamic batching the micro-batch width is the actual input width,
                # not self.seq_length, so derive the layout from input_seq_len.
                tokens_all = layers_topk_idx.size(1)
                local_seq_len = input_seq_len if input_seq_len is not None else self.seq_length // cp_size
                assert tokens_all % local_seq_len == 0, (
                    f"[RouterReplay] R2 recorded tokens {tokens_all} not divisible by "
                    f"local seq len {local_seq_len} (cp_size={cp_size})"
                )
                mbs = tokens_all // local_seq_len

                # Step 1: Correct sequence-major reshape: [1, S/cp_size*B, L, topk] → [S/cp_size, B, L, topk]
                layers_topk_idx = layers_topk_idx.squeeze(0)
                layers_topk_idx = layers_topk_idx.view(
                    local_seq_len, mbs, -1, layers_topk_idx.size(-1)
                )
                # Step 2: Permute to batch-major: [B, S/cp_size, L, topk]
                layers_topk_idx = layers_topk_idx.permute(1, 0, 2, 3).contiguous()

                # Step 3: CP gather to reconstruct full sequence: [B, S_actual, L, topk]
                if cp_size > 1:
                    layers_topk_idx = context_parallel_gather(layers_topk_idx, parallel_dim=1)

                # Step 4: pad back to seq_length so downstream (finalize cat, broadcast,
                # dynamic-batching narrow) sees a uniform layout.
                full_width = local_seq_len * cp_size
                if full_width != self.seq_length:
                    layers_topk_idx = adjust_sequence_length(
                        layers_topk_idx, self.seq_length, full_width, pad_value=0
                    )

            # Pinned + non_blocking: async D2H; stream ordering keeps the source safe until reuse.
            cpu_topk_idx = torch.empty_like(layers_topk_idx, device="cpu", pin_memory=True)
            cpu_topk_idx.copy_(layers_topk_idx, non_blocking=True)
            self.router_topk_indices_list.append(cpu_topk_idx)

    def inner_forward_step(self, loss_func, data_iterator: Iterator[DataProto], model):
        data = next(data_iterator)
        input_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        labels = data.batch["labels"] if "labels" in data.batch else None  # labels is only used for sft
        packed_seq_params = None

        # Get loss_mask early, priority: final_response_mask > response_mask > (labels != IGNORE_INDEX) > ones
        # For MTP training, loss_mask must match input_ids length
        if "final_response_mask" in data.batch:
            loss_mask = data.batch["final_response_mask"].float()
        elif "response_mask" in data.batch:
            loss_mask = data.batch["response_mask"].float()
        elif labels is not None:
            loss_mask = (labels != IGNORE_INDEX).float()
        else:
            loss_mask = torch.ones_like(input_ids)

        # Ensure loss_mask length matches input_ids length
        # This is important for MTP training where mtp_labels has the same length as input_ids
        if loss_mask.shape[1] != input_ids.shape[1]:
            if loss_mask.shape[1] < input_ids.shape[1]:
                # loss_mask is shorter (e.g., sliced with [:, 1:]), pad to match
                pad_length = input_ids.shape[1] - loss_mask.shape[1]
                # Pad at the beginning (position 0) since slicing was [:, 1:]
                loss_mask = torch.nn.functional.pad(loss_mask, (pad_length, 0), value=0.0)
            else:
                # loss_mask is longer, truncate to match
                loss_mask = loss_mask[:, :input_ids.shape[1]]

        # Save attention_mask before packing may set it to None,
        # needed later for packing layers_topk_idx in router replay.
        orig_attention_mask = attention_mask

        if self.use_sequence_packing:
            packed_input_ids, packed_seq_params, cu_seqlens, cu_seqlens_padded = self._pack_sequences(
                input_ids, attention_mask,
            )
            if labels is not None:
                labels, _, _, _ = self._pack_sequences(labels, attention_mask, pad_val=IGNORE_INDEX)
            loss_mask, _, _, _ = self._pack_sequences(loss_mask, attention_mask, pad_val=0)
            # Multimodal packs CP internally (MultimodalEmbeddingMixin), so keep the un-packed tensors.
            if not self.is_multimodal:
                input_ids = packed_input_ids
                attention_mask = None
        else:
            cu_seqlens_padded = None
            input_ids = self._get_feature_on_this_cp_rank(input_ids, "input_ids")
            attention_mask = self._get_feature_on_this_cp_rank(attention_mask, "attention_mask")
            if labels is not None:
                labels = self._get_feature_on_this_cp_rank(labels, "labels")
            loss_mask = self._get_feature_on_this_cp_rank(loss_mask, "loss_mask")
        position_ids = None
        forward_args = dict(data.meta_info.get("forward_args") or {})
        if "position_ids" in data.batch and data.batch["position_ids"].dim() == 3:  # qwen-vl/omni mrope
            position_ids = data.batch["position_ids"]
            if position_ids.size(1) == 4:
                position_ids = position_ids[:, 1:, :].contiguous()  # (bsz, 4, seqlen) -> (bsz, 3, seqlen)
            position_ids = position_ids.transpose(0, 1)  # (bsz, C, seqlen) -> (C, bsz, seqlen)
        if "multi_modal_inputs" in data.non_tensor_batch:
            multi_modal_inputs = data.non_tensor_batch["multi_modal_inputs"]
            multi_modal_data = defaultdict(list)
            # mm inputs of some samples would be empty to allow text and mm
            # mixed data
            for sample_mm_inputs in multi_modal_inputs:
                for key in sample_mm_inputs.keys():
                    multi_modal_data[key].append(sample_mm_inputs[key])
            for key in multi_modal_data.keys():
                assert key not in forward_args
                mm_data = multi_modal_data[key]
                # All mm fields are not padded in collator currently and some should be padded first
                # pixel_values/pixel_values_videos for images/videos are concated for packing
                # input_features for audios with shape `(bs, freqs, frames)` should be padded before concat
                need_padding = any(t.shape[-1] != mm_data[0].shape[-1] for t in mm_data[1:])
                if need_padding:  # input_features/feature_attention_mask
                    max_mm_len = max(t.shape[-1] for t in mm_data)
                    for i, t in enumerate(mm_data):
                        mm_data[i] = torch.nn.functional.pad(t, (0, max_mm_len - t.shape[-1]), "constant", 0)
                # DataProto.to('cuda') in upper frame not work for non_tensor_batch
                forward_args[key] = torch.concat(multi_modal_data[key], dim=0).to(input_ids.device)
            forward_args.update({"force_vit_image": True})

        unwrapped_model = unwrap_model(model)
        if hasattr(unwrapped_model, "vp_stage"):
            vp_rank = unwrapped_model.vp_stage
        else:
            vp_rank = 0

        if self.enable_router_replay:
            # Restore REPLAY_FORWARD after previous microbatch's backward recompute left it as REPLAY_BACKWARD
            if RouterReplayHelper.is_replay_backward_action(self.model.config, vp_rank):
                for router in RouterReplayHelper.get_micro_batch_router_list(self.model.config, vp_rank):
                    router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

            if RouterReplayHelper.is_replay_forward_action(self.model.config, vp_rank):
                if "routed_experts" in data.batch:
                    layers_topk_idx = data.batch["routed_experts"]
                    _tail_shape = data.meta_info.get("_routed_experts_tail_shape")
                    if _tail_shape is not None and layers_topk_idx.dim() == 2:
                        layers_topk_idx = layers_topk_idx.reshape(layers_topk_idx.shape[0], *_tail_shape)
                    if self.use_sequence_packing:
                        layers_topk_idx, _, _, _ = self._pack_sequences(
                            layers_topk_idx, orig_attention_mask, pad_val=0, distinct_topk_padding=True,
                        )
                    else:
                        # Padding slots past each sample's valid length hold constant
                        # values (R2 zero-pad / clamped sentinels); identical topk slots
                        # dedupe in the MoE dispatcher and desync the EP all-to-all split
                        # sizes. Give each slot a distinct expert id, mirroring
                        # _pack_sequences(distinct_topk_padding=True).
                        if orig_attention_mask is not None:
                            valid_lens = orig_attention_mask.sum(dim=-1)
                            width = layers_topk_idx.size(1)
                            if width > 0 and (valid_lens < width).any():
                                pos = torch.arange(width, device=valid_lens.device)
                                pad_mask = pos[None, :] >= valid_lens[:, None]
                                distinct = torch.arange(
                                    layers_topk_idx.size(-1),
                                    dtype=layers_topk_idx.dtype,
                                    device=layers_topk_idx.device,
                                )
                                layers_topk_idx[pad_mask] = distinct
                        cp_size = mpu.get_context_parallel_world_size()
                        if cp_size > 1:
                            layers_topk_idx = self._get_feature_on_this_cp_rank(layers_topk_idx, "routed_experts")
                    set_router_replay_data(layers_topk_idx, self.model.config, vp_rank)
                else:
                    logger.warning(
                        f"[RouterReplay] action is REPLAY_FORWARD on vp_rank={vp_rank} "
                        f"but routed_experts not found in micro-batch"
                    )

        # megatron_llama_core need loss_mask to compute aux loss
        forward_args["loss_mask"] = loss_mask
        if self._bounded_rl_token_statistics_enabled:
            if "response_mask" not in data.batch:
                raise ValueError("bounded RL token statistics require response_mask")
            block_kwargs = dict(forward_args.get("extra_block_kwargs") or {})
            if "rl_token_labels" in block_kwargs:
                raise ValueError("rl_token_labels is owned by MegatronStrategy")
            block_kwargs["rl_token_labels"] = self._build_next_token_labels(
                input_ids, data.batch["response_mask"]
            )
            forward_args["extra_block_kwargs"] = block_kwargs

        output_tensor = model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels,
            packed_seq_params=packed_seq_params, **forward_args
        )

        if RouterReplayHelper.is_r2_record_action(self.model.config, vp_rank):
            # input_ids is already CP-sharded (or packed); the non-packing collect
            # path uses its width to decode the recorded layout under dynamic batching.
            self._collect_r2_router_indices(
                vp_rank=vp_rank, cu_seqlens_padded=cu_seqlens_padded, input_seq_len=input_ids.shape[1]
            )
        elif self.enable_router_replay and RouterReplayHelper.is_replay_forward_action(self.model.config, vp_rank):
            # Switch to REPLAY_BACKWARD so gradient-checkpointing recompute uses the same routing
            for router in RouterReplayHelper.get_micro_batch_router_list(self.model.config, vp_rank):
                router.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)

        if self.use_sequence_packing:
            def loss_wrapper(output_tensor):
                sample_iter = self._unpack_sequences(
                    output_tensor, cu_seqlens_padded, cp_gather=False
                )
                loss_result = torch.tensor(0.0, device=output_tensor.device)
                metrics_result_list = []
                num_samples = len(data)
                for i in range(num_samples):
                    single_output_tensor, full_seq_len = next(sample_iter)
                    single_data = data[i:i+1]
                    for key, val in single_data.batch.items():
                        if not isinstance(val, torch.Tensor):
                            continue
                        single_data.batch[key] = adjust_sequence_length(
                            val, full_seq_len, self.seq_length,
                            pad_value=IGNORE_INDEX if key in {'labels', 'labels_for_loss'} else 0
                        )
                    loss, metrics = loss_func(single_data, single_output_tensor)
                    loss_result += loss
                    for key, val in metrics.items():
                        if isinstance(val, torch.Tensor):
                            metrics[key] = adjust_sequence_length(val, self.seq_length, full_seq_len, pad_value=0)
                    metrics_result_list.append(metrics)
                    del single_output_tensor
                metrics_result_dict = collate_fn_to_dict_list(metrics_result_list)
                if self.worker_config.apply_loss_scale:
                    loss_result *= data.meta_info['loss_scale']
                return loss_result, reduce_metrics(metrics_result_dict)

            return output_tensor, loss_wrapper
        else:
            def loss_wrapper(output_tensor):
                loss, metrics = loss_func(data, output_tensor)
                if self.worker_config.apply_loss_scale:
                    loss *= data.meta_info['loss_scale']
                return loss, metrics
            return output_tensor, loss_wrapper

    def broadcast_parameter(self, *args, **kwargs):
        pass

    def _get_offload_backend(self):
        """Get or initialize the offload backend (lazy, based on config).

        The constructor group is only the default for puts without an explicit
        one: dense tensors replicate over DP (with CP). Expert tensors dedup
        over the expert-DP group, passed per put by the offload patch, so EP
        or ETP settings never shrink (or corrupt) dense deduplication."""
        if self._offload_backend is None:
            from roll.distributed.store.factory import get_offload_backend
            dp_group = mpu.get_data_parallel_group(with_context_parallel=True)
            dp_rank = dist.get_rank(dp_group)
            self._offload_backend = get_offload_backend(
                self.worker, dp_rank=dp_rank, dp_group=dp_group)
        return self._offload_backend

    def _get_offload_key_prefix(self) -> str:
        """Key namespace within this process-private store; rank topology is
        not needed for correctness (one rank per process), cluster_name is
        kept only for log readability."""
        return f"megatron_{self.worker.cluster_name}"

    def load_states(self, include=None):
        if include is None or OffloadStateType.model_params in include:
            backend = self._get_offload_backend()
            reload_megatron_no_grad_module(model_chunks=self.model.get_models(), backend=backend)

    def offload_states(self, include=None):
        if include is None or OffloadStateType.model_params in include:
            backend = self._get_offload_backend()
            key_prefix = self._get_offload_key_prefix()
            offload_megatron_no_grad_module(
                model_chunks=self.model.get_models(),
                backend=backend,
                key_prefix=key_prefix,
            )
        RotaryEmbedding.forward.cache_clear()
        current_platform.empty_cache()

        self._log_offload_summary()

    def op_compute_log_probs(self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        input_ids [[p, p, r, r, r, 0, 0]] p: prompt, r: response, 0: pad
        response_mask [[0, 0, 1, 1, 1, 0, 0]]
        """
        if self._bounded_rl_token_statistics_enabled:
            if logits.ndim != 3 or logits.shape[-1] != 2:
                raise ValueError("bounded RL token statistics require a [batch, sequence, 2] tensor")
            return logits[..., 0][:, :-1] * attention_mask[:, 1:]

        labels = self._build_next_token_labels(input_ids, attention_mask)
        labels = self._get_feature_on_this_cp_rank(labels, "labels")
        # compute logprobs in remove padding token
        log_probs = vocab_parallel_logprobs(
            logits, labels,
            use_fused_kernel=self.megatron_train_args.cross_entropy_loss_fusion
        )
        if mpu.get_context_parallel_world_size() > 1:
            log_probs = context_parallel_gather(log_probs, parallel_dim=1)
        log_probs = log_probs[:, :-1] * attention_mask[:, 1:]
        return log_probs

    def op_compute_entropy(self, logits: torch.Tensor, attention_mask: torch.Tensor):
        if self._bounded_rl_token_statistics_enabled:
            if logits.ndim != 3 or logits.shape[-1] != 2:
                raise ValueError("bounded RL token statistics require a [batch, sequence, 2] tensor")
            return logits[..., 1][:, :-1] * attention_mask[:, 1:]
        entropy = vocab_parallel_entropy(
            logits,
            used_fp32=self.worker_config.logits_in_fp32,
            use_fused_kernel=self.megatron_train_args.cross_entropy_loss_fusion
        )
        if mpu.get_context_parallel_world_size() > 1:
            entropy = context_parallel_gather(entropy, parallel_dim=1)
        entropy = entropy[:, :-1] * attention_mask[:, 1:]
        return entropy

    @staticmethod
    def _build_next_token_labels(input_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        labels = input_ids[:, 1:].clone()
        labels[response_mask[:, 1:] == 0] = 0
        return torch.cat((labels, torch.zeros_like(labels[:, :1])), dim=1)

    def _validate_bounded_rl_token_statistics_request(self):
        options = self.worker_config.model_args.model_config_kwargs
        enabled = options.get("bounded_rl_token_statistics", False)
        configured_temperature = options.get("rl_token_statistics_temperature", 1.0)
        if (
            isinstance(configured_temperature, bool)
            or not isinstance(configured_temperature, (int, float))
            or float(configured_temperature) != 1.0
        ):
            raise ValueError("bounded RL token statistics use raw logits and require temperature=1")
        targets = self.worker_config.model_args.lora_target or []
        if isinstance(targets, str):
            targets = [targets]
        output_head_adapter = any(
            target in {"all", "all-linear", "output_layer", "lm_head"}
            or "output_layer" in target
            or "lm_head" in target
            for target in targets
        )
        return validate_bounded_rl_token_statistics(
            enabled=enabled,
            chunk_size=options.get("vocab_loss_chunk_size", 256),
            context_parallel_size=self.megatron_train_args.context_parallel_size or 1,
            sequence_packing=self.use_sequence_packing,
            mtp_num_layers=self.megatron_train_args.mtp_num_layers,
            output_head_adapter=output_head_adapter,
        )

    def _configure_bounded_rl_token_statistics(self):
        request = self._bounded_rl_token_statistics_request
        self._bounded_rl_token_statistics_enabled = bool(request["enabled"])
        if not self._bounded_rl_token_statistics_enabled:
            return
        for model in self.model.get_models():
            candidate = unwrap_model(model)
            if getattr(candidate, "bounded_rl_token_statistics_capability", None) != "qwen4_exp_v1":
                raise NotImplementedError(
                    "bounded_rl_token_statistics requires the Qwen4Exp bounded-statistics capability"
                )
            validate_bounded_rl_token_statistics(
                enabled=True,
                chunk_size=request["chunk_size"],
                context_parallel_size=candidate.config.context_parallel_size,
                sequence_packing=self.use_sequence_packing,
                mtp_num_layers=candidate.config.mtp_num_layers,
                output_head_adapter=not candidate._is_plain_output_head(),
            )
            candidate.config.bounded_rl_token_statistics = True
            candidate.config.vocab_loss_chunk_size = request["chunk_size"]
        self.model.config.bounded_rl_token_statistics = True
        self.model.config.vocab_loss_chunk_size = request["chunk_size"]

    def op_compute_language_loss_from_logits(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            reduction: str = "mean"
    ):
        """
        Compute cross-entropy language modeling loss with TP and CP support.

        Handles causal next-token prediction with proper sequence boundary alignment
        in distributed training scenarios.

        Args:
            logits (torch.Tensor): Shape [batch_size, local_seq_len, vocab_size/tp_size].
                                  TP-sharded (vocab) and CP-sharded (sequence).
            targets (torch.Tensor): Shape [batch_size, global_seq_len].
                                   Global vocab IDs, padding marked with IGNORE_INDEX.
            reduction (str): "mean" or "sum". Default: "mean".

        Returns:
            tuple: (loss, token_count)
                - loss: Scalar tensor based on reduction method
                - token_count: int64 tensor, number of valid tokens

        Sequence Alignment:
            - No CP: Simple shift, logits[:, :-1] predicts targets[:, 1:]
            - With CP (2 chunks/rank): Handle chunk boundaries carefully
                * Chunk 0: logits[:, :chunk_size-1] → targets[:, 1:chunk_size]
                * Chunk 1: logits[:, chunk_size:-1] → targets[:, chunk_size+1:]

        Note:
            - vocab_parallel_cross_entropy handles TP all-reduce internally
            - CP all-reduce performed explicitly for loss_sum and token_count
            - Assumes 2 chunks per rank in CP mode for load balancing
        """
        cp_size = mpu.get_context_parallel_world_size()

        # Slice targets to current CP rank's sequence portion
        targets = self._get_feature_on_this_cp_rank(targets, "targets")

        if cp_size == 1:
            # Simple causal shift: logits[t] predicts targets[t+1]
            logits = logits[:, :-1, :].contiguous()
            targets = targets[:, 1:].contiguous()
        else:
            # CP mode: Handle chunk boundaries with load balancing
            local_seq_len = logits.size(1)
            chunk_size = local_seq_len // 2  # 2 chunks per rank

            # Chunk 0: Remove last position (its target is in Chunk 1)
            chunk_0_logits = logits[:, :chunk_size - 1, :]
            chunk_0_targets = targets[:, 1:chunk_size]

            # Chunk 1: Remove last position and skip first target (belongs to Chunk 0)
            chunk_1_logits = logits[:, chunk_size:-1, :]
            chunk_1_targets = targets[:, chunk_size + 1:]

            # Merge chunks
            logits = torch.cat([chunk_0_logits, chunk_1_logits], dim=1)
            targets = torch.cat([chunk_0_targets, chunk_1_targets], dim=1)

        # Transpose to sequence-first layout for Megatron CE
        logits_tp = logits.transpose(0, 1).contiguous()
        labels_tp = targets.transpose(0, 1).contiguous()

        # Compute per-token CE loss (handles TP all-reduce)
        loss_per_token = vocab_parallel_cross_entropy(
            logits_tp, labels_tp, label_smoothing=0.0
        )

        # Apply ignore_index mask
        mask = (labels_tp != IGNORE_INDEX)
        loss_sum_local = (loss_per_token * mask).sum()
        token_count_local = mask.sum()

        # All-reduce across CP ranks
        if cp_size > 1:
            cp_group = mpu.get_context_parallel_group()
            stats_tensor = torch.stack([
                loss_sum_local.float(),
                token_count_local.float()
            ], dim=0)
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM, group=cp_group)
            loss_sum, token_count = stats_tensor[0], stats_tensor[1]
        else:
            loss_sum = loss_sum_local.float()
            token_count = token_count_local.float()

        # Apply reduction
        if reduction == "sum":
            loss = loss_sum
        elif reduction == "mean":
            loss = loss_sum / torch.clamp(token_count, min=1.0)
        else:
            raise ValueError(f"Unsupported reduction: {reduction}. Use 'mean' or 'sum'.")

        return loss, token_count.to(torch.int64)

    def op_compute_topk_logits(
            self,
            logits: torch.Tensor,
            topk: int = 0
    ):
        """
        Compute top-k logits with memory-efficient two-stage approach for TP and CP training.

        Strategy:
            - topk=0: Gather full vocab across TP ranks
            - topk>0: Two-stage TopK (local → gather K values → global TopK → CP gather)

        Args:
            logits (torch.Tensor): Shape [batch_size, local_seq_len, local_vocab_size].
                                  TP-sharded along vocabulary.
            topk (int): 0=full vocab, >0=top-k mode.

        Returns:
            tuple: (values, indices)
                - topk=0: (logits [B, S, V], None)
                - topk>0: (values [B, S, K], indices [B, S, K] in global vocab space)

        Note:
            - Indices adjusted to global vocabulary space
            - Intermediate tensors deleted early
            - CP gathering after TP operations
        """

        tp_size = mpu.get_tensor_model_parallel_world_size()
        cp_size = mpu.get_context_parallel_world_size()

        # ========== TopK Mode: Two-Stage Memory Optimization ==========
        if topk > 0:
            # Stage 1: Local TopK on each TP rank's vocabulary shard
            # Memory reduction: [B, local_seq, local_vocab] -> [B, local_seq, K]
            local_topk_values, local_topk_indices = torch.topk(
                logits, k=topk, dim=-1, sorted=False
            )

            # Adjust indices to global vocabulary space
            # Each TP rank owns a contiguous vocabulary range [vocab_start, vocab_end)
            vocab_start_index = mpu.get_tensor_model_parallel_rank() * logits.shape[-1]
            local_topk_indices = local_topk_indices + vocab_start_index

            # Release original logits immediately to save memory
            del logits

            # Stage 2: Gather local TopK results across TP ranks
            # Memory: [B, local_seq, K] -> [B, local_seq, K * tp_world_size]
            # Only gather K values per rank instead of full vocabulary
            gathered_values = local_topk_values
            gathered_indices = local_topk_indices
            if tp_size > 1:
                gathered_values = gather_from_tensor_model_parallel_region(local_topk_values)
                gathered_indices = gather_from_tensor_model_parallel_region(local_topk_indices)
            del local_topk_values, local_topk_indices

            # Stage 3: Global TopK on gathered candidates
            # Select final top-k from K * tp_size candidates
            # Memory: [B, local_seq, K * tp_world_size] -> [B, local_seq, K]
            final_topk_values, topk_positions = torch.topk(
                gathered_values, k=topk, dim=-1, sorted=True
            )
            # Use topk_positions to gather corresponding global indices
            final_topk_indices = torch.gather(
                gathered_indices, dim=-1, index=topk_positions
            )
            del gathered_values, gathered_indices, topk_positions

            # Stage 4: CP gather for sequence parallel training
            if cp_size > 1:
                final_topk_values = context_parallel_gather(final_topk_values, parallel_dim=1)
                final_topk_indices = context_parallel_gather(final_topk_indices, parallel_dim=1)

            return final_topk_values, final_topk_indices

        # ========== Full Vocabulary Mode: Traditional Gather Path ==========
        result = logits
        # Gather full vocabulary across TP ranks
        if tp_size > 1:
            result = gather_from_tensor_model_parallel_region(result)

        # Gather across CP ranks for sequence parallelism
        if cp_size > 1:
            result = context_parallel_gather(result, parallel_dim=1)

        # Return full vocabulary logits
        if topk == 0:
            return result, None

        # Fallback: TopK mode without TP optimization (when TP is not used)
        topk_values, topk_indices = torch.topk(result, k=topk, dim=-1)
        del result

        return topk_values, topk_indices

    def op_compute_gather_by_teacher_indices(
            self,
            student_logits: torch.Tensor,
            teacher_indices: torch.Tensor
    ):
        """
        Gather student logits at teacher indices with TP support via sparse gather.

        Strategy:
            - No TP: Direct torch.gather
            - TP mode: Sparse gather + all-reduce
                1. Mask indices belonging to local vocab shard
                2. Gather local values, zero out non-local
                3. All-reduce sum across TP ranks

        Args:
            student_logits (torch.Tensor): Shape [batch_size, seq_len, local_vocab_size].
                                           TP-sharded along vocabulary.
            teacher_indices (torch.Tensor): Shape [batch_size, seq_len, k] or [batch_size, seq_len].
                                           Global vocabulary indices (not sharded).

        Returns:
            torch.Tensor: Gathered logits matching teacher_indices shape.

        Note:
            - Returns original logits if teacher_indices is None
            - Handles 2D/3D indices, restores original shape
            - Vocab range per rank: [tp_rank * local_vocab_size, (tp_rank+1) * local_vocab_size)
        """

        # Early return if no teacher indices provided
        if teacher_indices is None:
            return student_logits

        # Ensure indices are long type for indexing
        if teacher_indices.dtype != torch.long:
            teacher_indices = teacher_indices.long()

        # Handle 2D input by adding dimension (will be removed before return)
        squeeze_output = False
        if teacher_indices.dim() == 2:
            teacher_indices = teacher_indices.unsqueeze(-1)
            squeeze_output = True

        tp_world_size = mpu.get_tensor_model_parallel_world_size()

        # Non-TP mode: Direct gather operation
        if tp_world_size == 1:
            gathered = torch.gather(student_logits, dim=-1, index=teacher_indices)
            return gathered.squeeze(-1) if squeeze_output else gathered

        # ========== TP-Sharded Sparse Gather ==========
        tp_rank = mpu.get_tensor_model_parallel_rank()
        local_vocab_size = student_logits.shape[-1]

        # Calculate vocabulary range owned by current TP rank
        vocab_start = tp_rank * local_vocab_size
        vocab_end = vocab_start + local_vocab_size

        # Create mask for indices that belong to local vocabulary shard
        local_mask = (teacher_indices >= vocab_start) & (teacher_indices < vocab_end)

        # Convert global indices to local vocabulary space
        # Clamp to valid range to avoid index errors (non-local indices will be masked out)
        local_indices = teacher_indices - vocab_start
        local_indices = torch.clamp(local_indices, 0, local_vocab_size - 1)

        # Gather values from local vocabulary shard
        local_gathered = torch.gather(student_logits, dim=-1, index=local_indices)

        # Mask out values that don't belong to local vocabulary
        # Non-local positions are set to zero (will not contribute to final sum)
        local_gathered = torch.where(local_mask, local_gathered, torch.zeros_like(local_gathered))

        # All-reduce sum across TP ranks (fully differentiable)
        # Forward: Sum contributions from all ranks (only one rank contributes non-zero per index)
        # Backward: Each rank receives full gradient, but only masked portion affects local parameters
        gathered = reduce_from_tensor_model_parallel_region(local_gathered)

        # Restore original shape if input was 2D
        return gathered.squeeze(-1) if squeeze_output else gathered

    def op_compute_various_divergence(
            self,
            loss_callable, logits, teacher_topk_probs, teacher_topk_log_probs, teacher_topk_indices,
            teacher_topk_inf_mask, labels, attention_mask=None, reduction="mean"
    ):
        """
        Compute divergence losses (KL, JSD, RKL, etc.) with TP and CP support.

        Strategy:
            1. Slice teacher outputs to current CP rank's sequence
            2. Gather student logits at teacher's top-k indices (TP-aware)
            3. Compute per-token divergence loss
            4. Gather loss across CP ranks
            5. Apply padding mask and reduction

        Args:
            loss_callable (callable): Divergence function (KL/JSD/RKL).
                                     Takes: logits, teacher_probs, teacher_log_probs, teacher_inf_mask.
            logits (torch.Tensor): Shape [batch_size, local_seq_len, local_vocab_size].
                                  TP and CP sharded.
            teacher_topk_probs (torch.Tensor): Shape [batch_size, global_seq_len, topk].
                                              Full tensor (not sharded).
            teacher_topk_log_probs (torch.Tensor): Shape [batch_size, global_seq_len, topk].
            teacher_topk_indices (torch.Tensor): Shape [batch_size, global_seq_len, topk].
                                                Global vocabulary indices.
            teacher_topk_inf_mask (torch.Tensor): Shape [batch_size, global_seq_len, topk].
            labels (torch.Tensor): Shape [batch_size, global_seq_len].
                                  Padding marked with IGNORE_INDEX.
            attention_mask (torch.Tensor, optional): Shape [batch_size, global_seq_len].
                                                    0=padding. Used if labels is None.
            reduction (str): "mean", "sum", or "none".

        Returns:
            tuple: (loss, token_count)
                - loss: Scalar (mean/sum) or tensor [B, S] (none)
                - token_count: Scalar, number of valid tokens

        Note:
            - Teacher outputs sliced to CP rank's sequence
            - Student logits TP-sharded, handled by sparse gather
            - Token count from full sequence for correct normalization
        """

        # Preserve full tensors for final mask computation
        labels_full = labels
        attention_mask_full = attention_mask

        # (1) Slice teacher outputs to current CP rank's sequence portion
        # Each CP rank processes a contiguous chunk of the sequence
        if teacher_topk_probs is not None:
            teacher_topk_probs = self._get_feature_on_this_cp_rank(teacher_topk_probs, "teacher_topk_probs")
        if teacher_topk_indices is not None:
            teacher_topk_indices = self._get_feature_on_this_cp_rank(teacher_topk_indices, "teacher_topk_indices")
        if teacher_topk_log_probs is not None:
            teacher_topk_log_probs = self._get_feature_on_this_cp_rank(teacher_topk_log_probs,"teacher_topk_log_probs")
        if teacher_topk_inf_mask is not None:
            teacher_topk_inf_mask = self._get_feature_on_this_cp_rank(teacher_topk_inf_mask, "teacher_topk_inf_mask")

        # (2) Gather student logits at teacher's top-k indices
        # Handles TP-sharded logits with sparse gather operation
        # Input: [batch_size, local_seq_len, local_vocab_size] (TP-sharded)
        # Output: [batch_size, local_seq_len, topk] (aligned with teacher indices)
        full_logits = self.op_compute_gather_by_teacher_indices(logits, teacher_topk_indices)

        # (3) Compute per-token divergence loss
        # loss_callable computes divergence (e.g., KL, JSD) between student and teacher distributions
        # Returns: [batch_size, local_seq_len] per-token loss
        kld_per_token = loss_callable(
            logits=full_logits,
            teacher_probs=teacher_topk_probs,
            teacher_log_probs=teacher_topk_log_probs,
            teacher_inf_mask=teacher_topk_inf_mask,
        )

        # (4) Gather per-token loss across CP ranks to restore full sequence
        # Input: [batch_size, local_seq_len] (CP-sharded sequence)
        # Output: [batch_size, global_seq_len] (full sequence)
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size > 1:
            kld_per_token = context_parallel_gather(kld_per_token, parallel_dim=1)

        # (5) Compute total number of valid (non-padded) tokens
        # Uses full labels/attention_mask to count across entire batch
        if labels_full is not None:
            # Padding positions marked with IGNORE_INDEX in labels
            pad_mask = labels_full.eq(IGNORE_INDEX)
        else:
            # Alternatively use attention_mask where 0 indicates padding
            pad_mask = attention_mask_full.eq(0)
        token_count = (~pad_mask).sum().float()

        # (6) Early return for 'none' reduction (per-token loss)
        if reduction == 'none':
            return kld_per_token, token_count

        # (7) Apply padding mask and compute aggregated loss
        # Mask out padding positions by setting their loss to 0
        kld_masked = kld_per_token.masked_fill_(pad_mask, 0.0)
        loss_sum = kld_masked.sum()

        # (8) Return loss based on reduction method
        if reduction == "sum":
            # Return sum of loss over all valid tokens
            return loss_sum, token_count
        elif reduction == "mean":
            # Return average loss per valid token
            # Clamp token_count to avoid division by zero
            return loss_sum / token_count.clamp(min=1.0), token_count
        else:
            raise ValueError(f"Unsupported reduction: {reduction}. Use 'mean', 'sum', or 'none'.")

    def op_compute_language_loss(self, losses: torch.Tensor, labels: torch.Tensor, batch_num_tokens: int):
        labels = self._get_feature_on_this_cp_rank(labels, "labels")

        loss_mask = (labels != IGNORE_INDEX).float()
        loss_mask = loss_mask.view(-1).float()
        losses = torch.sum(losses.view(-1) * loss_mask)

        if mpu.get_context_parallel_world_size() > 1:
            loss_info = torch.cat([losses.view(1)])
            torch.distributed.all_reduce(
                loss_info, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group()
            )
            losses = loss_info[0]

        loss = losses.clone() / batch_num_tokens# clone to make sure loss is not a view

        metrics = {f"{self.worker_config.name}/loss@sum": loss.clone().detach().item()}

        return loss, metrics

class MegatronTrainStrategy(MegatronInferStrategy, TrainStrategy):
    strategy_name = "megatron_train"

    def __init__(self, worker: Worker):
        super().__init__(worker)
        self.models_wrapped = None
        self.models_unwrapped = None
        self.processor = None
        self._validate_access_integrity = True

        # 新增：Router Replay 配置（用于 R2 和 R3 的 REPLAY）
        # 注意：这里会覆盖父类的配置，因为训练阶段的行为不同
        self.router_replay_config = self.worker_config.router_replay
        self.enable_router_replay = (self.router_replay_config.mode in ["R2", "R3"])
        self.router_replay_mode = self.router_replay_config.mode

        if self.enable_router_replay:
            logger.info(f"Router Replay {self.router_replay_mode} mode: REPLAY enabled in MegatronTrainStrategy")

    def initialize(self, model_provider):
        self.seq_length = self.worker.pipeline_config.sequence_length
        self.weight_updaters: dict[str, MegatronWeightUpdater] = {}

        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.processor = default_processor_provider(model_args=self.worker_config.model_args)
        # model provider will initialize megatron distributed groups
        self.model: "VirtualModels" = model_provider(
            tokenizer=self.tokenizer,
            model_args=self.worker_config.model_args,
            training_args=self.megatron_train_args,
            is_trainable=True,
        )
        self.forward_backward_func = get_forward_backward_func()
        self.model.config.finalize_model_grads_func = finalize_model_grads
        if self.megatron_train_args.optimizer_cpu_offload and not self.megatron_train_args.overlap_grad_reduce:
            from roll.third_party.megatron.grad_sync_memory import FirstGradSyncCacheRelease

            self.model.config.finalize_model_grads_func = FirstGradSyncCacheRelease(finalize_model_grads)

        # Inject mtp_training_mode from WorkerConfig to model config
        if hasattr(self.worker_config, "mtp_training_mode"):
            self.model.config.mtp_training_mode = self.worker_config.mtp_training_mode

        # mtp_only: freeze the main model and keep only MTP parameters trainable.
        # Must happen before DDP wrapping so grad buckets and optimizer param groups
        # are built from trainable params only.
        if getattr(self.worker_config, "mtp_training_mode", "disabled") == "mtp_only":
            assert self.megatron_train_args.mtp_num_layers, (
                "mtp_training_mode='mtp_only' requires mtp_num_layers > 0 in strategy_config"
            )
            trainable_count, frozen_count = 0, 0
            for m in self.model.get_models():
                chunk_trainable, chunk_frozen = freeze_except_mtp(m)
                trainable_count += chunk_trainable
                frozen_count += chunk_frozen
            # With PP>1 MTP params only live on the last pipeline stage, so
            # validate the trainable count globally across the pipeline group.
            trainable_total = torch.tensor(
                [trainable_count], dtype=torch.long, device=torch.device("cuda", torch.cuda.current_device())
            )
            dist.all_reduce(trainable_total, group=mpu.get_pipeline_model_parallel_group())
            assert trainable_total.item() > 0, "mtp_only mode found no trainable MTP parameters"
            logger.info(
                f"mtp_only: froze {frozen_count} main-model parameters, "
                f"{trainable_count} MTP parameters trainable"
            )

        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=self.megatron_train_args.accumulate_allreduce_grads_in_fp32,
            overlap_grad_reduce=self.megatron_train_args.overlap_grad_reduce,
            use_distributed_optimizer=self.megatron_train_args.use_distributed_optimizer,
            check_for_nan_in_grad=self.megatron_train_args.check_for_nan_in_loss_and_grad,
            bucket_size=self.megatron_train_args.ddp_bucket_size,
        )
        self.models_wrapped = [
            DistributedDataParallel(
                config=m.config,
                ddp_config=ddp_config,
                module=m,
                # Turn off bucketing for model_chunk 2 onwards, since communication for these
                # model chunks is overlapped with compute anyway.
                disable_bucketing=(model_index > 0),
            )
            for model_index, m in enumerate(self.model.get_models())
        ]
        self.models_unwrapped = self.model.get_models()
        self._configure_bounded_rl_token_statistics()
        self.model.models = self.models_wrapped
        self.is_multimodal = self.processor is not None
        self._validate_vlm_packing_support()
        self._rebuild_router_replay_instances()

        params_dtype = (
            torch.float16
            if self.megatron_train_args.fp16
            else torch.bfloat16 if self.megatron_train_args.bf16 else torch.float32
        )
        optimizer_config = build_optimizer_config(
            OptimizerConfig,
            {
                "optimizer": self.megatron_train_args.optimizer,
                "lr": self.megatron_train_args.learning_rate,
                "min_lr": self.megatron_train_args.lr_scheduler_kwargs.get("min_lr", 0.0),
                "weight_decay": self.megatron_train_args.weight_decay,
                "adam_beta1": self.megatron_train_args.adam_beta1,
                "adam_beta2": self.megatron_train_args.adam_beta2,
                "adam_eps": self.megatron_train_args.adam_epsilon,
                "fp16": self.megatron_train_args.fp16,
                "bf16": self.megatron_train_args.bf16,
                "params_dtype": params_dtype,
                "use_distributed_optimizer": self.megatron_train_args.use_distributed_optimizer,
                "clip_grad": self.megatron_train_args.max_grad_norm,
            },
            self.megatron_train_args,
        )
        self.optimizer: MegatronOptimizer = get_megatron_optimizer(optimizer_config, self.models_wrapped)

        logger.info(f"megatron optimizer: {self.optimizer}")

        bind_megatron_offload_states_func(optimizer=self.optimizer)


        self.worker.rank_info.dp_rank = mpu.get_data_parallel_rank()
        self.worker.rank_info.dp_size = mpu.get_data_parallel_world_size()
        self.worker.rank_info.tp_rank = mpu.get_tensor_model_parallel_rank()
        self.worker.rank_info.tp_size = mpu.get_tensor_model_parallel_world_size()
        self.worker.rank_info.pp_rank = mpu.get_pipeline_model_parallel_rank()
        self.worker.rank_info.pp_size = mpu.get_pipeline_model_parallel_world_size()
        self.worker.rank_info.cp_size = mpu.get_context_parallel_world_size()
        self.worker.rank_info.cp_rank = mpu.get_context_parallel_rank()

        logger.info(f"max steps pipeline {self.worker_config.training_args.max_steps}")
        self.worker_config.training_args.max_steps = (
            self.worker_config.training_args.max_steps // self.worker.rank_info.dp_size
        )
        self.megatron_train_args.max_steps = self.worker_config.training_args.max_steps
        logger.info(f"max steps worker train {self.worker_config.training_args.max_steps}")

        self.scheduler = get_megatron_lr_scheduler(
            self.megatron_train_args, self.megatron_train_args.max_steps, optimizer=self.optimizer
        )

        if self.megatron_train_args.use_distributed_optimizer:
            self.save_strategy = FullyParallelSaveStrategyWrapper(
                streaming_save_strategy(),
                mpu.get_data_parallel_group(with_context_parallel=True),
                do_cache_distribution=True,
            )
            self.ckpt_sharding_metadata = build_sharded_state_dict_metadata(self.megatron_train_args)

        if self.megatron_train_args.overlap_grad_reduce:
            model_config = self.model.config
            assert model_config.no_sync_func is None, (
                "When overlap_grad_reduce is True, config.no_sync_func must be None; "
                "a custom no_sync_func is not supported when overlapping grad-reduce"
            )
            model_config.no_sync_func = [model_wrapped.no_sync for model_wrapped in self.models_wrapped]
            if len(self.models_wrapped) == 1:
                model_config.no_sync_func = model_config.no_sync_func[0]
            if self.megatron_train_args.delay_grad_reduce:
                model_config.grad_sync_func = [model_wrapped.start_grad_sync for model_wrapped in self.models_wrapped]
                if len(self.models_wrapped) == 1:
                    model_config.grad_sync_func = model_config.grad_sync_func[0]

        if (self.worker_config.use_dynamic_batching_in_train or self.worker_config.use_sequence_packing) and self.worker.rank_info.pp_size > 1:
            self.model.config.variable_seq_lengths = True
            logger.info("Set variable_seq_lengths to True when use dynamic batching and pipeline parallel.")

        logger.info(f"{self.model.get_models()}")
        if self.megatron_train_args.compile_warmup and self.worker.rank_info.pp_size > 1:
            compile_warmup_pipeline_stages(self)

        self._warmup_p2p_comms()
        dist.barrier()
        clear_memory(clear_host_memory=True)

    def train_step(self, batch: DataProto, loss_func: Callable):
        self.model.train()

        if self.enable_router_replay:
            assert "routed_experts" in batch.batch
            RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

        global_step = batch.meta_info.get("global_step", 0)
        is_offload_optimizer_states_in_train_step = batch.meta_info.get("is_offload_optimizer_states_in_train_step", True)
        batch.meta_info['batch_num_tokens'] = self._get_batch_num_tokens(batch, dp_group=mpu.get_data_parallel_group())
        batch.meta_info['global_valid_samples'] = self._get_global_valid_samples(batch, dp_group=mpu.get_data_parallel_group())

        if self.worker_config.use_dynamic_batching_in_train:
            micro_batches_list = list(make_micro_batch_iter_for_dynamic_batching(batch))
            num_microbatches = batch.meta_info["num_micro_batchs"]
            mini_batch_size = 1
        elif self.use_sequence_packing:
            vp_size = self.worker_config.strategy_args.strategy_config['virtual_pipeline_model_parallel_size']\
                if 'virtual_pipeline_model_parallel_size' in self.worker_config.strategy_args.strategy_config else 1
            # Flatten [B,S,L,K] -> [B,S*L*K] (zero-copy) for fast per-microbatch row
            # gather in the packer; restored in inner_forward_step.
            if "routed_experts" in batch.batch and batch.batch["routed_experts"].dim() >= 3:
                _re = batch.batch["routed_experts"]
                batch.meta_info["_routed_experts_tail_shape"] = list(_re.shape[1:])
                batch.batch["routed_experts"] = _re.reshape(_re.shape[0], -1)
            micro_batches_list = list(make_micro_batch_iter_for_sequence_packing(batch, tp_size=self.worker.rank_info.tp_size,
                                                                cp_size=self.worker.rank_info.cp_size,
                                                                vp_size=vp_size, is_train=True,
                                                                dp_group=mpu.get_data_parallel_group(with_context_parallel=True),
                                                                micro_batch_size=self.worker_config.training_args.per_device_train_batch_size,
                                                                                 config=self.worker_config.sequence_packing_args,
                                                                                 pp_size=self.worker.rank_info.pp_size))
            self._precompute_pack_layouts(micro_batches_list)
            num_microbatches = micro_batches_list[0].meta_info["num_micro_batchs"]
            mini_batch_size = 1
        else:
            mini_batch_size = self.worker_config.training_args.per_device_train_batch_size
            num_microbatches = batch.batch.batch_size[0] // self.worker_config.training_args.per_device_train_batch_size
            assert (
                num_microbatches == self.megatron_train_args.gradient_accumulation_steps
            ), f"num_microbatches={num_microbatches} gradient_accumulation_steps={self.megatron_train_args.gradient_accumulation_steps}"
            micro_batches_list = batch.chunk(chunks=num_microbatches)

        for micro_batch in micro_batches_list:
            micro_batch.meta_info['loss_scale'] = num_microbatches * mpu.get_data_parallel_world_size()
            micro_batch.meta_info['micro_batch_size'] = micro_batch.batch.batch_size[0]

        data_iterator = [iter(micro_batches_list) for _ in range(len(self.model))]

        metrics_tensors: List[Dict[str, "torch.Tensor"]] = self.forward_backward_func(
            forward_step_func=partial(self.inner_forward_step, loss_func),
            data_iterator=data_iterator,
            model=self.model.get_models(),
            num_microbatches=num_microbatches,
            seq_length=self.seq_length,
            micro_batch_size=mini_batch_size,
            forward_only=False,
        )

        # clear global router replay action
        if self.enable_router_replay:
            RouterReplay.clear_global_router_replay_action()
            RouterReplay.clear_global_indices()

        # 只有step的时候需要load optimizer states
        self.load_states(include=[OffloadStateType.optimizer_states])

        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()

        if is_offload_optimizer_states_in_train_step:
            self.offload_states(include=[OffloadStateType.optimizer_states])

        if update_successful:
            self.scheduler.step()
        else:
            raise NotImplementedError("megatron optimizer step failed!")

        # Clean up stale model parameters from backend after gradient update
        cleanup_ddp_buffers(self.optimizer, backend=self._get_offload_backend())

        for model in self.model:
            for bucket_group in model.bucket_groups + model.expert_parallel_bucket_groups:
                if hasattr(bucket_group, "per_param_grad_ready_counts") and hasattr(bucket_group, "is_first_batch"):
                    if bucket_group.is_first_batch and len(bucket_group.per_param_grad_ready_counts) > 0:
                        # Fill in any missing params with count=1 so the assertion passes.
                        for param in bucket_group.params:
                            if param not in bucket_group.per_param_grad_ready_counts:
                                bucket_group.per_param_grad_ready_counts[param] = 1
            model.zero_grad_buffer()
            # Offload/reload does not update cached_param_buffer_shard_list/cached_grad_buffer_shard_list,
            # resulting using old params in `start_param_sync`, which leads to wrong results. So we clear the cache.
            for bucket_group in model.bucket_groups + model.expert_parallel_bucket_groups:
                if hasattr(bucket_group, "cached_param_buffer_shard_list"):
                    bucket_group.cached_param_buffer_shard_list = [None] * len(bucket_group.buckets)
                if hasattr(bucket_group, "cached_grad_buffer_shard_list"):
                    bucket_group.cached_grad_buffer_shard_list = [None] * len(bucket_group.buckets)
        self.optimizer.zero_grad()

        metrics = {}
        for mini_metrics in metrics_tensors:
            append_to_dict(metrics, mini_metrics)

        metrics.update(
            {  # type of grad_norm differs between different mcore+te versions
                self.worker_config.name + "/" + "grad_norm": grad_norm.item()
                if isinstance(grad_norm, torch.Tensor)
                else grad_norm
            }
        )

        if self.model.config.num_moe_experts is not None and self.model.config.num_moe_experts > 1:
            tracker = get_moe_layer_wise_logging_tracker()
            # With PP>1 (e.g. mtp_only) some pp stages may track no aux losses at all;
            # gather the union of names across stages and pad zeros for missing ones so
            # every rank participates in the same reduce collectives.
            gathered_names = [None] * mpu.get_pipeline_model_parallel_world_size()
            dist.all_gather_object(
                gathered_names,
                list(tracker),
                group=mpu.get_pipeline_model_parallel_group(),
            )
            track_names = sorted({name for names in gathered_names for name in names})
            device = next(self.models_unwrapped[0].parameters()).device
            for name in set(track_names) - set(tracker):
                save_to_aux_losses_tracker(
                    name,
                    torch.zeros((), device=device),
                    layer_number=1,
                    num_layers=self.model.config.num_layers + (self.model.config.mtp_num_layers or 0),
                )
            reduce_aux_losses_tracker_across_ranks(track_names=track_names)
            tracker = get_moe_layer_wise_logging_tracker()
            loss_scale = 1 / self.megatron_train_args.gradient_accumulation_steps
            moe_losses = {
                self.worker_config.name + "/" + k: (v["values"].float() * loss_scale).mean().item()
                for k, v in tracker.items()
            }
            clear_aux_losses_tracker()
            metrics.update(moe_losses)

        if self.model.config.mtp_num_layers is not None and self.model.config.mtp_num_layers > 0:
            mtp_total_loss_dict = {}
            MTPLossLoggingHelper.reduce_loss_in_tracker()
            tracker = MTPLossLoggingHelper.tracker
            if "values" in tracker:
                loss_scale = 1 / self.megatron_train_args.gradient_accumulation_steps
                mtp_losses = tracker["values"] * loss_scale
                mtp_num_layers = mtp_losses.shape[0]
                for i in range(mtp_num_layers):
                    name = self.worker_config.name + "/" + f"mtp_{i+1} loss"
                    mtp_total_loss_dict[name] = mtp_losses[i].item()
                MTPLossLoggingHelper.clean_loss_in_tracker()
                metrics.update(mtp_total_loss_dict)
        seq_packing_metrics = batch.meta_info.pop('sequence_packing_metrics', {})
        if seq_packing_metrics:
            sp_prefix = f"sequence_packing/{self.worker_config.name}"
            metrics.update({f"{sp_prefix}/{k}": v for k, v in seq_packing_metrics.items()})
        return metrics

    @contextmanager
    def _materialized_model_params(self):
        """Model params resident for the duration; restores entry residency on exit.
        No-op when already resident. Needed for code that reads live params
        outside a state_offload_manger wrapper — setup_model_update gathers
        per-param weight meta (shape/dtype) and runs AFTER the worker-init
        offload, when param.data is empty."""
        offloaded = is_model_params_offloaded(self.optimizer)
        if offloaded:
            self.load_states(include=[OffloadStateType.model_params])
        try:
            yield
        finally:
            if offloaded:
                self.offload_states(include=[OffloadStateType.model_params])

    def model_update(self, model_update_name: str):
        if self.worker_config.strategy_args.strategy_config.get("offload_model_from_cpu_master", False):
            from roll.third_party.megatron.cpu_master_params import CpuMasterWeightProvider

            # Weights are streamed from the Hybrid optimizer's CPU FP32 masters;
            # the bf16 DDP image stays a placeholder and is never materialized.
            provider = CpuMasterWeightProvider(self.models_unwrapped, self.optimizer,
                                               device=current_platform.device_type)
            return self.weight_updaters[model_update_name].model_update(weight_provider=provider)
        with self._materialized_model_params():
            return self.weight_updaters[model_update_name].model_update()

    def get_model_update_load_kwargs(self):
        kwargs = super().get_model_update_load_kwargs()
        if self.worker_config.strategy_args.strategy_config.get("offload_model_from_cpu_master", False):
            # Export each weight from CPU optimizer masters as its bucket is
            # converted; keep the complete actor off GPU while vLLM wakes.
            kwargs["include"] = []
            return kwargs
        if is_peft_available() and isinstance(self.models_unwrapped[0], PeftModel):
            # Adapter export does not read the frozen backbone. Keep it on CPU
            # while the colocated inference engine wakes to admit the adapter.
            kwargs["include_frozen_parameters"] = False
        return kwargs

    def load_states(self, include=None, include_frozen_parameters=True):
        """Load states from offload backend. Already-resident sections no-op via
        the optimizer's and the model chunks' offloaded_states sets.

        ``include_frozen_parameters=False`` leaves the frozen (no-grad) backbone
        parked so a LoRA adapter export never wakes it alongside the inference engine.
        """
        backend = self._get_offload_backend()
        self.optimizer._offload_backend = backend

        megatron_include = _to_megatron_include(include)
        if megatron_include:
            self.optimizer.reload_states(include=megatron_include)

        if include_frozen_parameters and (include is None or OffloadStateType.model_params in include):
            reload_megatron_no_grad_module(model_chunks=self.model.get_models(), backend=backend)

    def offload_states(self, include=None, include_frozen_parameters=True):
        """Offload states to backend."""
        backend = self._get_offload_backend()
        key_prefix = self._get_offload_key_prefix()
        self.optimizer._offload_backend = backend
        self.optimizer._offload_key_prefix = key_prefix

        megatron_include = _to_megatron_include(include)

        if megatron_include:
            self.optimizer.offload_states(include=megatron_include)

        if include_frozen_parameters and (include is None or OffloadStateType.model_params in include):
            offload_megatron_no_grad_module(
                model_chunks=self.model.get_models(),
                backend=backend,
                key_prefix=key_prefix,
            )

        # empty_cache only when GPU memory is being handed back (phase boundary),
        # not on the per-iteration optimizer_states-only offload in train_step.
        if include is None or any(
            s in include for s in (OffloadStateType.model_params, OffloadStateType.other_params)
        ):
            RotaryEmbedding.forward.cache_clear()
            current_platform.empty_cache()

        self._log_offload_summary()

    def setup_model_update(self, infer_cluster, model_update_name: str):
        assert model_update_name not in self.weight_updaters
        # Updater setup gathers per-param weight meta from live module params,
        # and the worker offloads states at the end of initialize — materialize
        # the params first or every meta shape is recorded as (0).
        with self._materialized_model_params():
            self.weight_updaters[model_update_name] = MegatronWeightUpdater(
                pipeline_config=self.worker.pipeline_config,
                worker_config=self.worker_config,
                model_update_name=model_update_name,
                models_unwrapped=self.models_unwrapped,
                infer_cluster=infer_cluster,
            )

    def save_checkpoint(self, save_dir, global_step, ckpt_id, tag="checkpoint", local_state_path=None, **kwargs):
        logger.info(f"save_dir: {save_dir}")
        if local_state_path is None:
            local_state_path = save_dir
        with Timer("load") as load_timer:
            self.load_states()

        # Only during checkpoint: release pinned memory cache to free CPU memory
        # for the upcoming serialization. Normal training keeps the cache for reuse.
        clear_memory(clear_host_memory=True)

        is_last_step = kwargs.get("is_last_step", False)

        if self.megatron_train_args.save_hf_model:
            self.model.save_pretrained_as_hf(save_dir)
            clear_memory(clear_host_memory=True)

        ckpt_format = self.megatron_train_args.ckpt_format
        # save model and tokenizer
        if len(self.models_unwrapped) == 1:
            if is_peft_available() and isinstance(self.models_unwrapped[0], PeftModel):
                for adapter_name, peft_config in self.models_unwrapped[0].peft_config.items():
                    adapter_save_directory = os.path.join(save_dir, adapter_name)
                    peft_config.save_pretrained(adapter_save_directory)
                    peft_state_dict = get_peft_model_state_dict(
                        self.models_unwrapped[0], self.models_unwrapped[0].state_dict_for_save_checkpoint(), adapter_name
                    )
                    self.models_unwrapped[0].base_model.model.save_pretrained(
                        adapter_save_directory, state_dict={"model": peft_state_dict}
                    )
                self.models_unwrapped[0].config.save_pretrained(save_dir)
                # Training state lives at the checkpoint root, including the
                # frozen asset identity validated before restoring any adapter.
                asset_hook = getattr(self.models_unwrapped[0], "save_external_assets", None)
                if asset_hook is not None:
                    asset_hook(save_dir)
            else:
                self.models_unwrapped[0].save_pretrained(save_dir, ckpt_format=ckpt_format)
        else:
            state_dict = {f"model{i}": generate_model_state_dict(model, ckpt_format=ckpt_format) for i, model in enumerate(self.models_unwrapped)}
            self.models_unwrapped[0].save_pretrained(
                save_dir, state_dict=state_dict, ckpt_format=ckpt_format,
                external_asset_models=self.models_unwrapped,
            )
            del state_dict
        if dist.get_rank() == 0:
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(save_dir)
            if self.processor is not None:
                self.processor.save_pretrained(save_dir)
        clear_memory(clear_host_memory=True)

        # save optimizer
        if not self.megatron_train_args.save_only_model:
            if self.megatron_train_args.optimizer_cpu_offload:
                from roll.third_party.megatron.checkpoint_optimizer_state import prepare_cpu_adam_for_checkpoint

                prepare_cpu_adam_for_checkpoint(self.optimizer)
            checkpoint_dir = get_checkpoint_dir(save_dir,
                                                return_base_dir=self.megatron_train_args.use_distributed_optimizer)
            if self.megatron_train_args.use_distributed_optimizer:
                checkpoint_dir = os.path.join(checkpoint_dir, DIST_OPTIMIZER_DIR)
            os.makedirs(checkpoint_dir, exist_ok=True)
            if self.megatron_train_args.use_distributed_optimizer:
                model_shared_state_dict = self.model.sharded_state_dict()
                optimizer_state_dict = self.optimizer.sharded_state_dict(
                    model_shared_state_dict, metadata=self.ckpt_sharding_metadata
                )
                dist_checkpointing.save(
                    optimizer_state_dict,
                    checkpoint_dir=checkpoint_dir,
                    sharded_strategy=self.save_strategy,
                    async_sharded_save=False,
                    validate_access_integrity=self._validate_access_integrity,
                )
                del model_shared_state_dict, optimizer_state_dict
                self._validate_access_integrity = False
            elif not dist.is_initialized() or mpu.get_expert_data_parallel_world_size() == 0:
                torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, OPTIMIZER_NAME))
                logger.info(f"Saving optimizer state to {os.path.join(checkpoint_dir, OPTIMIZER_NAME)}")

            if dist.is_initialized():
                dist.barrier()

            # save lr_scheduler
            if dist.get_rank() == 0:
                torch.save(self.scheduler.state_dict(), os.path.join(save_dir, SCHEDULER_NAME))

            # save rng state
            rng_states = {
                "random_rng_state": random.getstate(),
                "np_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": current_platform.get_rng_state(),
                "rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states(),
            }
            rng_path = os.path.join(save_dir, RNG_STATE_DIR, f"rng_state_{dist.get_rank()}.pth")
            os.makedirs(os.path.dirname(rng_path), exist_ok=True)
            torch.save(rng_states, rng_path)
            del rng_states

        clear_memory(clear_host_memory=True)
        if dist.is_initialized():
            dist.barrier()

        if self.worker_config.checkpoint_config.get("async_upload", True) and not is_last_step:
            self.thread_executor.submit(self.checkpoint_manager.upload, ckpt_id=ckpt_id, local_state_path=local_state_path)
        else:
            self.checkpoint_manager.upload(ckpt_id=ckpt_id, local_state_path=local_state_path)

        metrics = {
            "load": load_timer.last,
        }
        clear_memory(clear_host_memory=True)
        return metrics

    def load_checkpoint(self, load_dir, tag="checkpoint", **kwargs):
        logger.info(f"load checkpoint from {load_dir}")

        # Every rank must restore its saved RNG before exact training can
        # continue. Agree on missing files before entering state-load collectives.
        rng_file = os.path.join(load_dir, RNG_STATE_DIR, f"rng_state_{dist.get_rank()}.pth")
        missing_rng = None if os.path.isfile(rng_file) else rng_file
        missing_rng_files = [None] * dist.get_world_size()
        dist.all_gather_object(missing_rng_files, missing_rng)
        missing_rng_files = [path for path in missing_rng_files if path is not None]
        if missing_rng_files:
            raise FileNotFoundError(f"Incomplete training checkpoint: missing RNG state {missing_rng_files}")

        # Frozen external assets are part of checkpoint identity. Validate them
        # before loading mutable training state, using the unwrapped model hooks.
        adapter_state_dict = None
        self.model.models = self.models_unwrapped
        try:
            self.model.load_external_assets(load_dir, external_asset_path=kwargs.get("external_asset_path"))
            adapter_error = None
            try:
                if len(self.models_unwrapped) == 1 and is_peft_available() and isinstance(
                    self.models_unwrapped[0], PeftModel
                ):
                    # Match save_checkpoint's per-adapter legacy checkpoint layout
                    # and VirtualModels.load_state_dict's adapter-name mapping.
                    adapter_state_dict = {}
                    for adapter_name in self.models_unwrapped[0].peft_config:
                        adapter_dir = os.path.join(load_dir, adapter_name)
                        adapter_state = load_state_dict_from_checkpoint(adapter_dir)
                        if adapter_state is None:
                            raise FileNotFoundError(f"Missing adapter checkpoint: {adapter_dir}")
                        expected = get_peft_model_state_dict(
                            self.models_unwrapped[0], self.models_unwrapped[0].state_dict_for_save_checkpoint(),
                            adapter_name,
                        )
                        actual = adapter_state.get("model", adapter_state)
                        missing = sorted(expected.keys() - actual.keys())
                        unexpected = sorted(actual.keys() - expected.keys())
                        mismatched = [key for key in expected.keys() & actual.keys()
                                      if isinstance(expected[key], torch.Tensor) and
                                      (not isinstance(actual[key], torch.Tensor) or
                                       expected[key].shape != actual[key].shape)]
                        if missing or unexpected or mismatched:
                            raise ValueError(
                                f"Invalid adapter checkpoint {adapter_name}: missing={missing}, "
                                f"unexpected={unexpected}, shape_mismatch={sorted(mismatched)}"
                            )
                        adapter_state_dict[adapter_name] = adapter_state
            except Exception as exc:
                adapter_error = exc

            # Adapter files are read locally. Agree on every rank's validation
            # result before any rank loads optimizer, scheduler, or model state.
            adapter_errors = [None] * dist.get_world_size()
            dist.all_gather_object(
                adapter_errors,
                None if adapter_error is None else f"{type(adapter_error).__name__}: {adapter_error}",
            )
            adapter_errors = [f"rank {rank}: {error}" for rank, error in enumerate(adapter_errors) if error]
            if adapter_errors:
                if adapter_error is not None:
                    raise adapter_error
                raise ValueError("Invalid adapter checkpoint across ranks: " + "; ".join(adapter_errors))
        finally:
            self.model.models = self.models_wrapped

        # load optimizer
        optimizer_checkpoint = get_checkpoint_dir(
            load_dir, iteration=1, return_base_dir=self.megatron_train_args.use_distributed_optimizer
        )
        if self.megatron_train_args.use_distributed_optimizer:
            optimizer_checkpoint = os.path.join(optimizer_checkpoint, DIST_OPTIMIZER_DIR)
        logger.info(
            f"Loading optimizer from {optimizer_checkpoint}, process_index: {self.megatron_train_args.process_index}"
        )

        if self.megatron_train_args.use_distributed_optimizer:
            model_shared_state_dict = self.model.sharded_state_dict()
            sharded_state_dict = self.optimizer.sharded_state_dict(
                model_shared_state_dict, is_loading=True, metadata=self.ckpt_sharding_metadata
            )
            load_strategy = dist_checkpointing.serialization.get_default_load_sharded_strategy(optimizer_checkpoint)
            load_strategy = FullyParallelLoadStrategyWrapper(
                load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
            )
            state_dict = dist_checkpointing.load(sharded_state_dict, optimizer_checkpoint, load_strategy)
        else:
            state_dict = torch.load(
                os.path.join(optimizer_checkpoint, OPTIMIZER_NAME), map_location=self.megatron_train_args.device,
                weights_only=False
            )
        self.optimizer.load_state_dict(state_dict)

        # load lr_scheduler
        scheduler_state = torch.load(os.path.join(load_dir, SCHEDULER_NAME))
        # Megatron's loader calls step(increment=saved_num_steps). Restoring
        # into an existing strategy must use the saved absolute position.
        self.scheduler.num_steps = 0
        self.scheduler.load_state_dict(scheduler_state)

        # load model state dict
        release_gradients = (
            self.megatron_train_args.use_distributed_optimizer
            and self.megatron_train_args.bounded_cpu_grad_staging
            and adapter_state_dict is None
        )
        if release_gradients:
            self.optimizer._offload_backend = self._get_offload_backend()
            self.optimizer._offload_key_prefix = self._get_offload_key_prefix()
        gradient_context = checkpoint_grad_buffer_offload(self.optimizer) if release_gradients else nullcontext()
        with gradient_context:
            self.model.models = self.models_unwrapped
            model_state = None
            try:
                # Distributed model checkpoints use the unwrapped model's
                # shard keys. Factories may allocate merged GDN/MLP tensors.
                model_state = (adapter_state_dict if adapter_state_dict is not None
                               else load_state_dict_from_checkpoint(load_dir, models=self.model))
                assert model_state is not None, "No model state_dict found in checkpoint."
                self.model.load_state_dict(model_state)
            finally:
                self.model.models = self.models_wrapped
                # Drop merge results before the context recreates gradients.
                del model_state

        # load rng state
        if os.path.exists(rng_file):
            logger.info(f"Loading rng states from {rng_file}")
            checkpoint_rng_state = torch.load(rng_file, weights_only=False)
            random.setstate(checkpoint_rng_state["random_rng_state"])
            np.random.set_state(checkpoint_rng_state["np_rng_state"])
            torch.set_rng_state(checkpoint_rng_state["torch_rng_state"])
            current_platform.set_rng_state(checkpoint_rng_state["cuda_rng_state"])
            # Check for empty states array
            if not checkpoint_rng_state["rng_tracker_states"]:
                raise KeyError
            tensor_parallel.get_cuda_rng_tracker().set_states(checkpoint_rng_state["rng_tracker_states"])
        else:
            raise FileNotFoundError(f"Checkpoint RNG state disappeared during restoration: {rng_file}")
