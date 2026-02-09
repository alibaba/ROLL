import math
import os
import random
from collections import defaultdict
from contextlib import nullcontext
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Tuple

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
)
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.core.packed_seq_params import PackedSeqParams

from mcore_adapter import TrainingArguments
from mcore_adapter.checkpointing import get_checkpoint_dir, load_state_dict_from_checkpoint
from mcore_adapter.parallel_functions import context_parallel_gather, vocab_parallel_logprobs
from mcore_adapter.patcher import patch_torch_find_nd_overlapping_shards, patch_torch_validate_global_plan
from mcore_adapter.trainer.utils import get_megatron_lr_scheduler
from roll.datasets.collator import collate_fn_to_dict_list
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_processor_provider, default_tokenizer_provider
from roll.platforms import current_platform
from roll.third_party.megatron.model_update import MegatronWeightUpdater
from roll.third_party.megatron.offload_states_patch import (
    MegatronOffloadStateType,
    bind_megatron_offload_states_func,
    offload_megatron_no_grad_module,
    reload_megatron_no_grad_module,
)
from roll.third_party.megatron.optimizer import get_megatron_optimizer
from roll.third_party.megatron.tensor_parallel import vocab_parallel_entropy
from roll.utils.constants import (
    DIST_OPTIMIZER_DIR,
    IGNORE_INDEX,
    OPTIMIZER_NAME,
    RNG_STATE_DIR,
    SCHEDULER_NAME,
)
from roll.utils.context_managers import disable_gradients
from roll.utils.dynamic_batching import make_micro_batch_iter_for_dynamic_batching
from roll.utils.functionals import append_to_dict, reduce_metrics, adjust_sequence_length
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType
from roll.utils.sequence_packing import make_micro_batch_iter_for_sequence_packing, restore_results_order


if TYPE_CHECKING:
    from mcore_adapter.models.model_factory import VirtualModels

logger = get_logger()


class MegatronInferStrategy(InferenceStrategy):
    strategy_name = "megatron_infer"

    def __init__(self, worker: Worker):
        #TODO remove the patches when the latest pytorch version > v2.9.1
        patch_torch_find_nd_overlapping_shards()
        patch_torch_validate_global_plan()
        super().__init__(worker)
        config_dict = self.worker_config.training_args.to_dict()
        config_dict.update(self.worker_config.strategy_args.strategy_config)
        # maybe put max_grad_norm into training_args as transformers do, rather
        # than in pipeline_config (PPOConfig)
        config_dict.update({"max_grad_norm": self.worker.pipeline_config.max_grad_norm})
        logger.info(f"training_args: {config_dict}")
        self.megatron_train_args = TrainingArguments(**config_dict)
        self.model = None
        self.forward_backward_func = None
        self.seq_length = None
        self.use_sequence_packing = self.worker_config.use_sequence_packing
        # hard to impl with offload states
        assert not self.megatron_train_args.overlap_param_gather, "overlap_param_gather is not supported"

    def initialize(self, model_provider):
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.model: "VirtualModels" = model_provider(
            tokenizer=self.tokenizer,
            model_args=self.worker_config.model_args,
            training_args=self.megatron_train_args,
            is_trainable=False,
        )
        self.model.config.finalize_model_grads_func = finalize_model_grads

        self.models_unwrapped = self.model.get_models()
        self.forward_backward_func = get_forward_backward_func()

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

        logger.info(f"{self.model.get_models()}")
        dist.barrier()

    def get_data_input(self, batch: DataProto):
        def broadcast_obj(obj, group):
            obj_list = [obj if dist.get_rank(group) == 0 else None]
            src_rank = dist.get_process_group_ranks(group)[0]
            dist.broadcast_object_list(obj_list, src=src_rank, group=group)
            return obj_list[0]

        # to avoid making side-effect on LLM, if want to broadcast non_tensor_batch,
        # set _broadcast_non_tensor_batch into meta_info
        broadcast_non_tensor_batch = batch.meta_info.get("_broadcast_non_tensor_batch", False)

        if mpu.get_pipeline_model_parallel_rank() == 0 and mpu.get_tensor_and_context_parallel_world_size() > 1:
            if broadcast_non_tensor_batch:
                tmp_batch = broadcast_obj(batch, mpu.get_tensor_and_context_parallel_group())
                batch.batch = tmp_batch.batch
                batch.non_tensor_batch = tmp_batch.non_tensor_batch
            else:
                batch.batch = broadcast_obj(batch.batch, mpu.get_tensor_and_context_parallel_group())

        if mpu.get_pipeline_model_parallel_world_size() > 1:
            if broadcast_non_tensor_batch:
                tmp_batch = broadcast_obj(batch, mpu.get_pipeline_model_parallel_group())
                batch.batch = tmp_batch.batch
                batch.non_tensor_batch = tmp_batch.non_tensor_batch
            else:
                batch.batch = broadcast_obj(batch.batch, mpu.get_pipeline_model_parallel_group())

        return batch

    def forward_step(
        self,
        batch: DataProto,
        forward_func: Callable[[DataProto, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
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
            micro_batches_list = list(
                make_micro_batch_iter_for_sequence_packing(batch, tp_size=self.worker.rank_info.tp_size,
                                                           cp_size=self.worker.rank_info.cp_size,
                                                           vp_size=vp_size, is_train=False,
                                                           dp_group=mpu.get_data_parallel_group(with_context_parallel=True),
                                                           micro_batch_size=batch.meta_info["micro_batch_size"],
                                                           config=self.worker_config.sequence_packing_args))
            num_microbatches = micro_batches_list[0].meta_info["num_micro_batchs"]
            micro_batch_size = 1
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

        if self.use_sequence_packing:
            results = restore_results_order(results, micro_batches_list[0].meta_info['partition_indices_list'],
                                  self.worker_config.sequence_packing_args)


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

    def _pack_sequences(self, input_tensor, attention_mask, pad_packed_seq_to=None, pad_val=0):
        """
        Pack multiple sequences into a single continuous sequence by removing padding.

        Implements sequence packing for efficient batch processing with variable-length sequences.
        Removes per-sample padding and concatenates sequences while maintaining cumulative length info.

        Args:
            input_tensor (torch.Tensor): Shape [batch_size, seq_len, ...], padded sequences.
            attention_mask (torch.Tensor): Shape [batch_size, seq_len], 1=valid, 0=padding.
            pad_packed_seq_to (int, optional): Target length for packed sequence. Defaults to None.
            pad_val (int): Padding value. Defaults to 0.

        Returns:
            tuple: (packed_input_tensor, packed_seq_params, cu_seqlens, cu_seqlens_padded)
                - packed_input_tensor: Shape [1, total_packed_length, ...], ready for current CP rank
                - packed_seq_params: PackedSeqParams with cumulative lengths and max_seqlen
                - cu_seqlens: Shape [batch_size + 1], cumulative lengths of original sequences
                - cu_seqlens_padded: Shape [batch_size + 1], cumulative lengths after alignment

        Note:
            - Sequences padded to alignment boundaries if pad_factor > 1 or pad_packed_seq_to is set
            - For CP training, sequences distributed across CP ranks
            - attention_mask not needed after packing
        """

        batch_size = input_tensor.shape[0]
        seq_lens = attention_mask.sum(dim=-1)
        pad_factor = self._get_pad_factor()

        # Remove padding from each sequence
        # Note: attention_mask is not needed in sequence packing mode
        input_tensor_unpadded = [input_tensor[b][:seq_lens[b]] for b in range(batch_size)]

        # Build cumulative sequence lengths
        cu_seqlens = [0]
        cu_seqlens_padded = ([0] if pad_factor > 1 or pad_packed_seq_to is not None
                             else None
                             )

        # Calculate cumulative lengths for both original and padded sequences
        for b in range(batch_size):
            seq_len = seq_lens[b].item() if torch.is_tensor(seq_lens[b]) else seq_lens[b]
            cu_seqlens.append(cu_seqlens[-1] + seq_len)
            if pad_factor > 1 or pad_packed_seq_to is not None:
                # Pad sequence length to multiple of pad_factor
                padded_seq_len = ((seq_len + pad_factor - 1) // pad_factor) * pad_factor
                cu_seqlens_padded.append(cu_seqlens_padded[-1] + padded_seq_len)

        # Convert to tensors
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=current_platform.device_type)
        if pad_factor > 1 or pad_packed_seq_to is not None:
            cu_seqlens_padded = torch.tensor(cu_seqlens_padded, dtype=torch.int32, device=current_platform.device_type)
            if pad_packed_seq_to is not None:
                cu_seqlens_padded[-1] = pad_packed_seq_to

        # Calculate maximum sequence length
        if pad_factor > 1 or pad_packed_seq_to is not None:
            seq_lens_padded = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
            max_seqlen = seq_lens_padded.max().item()
        else:
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            max_seqlen = seq_lens.max().item()

        cp_size = mpu.get_context_parallel_world_size()

        # Track running sequence length for padding
        running_seq_len = 0
        all_input_tensor_padded = []
        padded_tokens = []
        for b in range(batch_size):
            seq_len = seq_lens[b].item() if torch.is_tensor(seq_lens[b]) else seq_lens[b]
            if b == batch_size - 1 and pad_packed_seq_to is not None:
                # Different from original implementation: calculate remaining length
                padded_seq_len = pad_packed_seq_to - running_seq_len
            else:
                # Align to pad_factor boundary
                padded_seq_len = ((seq_len + pad_factor - 1) // pad_factor) * pad_factor

            running_seq_len += padded_seq_len

            seq_tokens = input_tensor_unpadded[b]

            # Pad sequence if needed
            if padded_seq_len > seq_len:
                seq_tokens = torch.nn.functional.pad(
                    seq_tokens, (0, padded_seq_len - seq_len), value=pad_val
                )
            all_input_tensor_padded.append(seq_tokens)

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
        all_input_tensor_padded = torch.cat(all_input_tensor_padded, dim=0).unsqueeze(0)

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

    def _unpack_sequences(self, output_tensor, cu_seqlens_padded):
        """
        Unpack concatenated sequences into individual padded sequences.
        """
        cp_size = mpu.get_context_parallel_world_size()
        seq_starts = cu_seqlens_padded[:-1] // cp_size
        seq_ends = cu_seqlens_padded[1:] // cp_size

        for seq_idx, (seq_start, seq_end) in enumerate(zip(seq_starts, seq_ends)):
            local_chunk = output_tensor[:, seq_start:seq_end]
            yield local_chunk

    def inner_forward_step(self, loss_func, data_iterator: Iterator[DataProto], model):
        data = next(data_iterator)
        input_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        labels = data.batch["labels"] if "labels" in data.batch else None  # labels is only used for sft
        packed_seq_params = None

        if self.use_sequence_packing:
            input_ids, packed_seq_params, cu_seqlens, cu_seqlens_padded = self._pack_sequences(
                input_ids, attention_mask,
            )
            if labels is not None:
                labels, _, _, _ = self._pack_sequences(labels, attention_mask, pad_val=IGNORE_INDEX)
            attention_mask = None
        else:
            input_ids = self._get_feature_on_this_cp_rank(input_ids, "input_ids")
            attention_mask = self._get_feature_on_this_cp_rank(attention_mask, "attention_mask")
            if labels is not None:
                labels = self._get_feature_on_this_cp_rank(labels, "labels")
        position_ids = None
        # attention_mask: SelfAttention defalt to te DotProductAttention with
        # AttnMaskType.causal in which attention_mask would not be used, pass
        # it mainly for moe aux loss without pad token and it is 2D
        # position_ids: not used in LLM
        # While MCA Qwen2VlModel requires 4D attention_mask, and
        # attention_mask and position_ids would be chunked for cp with dim 2 as
        # seq dim in it if they are provided
        forward_args = data.meta_info.get("forward_args", {})
        if "position_ids" in data.batch.keys() and data.batch["position_ids"].dim() == 3:  # qwen2vl mrope
            # not support MoE VLM, not used temperarily
            attention_mask = None
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
                # DataProto.to('cuda') in upper frame not work for non_tensor_batch
                forward_args[key] = torch.concat(multi_modal_data[key], dim=0).to(input_ids.device)
            forward_args.update({"force_vit_image": True})

        # megatron_llama_core need loss_mask to compute aux loss
        if "loss_mask" not in forward_args:
            if labels is not None:
                forward_args["loss_mask"] = (labels != IGNORE_INDEX).float()
            else:
                forward_args["loss_mask"] = torch.ones_like(input_ids)

        output_tensor = model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels,
            packed_seq_params=packed_seq_params, **forward_args
        )

        if self.use_sequence_packing:
            cp_size = mpu.get_context_parallel_world_size()
            def loss_wrapper(output_tensor):
                unpacked_output_iter = self._unpack_sequences(
                    output_tensor,
                    cu_seqlens_padded,
                )
                loss_result = torch.tensor(0.0, device=output_tensor.device)
                metrics_result_list = []
                num_samples = len(data)
                for i in range(num_samples):
                    single_output_tensor = next(unpacked_output_iter)
                    full_seq_len = single_output_tensor.size(1) * cp_size
                    if full_seq_len == 0:
                    # Create a mock output tensor when the sample is empty to ensure the subsequent pipeline works correctly.
                        full_seq_len = self._get_pad_factor()
                        local_seq_len = max(1, full_seq_len // cp_size)
                        new_shape = list(single_output_tensor.shape)
                        new_shape[1] = local_seq_len
                        single_output_tensor = torch.zeros(new_shape, dtype=single_output_tensor.dtype,
                                                           device=single_output_tensor.device)
                    single_data = data[i:i+1]
                    for key, val in single_data.batch.items():
                        single_data.batch[key] = adjust_sequence_length(val, full_seq_len, self.seq_length, pad_value=IGNORE_INDEX
                                                                  if key in {'labels', 'labels_for_loss'} else 0)
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

    def load_states(self, include=None, non_blocking=False):
        reload_megatron_no_grad_module(model_chunks=self.model.get_models())

    def offload_states(self, include=None, non_blocking=False):
        if include is None or OffloadStateType.model_params in include:
            offload_megatron_no_grad_module(model_chunks=self.model.get_models())
        RotaryEmbedding.forward.cache_clear()
        current_platform.empty_cache()

    def op_compute_log_probs(self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        input_ids [[p, p, r, r, r, 0, 0]] p: prompt, r: response, 0: pad
        response_mask [[0, 0, 1, 1, 1, 0, 0]]
        """
        ori_seq_length = attention_mask.size(1)
        cp_size = mpu.get_context_parallel_world_size()
        seq_len = ori_seq_length

        labels: torch.Tensor = input_ids[:, 1:].clone()
        labels[attention_mask[:, 1:seq_len] == 0] = 0  # avoid invalid token id
        # TODO: don't pad here but process this shift after generation
        labels = torch.cat([labels, torch.zeros_like(labels[:, :1])], dim=1)
        labels = self._get_feature_on_this_cp_rank(labels, "labels")
        # compute logprobs in remove padding token
        log_probs = vocab_parallel_logprobs(logits, labels)
        if mpu.get_context_parallel_world_size() > 1:
            log_probs = context_parallel_gather(log_probs, parallel_dim=1)
        log_probs = log_probs[:, :-1] * attention_mask[:, 1:]
        return log_probs

    def op_compute_entropy(self, logits: torch.Tensor, attention_mask: torch.Tensor):
        if self.worker_config.logits_in_fp32:
            logits = logits.float()
        entropy = vocab_parallel_entropy(logits)
        if mpu.get_context_parallel_world_size() > 1:
            entropy = context_parallel_gather(entropy, parallel_dim=1)
        entropy = entropy[:, :-1] * attention_mask[:, 1:]
        return entropy

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
        self.model.models = self.models_wrapped

        params_dtype = (
            torch.float16
            if self.megatron_train_args.fp16
            else torch.bfloat16 if self.megatron_train_args.bf16 else torch.float32
        )
        optimizer_config = OptimizerConfig(
            optimizer=self.megatron_train_args.optimizer,
            lr=self.megatron_train_args.learning_rate,
            min_lr=self.megatron_train_args.lr_scheduler_kwargs.get("min_lr", 0.0),
            weight_decay=self.megatron_train_args.weight_decay,
            adam_beta1=self.megatron_train_args.adam_beta1,
            adam_beta2=self.megatron_train_args.adam_beta2,
            adam_eps=self.megatron_train_args.adam_epsilon,
            fp16=self.megatron_train_args.fp16,
            bf16=self.megatron_train_args.bf16,
            params_dtype=params_dtype,
            use_distributed_optimizer=self.megatron_train_args.use_distributed_optimizer,
            clip_grad=self.megatron_train_args.max_grad_norm,
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
                dist_checkpointing.serialization.get_default_save_sharded_strategy(),
                mpu.get_data_parallel_group(with_context_parallel=True),
                do_cache_distribution=True,
            )

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

        if (self.worker_config.use_dynamic_batching_in_train or self.worker_config.use_sequence_packing or
            self.worker_config.use_sequence_packing) and self.worker.rank_info.pp_size > 1:
            self.model.config.variable_seq_lengths = True
            logger.info("Set variable_seq_lengths to True when use dynamic batching and pipeline parallel.")

        logger.info(f"{self.model.get_models()}")
        dist.barrier()

    def train_step(self, batch: DataProto, loss_func: Callable):
        self.model.train()

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
            micro_batches_list = list(make_micro_batch_iter_for_sequence_packing(batch, tp_size=self.worker.rank_info.tp_size,
                                                                cp_size=self.worker.rank_info.cp_size,
                                                                vp_size=vp_size, is_train=True,
                                                                dp_group=mpu.get_data_parallel_group(with_context_parallel=True),
                                                                micro_batch_size=self.worker_config.training_args.per_device_train_batch_size,
                                                                                 config=self.worker_config.sequence_packing_args))
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

        # 只有step的时候需要load optimizer states
        self.load_states(include=[OffloadStateType.optimizer_states])

        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
        if is_offload_optimizer_states_in_train_step:
            self.offload_states(include=[OffloadStateType.optimizer_states], non_blocking=True)

        if update_successful:
            self.scheduler.step()
        else:
            raise NotImplementedError("megatron optimizer step failed!")

        for model in self.model:
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

        metrics.update({self.worker_config.name + "/" + "grad_norm": grad_norm})

        if self.model.config.num_moe_experts is not None and self.model.config.num_moe_experts > 1:
            reduce_aux_losses_tracker_across_ranks()
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
        return metrics

    def model_update(self, model_update_name: str):
        return self.weight_updaters[model_update_name].model_update()

    def load_states(self, include=None, non_blocking=False):
        if include is not None:
            include_states = []
            if OffloadStateType.model_params in include:
                reload_megatron_no_grad_module(model_chunks=self.model.get_models())
                include_states.append(MegatronOffloadStateType.model_params)
            if OffloadStateType.other_params in include:
                include_states.append(MegatronOffloadStateType.other_params)
            if OffloadStateType.optimizer_states in include:
                include_states.append(MegatronOffloadStateType.optimizer_states)
            include = include_states
        self.optimizer.reload_states(include=include, non_blocking=non_blocking)

    def offload_states(self, include=None, non_blocking=False, pin_memory=True):
        if include is not None:
            include_states = []
            if OffloadStateType.model_params in include:
                offload_megatron_no_grad_module(model_chunks=self.model.get_models(), pin_memory=pin_memory)
                include_states.append(MegatronOffloadStateType.model_params)
            if OffloadStateType.other_params in include:
                include_states.append(MegatronOffloadStateType.other_params)
            if OffloadStateType.optimizer_states in include:
                include_states.append(MegatronOffloadStateType.optimizer_states)
            include = include_states
        self.optimizer.offload_states(include=include, non_blocking=non_blocking, pin_memory=pin_memory)
        RotaryEmbedding.forward.cache_clear()
        current_platform.empty_cache()

    def setup_model_update(self, infer_cluster, model_update_name: str):
        assert model_update_name not in self.weight_updaters
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

        is_last_step = kwargs.get("is_last_step", False)

        if self.megatron_train_args.save_hf_model:
            self.model.save_pretrained_as_hf(save_dir)

        # save model and tokenizer
        if len(self.models_unwrapped) == 1:
            self.models_unwrapped[0].save_pretrained(save_dir)
        else:
            state_dict = {f"model{i}": model.state_dict_for_save_checkpoint() for i, model in
                          enumerate(self.models_unwrapped)}
            self.models_unwrapped[0].save_pretrained(save_dir, state_dict=state_dict)
        if dist.get_rank() == 0:
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(save_dir)
            if self.processor is not None:
                self.processor.save_pretrained(save_dir)

        # save optimizer
        checkpoint_dir = get_checkpoint_dir(save_dir,
                                            return_base_dir=self.megatron_train_args.use_distributed_optimizer)
        if self.megatron_train_args.use_distributed_optimizer:
            checkpoint_dir = os.path.join(checkpoint_dir, DIST_OPTIMIZER_DIR)
        os.makedirs(checkpoint_dir, exist_ok=True)
        if self.megatron_train_args.use_distributed_optimizer:
            model_shared_state_dict = self.model.sharded_state_dict()
            optimizer_state_dict = self.optimizer.sharded_state_dict(model_shared_state_dict,
                                                                     sharding_type="fully_sharded_model_space")
            dist_checkpointing.save(
                optimizer_state_dict,
                checkpoint_dir=checkpoint_dir,
                sharded_strategy=self.save_strategy,
                async_sharded_save=False,
                validate_access_integrity=self._validate_access_integrity,
            )
            self._validate_access_integrity = False
        elif not dist.is_initialized() or mpu.get_data_modulo_expert_parallel_rank() == 0:
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
        rgn_path = os.path.join(save_dir, RNG_STATE_DIR, f"rng_state_{dist.get_rank()}.pth")
        os.makedirs(os.path.dirname(rgn_path), exist_ok=True)
        torch.save(rng_states, rgn_path)

        if self.worker_config.checkpoint_config.get("async_upload", True) and not is_last_step:
            self.thread_executor.submit(self.checkpoint_manager.upload, ckpt_id=ckpt_id, local_state_path=local_state_path)
        else:
            self.checkpoint_manager.upload(ckpt_id=ckpt_id, local_state_path=local_state_path)

        metrics = {
            "load": load_timer.last,
        }
        return metrics

    def load_checkpoint(self, load_dir, tag="checkpoint", **kwargs):
        logger.info(f"load checkpoint from {load_dir}")

        # load optimizer
        optimizer_checkpoint = get_checkpoint_dir(
            load_dir, iteration=1, return_base_dir=self.megatron_train_args.use_distributed_optimizer
        )
        if self.megatron_train_args.use_distributed_optimizer:
            optimizer_checkpoint = os.path.join(optimizer_checkpoint, DIST_OPTIMIZER_DIR)
        logger.info(
            f"Loading optimizer from {optimizer_checkpoint}, process_index: {self.megatron_train_args.process_index}"
        )

        self.offload_states()

        if self.megatron_train_args.use_distributed_optimizer:
            model_shared_state_dict = self.model.sharded_state_dict()
            sharded_state_dict = self.optimizer.sharded_state_dict(
                model_shared_state_dict, is_loading=True, sharding_type="fully_sharded_model_space"
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
        self.scheduler.load_state_dict(torch.load(os.path.join(load_dir, SCHEDULER_NAME)))

        # load model state dict
        state_dict = load_state_dict_from_checkpoint(load_dir)
        assert state_dict is not None, "No model state_dict found in checkpoint."
        self.model.models = self.models_unwrapped
        self.model.load_state_dict(state_dict)
        self.model.models = self.models_wrapped

        # load rng state
        rng_file = os.path.join(load_dir, RNG_STATE_DIR, f"rng_state_{dist.get_rank()}.pth")
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
            logger.info(f"not load rng state, not found file: {rng_file}")

        self.load_states()
