import math
import os
import random
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, Iterator, List, Tuple

import numpy as np
import ray
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
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region
from megatron.core.transformer.moe.moe_utils import (
    clear_aux_losses_tracker,
    get_moe_layer_wise_logging_tracker,
    reduce_aux_losses_tracker_across_ranks,
)

from mcore_adapter import TrainingArguments
from mcore_adapter.checkpointing import get_checkpoint_dir, load_state_dict_from_checkpoint
from mcore_adapter.parallel_functions import context_parallel_gather, vocab_parallel_logprobs
from mcore_adapter.trainer.utils import get_megatron_lr_scheduler
from roll.datasets.collator import collate_fn_to_dict_list
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.driver_utils import Barrier
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_processor_provider, default_tokenizer_provider
from roll.third_party.megatron.offload_states_patch import (
    MegatronOffloadStateType,
    bind_megatron_offload_states_func,
    offload_megatron_no_grad_module,
    reload_megatron_no_grad_module,
)
from roll.third_party.megatron.optimizer import get_megatron_optimizer
from roll.third_party.megatron.tensor_parallel import vocab_parallel_entropy
from roll.utils.collective import collective
from roll.utils.constants import DIST_OPTIMIZER_DIR, IGNORE_INDEX, OPTIMIZER_NAME, RNG_STATE_DIR, SCHEDULER_NAME, RAY_NAMESPACE, BARRIER_NAME
from roll.utils.context_managers import disable_gradients
from roll.utils.functionals import append_to_dict
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType


logger = get_logger()


class MegatronInferStrategy(InferenceStrategy):
    strategy_name = "megatron_infer"

    def __init__(self, worker: Worker):
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
        self.use_remove_padding = self.worker_config.use_remove_padding
        # hard to impl with offload states
        assert not self.megatron_train_args.overlap_param_gather, "overlap_param_gather is not supported"
        if self.worker_config.use_remove_padding:
            assert self.megatron_train_args.allow_variable_seq_lengths(), "when use_remove_padding=True, must set variable_seq_lengths=True for megatron."

    def initialize(self, model_provider):
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.model = model_provider(
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
        batch_size = batch.batch.batch_size[0]
        micro_batch_size = batch.meta_info["micro_batch_size"]
        output_on_all_tp_ranks = batch.meta_info.get("output_on_all_tp_ranks", False)
        num_microbatches = max(batch_size // micro_batch_size, 1)
        micro_batches = batch.chunk(chunks=num_microbatches)
        data_iterator = [iter(micro_batches) for _ in range(len(self.model))]
        with disable_gradients(models=self.model.get_models()):
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
        results = collate_fn_to_dict_list(losses_reduced)

        if not (
                (self.worker.rank_info.tp_rank == 0 or output_on_all_tp_ranks)
                and self.worker.rank_info.cp_rank == 0
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

    def inner_forward_step(self, loss_func, data_iterator: Iterator[DataProto], model):
        data = next(data_iterator)
        input_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        if self.use_remove_padding:
            unpad_seq_len = self._get_unpad_seqlen(attention_mask=attention_mask)
            input_ids = input_ids[:, :unpad_seq_len].contiguous()
            attention_mask = attention_mask[:, :unpad_seq_len].contiguous()

        input_ids = self._get_feature_on_this_cp_rank(input_ids, "input_ids")
        attention_mask = self._get_feature_on_this_cp_rank(attention_mask, "attention_mask")
        labels = data.batch["labels"] if "labels" in data.batch else None # labels is only used for sft
        if labels is not None:
            labels = self._get_feature_on_this_cp_rank(labels, "labels")
        position_ids = None
        # attention_mask: SelfAttention defalt to te DotProductAttention with
        # AttnMaskType.causal in which attention_mask would not be used, pass
        # it mainly for moe aux loss without pad token and it is 2D
        # position_ids: not used in LLM
        # While TransformerTurbo Qwen2VlModel requires 4D attention_mask, and
        # attention_mask and position_ids would be chunked for cp with dim 2 as
        # seq dim in it if they are provided
        forward_args = data.meta_info.get("forward_args", {})
        if "position_ids" in data.batch.keys() and data.batch["position_ids"].dim() == 3:  # qwen2vl mrope
            # not support MoE VLM, not used temperarily
            attention_mask = None
            position_ids = data.batch["position_ids"]
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)
            if self.use_remove_padding:
                position_ids = position_ids[:, :, :unpad_seq_len].contiguous()
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
        
        output_tensor = model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels, **forward_args
        )

        return output_tensor, partial(loss_func, data)

    def broadcast_parameter(self, model_update_name, src_pp_rank, dtype, shape, parameter_name):
        pass

    def broadcast_bucket(self, model_update_name, src_pp_rank, meta_infos, bucket_size):
        raise NotImplementedError

    def load_states(self, include=None, non_blocking=False):
        reload_megatron_no_grad_module(model_chunks=self.model.get_models())

    def offload_states(self, include=None, non_blocking=False):
        if include is None or OffloadStateType.model_params in include:
            offload_megatron_no_grad_module(model_chunks=self.model.get_models())
        RotaryEmbedding.forward.cache_clear()
        torch.cuda.empty_cache()

    def op_compute_log_probs(self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        input_ids [[p, p, r, r, r, 0, 0]] p: prompt, r: response, 0: pad
        response_mask [[0, 0, 1, 1, 1, 0, 0]]
        """
        ori_seq_length = attention_mask.size(1)
        cp_size = mpu.get_context_parallel_world_size()
        seq_len = logits.size(1) * cp_size if self.use_remove_padding else ori_seq_length
        # remove padding token
        if self.use_remove_padding:
            input_ids = input_ids[:, :seq_len]

        labels: torch.Tensor = input_ids[:, 1:].clone()
        labels[attention_mask[:, 1:seq_len] == 0] = 0  # avoid invalid token id
        # TODO: don't pad here but process this shift after generation
        labels = torch.cat([labels, torch.zeros_like(labels[:, :1])], dim=1)
        labels = self._get_feature_on_this_cp_rank(labels, "labels")
        # compute logprobs in remove padding token
        log_probs = vocab_parallel_logprobs(logits, labels)
        if mpu.get_context_parallel_world_size() > 1:
            log_probs = context_parallel_gather(log_probs, parallel_dim=1)
        # add pad to recover tensor shape
        if self.use_remove_padding:
            pad_token_num = ori_seq_length - seq_len
            log_probs = torch.nn.functional.pad(log_probs, pad=(0, pad_token_num), value=0)
        log_probs = log_probs[:, :-1] * attention_mask[:, 1:]
        return log_probs

    def op_compute_entropy(self, logits: torch.Tensor, attention_mask: torch.Tensor):
        entropy = vocab_parallel_entropy(logits)
        if mpu.get_context_parallel_world_size() > 1:
            entropy = context_parallel_gather(entropy, parallel_dim=1)
        # add pad to recover shape
        if self.use_remove_padding:
            pad_token_num = attention_mask.size(1) - entropy.size(1)
            entropy = torch.nn.functional.pad(entropy, pad=(0, pad_token_num), value=0)
        entropy = entropy[:, :-1] * attention_mask[:, 1:]
        return entropy

    def op_compute_logits(self, logits: torch.Tensor):
        full_logits = gather_from_tensor_model_parallel_region(logits)
        #TODO: support CP & use remove padding
        return full_logits

    def op_compute_language_loss(self, losses: torch.Tensor, labels: torch.Tensor):
        labels = self._get_feature_on_this_cp_rank(labels, "labels")
        
        loss_mask = (labels != IGNORE_INDEX).float()
        loss_mask = loss_mask.view(-1).float()
        losses = torch.sum(losses.view(-1) * loss_mask)
        loss_mask = loss_mask.sum()

        if mpu.get_context_parallel_world_size() > 1:
            loss_info = torch.cat([losses.view(1), loss_mask.view(1)])
            torch.distributed.all_reduce(
                loss_info, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group()
            )
            losses, loss_mask = loss_info[0], loss_info[1]

        loss = losses.clone() # clone to make sure loss is not a view 

        local_num_tokens = loss_mask.clone().detach()
        if local_num_tokens == 0:
            local_num_tokens += 1  # avoid divide by zero

        metrics = {f"{self.worker_config.name}/loss": (loss / local_num_tokens).clone().detach().unsqueeze(0)}
        
        return loss, local_num_tokens.int(), metrics


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

        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.processor = default_processor_provider(model_args=self.worker_config.model_args)
        # model provider will initialize megatron distributed groups
        self.model = model_provider(
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

        self.barrier = Barrier.options(
            name=BARRIER_NAME, get_if_exists=True, namespace=RAY_NAMESPACE
        ).remote(self.worker.world_size / self.worker.rank_info.pp_size)

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

        logger.info(f"{self.model.get_models()}")
        dist.barrier()

    def train_step(self, batch: DataProto, loss_func: Callable):
        self.model.train()

        mini_batch_size = self.worker_config.training_args.per_device_train_batch_size
        num_microbatches = batch.batch.batch_size[0] // self.worker_config.training_args.per_device_train_batch_size
        is_offload_optimizer_states_in_train_step = batch.meta_info.get("is_offload_optimizer_states_in_train_step", True)
        assert (
            num_microbatches == self.megatron_train_args.gradient_accumulation_steps
        ), f"num_microbatches={num_microbatches} gradient_accumulation_steps={self.megatron_train_args.gradient_accumulation_steps}"

        data_iterator = [
            batch.make_iterator(mini_batch_size=mini_batch_size, epochs=1) for _ in range(len(self.model))
        ]
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

        return metrics

    def model_update(self, model_update_name, tgt_workers, broadcast_tgt_devices, p2p_tgt_devices):
        comm_plan = self.model_update_comm_plan[model_update_name][self.worker.rank_info.pp_rank]
        broadcast_time_cost = 0
        with Timer("model_update_total") as timer_total:
            for meta_infos, buffer in self.model.all_gather_weights_as_hf_bucket(
                models=self.models_unwrapped, bucket_size=256 * 1024 * 1024
            ):
                ray.get(self.barrier.wait.remote())
                refs = []
                with Timer("broadcast") as timer_broadcast:
                    for p2p_tgt_device in p2p_tgt_devices:
                        p2p_tgt_worker = tgt_workers[p2p_tgt_device["rank"]]
                        ref = p2p_tgt_worker.update_parameter_in_bucket.remote(model_update_name=model_update_name,
                            meta_infos=meta_infos, buffer=buffer, ranks_in_worker=[p2p_tgt_device["device"]["rank"]]
                        )
                        refs.append(ref)

                    if (
                        self.worker.rank_info.tp_rank == 0
                        and self.worker.rank_info.cp_rank == 0
                        and self.worker.rank_info.dp_rank == 0
                    ):
                        for worker in tgt_workers:
                            ref = worker.broadcast_bucket.remote(
                                model_update_name=model_update_name,
                                src_pp_rank=self.worker.rank_info.pp_rank,
                                meta_infos=meta_infos,
                                bucket_size=buffer.numel() * buffer.element_size(),
                            )
                            refs.append(ref)
                    if len(broadcast_tgt_devices) > 0:
                        collective.broadcast(tensor=buffer, src_rank=0, group_name=comm_plan["group_name"])
                    ray.get(refs)
                ray.get(self.barrier.wait.remote())
                broadcast_time_cost += timer_broadcast.last

        metrics = {
            "all_gather": timer_total.last - broadcast_time_cost,
            "broadcast": broadcast_time_cost,
        }
        return metrics

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
        torch.cuda.empty_cache()

    def save_checkpoint(self, save_dir, global_step, ckpt_id, tag="checkpoint", local_state_path=None, **kwargs):
        logger.info(f"save_dir: {save_dir}")
        if local_state_path is None:
            local_state_path = save_dir
        with Timer("load") as load_timer:
            self.load_states()

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
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states(),
        }
        rgn_path = os.path.join(save_dir, RNG_STATE_DIR, f"rng_state_{dist.get_rank()}.pth")
        os.makedirs(os.path.dirname(rgn_path), exist_ok=True)
        torch.save(rng_states, rgn_path)

        if self.worker_config.checkpoint_config.get("async_upload", True):
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
            torch.cuda.set_rng_state(checkpoint_rng_state["cuda_rng_state"])
            # Check for empty states array
            if not checkpoint_rng_state["rng_tracker_states"]:
                raise KeyError
            tensor_parallel.get_cuda_rng_tracker().set_states(checkpoint_rng_state["rng_tracker_states"])
        else:
            logger.info(f"not load rng state, not found file: {rng_file}")
