import math
import os
import random
import sys
import time
import warnings
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from accelerate import skip_first_batches
from accelerate.data_loader import SeedableRandomSampler, prepare_data_loader
from megatron.core import dist_checkpointing, mpu, tensor_parallel
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig, finalize_model_grads
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.transformer.moe.moe_utils import clear_aux_losses_tracker, reduce_aux_losses_tracker_across_ranks
from torch._tensor import Tensor
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import PreTrainedTokenizerBase
from transformers.trainer import OPTIMIZER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME, Trainer, safe_globals
from transformers.trainer_callback import ExportableState, TrainerState
from transformers.trainer_pt_utils import get_dataloader_sampler, get_model_param_count, reissue_pt_warnings
from transformers.trainer_utils import (
    EvalLoopOutput,
    TrainOutput,
    has_length,
    seed_worker,
    speed_metrics,
)

from ..checkpointing import get_checkpoint_dir, load_state_dict_from_checkpoint
from ..constants import DIST_OPTIMIZER_DIR, IGNORE_INDEX
from ..initialize import initialize_megatron
from ..training_args import TrainingArguments
from ..utils import distributed_reduce, get_logger
from .utils import get_ltor_masks_and_position_ids, get_megatron_lr_scheduler, get_seqlens_in_batch


if TYPE_CHECKING:
    from megatron.core.optimizer import MegatronOptimizer

    from ..models import VirtualModels

logger = get_logger(__name__)


class McaTrainer(Trainer):
    metrics_keys = ["loss"]
    _language_input_names = ["input_ids", "attention_mask", "labels", "position_ids"]
    ckpt_sharding_type = "fully_sharded_model_space"

    def __init__(
        self,
        model: "VirtualModels" = None,
        args: TrainingArguments = None,
        **kwargs,
    ):
        initialize_megatron(args=args)
        self.args = args
        super().__init__(
            model[0],  # hack to avoid panic
            args=args,
            **kwargs,
        )
        self.model = model
        self.model_impl = self.model.config.transformer_impl
        self.models_wrapped = self._prepare_model(model)
        self.forward_backward_func = get_forward_backward_func()
        if self.args.use_distributed_optimizer:
            self.save_strategy = FullyParallelSaveStrategyWrapper(
                dist_checkpointing.serialization.get_default_save_sharded_strategy(),
                mpu.get_data_parallel_group(with_context_parallel=True),
                do_cache_distribution=True,  # don't support change model structure during training
            )
        if self.accelerator.dispatch_batches:
            self.accelerator.dispatch_batches = False
            logger.warning("Currently, accelerator.dispatch_batches must be set to False!")
        if self.args.sequence_packing and self.model_impl != "transformer_engine":
            raise ValueError("Currently, sequence_packing only support transformer_engine model!")

        if getattr(self, "processing_class", None) is None:
            self.processing_class = self.tokenizer

    def _prepare_model(self, models: "VirtualModels") -> List["DistributedDataParallel"]:
        config = models.config
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=self.args.accumulate_allreduce_grads_in_fp32,
            overlap_grad_reduce=self.args.overlap_grad_reduce,
            use_distributed_optimizer=self.args.use_distributed_optimizer,
            check_for_nan_in_grad=self.args.check_for_nan_in_loss_and_grad,
            bucket_size=self.args.ddp_bucket_size,
            average_in_collective=self.args.ddp_average_in_collective,
            overlap_param_gather=self.args.overlap_param_gather,
        )
        return [
            DistributedDataParallel(
                config=config,
                ddp_config=ddp_config,
                module=model,
                # Turn off bucketing for model_chunk 2 onwards, since communication for these
                # model chunks is overlapped with compute anyway.
                disable_bucketing=(model_index > 0),
            )
            for model_index, model in enumerate(models)
        ]

    def disable_ddp_forward_pre_hook(
        self, model_chunks: Optional[List["DistributedDataParallel"]] = None, param_sync=True
    ):
        """
        disable the overlap param gather pre-hook of DDP for 3 reasons:
            1. param sync: force sync model params. using before save ckpt.
            2. eval mode
            3. first training step before optimizer step
        """
        if not (self.args.use_distributed_optimizer and self.args.overlap_param_gather):
            return
        model_chunks = model_chunks or self.models_wrapped
        for model_chunk in model_chunks:
            assert isinstance(model_chunk, DistributedDataParallel)
            # TODO: add param_sync in core0.11.0
            model_chunk.disable_forward_pre_hook()

    def enable_ddp_forward_pre_hook(self, model_chunks: Optional[List["DistributedDataParallel"]] = None):
        if not (self.args.use_distributed_optimizer and self.args.overlap_param_gather):
            return
        model_chunks = model_chunks or self.models_wrapped
        for model_chunk in model_chunks:
            assert isinstance(model_chunk, DistributedDataParallel)
            model_chunk.enable_forward_pre_hook()

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            if not self.args.dataloader_drop_last:
                logger.warning("Currently, train dataloader drop_last must be set to True!")
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = True
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return prepare_data_loader(
            DataLoader(train_dataset, **dataloader_params),
            device=self.args.device,
            num_processes=mpu.get_data_parallel_world_size(with_context_parallel=False),
            process_index=mpu.get_data_parallel_rank(with_context_parallel=False),
            split_batches=self.accelerator.split_batches,
            put_on_device=True,
            rng_types=self.accelerator.rng_types,
            dispatch_batches=False,
        )

    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator
        dataloader_params = {
            "batch_size": self.args.per_device_eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = True
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return prepare_data_loader(
            DataLoader(eval_dataset, **dataloader_params),
            device=self.args.device,
            num_processes=mpu.get_data_parallel_world_size(with_context_parallel=False),
            process_index=mpu.get_data_parallel_rank(with_context_parallel=False),
            split_batches=self.accelerator.split_batches,
            put_on_device=True,
            rng_types=self.accelerator.rng_types,
            dispatch_batches=False,
        )

    def _get_batch_on_this_cp_rank(self, batch: Dict[str, Tensor]):
        dim3_keys = [] if self.model_impl == "transformer_engine" else ["attention_mask"]
        return self.model.get_batch_on_this_cp_rank(batch, dim3_keys=dim3_keys)

    def _prepare_train_inputs(self, data_iterator: Iterator) -> Dict[str, Tensor | Any]:
        inputs = next(data_iterator)
        inputs = {**inputs}  # avoid repeated modifications
        if self.args.sequence_packing:
            inputs = self._packing_sequence(inputs)
        else:
            attention_mask = inputs["attention_mask"] if "attention_mask" in inputs else None

            # causal attention impl in transformer engine don't need attention mask
            attention_mask, _ = get_ltor_masks_and_position_ids(
                inputs["input_ids"],
                build_attention_mask=self.model_impl != "transformer_engine",
                attn_mask_1D=attention_mask,
            )
            if not self.model.config.num_moe_experts and self.model_impl == "transformer_engine":
                attention_mask = None
            inputs["attention_mask"] = attention_mask

        if "position_ids" not in inputs:
            inputs["position_ids"] = None
        inputs = self._get_batch_on_this_cp_rank(inputs)
        return inputs

    def _pre_compute_loss(self, data_iterator: Iterator, model: DistributedDataParallel):
        inputs = self._prepare_train_inputs(data_iterator)
        loss_mask = (inputs["labels"] != IGNORE_INDEX).float()
        output_tensor = model(**inputs)
        return output_tensor, loss_mask

    def _post_compute_loss(self, loss_mask, losses):
        loss_mask = loss_mask.view(-1).float()
        cp_size = self.model.config.context_parallel_size
        losses = torch.sum(losses.view(-1) * loss_mask)
        loss_mask = loss_mask.sum()
        if cp_size > 1:
            loss_info = torch.cat([losses.view(1), loss_mask.view(1)])
            torch.distributed.all_reduce(
                loss_info, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group()
            )
            losses, loss_mask = loss_info[0], loss_info[1]
        loss = losses.clone()  # clone to make sure loss is not a view
        local_num_tokens = loss_mask.clone().detach()
        if local_num_tokens == 0:
            local_num_tokens += 1  # avoid divide by zero
        if self.args.calculate_per_token_loss:
            metrics = {"loss": (loss.clone().detach(), local_num_tokens)}
        else:
            metrics = {"loss": (loss / local_num_tokens).clone().detach()}
        return loss, local_num_tokens.int(), metrics

    def _inner_forward_step(self, data_iterator: Iterator, model: DistributedDataParallel):
        outputs = self._pre_compute_loss(data_iterator, model)
        return outputs[0], partial(self._post_compute_loss, *outputs[1:])

    def _packing_sequence(self, inputs: Dict[str, Tensor | Any]):
        if not self.args.sequence_packing:
            return inputs
        attention_mask = inputs.pop("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(inputs["input_ids"])
        seqlens, max_seq_len = get_seqlens_in_batch(attention_mask)

        packing_inputs = {
            k: v.view(1, -1, *v.shape[2:]) if v is not None and isinstance(v, Tensor) else v
            for k, v in inputs.items()
            if k in self._language_input_names
        }
        inputs.update(
            {
                **packing_inputs,
                "packed_seq_params": PackedSeqParams(
                    qkv_format="thd",
                    cu_seqlens_kv=seqlens,
                    cu_seqlens_q=seqlens,
                    cu_seqlens_q_padded=seqlens,
                    cu_seqlens_kv_padded=seqlens,
                    max_seqlen_q=max_seq_len,
                    max_seqlen_kv=max_seq_len,
                ),
                "attention_mask": None,
            }
        )
        return inputs

    def _get_step_iterator_and_seq_length(self, epoch_iterator: Iterator[Dict[str, Tensor | Any]], standard_batch_size: Optional[int] = None):
        """
        construct data iterator for gradient accumulation
        """
        step_inputs = []
        max_seq_length = 0
        standard_batch_size = standard_batch_size or self.args.per_device_train_batch_size
        for _ in range(self.args.gradient_accumulation_steps):
            try:
                inputs = next(epoch_iterator)
            except StopIteration:
                self.control.should_epoch_stop = True
                logger.warning("epoch_iterator has stopped, epoch will stop!")
                break
            if inputs is None:
                self.control.should_epoch_stop = True
                logger.warning("Insufficient data in dataset, epoch will stop!")
                break
            main_inputs = inputs[self.model.main_input_name]
            batch_size, seq_length = main_inputs.size(0), main_inputs.size(1)
            if batch_size != standard_batch_size:
                # iterable dataloader can't drop last
                self.control.should_epoch_stop = True
                logger.warning(
                    f"batch_size {batch_size} not equal to standard_batch_size {standard_batch_size}, epoch will stop!"
                )
                break

            step_inputs.append(inputs)
            max_seq_length = max(max_seq_length, seq_length)

        if len(step_inputs) < self.args.gradient_accumulation_steps:
            return None, 0

        if not self.args.allow_variable_seq_lengths():
            step_inputs = [self._pad_batched_inputs(inputs, max_seq_length) for inputs in step_inputs]
        for inputs in step_inputs:
            self.current_flos += float(self.floating_point_ops(inputs))
        return iter(step_inputs), max_seq_length

    def _pad_batched_inputs(self, inputs: Dict[str, Tensor | Any], seq_length: int):
        padding_inputs = {
            k: v.tolist() if v is not None and isinstance(v, Tensor) else v
            for k, v in inputs.items()
            if k in self._language_input_names
        }
        if "labels" in padding_inputs:
            padding_inputs["labels"] = [
                labels + [IGNORE_INDEX] * (seq_length - len(labels)) for labels in padding_inputs["labels"]
            ]

        tokenizer = (
            self.processing_class
            if isinstance(self.processing_class, PreTrainedTokenizerBase)
            else getattr(self.processing_class, "tokenizer", self.processing_class)
        )
        padding_inputs = tokenizer.pad(padding_inputs, padding="max_length", max_length=seq_length, return_tensors="pt").to(
            self.args.device
        )
        inputs.update(padding_inputs)
        return inputs

    def _stream_eval_inputs(self, eval_dataloader: DataLoader, standard_batch_size: Optional[int] = None):
        collected_inputs = []
        max_seq_length = 0
        standard_batch_size = standard_batch_size or self.args.per_device_eval_batch_size

        pad_func = lambda x, length: [self._pad_batched_inputs(i, length) for i in x]
        end_flag = torch.tensor(0, device=self.args.device)
        for inputs in eval_dataloader:
            main_inputs = inputs[self.model.main_input_name]
            batch_size, seq_length = main_inputs.size()
            if batch_size == standard_batch_size:
                collected_inputs.append(inputs)
                max_seq_length = max(max_seq_length, seq_length)
                if len(collected_inputs) == self.args.gradient_accumulation_steps:
                    dist.all_reduce(end_flag, op=dist.ReduceOp.MAX)
                    if end_flag > 0:
                        return
                    yield pad_func(collected_inputs, max_seq_length), max_seq_length, standard_batch_size
                    collected_inputs, max_seq_length = [], 0
        end_flag = torch.ones_like(end_flag)
        dist.all_reduce(end_flag, op=dist.ReduceOp.MAX)

    def training_step(self, models: List[DistributedDataParallel], data_iterator, seq_length):
        # a real step not a minibatch of gradient accumulation
        for model in models:
            model.train()
            model.zero_grad_buffer()
        self.optimizer.zero_grad()

        if len(models) > 1:
            data_list = list(data_iterator)
            data_iterator = [iter(data_list) for _ in range(len(models))]
        metrics_tensors: List[Dict[str, Tensor]] = self.forward_backward_func(
            forward_step_func=self._inner_forward_step,
            data_iterator=data_iterator,
            model=models,
            num_microbatches=self.args.gradient_accumulation_steps,
            seq_length=seq_length,
            micro_batch_size=self.args.per_device_train_batch_size,
            forward_only=False,
        )
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
        if update_successful:
            self.lr_scheduler.step()
            skipped_iter = 0
        else:
            skipped_iter = 1

        if len(metrics_tensors) > 0 and "loss" in metrics_tensors[0]:
            first_val = metrics_tensors[0]["loss"]
            if isinstance(first_val, tuple) or isinstance(first_val, list):
                assert len(first_val) == 2, f"metrics_tensors: {metrics_tensors} has wrong format"
                loss = torch.stack([m["loss"][0] for m in metrics_tensors]).view(-1).sum()
                loss_scale = torch.stack([m["loss"][1] for m in metrics_tensors]).view(-1).sum()
                loss = torch.stack([loss, loss_scale]).view(-1)  # scale after reducing cross dp and steps
            else:
                loss = torch.stack([m["loss"] for m in metrics_tensors]).view(-1).mean()
        else:
            loss = torch.tensor(0.0, device=self.args.device)
        return loss, metrics_tensors, skipped_iter, grad_norm, num_zeros_in_grad

    def gather_metrics(self, metrics_tensors: List[Dict[str, Tensor]]) -> Dict[str, float]:
        metrics = {}
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            get_metrics_keys = metrics_tensors[0].keys()
            assert all(
                key in get_metrics_keys for key in self.metrics_keys
            ), f"some keys in self.metrics_keys: {self.metrics_keys} not get in metrics_tensors: {get_metrics_keys}"
            diff_keys = set(self.metrics_keys) - set(get_metrics_keys)
            if len(diff_keys) > 0 and not getattr(self, "warned_metrics", False):
                logger.warning(f"some metrics_tensors: {diff_keys} not set in self.metrics_keys: {self.metrics_keys}")
                setattr(self, "warned_metrics", True)
            for key in self.metrics_keys:
                metrics[key] = torch.stack(
                    [
                        m[key] if isinstance(m[key], torch.Tensor) else m[key][0] / m[key][1]  # per token loss
                        for m in metrics_tensors
                    ]
                ).mean(0)
            metrics = distributed_reduce(metrics, group=mpu.get_data_parallel_group(), op=dist.ReduceOp.AVG)
        else:
            metrics = {k: torch.tensor(0.0, device=self.args.device, dtype=torch.float32) for k in self.metrics_keys}
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            metrics = distributed_reduce(metrics, group=mpu.get_pipeline_model_parallel_group())
        metrics = {k: v.cpu().item() for k, v in metrics.items()}
        return metrics

    def create_optimizer(self):
        params_dtype = torch.float16 if self.args.fp16 else torch.bfloat16 if self.args.bf16 else torch.float32
        config = OptimizerConfig(
            optimizer=self.args.optimizer,
            lr=self.args.learning_rate,
            min_lr=self.args.lr_scheduler_kwargs.get("min_lr", 0.0),
            weight_decay=self.args.weight_decay,
            adam_beta1=self.args.adam_beta1,
            adam_beta2=self.args.adam_beta2,
            adam_eps=self.args.adam_epsilon,
            fp16=self.args.fp16,
            bf16=self.args.bf16,
            params_dtype=params_dtype,
            use_distributed_optimizer=self.args.use_distributed_optimizer,
            clip_grad=self.args.max_grad_norm,
        )
        self.optimizer = get_megatron_optimizer(config, self.models_wrapped)
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: "MegatronOptimizer" = None):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_megatron_lr_scheduler(self.args, num_training_steps, optimizer)
        return self.lr_scheduler

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        # TODO: support resume _CUDA_RNG_STATE_TRACKER (which is needed for dropout/init model weights)
        model = model or self.model
        logger.info(f"Loading model from {resume_from_checkpoint}.")
        state_dict = load_state_dict_from_checkpoint(resume_from_checkpoint)
        assert state_dict is not None, "No model state_dict found in checkpoint."
        model.load_state_dict(state_dict)

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is None:
            return
        optimizer_checkpoint = get_checkpoint_dir(
            checkpoint, iteration=1, return_base_dir=self.args.use_distributed_optimizer
        )
        if self.args.use_distributed_optimizer:
            optimizer_checkpoint = os.path.join(optimizer_checkpoint, DIST_OPTIMIZER_DIR)
        logger.info(f"Loading optimizer from {optimizer_checkpoint}, process_index: {self.args.process_index}")
        if self.args.use_distributed_optimizer:
            model_shared_state_dict = self.model.sharded_state_dict()
            sharded_state_dict = self.optimizer.sharded_state_dict(
                model_shared_state_dict, is_loading=True, sharding_type=self.ckpt_sharding_type
            )
            load_strategy = dist_checkpointing.serialization.get_default_load_sharded_strategy(optimizer_checkpoint)
            load_strategy = FullyParallelLoadStrategyWrapper(
                load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
            )
            state_dict = dist_checkpointing.load(sharded_state_dict, optimizer_checkpoint, load_strategy)
        else:
            state_dict = torch.load(os.path.join(optimizer_checkpoint, OPTIMIZER_NAME), map_location=self.args.device)
        self.optimizer.load_state_dict(state_dict)

        with warnings.catch_warnings(record=True) as caught_warnings:
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME)))
        reissue_pt_warnings(caught_warnings)

    def _save_rng_state(self, output_dir):
        rng_states = {
            "random_rng_state": random.getstate(),
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states(),
        }
        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))

    def _load_rng_state(self, checkpoint):
        # Load RNG states from `checkpoint`
        if checkpoint is None:
            return

        if self.args.world_size > 1:
            process_index = self.args.process_index
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    f"Didn't find an RNG file for process {process_index}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return

        with safe_globals():
            checkpoint_rng_state = torch.load(rng_file)
        random.setstate(checkpoint_rng_state["random_rng_state"])
        np.random.set_state(checkpoint_rng_state["np_rng_state"])
        torch.set_rng_state(checkpoint_rng_state["torch_rng_state"])
        torch.cuda.set_rng_state(checkpoint_rng_state["cuda_rng_state"])
        # Check for empty states array
        if not checkpoint_rng_state["rng_tracker_states"]:
            raise KeyError
        tensor_parallel.get_cuda_rng_tracker().set_states(checkpoint_rng_state["rng_tracker_states"])

    def _prepare_train_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        total_train_batch_size = (
            self._train_batch_size
            * args.gradient_accumulation_steps
            * mpu.get_data_parallel_world_size(with_context_parallel=False)
        )
        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(self.model, trainable_only=True):,}")

        self.state.epoch = 0
        epochs_trained = 0
        batches_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                batches_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                batches_trained_in_current_epoch *= self.args.gradient_accumulation_steps
            else:
                batches_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {batches_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader

        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step

        self.model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler, SeedableRandomSampler]
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        model_config = self.model.config
        model_config.grad_scale_func = self.optimizer.scale_loss
        if self.args.overlap_grad_reduce:
            assert model_config.no_sync_func is None, (
                "When overlap_grad_reduce is True, config.no_sync_func must be None; "
                "a custom no_sync_func is not supported when overlapping grad-reduce"
            )
            model_config.no_sync_func = [model_wrapped.no_sync for model_wrapped in self.models_wrapped]
            if len(self.models_wrapped) == 1:
                model_config.no_sync_func = model_config.no_sync_func[0]
            if self.args.delay_grad_reduce:
                model_config.grad_sync_func = [model_wrapped.start_grad_sync for model_wrapped in self.models_wrapped]
                if len(self.models_wrapped) == 1:
                    model_config.grad_sync_func = model_config.grad_sync_func[0]
        model_config.finalize_model_grads_func = finalize_model_grads
        return (
            epochs_trained,
            num_train_epochs,
            train_dataloader,
            len_dataloader,
            batches_trained_in_current_epoch,
            max_steps,
            num_train_samples,
            num_train_tokens,
        )

    def _get_train_cyclic_iterator(self, train_dataloader):
        while True:
            for x in train_dataloader:
                yield x
            self.control.should_epoch_stop = True

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        start_time = time.time()
        (
            epochs_trained,
            num_train_epochs,
            train_dataloader,
            len_dataloader,
            batches_trained_in_current_epoch,
            max_steps,
            num_train_samples,
            num_train_tokens,
        ) = self._prepare_train_loop(batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
        if self.args.calculate_per_token_loss:
            # [loss, loss_scale]
            tr_loss = torch.tensor([0.0, 0.0], device=self.args.device)
        else:
            tr_loss = torch.tensor(0.0, device=self.args.device)
        grad_norm = None
        for epoch in range(epochs_trained, num_train_epochs):
            cyclic_iterator = self._get_train_cyclic_iterator(train_dataloader)
            if hasattr(train_dataloader, "set_epoch"):
                train_dataloader.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(train_dataloader) // args.gradient_accumulation_steps
                if len_dataloader is not None
                else args.max_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)
            if epoch == epochs_trained and resume_from_checkpoint is not None and batches_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)
            rng_to_sync = False
            steps_skipped = 0
            if batches_trained_in_current_epoch > 0:
                cyclic_iterator = self._get_train_cyclic_iterator(
                    skip_first_batches(train_dataloader, batches_trained_in_current_epoch)
                )
                steps_skipped = batches_trained_in_current_epoch // args.gradient_accumulation_steps
                batches_trained_in_current_epoch = 0
                rng_to_sync = True

            self.disable_ddp_forward_pre_hook(param_sync=False)
            step = -1
            first_step = True
            while True:
                step_iterator, seq_length = self._get_step_iterator_and_seq_length(cyclic_iterator)
                if step_iterator is None:
                    break
                step += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False
                self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                tr_loss_step, metrics_tensors, skipped_iter, grad_norm, num_zeros_in_grad = self.training_step(
                    self.models_wrapped, step_iterator, seq_length
                )

                if first_step:
                    self.enable_ddp_forward_pre_hook()
                    first_step = False

                if args.logging_nan_inf_filter and (
                    torch.isnan(tr_loss_step).any() or torch.isinf(tr_loss_step).any()
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    tr_loss += tr_loss_step

                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                logs = {"skipped_iter": skipped_iter, "num_zeros_in_grad": num_zeros_in_grad or 0}
                self._maybe_log_save_evaluate(
                    tr_loss,
                    grad_norm,
                    self.model,
                    trial,
                    epoch,
                    ignore_keys_for_eval,
                    other_logs=logs,
                    metrics_tensors=metrics_tensors,
                )
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, self.model_wrapped, trial, epoch, ignore_keys_for_eval)
            if self.control.should_training_stop:
                break

        # add remaining tr_loss
        if self.args.calculate_per_token_loss:
            tr_loss = tr_loss[0] / tr_loss[1]
            tr_loss = tr_loss if tr_loss.isfinite() else torch.zeros_like(tr_loss)
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step
        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        other_logs: Dict[str, float] = {},
        metrics_tensors: Optional[List[Dict[str, Tensor]]] = None,
    ):
        eval_or_save = self.control.should_evaluate or self.control.should_save
        if eval_or_save:
            self.disable_ddp_forward_pre_hook()

        moe_losses = {}
        if self.model.config.num_moe_experts is not None and self.model.config.num_moe_experts > 1:
            if self.control.should_log:
                reduce_aux_losses_tracker_across_ranks()
                tracker = mpu.get_moe_layer_wise_logging_tracker()
                loss_scale = 1 / self.args.gradient_accumulation_steps
                moe_losses = {k: (v['values'].float() * loss_scale).mean().item() for k, v in tracker.items()}

            clear_aux_losses_tracker()

        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs = {}
            loss = tr_loss.clone().detach()
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                loss = distributed_reduce(loss, group=mpu.get_data_parallel_group(), op=torch.distributed.ReduceOp.AVG)
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                loss = distributed_reduce(loss, group=mpu.get_pipeline_model_parallel_group())
            if self.args.calculate_per_token_loss:
                assert len(loss) == 2, f"Per token loss must be a tensor of [loss, num_tokens] but got {loss}"
                loss = loss[0] / loss[1]
            tr_loss_scalar = loss.item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()
            logs.update(moe_losses)
            if metrics_tensors is not None and len(self.metrics_keys) > 1:  # metrics except loss
                metrics = self.gather_metrics(metrics_tensors)
                metrics.pop("loss", None)
                logs.update(metrics)
            logs.update(other_logs)
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(trial, ignore_keys_for_eval)

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        if eval_or_save:
            self.enable_ddp_forward_pre_hook()

    @torch.no_grad()
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
        assert prediction_loss_only, "Evaluation with `prediction_loss_only=False` is not supported."
        models = self.model
        models.eval()
        metrics_tensors: List[Dict[str, Tensor]] = []
        for step_inputs, seq_length, batch_size in self._stream_eval_inputs(dataloader):
            num_microbatches = len(step_inputs)
            data_iterator = [iter(step_inputs) for _ in range(len(models))]
            step_metrics_tensors = self.forward_backward_func(
                forward_step_func=self._inner_forward_step,
                data_iterator=data_iterator,
                model=models.get_models(),
                num_microbatches=num_microbatches,
                seq_length=seq_length,
                micro_batch_size=batch_size,
                forward_only=True,
            )
            metrics_tensors.extend(step_metrics_tensors)
        num_samples = torch.tensor(len(metrics_tensors)).to(self.args.device)
        num_samples = distributed_reduce(num_samples, group=mpu.get_data_parallel_group()).cpu().item()
        metrics = self.gather_metrics(metrics_tensors)
        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples)

    def _save_optimizer_and_scheduler(self, output_dir):
        if dist.is_initialized():
            dist.barrier()
        checkpoint_dir = get_checkpoint_dir(output_dir, return_base_dir=self.args.use_distributed_optimizer)
        if self.args.use_distributed_optimizer:
            checkpoint_dir = os.path.join(checkpoint_dir, DIST_OPTIMIZER_DIR)
        os.makedirs(checkpoint_dir, exist_ok=True)
        if self.args.use_distributed_optimizer:
            model_shared_state_dict = self.model.sharded_state_dict()
            state_dict = self.optimizer.sharded_state_dict(
                model_shared_state_dict, sharding_type=self.ckpt_sharding_type
            )
            # validate access integrity in the first time
            validate_access_integrity = getattr(self, "_validate_access_integrity", True)
            dist_checkpointing.save(
                state_dict,
                checkpoint_dir=checkpoint_dir,
                sharded_strategy=self.save_strategy,
                async_sharded_save=False,
                validate_access_integrity=validate_access_integrity,
            )
            self._validate_access_integrity = False
        elif not dist.is_initialized() or mpu.get_expert_data_parallel_rank() == 0:
            torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, OPTIMIZER_NAME))
            logger.info(f"Saving optimizer state to {os.path.join(checkpoint_dir, OPTIMIZER_NAME)}")

        if dist.is_initialized():
            dist.barrier()

        if self.args.should_save:
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)

    def save_model(self, output_dir: str = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        if not (self.args.save_only_model and self.args.save_hf_model):
            self.model.save_pretrained(output_dir)
        if self.args.save_hf_model:
            self.model.save_pretrained_as_hf(output_dir)
        if self.args.should_save:
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def estimate_tokens(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        if not hasattr(self.model, "estimate_tokens"):
            return 0
        return self.model.estimate_tokens(inputs)
