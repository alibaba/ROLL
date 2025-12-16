import os
from typing import Union, Optional, Dict
from tensordict import TensorDict

import ray
import torch
import torch.distributed as dist
from codetiming import Timer

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import register, Dispatch
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_actor_model_provider
from roll.utils.context_managers import state_offload_manger
from roll.utils.functionals import (
    append_to_dict,
)
from roll.utils.offload_states import OffloadStateType
from roll.pipeline.distill.various_divergence import VariousDivergence
from roll.utils.collective import collective
from roll.utils.cuda_ipc_utils import MultiprocessingSerializer
from roll.platforms import current_platform



class StudentWorker(Worker):

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None
        self.kl_loss_func = None
        self.probs_cache = LogitsCache(self.logger)
        self.log_probs_cache = LogitsCache(self.logger)
        self.topk_indices_cache = LogitsCache(self.logger)
        self.inf_mask_cache = LogitsCache(self.logger)
        self.tensor_name_to_cache_name = {"topk_probs": "probs_cache", "topk_log_probs": "log_probs_cache",
                                          "topk_indices": "topk_indices_cache", "topk_inf_mask": "inf_mask_cache"}
        self.teacher_probs = None
        self.teacher_log_probs = None
        self.teacher_topk_indices = None
        self.teacher_inf_mask = None
        self.teacher_probs_iterator = None
        self.teacher_log_probs_iterator = None
        self.teacher_topk_indices_iterator = None
        self.teacher_inf_mask_iterator = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)

        self.strategy = create_strategy(worker=self)

        self.strategy.initialize(model_provider=default_actor_model_provider)
        self.tokenizer = self.strategy.tokenizer

        if self.pipeline_config.resume_from_checkpoint:
            load_dir = os.path.join(self.pipeline_config.resume_from_checkpoint, self.cluster_name)
            self.strategy.load_checkpoint(load_dir=load_dir, tag="checkpoint")

        self.logger.info(f"{self.worker_name} initialized")

        self.strategy.offload_states()

        self.kl_loss_func = VariousDivergence(self.pipeline_config)

    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST, clear_cache=False)
    def train_step(self, data: DataProto):
        """
        return DataProto(meta_info={'metrics': metrics})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        micro_batch_size = self.worker_config.training_args.per_device_train_batch_size

        # Retrieve the teacher logits
        if self.rank_info.is_pipeline_last_stage:
            self.teacher_probs = self.probs_cache.pop_full_logits()
            self.teacher_probs_iterator = iter(self.teacher_probs.split(micro_batch_size, dim=0))
            self.teacher_log_probs = self.log_probs_cache.pop_full_logits()
            self.teacher_log_probs_iterator = iter(self.teacher_log_probs.split(micro_batch_size, dim=0))
        # Retrieve the teacher_topk_indices
        if self.rank_info.is_pipeline_last_stage:
            self.teacher_topk_indices = self.topk_indices_cache.pop_full_logits()
            if self.pipeline_config.logits_topk != 0:
                self.teacher_topk_indices_iterator = iter(self.teacher_topk_indices.split(micro_batch_size, dim=0))
            self.teacher_inf_mask = self.inf_mask_cache.pop_full_logits()
            self.teacher_inf_mask_iterator = iter(self.teacher_inf_mask.split(micro_batch_size, dim=0))
        self.logger.info(f"is_offload_states: {is_offload_states}")
        with state_offload_manger(
                strategy=self.strategy,
                metrics=metrics,
                metric_infix=f"{self.cluster_name}/train_step",
                is_offload_states=is_offload_states,
                load_kwargs={"include": None},
        ):
            data = data.to(current_platform.device_type)
            data = self.strategy.get_data_input(data)
            if "labels" in data.batch.keys():
                # rename key: labels -> labels_for_loss
                data.batch.rename_key_("labels", "labels_for_loss")
            self.logger.info(f"global_step: {data.meta_info.get('global_step',0)}")
            per_device_train_batch_size = self.worker_config.training_args.per_device_train_batch_size
            backward_batch_size = (
                    per_device_train_batch_size * self.worker_config.training_args.gradient_accumulation_steps
            )

            loss_func = self.loss_func
            if self.worker_config.use_sequence_packing:
                from roll.utils.sequence_packing import SequencePackingDistillLossWrapper
                loss_func = SequencePackingDistillLossWrapper(self.strategy, loss_func)
            student_metrics = self.strategy.train_step(batch=data, loss_func=loss_func)
            append_to_dict(metrics, student_metrics)

            data.to("cpu")
            metrics["student/lr"] = self.strategy.scheduler.get_last_lr()[0]

        output = DataProto(meta_info={"metrics": metrics}).to("cpu")

        return output

    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        """
        Loss function interface definition:
            data: DataProto, passed through unchanged from train_step  
            output_tensor: torch.Tensor, the tensor returned by model.forward()
        """

        student_logits = output_tensor
        labels = data.batch['labels_for_loss']

        # language loss
        gpt_loss, _ = self.strategy.op_compute_language_loss_from_logits(student_logits, labels)

        # distill loss
        if self.teacher_probs_iterator is not None:
            teacher_probs = next(self.teacher_probs_iterator)
        else:
            teacher_probs = None
        if self.teacher_log_probs_iterator is not None:
            teacher_log_probs = next(self.teacher_log_probs_iterator)
        else:
            teacher_log_probs = None
        if self.teacher_topk_indices_iterator is not None:
            teacher_topk_indices = next(self.teacher_topk_indices_iterator)
        else:
            teacher_topk_indices = None
        if self.teacher_inf_mask_iterator is not None:
            teacher_inf_mask = next(self.teacher_inf_mask_iterator)
        else:
            teacher_inf_mask = None

        distill_loss, _ = self.strategy.op_compute_various_divergence(self.kl_loss_func, student_logits, teacher_probs,
                                                                teacher_log_probs, teacher_topk_indices, teacher_inf_mask
                                                                , labels, attention_mask=None,)

        loss = ((1 - self.pipeline_config.distill_loss_weight) * gpt_loss
                + self.pipeline_config.distill_loss_weight * distill_loss)
        student_metrics = {
            "train/loss": loss.detach().item(),
            "train/train_distill_loss": distill_loss.detach().item(),
            "train/train_student_loss": gpt_loss.detach().item(),
        }
        return loss, student_metrics

    @register(Dispatch.DP_MP_DISPATCH_FIRST, clear_cache=False)
    def val_step(self, data: DataProto):
        data = data.to(current_platform.device_type)
        data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size
        data = self.strategy.get_data_input(data)
        if "labels" in data.batch.keys():
            # rename key: labels -> labels_for_loss
            data.batch.rename_key_("labels", "labels_for_loss")
        metrics = self.strategy.forward_step(batch=data, forward_func=self.loss_func_for_eval)
        output = DataProto(meta_info={"metrics": metrics}).to("cpu")
        return output
    
    def loss_func_for_eval(self, data: DataProto, output_tensor: torch.Tensor):
        labels = data.batch['labels_for_loss']
        gpt_loss, _ = self.strategy.op_compute_language_loss_from_logits(output_tensor, labels)
        student_metrics = {
            "student/val_loss": gpt_loss.detach().item(),
        }
        return gpt_loss, student_metrics
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def do_checkpoint(self, global_step):
        with Timer("do_checkpoint") as total_timer:
            ckpt_id = f"checkpoint-{global_step}"
            save_dir = os.path.join(self.pipeline_config.output_dir, self.worker_name, ckpt_id, self.cluster_name)
            self.logger.info(f"save checkpoint-{global_step} to {save_dir}")
            exec_metrics: Dict = self.strategy.save_checkpoint(save_dir, global_step, ckpt_id)

        metrics = {
            f"time/{self.cluster_name}/do_checkpoint/total": total_timer.last,
        }
        metric_prefix = f"time/{self.cluster_name}/do_checkpoint"
        metrics.update({f"{metric_prefix}/{k}": v for k, v in exec_metrics.items()})
        output = DataProto(meta_info={"metrics": metrics})
        return output

    def receive_broadcast_logits(self, slice_info, backend, tensor_name_for_transfer=None ,tensor_slice=None, tensor_shape=None, tensor_dtype=None,
                                 group_name=None):
        cache_name = self.tensor_name_to_cache_name[tensor_name_for_transfer]
        assert hasattr(self, cache_name), f"Receive broadcast logits: student worker doesn't have attr {cache_name}"
        logits_cache = getattr(self, cache_name)
        self.logger.info(
            f"[Student][receive_broadcast_logits] rank={dist.get_rank()}, slice_info={slice_info}, backend={backend}, "
            f"tensor_shape={tensor_shape}, tensor_dtype={tensor_dtype}, group_name={group_name}"
        )
        if backend == "ray":
            self.logger.info("[Student][receive_broadcast_logits][Ray] caching tensor slice directly")
            logits_cache.cache(
                slice_info, tensor_slice=tensor_slice, tensor_shape=tensor_shape, tensor_dtype=tensor_dtype
            )
        elif backend in ('nccl-only', 'ipc+nccl'):
            logits_cache.cache(
                slice_info, tensor_slice=None, tensor_shape=tensor_shape, tensor_dtype=tensor_dtype
            )
            assert group_name is not None, "StudentWorker receive_broadcast_logits: group name is None"
            src_rank, slice_index, total_slices, slice_type = slice_info
            self.logger.info(
                f"[Student][receive_broadcast_logits][NCCL] about to broadcast, src_rank={src_rank}, slice_index={slice_index}, group_name={group_name}")
            collective.broadcast(
                tensor=logits_cache.get_slice_view(slice_index),
                src_rank=0,
                group_name=group_name
            )
            self.logger.info(f"[Student][receive_broadcast_logits][NCCL] broadcast done for slice_index={slice_index}")
            logits_cache.add_receive_count()
            logits_cache.try_finalize_full_logits()
            self.logger.info(
                "[Student][receive_broadcast_logits][NCCL] receive_count updated, try_finalize_full_logits done")
        else:
            raise RuntimeError(
                "StudentWorker receive_broadcast_logits: backend must be 'ipc+nccl', 'ray' or 'nccl-only'")
    def receive_p2p_logits(self,  slice_info, backend, tensor_name_for_transfer=None, tensor_slice_handle=None, tensor_shape=None, tensor_dtype=None):
        cache_name = self.tensor_name_to_cache_name[tensor_name_for_transfer]
        assert hasattr(self, cache_name), f"Receive broadcast logits: student worker doesn't have attr {cache_name}"
        logits_cache = getattr(self, cache_name)
        assert backend == "ipc+nccl", "StudentWorker receive_p2p_logits: backend must be 'ipc+nccl'"
        tensor_slice = MultiprocessingSerializer.deserialize(tensor_slice_handle)
        logits_cache.cache(slice_info, tensor_slice=tensor_slice,
                                        tensor_shape=tensor_shape, tensor_dtype=tensor_dtype)
        current_platform.synchronize()
        logits_cache.try_finalize_full_logits()

    def broadcast_logits(self, tensor_name_for_transfer, tp=False, cp=False):
        assert tp ^ cp, f"Logits broadcasting can only occur in either the TP group or the CP group at the same time, but not both."
        cache_name = self.tensor_name_to_cache_name[tensor_name_for_transfer]
        assert hasattr(self, cache_name), f"Receive broadcast logits: student worker doesn't have attr {cache_name}"
        logits_cache = getattr(self, cache_name)
        rank_info = self.rank_info
        self.logger.info(
            f"[Student][broadcast_logits] rank={dist.get_rank()}, pp={rank_info.pp_rank}, dp={rank_info.dp_rank}, tp={rank_info.tp_rank}, "
            f"is_pipeline_last_stage={rank_info.is_pipeline_last_stage}, tp_size={rank_info.tp_size}"
        )
        if rank_info.is_pipeline_last_stage and rank_info.tp_size > 1:
            assert self.strategy.strategy_name == "megatron_train", \
                f"Error in DistillWorker broadcast_logits: {self.strategy.strategy_name} with tp_size == {rank_info.tp_size}"
            from megatron.core import mpu
            if tp:
                group = mpu.get_tensor_model_parallel_group()
                rank = rank_info.tp_rank
            else:
                group = mpu.get_context_parallel_group()
                rank = rank_info.cp_rank
            self.logger.info(
                f"[Student][broadcast_logits] calling logits_cache.broadcast_from_dynamic_holder(), tp={tp}, cp={cp}, group={group}, rank={rank}"
            )

            logits_cache.broadcast_from_dynamic_holder(group=group, rank=rank)
            self.logger.info("[Student][broadcast_logits] broadcast_from_dynamic_holder() finished")


class LogitsCache:
    def __init__(self, logger):
        self.full_buffer = None
        self.total_slices = None
        self.slice_len = None
        self.received_count = 0
        self.slice_info = None
        self._finalized = False
        self._owner = None
        self.logger = logger

    def add_receive_count(self):
        self.received_count += 1

    def cache(self, slice_info, tensor_slice=None, tensor_shape=None, tensor_dtype=None):
        src_rank, slice_index, total_slices, slice_type = slice_info
        self.slice_info = slice_info

        # full logits mode
        if slice_type == "full":
            self.total_slices = 1
            if tensor_slice is not None:
                self.full_buffer = tensor_slice.contiguous()
                self.slice_len = tensor_slice.size(0)
                self.received_count = 1
            else:
                assert tensor_shape is not None and tensor_dtype is not None
                self.slice_len = tensor_shape[0]
                self.full_buffer = torch.empty(
                    tensor_shape, dtype=tensor_dtype, device=current_platform.device_type
                )
            self.try_finalize_full_logits()
            return

        # teacher_send_slice mode
        if slice_type == "teacher_send_slice":
            self.total_slices = 1
            if tensor_slice is not None:
                self.full_buffer = tensor_slice.contiguous()
                self.slice_len = tensor_slice.size(0) // total_slices
                self.received_count = 1
            else:
                assert tensor_shape is not None and tensor_dtype is not None
                self.slice_len = tensor_shape[0] // total_slices
                self.full_buffer = torch.empty(
                    tensor_shape, dtype=tensor_dtype, device=current_platform.device_type
                )
            self.try_finalize_full_logits()
            return

        if slice_type == "student_receive_slice":
            if self.full_buffer is None:
                self.total_slices = total_slices
                if tensor_slice is not None:
                    self.slice_len = tensor_slice.size(0)
                    rest_shape = tensor_slice.shape[1:]
                    dtype = tensor_slice.dtype
                    device = tensor_slice.device
                else:
                    assert tensor_shape is not None and tensor_dtype is not None
                    self.slice_len = tensor_shape[0]
                    rest_shape = tuple(tensor_shape[1:])
                    dtype = tensor_dtype
                    device = torch.device(current_platform.device_type)
                full_batch_size = self.slice_len * total_slices
                self.full_buffer = torch.empty(
                    (full_batch_size, *rest_shape), dtype=dtype, device=device
                )

            if tensor_slice is not None:
                start = slice_index * self.slice_len
                end = start + tensor_slice.size(0)
                self.full_buffer[start:end] = tensor_slice
                self.received_count += 1
            self.try_finalize_full_logits()

    def is_complete(self):
        return self.received_count == self.total_slices

    def try_finalize_full_logits(self):
        if self._finalized or not self.is_complete():
            return

        src_rank, slice_index, total_slices, slice_type = self.slice_info
        if slice_type == "teacher_send_slice":
            slice_len = self.full_buffer.size(0) // total_slices
            start = slice_index * slice_len
            end = start + slice_len
            self.full_buffer = self.full_buffer[start:end].contiguous()

        self._finalized = True
        return

    def clear(self):
        self.full_buffer = None
        self.total_slices = None
        self.slice_len = None
        self.received_count = 0
        self.slice_info = None
        self._finalized = False


    def pop_full_logits(self):
        self.try_finalize_full_logits()
        logits = self.full_buffer
        assert self.is_complete(), "StudentWorker pop_full_logits: logits not complete"
        assert self._finalized, "StudentWorker pop_full_logits: logits not finalized"
        self.clear()
        return logits

    def get_slice_view(self, slice_index):
        if self.full_buffer is None:
            raise RuntimeError("full_buffer not allocated")
        if self.slice_len is None:
            raise RuntimeError("slice_len not initialized")
        src_rank, slice_index, total_slices, slice_type = self.slice_info
        if slice_type == "student_receive_slice":
            if slice_index >= self.total_slices:
                raise IndexError(f"slice_index={slice_index} >= total_slices={self.total_slices}")
            start = slice_index * self.slice_len
            end = start + self.slice_len
            return self.full_buffer[start:end]
        else:
            return self.full_buffer

    @property
    def has_logits(self):
        return self.full_buffer is not None and self.slice_info is not None

    def broadcast_from_dynamic_holder(self, group, rank):
        has_logits = self.has_logits

        # get broadcast src_rank
        holder_group_rank_tensor = torch.tensor(
            rank if has_logits else -1,
            dtype=torch.int, device=current_platform.device_type
        )
        dist.all_reduce(holder_group_rank_tensor, op=dist.ReduceOp.MAX, group=group)
        holder_group_rank = holder_group_rank_tensor.item()

        # pass when none of the ranks hold tensor
        if holder_group_rank == -1:
            return

        # allocate buffer in other ranks
        if rank == holder_group_rank:
            mock_slice_info = (0, 0, 1, "full")
            meta = [mock_slice_info, tuple(self.full_buffer.shape), str(self.full_buffer.dtype)]
        else:
            meta = [None, None, None]
        dist.broadcast_object_list(meta, group=group, group_src=holder_group_rank)

        slice_info = meta[0]
        shape_tuple = meta[1]
        dtype = getattr(torch, meta[2].split('.')[-1])

        if rank != holder_group_rank:
            self.cache(slice_info, tensor_shape=shape_tuple, tensor_dtype=dtype)

        # perform broadcast
        dist.broadcast(self.full_buffer, group=group, group_src=holder_group_rank)

        # add receive count to make sure that is_complete() returns True
        if not has_logits:
            self.add_receive_count()

        # try to finalize full logits in all ranks
        self.try_finalize_full_logits()

class TeacherWorker(Worker):

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None
        # Store the output tensors to prevent their GPU memory from being released.
        self.topk_probs = None
        self.topk_log_probs = None
        self.topk_indices = None
        self.topk_inf_mask = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)

        self.strategy = create_strategy(worker=self)

        self.strategy.initialize(model_provider=default_actor_model_provider)
        self.tokenizer = self.strategy.tokenizer

        if self.pipeline_config.resume_from_checkpoint:
            load_dir = os.path.join(self.pipeline_config.resume_from_checkpoint, self.cluster_name)
            self.strategy.load_checkpoint(load_dir=load_dir, tag="checkpoint")

        self.logger.info(f"{self.worker_name} initialized")

        self.strategy.offload_states()

    def get_tensor_name_list_for_transfer(self):
        return ['topk_probs', 'topk_log_probs', 'topk_indices', 'topk_inf_mask']

    def forward_func(self, data: DataProto, output_tensor: torch.Tensor, non_loss_data: bool = True):
        topk_probs, topk_log_probs, topk_indices, topk_inf_mask = self.strategy.op_compute_topk_probs_and_indices(
            output_tensor,
            topk=self.pipeline_config.logits_topk,
            target_vocab_size=self.pipeline_config.target_vocab_size,
            kd_temperature=self.pipeline_config.kd_temperature,
            teacher_temperature=self.pipeline_config.teacher_temperature
        )
        return torch.tensor(0., device=output_tensor.device), {
            'topk_probs': topk_probs.detach(),
            'topk_log_probs': topk_log_probs.detach(),
            'topk_indices': topk_indices.detach(),
            'topk_inf_mask': topk_inf_mask.detach()
        }

    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST_COLLECT_ALL, clear_cache=False)
    def forward(self, data: DataProto):
        data = self.strategy.get_data_input(data)
        if "labels" in data.batch.keys():
            keep_keys = [k for k in data.batch.keys() if k != "labels"]
            data = data.select(batch_keys=keep_keys, deepcopy=False)
        is_offload_states = data.meta_info.get("is_offload_states", False)
        metrics = {}
        with state_offload_manger(
                strategy=self.strategy,
                metrics=metrics,
                metric_infix=f"{self.cluster_name}/teacher_forward",
                is_offload_states=is_offload_states,
                load_kwargs={"include": None},
        ):
            data = data.to(current_platform.device_type)
            data.meta_info["micro_batch_size"] = self.pipeline_config.teacher.training_args.per_device_train_batch_size
            assert self.pipeline_config.teacher.training_args.per_device_train_batch_size <= \
                   self.pipeline_config.student.training_args.per_device_train_batch_size, \
                "Teacher's per_device_train_batch_size must be less than or equal to student's."

            data.meta_info["output_on_all_tp_cp_ranks"] = True
            self.logger.info(f"global_step: {data.meta_info.get('global_step', 0)}")

            forward_func = self.forward_func
            if self.worker_config.use_sequence_packing:
                from roll.utils.sequence_packing import SequencePackingDistillForwardWrapper
                forward_func = SequencePackingDistillForwardWrapper(self.strategy, forward_func)

            with torch.no_grad():
                forward_output = self.strategy.forward_step(batch=data, forward_func=forward_func)
            self.topk_probs = None
            self.topk_log_probs = None
            self.topk_indices = None
            self.topk_inf_mask = None
            if forward_output:
                self.topk_probs = forward_output['topk_probs']
                self.topk_log_probs = forward_output['topk_log_probs']
                self.topk_indices = forward_output['topk_indices']
                self.topk_inf_mask = forward_output['topk_inf_mask']

        output = DataProto(meta_info={"metrics": metrics}).to("cpu")

        return output

    def logits_transfer(self, tensor_name_for_transfer, model_update_name, broadcast_comm_plan_args, p2p_tgt_workers, p2p_entry_list, backend):
        rank_info = self.rank_info
        assert hasattr(self, tensor_name_for_transfer), f"Logits transfer: teacher worker doesn't have attr {tensor_name_for_transfer}"
        logits = getattr(self, tensor_name_for_transfer)
        self.logger.info(
            f"[Teacher][logits_transfer] start. "
            f"rank={dist.get_rank()}, pp={rank_info.pp_rank}, dp={rank_info.dp_rank}, tp={rank_info.tp_rank}, "
            f"logits_shape={tuple(logits.shape) if logits is not None else None}, "
            f"dtype={getattr(logits, 'dtype', None)}, device={getattr(logits, 'device', None)}"
        )

        # ---- Process P2P First----
        if len(p2p_tgt_workers) > 0:
            current_platform.synchronize()
            self.logger.info(f"[Teacher][P2P] sending to {len(p2p_tgt_workers)} workers, backend={backend}")
            logits_handle = MultiprocessingSerializer.serialize(logits)
            refs = []
            for idx, p2p_tgt_worker in enumerate(p2p_tgt_workers):
                entry = p2p_entry_list[idx]
                slice_info = [entry['t_dp'], entry['slice_index'], entry["total_slices"], entry["slice_type"]]
                self.logger.info(f"[Teacher][P2P] target_worker={idx}, slice_info={slice_info}")
                ref = p2p_tgt_worker.receive_p2p_logits.remote(
                    slice_info, backend,
                    tensor_name_for_transfer=tensor_name_for_transfer,
                    tensor_slice_handle=logits_handle,
                    tensor_shape=logits.shape, tensor_dtype=logits.dtype,
                )
                refs.append(ref)
            ray.get(refs)
            self.logger.info("[Teacher][P2P] all sends completed")

        # ---- Then Broadcast ----
        if broadcast_comm_plan_args is not None:
            broadcast_tgt_workers = broadcast_comm_plan_args["tgt_workers"]
            slice_info_list = broadcast_comm_plan_args["slice_info"]
            self.logger.info(f"[Teacher][Broadcast] target_workers={len(broadcast_tgt_workers)}, backend={backend}")

            if backend == "ray":
                refs = []
                for idx, tgt_worker in enumerate(broadcast_tgt_workers):
                    slice_info = slice_info_list[idx]
                    self.logger.info(f"[Teacher][Broadcast][Ray] target_worker={idx}, slice_info={slice_info}")
                    ref = tgt_worker.receive_broadcast_logits.remote(
                        slice_info, backend,
                        tensor_name_for_transfer=tensor_name_for_transfer,
                        tensor_slice=logits,
                        tensor_shape=logits.shape, tensor_dtype=logits.dtype,
                    )
                    refs.append(ref)
                ray.get(refs)
                self.logger.info("[Teacher][Broadcast][Ray] all sends completed")
            else:
                refs = []
                for idx, tgt_worker in enumerate(broadcast_tgt_workers):
                    slice_info = slice_info_list[idx]
                    self.logger.info(
                        f"[Teacher][Broadcast][NCCL] target_worker={idx}, slice_info={slice_info}, group_name={broadcast_comm_plan_args['group_name']}")
                    ref = tgt_worker.receive_broadcast_logits.remote(
                        slice_info, backend,
                        tensor_name_for_transfer=tensor_name_for_transfer,
                        tensor_shape=logits.shape, tensor_dtype=logits.dtype,
                        group_name=broadcast_comm_plan_args['group_name']
                    )
                    refs.append(ref)
                self.logger.info("[Teacher][Broadcast][NCCL] calling collective.broadcast() as src_rank=0 ...")
                collective.broadcast(
                    tensor=logits, src_rank=0, group_name=broadcast_comm_plan_args['group_name']
                )
                self.logger.info("[Teacher][Broadcast][NCCL] broadcast() done")
                ray.get(refs)
                self.logger.info("[Teacher][Broadcast][NCCL] all sends completed")





