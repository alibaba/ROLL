import os
from typing import Union, Optional, Dict
from tensordict import TensorDict

import ray
import torch
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
from roll.utils.cuda_ipc_utils import MultiprocessingSerializer
from roll.utils.offload_states import OffloadStateType
from roll.pipeline.distill.various_divergence import VariousDivergence, GPTLMLoss



class StudentWorker(Worker):

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None
        self.kl_loss_func = None
        self.teacher_logits = None

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
        self.gpt_loss_func = GPTLMLoss()

    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST, clear_cache=False)
    def train_step(self, data: DataProto):
        """
        return DataProto(meta_info={'metrics': metrics})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        # 获取teacher logits
        self.teacher_logits = MultiprocessingSerializer.deserialize(
            data.meta_info.get("teacher_logits_handles")[self.rank_info.rank])
        metrics = {}
        self.logger.info(f"is_offload_states: {is_offload_states}")
        with state_offload_manger(
                strategy=self.strategy,
                metrics=metrics,
                metric_infix=f"{self.cluster_name}/train_step",
                is_offload_states=is_offload_states,
                load_kwargs={"include": None},
        ):
            data = data.to("cuda")
            data = self.strategy.get_data_input(data)
            if "labels" in data.batch.keys():
                # rename key: labels -> labels_for_loss
                data.batch.rename_key_("labels", "labels_for_loss")
            self.logger.info(f"global_step: {data.meta_info.get('global_step',0)}")
            per_device_train_batch_size = self.worker_config.training_args.per_device_train_batch_size
            backward_batch_size = (
                    per_device_train_batch_size * self.worker_config.training_args.gradient_accumulation_steps
            )
            student_metrics = self.strategy.train_step(batch=data, loss_func=self.loss_func)
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

        teacher_logits = self.teacher_logits
        student_logits = self.strategy.op_compute_logits(output_tensor)

        labels = data.batch['labels_for_loss']
        attention_mask = data.batch['attention_mask']
        gpt_loss = self.gpt_loss_func(student_logits, labels)
        if teacher_logits.shape[-1] != student_logits.shape[-1]:
            teacher_logits = teacher_logits[:, :, : min(student_logits.shape[-1], teacher_logits.shape[-1])]
        distill_loss = self.kl_loss_func(student_logits, teacher_logits, labels, attention_mask)
        loss = ((1 - self.pipeline_config.distill_loss_weight) * gpt_loss
                + self.pipeline_config.distill_loss_weight * distill_loss)
        student_metrics = {
            "train/loss": loss.detach().item(),
            "train/distill_loss": distill_loss.detach().item(),
            "train/student_loss": gpt_loss.detach().item(),
        }
        return loss, student_metrics

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


class TeacherWorker(Worker):

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None
        # Store the output logits to prevent their GPU memory from being released.
        self.logits = None

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

    def forward_func(self, data: DataProto, output_tensor: torch.Tensor, non_loss_data: bool=True):
        logits = self.strategy.op_compute_logits(output_tensor)
        return torch.tensor(0, dtype=float, device=output_tensor.device), {'logits': logits.detach()}

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
            data = data.to("cuda")
            data.meta_info["micro_batch_size"] = self.pipeline_config.teacher.training_args.per_device_train_batch_size
            data.meta_info["output_on_all_tp_ranks"] = True
            self.logger.info(f"global_step: {data.meta_info.get('global_step', 0)}")
            with torch.no_grad():
                forward_output = self.strategy.forward_step(batch=data, forward_func=self.forward_func)
            self.logits = None
            if forward_output:
                self.logits = forward_output['logits']
        return MultiprocessingSerializer.serialize(self.logits)

