import os
from typing import Dict, Union, Optional

import torch
from codetiming import Timer

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import register, Dispatch
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.utils.functionals import reduce_metrics
from roll.models.model_providers import default_actor_model_provider
from roll.platforms import current_platform


class SFTWorker(Worker):
    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

    @register(Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)
        self.strategy = create_strategy(worker=self)
        self.strategy.initialize(model_provider=default_actor_model_provider)
        self.logger.info(f"{self.worker_name} initialized")

    @register(Dispatch.DP_MP_DISPATCH_FIRST, clear_cache=False)
    def train_step(self, data: DataProto):
        data = data.to(current_platform.device_type)
        data = self.strategy.get_data_input(data)

        metrics = self.strategy.train_step(batch=data, loss_func=self.loss_func)

        output = DataProto(meta_info={"metrics": metrics}).to("cpu")
        return output

    @register(Dispatch.DP_MP_DISPATCH_FIRST, clear_cache=False)
    def val_step(self, data: DataProto):
        data = data.to(current_platform.device_type)
        data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size
        data = self.strategy.get_data_input(data)
        metrics = self.strategy.forward_step(batch=data, forward_func=self.loss_func)
        if metrics is None:
            metrics = {}
        metrics = reduce_metrics(metrics)
        output = DataProto(meta_info={"metrics": metrics}).to("cpu")
        return output

    @register(Dispatch.ONE_TO_ALL)
    def do_checkpoint(self, global_step, is_last_step=False):
        with Timer("do_checkpoint") as total_timer:
            ckpt_id = f"checkpoint-{global_step}"
            save_dir = os.path.join(self.pipeline_config.output_dir, self.worker_name, ckpt_id, self.cluster_name)
            self.logger.info(f"save checkpoint-{global_step} to {save_dir}")
            exec_metrics: Dict = self.strategy.save_checkpoint(save_dir, global_step, ckpt_id, is_last_step=is_last_step)

        metrics = {
            f"time/{self.cluster_name}/do_checkpoint/total": total_timer.last,
        }
        metric_prefix = f"time/{self.cluster_name}/do_checkpoint"
        metrics.update({f"{metric_prefix}/{k}": v for k, v in exec_metrics.items()})
        output = DataProto(meta_info={"metrics": metrics})
        return output

    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        labels = data.batch["labels"]
        batch_num_tokens = data.meta_info['batch_num_tokens']['labels']
        loss, metrics = self.strategy.op_compute_language_loss(output_tensor, labels, batch_num_tokens)
        return loss, metrics
