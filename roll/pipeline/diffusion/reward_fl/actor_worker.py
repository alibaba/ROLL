import numpy as np
import torch

from tqdm import tqdm

from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.base_worker import ActorWorker as BaseActorWorker
from roll.utils.functionals import append_to_dict
from roll.platforms import current_platform


class ActorWorker(BaseActorWorker):
    
    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST, clear_cache=False)
    def train_step(self, data: DataProto):
        
        global_step = data.meta_info.get("global_step", 0)
        metrics = {}

        data = data.to(current_platform.device_type)

        per_device_train_batch_size = self.worker_config.training_args.per_device_train_batch_size
        backward_batch_size = (
            per_device_train_batch_size * self.worker_config.training_args.gradient_accumulation_steps
        )

        dataloader = data.make_iterator(
            mini_batch_size=backward_batch_size,
            epochs=1,
            dataloader_kwargs={"shuffle": False},
        )

        for batch_idx, data in tqdm(
            enumerate(dataloader),
            desc=f"{self.worker_name} train global step {global_step}",
            total=data.batch.batch_size[0] // backward_batch_size,
        ):
            pg_metrics = self.strategy.train_step(batch=data, loss_func=self.loss_func)
            append_to_dict(metrics, pg_metrics)

        metrics["actor/loss"] = np.mean(metrics["actor/loss"])
        metrics["actor/face_score"] = np.mean(metrics["actor/face_score"])
        metrics["actor/kl_loss"] = np.mean(metrics["actor/kl_loss"])
        data.to("cpu")

        output = DataProto(meta_info={"metrics": metrics})
        return output

    def loss_func(self, data: DataProto, loss: torch.Tensor, face_score: torch.Tensor, kl_loss: torch.Tensor):
        """
            data: DataProto, from train_step
        """
        metrics = {
            "actor/loss": loss.float().detach().cpu().item(),
            "actor/face_score": face_score.float().detach().cpu().item(),
            "actor/kl_loss": kl_loss.float().detach().cpu().item(),
        }

        return loss, metrics
