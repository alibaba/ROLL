import os
import torch
import torchvision
import torch.distributed as dist

from typing import Callable, Dict, Tuple
from codetiming import Timer

from roll.distributed.strategy.deepspeed_strategy import DeepSpeedTrainStrategy as BaseDeepSpeedTrainStrategy
from roll.distributed.scheduler.protocol import DataProto
from roll.utils.functionals import append_to_dict
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType


logger = get_logger()


class DeepSpeedTrainStrategy(BaseDeepSpeedTrainStrategy):
    
    strategy_name = "diffusion_deepspeed_train"
    
    def train_step(
        self,
        batch: DataProto,
        loss_func: Callable[[DataProto, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
    ):
        mini_batch_size = self.worker_config.training_args.per_device_train_batch_size
        data_iter = batch.make_iterator(mini_batch_size=mini_batch_size, epochs=1)
        mini_steps = batch.batch.batch_size[0] // self.worker_config.training_args.per_device_train_batch_size
        metrics = {}
        
        for step in range(mini_steps):
            data: DataProto = next(data_iter)
            
            # convert data to dict for DiffusionTrainingModule
            prompt = data.non_tensor_batch["prompt"][0]
            video = list(torch.unbind(data.batch["video"][0], dim=0))
            video = [torchvision.transforms.functional.to_pil_image(v) for v in video]
            data = {"prompt": prompt, "video": video}

            loss, face_score, kl_loss = self.model(data)
            loss, loss_reduced = loss_func(data, loss, face_score, kl_loss)
            append_to_dict(metrics, loss_reduced)
            
            self.model.backward(loss)

            is_gradient_accumulation_boundary = self.model.is_gradient_accumulation_boundary()
            if is_gradient_accumulation_boundary:
                self.load_states(include=[OffloadStateType.optimizer_states])
            self.model.step()
            if is_gradient_accumulation_boundary:
                # global_grad_norm is calculated in optimizer.step thus put it
                # into metrics after optimizer.step
                metrics.update({self.worker_config.name + "/" + "grad_norm": self.model.get_global_grad_norm().item()})
        return metrics

    def offload_states(self):
        pass

    def save_checkpoint(self, save_dir, global_step, ckpt_id, tag="checkpoint", local_state_path=None, **kwargs):
        assert not self.ds_config.is_zero3(), "zero3 is not supported yet"

        logger.info(f"save_dir: {save_dir}")
        if local_state_path is None:
            local_state_path = save_dir

        with Timer("load") as load_timer:
            self.load_states()

        from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
        state_dict = clone_tensors_for_torch_save(self.unwrap_model().state_dict())
        state_dict = self.unwrap_model().export_trainable_state_dict(state_dict, remove_prefix='pipe.dit2.')
        
        # save DiffusionTrainingModule
        if dist.get_rank() == 0:
            os.makedirs(local_state_path, exist_ok=True)
            torch.save(state_dict, os.path.join(local_state_path, "diffusion_module.pth"))

        metrics = {
            "load": load_timer.last,
        }
        return metrics
