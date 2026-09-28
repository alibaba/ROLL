import torch
from megatron.core import parallel_state
from torch.utils.data import DataLoader

from ..parallel_functions.context_parallel import context_parallel_gather
from .trainer import McaTrainer


class RewardTrainer(McaTrainer):
    def __init__(self, model=None, args=None, **kwargs):
        super().__init__(model, args, **kwargs)
        if self.args.calculate_per_token_loss:
            raise ValueError("It's not supported to calculate per token loss in Reward training.")
        if self.args.sequence_packing:
            raise ValueError("It's not supported to use sequence packing in Reward training.")

    def _get_step_iterator_and_seq_length(self, epoch_iterator, standard_batch_size=None):
        standard_batch_size = standard_batch_size or self.args.per_device_train_batch_size * 2
        return super()._get_step_iterator_and_seq_length(epoch_iterator, standard_batch_size)

    def _stream_eval_inputs(self, eval_dataloader: "DataLoader", standard_batch_size=None):
        standard_batch_size = standard_batch_size or self.args.per_device_eval_batch_size * 2
        yield from super()._stream_eval_inputs(eval_dataloader, standard_batch_size)

    def training_step(self, models, data_iterator, seq_length):
        # a real step not a minibatch of gradient accumulation
        for model in models:
            model.train()
            model.zero_grad_buffer()
        self.optimizer.zero_grad()
        if len(models) > 1:
            data_list = list(data_iterator)
            data_iterator = [iter(data_list) for _ in range(len(models))]
        metrics_tensors: list[dict[str, torch.Tensor]] = self.forward_backward_func(
            forward_step_func=self._inner_forward_step,
            data_iterator=data_iterator,
            model=models,
            num_microbatches=self.args.gradient_accumulation_steps,
            seq_length=seq_length,
            micro_batch_size=self.args.per_device_train_batch_size*2,
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

    def _pre_compute_loss(self, data_iterator, model):
        inputs = next(data_iterator)
        attention_mask_2d = inputs.get("attention_mask")
        inputs.pop("labels", None)
        inputs = self._prepare_train_inputs(iter((inputs,)))
        output_tensor = model(**inputs)
        return output_tensor, attention_mask_2d

    def _post_compute_loss(self, attention_mask_2d: torch.Tensor, values: torch.Tensor):
        cp_size = self.model.config.context_parallel_size
        batch_size = attention_mask_2d.size(0) // 2
        attention_mask_2d = attention_mask_2d.sum(dim=-1, keepdim=True) - 1
        if cp_size > 1:
            values = context_parallel_gather(values, parallel_dim=1)
        scores = values.squeeze(-1).gather(dim=1, index=attention_mask_2d).squeeze(-1)
        chosen_scores, rejected_scores = torch.split(scores, batch_size, dim=0)
        loss = -torch.nn.functional.logsigmoid(chosen_scores.float() - rejected_scores.float()).mean()
        return loss, {"loss": loss.clone().detach()}
