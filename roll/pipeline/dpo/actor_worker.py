import numpy as np
import torch
from tqdm import tqdm
from typing import Dict

from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.base_worker import ActorWorker as BaseActorWorker
from roll.platforms import current_platform
from roll.utils.context_managers import state_offload_manger
from roll.utils.functionals import append_to_dict
from roll.utils.offload_states import OffloadStateType


def get_logps(
    per_token_logps: torch.LongTensor,
    attention_mask,
    prompt_id_lens,
) -> torch.FloatTensor:
    loss_masks = attention_mask.clone().bool()
    for mask, source_len in zip(loss_masks, prompt_id_lens):
        mask[:source_len] = False
    loss_masks = loss_masks[:, 1:]

    logprobs_sums = (per_token_logps * loss_masks).sum(-1)

    chosen_logps = logprobs_sums[0::2]
    rejected_logps = logprobs_sums[1::2]

    return chosen_logps, rejected_logps


def loss_fn(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    ipo: bool = False,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios

    if ipo:
        losses = (logits - 1 / (2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = (
            -torch.nn.functional.logsigmoid(beta * logits) * (1 - label_smoothing)
            - torch.nn.functional.logsigmoid(-beta * logits) * label_smoothing
        )

    loss = losses.mean()
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return loss, chosen_rewards, rejected_rewards


class ActorWorker(BaseActorWorker):
    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST, clear_cache=False)
    def train_step(self, data: DataProto):
        """
        return DataProto(meta_info={'metrics': metrics})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", False)
        metrics = {}
        self.logger.info(f"{self.worker_name} generate global step {global_step}")

        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/train_step",
            is_offload_states=is_offload_states,
        ):
            data = data.to(current_platform.device_type)
            data = self.strategy.get_data_input(data)
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

            metrics["actor/lr"] = self.strategy.scheduler.get_last_lr()[0]
            data.to("cpu")

        output = DataProto(meta_info={"metrics": metrics})
        return output

    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST, clear_cache=False)
    def compute_log_probs(self, data: DataProto):
        """
        return DataProto.from_dict(tensors={'log_probs': output})
        """

        data = self.strategy.get_data_input(data)
        data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", False)
        metrics = {}

        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/compute_log_probs",
            is_offload_states=is_offload_states,
        ):
            data = data.to(current_platform.device_type)
            data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size

            with torch.no_grad():
                results: Dict[str, torch.Tensor] = self.strategy.forward_step(
                    batch=data, forward_func=self.forward_func_log_probs
                )
            if results is None:
                return DataProto(batch=None, meta_info={"metrics": metrics})
            output = DataProto.from_dict(tensors={"log_probs": results["log_probs"]})
            output = output.to("cpu")
            data.to("cpu")
        output.meta_info = {"metrics": metrics}
        return output

    def forward_func_log_probs(self, data: DataProto, output_tensor: torch.Tensor):
        """
        forward func 接口定义:
            data: DataProto, 由forward_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """

        log_probs = self.strategy.op_compute_log_probs(
            logits=output_tensor, input_ids=data.batch["input_ids"], attention_mask=data.batch["attention_mask"]
        )

        return log_probs, {"log_probs": log_probs}

    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        """
        loss func接口定义:
            data: DataProto, 由train_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """
        input_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        prompt_id_lens = data.batch["prompt_id_lens"]
        reference_log_probs = data.batch["reference_log_probs"]
        reference_chosen_logps, reference_rejected_logps = get_logps(
            reference_log_probs, attention_mask, prompt_id_lens
        )

        log_probs = self.strategy.op_compute_log_probs(
            logits=output_tensor, input_ids=input_ids, attention_mask=attention_mask
        )
        chosen_logps, rejected_logps = get_logps(log_probs, attention_mask, prompt_id_lens)

        ipo = data.meta_info.get("ipo", False)
        beta = data.meta_info.get("beta", 0.1)
        label_smoothing = data.meta_info.get("label_smoothing", 0.0)

        loss, chosen_rewards, reject_rewards = loss_fn(
            chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps, ipo, beta, label_smoothing
        )

        acc = (chosen_rewards > reject_rewards).float().mean().item()

        metrics = {
            "actor/loss": loss.item(),
            "actor/acc": acc,
            "actor/chosen_reward": chosen_rewards.mean().item(),
            "actor/reject_reward": reject_rewards.mean().item(),
        }

        return loss, metrics
