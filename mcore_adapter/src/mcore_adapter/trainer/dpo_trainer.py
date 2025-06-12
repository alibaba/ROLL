from functools import partial
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel

from ..constants import IGNORE_INDEX
from ..parallel_functions.vocab_parallel import vocab_parallel_logprobs
from ..utils import get_logger
from .trainer import McaTrainer


if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import DataLoader

    from ..models import VirtualModels
    from ..training_args import TrainingArguments
    from .dpo_config import DPOConfig

logger = get_logger(__name__)


class DPOTrainer(McaTrainer):
    metrics_keys = [
        "loss",
        "rewards/chosen",
        "rewards/rejected",
        "rewards/accuracies",
        "rewards/margins",
        "logps/chosen",
        "logps/rejected",
        "sft_loss",
    ]

    def __init__(
        self,
        model: "VirtualModels" = None,
        train_config: "DPOConfig" = None,
        ref_model: Optional["VirtualModels"] = None,
        args: "TrainingArguments" = None,
        **kwargs,
    ):
        self.ref_model = ref_model
        if ref_model is not None:
            self.ref_model.eval()
        else:
            assert (
                not train_config.use_ref_model
            ), f"ref_model must be provided when using pref_loss: {train_config.pref_loss}"
        self.train_config = train_config
        super().__init__(
            model=model,
            args=args,
            **kwargs,
        )

        if self.args.calculate_per_token_loss:
            raise ValueError("It's not supported to calculate per token loss in DPO training.")
        if self.args.sequence_packing:
            raise ValueError("It's not supported to use sequence packing in DPO training.")

    def _get_batch_on_this_cp_rank(self, batch: Dict[str, "Tensor"]):
        not_cp_parallel_keys = ["reference_chosen_logps", "reference_rejected_logps"]
        not_cp_parallel_dict = {key: batch.pop(key) for key in not_cp_parallel_keys if key in batch}
        batch = self.model.get_batch_on_this_cp_rank(batch)
        return {**batch, **not_cp_parallel_dict}

    def _pre_compute_loss(self, data_iterator: Iterator, model: DistributedDataParallel, compute_ref_logps=False):
        inputs = self._prepare_train_inputs(data_iterator)
        labels = inputs.pop("labels").clone()
        loss_mask = (labels != IGNORE_INDEX).float()
        labels[labels == IGNORE_INDEX] = 0
        outputs = (labels, loss_mask)
        if not compute_ref_logps:
            outputs += (inputs.pop("reference_chosen_logps", None), inputs.pop("reference_rejected_logps", None))
        output_tensor = model(**inputs)
        return output_tensor, *outputs

    def _post_compute_log_probs(self, labels: "torch.Tensor", loss_mask: "torch.Tensor", logits: "torch.Tensor", non_loss_data: bool=False):
        batch_size = labels.size(0) // 2
        logprobs = vocab_parallel_logprobs(logits, labels)
        logprobs = (logprobs * loss_mask).sum(-1)
        if mpu.get_context_parallel_world_size() > 1:
            dist.all_reduce(logprobs, group=mpu.get_context_parallel_group())
        chosen_logps, rejected_logps = torch.split(logprobs.clone().detach(), batch_size, 0)
        if non_loss_data:
            return {"chosen_logps": chosen_logps, "rejected_logps": rejected_logps}
        return logprobs, {"chosen_logps": chosen_logps, "rejected_logps": rejected_logps}

    def odds_ratio_loss(
        self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor", response_lens: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes ORPO's odds ratio (OR) loss for batched log probabilities of the policy model.
        modified from LLaMA-Factory
        """
        batch_size = response_lens.size(0) // 2
        chosen_length, rejected_length = response_lens.split(batch_size, 0)
        chosen_logps = chosen_logps / chosen_length
        rejected_logps = rejected_logps / rejected_length
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.train_config.beta * odds_ratio_loss
        chosen_rewards = self.train_config.beta * chosen_logps.detach()
        rejected_rewards = self.train_config.beta * rejected_logps.detach()
        return orpo_loss, chosen_rewards, rejected_rewards

    def dpo_loss(
        self,
        policy_chosen_logps: "torch.FloatTensor",
        policy_rejected_logps: "torch.FloatTensor",
        reference_chosen_logps: "torch.FloatTensor",
        reference_rejected_logps: "torch.FloatTensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        "modified from trl"
        # chosen_logratios = policy_chosen_logps - reference_chosen_logps
        # rejected_logratios = policy_rejected_logps - reference_rejected_logps
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        losses = (
            -F.logsigmoid(self.train_config.beta * logits) * (1 - self.train_config.label_smoothing)
            - F.logsigmoid(-self.train_config.beta * logits) * self.train_config.label_smoothing
        )
        chosen_rewards = self.train_config.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.train_config.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards

    def _post_compute_loss(
        self,
        labels: "torch.Tensor",
        loss_mask: "torch.Tensor",
        ref_chosen_logps: "torch.Tensor",
        ref_rejected_logps: "torch.Tensor",
        logits: "torch.Tensor",
    ):
        batch_size = labels.size(0) // 2
        logprobs = vocab_parallel_logprobs(logits, labels)
        logprobs = (logprobs * loss_mask).sum(-1)
        response_lens = loss_mask.sum(-1)
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size > 1:
            dist.all_reduce(logprobs, group=mpu.get_context_parallel_group())
            dist.all_reduce(response_lens, group=mpu.get_context_parallel_group())
        chosen_logps, rejected_logps = torch.split(logprobs, batch_size, 0)

        if self.train_config.pref_loss == "sigmoid":
            loss, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps=chosen_logps,
                policy_rejected_logps=rejected_logps,
                reference_chosen_logps=ref_chosen_logps,
                reference_rejected_logps=ref_rejected_logps,
            )
        elif self.train_config.pref_loss == "orpo":
            loss, chosen_rewards, rejected_rewards = self.odds_ratio_loss(
                chosen_logps=chosen_logps, rejected_logps=rejected_logps, response_lens=response_lens
            )
        else:
            raise ValueError(f"pref_loss: {self.train_config.pref_loss} is not supported.")

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        loss = loss.mean()
        chosen_logps_avg = chosen_logps / response_lens.split(batch_size, 0)[0]
        sft_loss = -chosen_logps_avg.mean()
        metrics = {
            "loss": loss.clone().detach(),
            "rewards/chosen": chosen_rewards.mean().detach(),
            "rewards/rejected": rejected_rewards.mean().detach(),
            "rewards/accuracies": reward_accuracies.mean().detach(),
            "rewards/margins": (chosen_rewards - rejected_rewards).mean().detach(),
            "logps/chosen": chosen_logps.mean().detach(),
            "logps/rejected": rejected_logps.mean().detach(),
            "sft_loss": sft_loss.detach(),
            # TODO: is logits mean needed for metrics? it needs more calculation and communication
        }
        return loss, metrics

    def _inner_compute_log_probs_forward_step(self, data_iterator: Iterator, model):
        outputs = self._pre_compute_loss(data_iterator, model, compute_ref_logps=True)
        return outputs[0], partial(self._post_compute_log_probs, *outputs[1:])

    @torch.no_grad()
    def compute_reference_log_probs(
        self, models: "VirtualModels", data_list: List[Dict[str, Any]], seq_length: int, micro_batch_size: int
    ) -> Optional[List[Dict[str, "torch.Tensor"]]]:
        data_iterator = [iter(data_list) for _ in range(len(models))]
        metrics_tensors: List[Dict[str, "torch.Tensor"]] = self.forward_backward_func(
            forward_step_func=self._inner_compute_log_probs_forward_step,
            data_iterator=data_iterator,
            model=models.get_models(),
            num_microbatches=len(data_list),
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            forward_only=True,
            collect_non_loss_data=True,
        )
        if not mpu.is_pipeline_last_stage():
            # only the last stage needs the reference log probs to compute the loss
            return None
        return metrics_tensors

    def training_step(self, models: List[DistributedDataParallel], data_iterator, seq_length):
        data_list = list(data_iterator)
        # 1. gather reference log probs
        if self.train_config.use_ref_model:
            ref_log_probs = self.compute_reference_log_probs(
                self.ref_model, data_list, seq_length, micro_batch_size=self.args.per_device_train_batch_size * 2
            )
            if ref_log_probs is not None:
                for i in range(len(ref_log_probs)):
                    data_list[i]["reference_chosen_logps"] = ref_log_probs[i]["chosen_logps"]
                    data_list[i]["reference_rejected_logps"] = ref_log_probs[i]["rejected_logps"]

        # 2. train
        for model in models:
            model.train()
            model.zero_grad_buffer()
        self.optimizer.zero_grad()

        data_iterator = [iter(data_list) for _ in range(len(models))]
        metrics_tensors: List[Dict[str, "torch.Tensor"]] = self.forward_backward_func(
            forward_step_func=self._inner_forward_step,
            data_iterator=data_iterator,
            model=models,
            num_microbatches=self.args.gradient_accumulation_steps,
            seq_length=seq_length,
            micro_batch_size=self.args.per_device_train_batch_size * 2,
            forward_only=False,
        )
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
        if update_successful:
            self.lr_scheduler.step()
            skipped_iter = 0
        else:
            skipped_iter = 1

        if len(metrics_tensors) > 0 and "loss" in metrics_tensors[0]:
            loss = torch.stack([metrics["loss"] for metrics in metrics_tensors]).view(-1).mean()
        else:
            loss = torch.tensor(0.0, device=self.args.device)
        return loss, metrics_tensors, skipped_iter, grad_norm, num_zeros_in_grad

    def _get_step_iterator_and_seq_length(self, epoch_iterator, standard_batch_size = None):
        standard_batch_size = standard_batch_size or self.args.per_device_train_batch_size * 2
        return super()._get_step_iterator_and_seq_length(epoch_iterator, standard_batch_size)

    def _stream_eval_inputs(self, eval_dataloader: "DataLoader", standard_batch_size=None):
        standard_batch_size = standard_batch_size or self.args.per_device_eval_batch_size * 2
        models = self.ref_model
        for step_inputs, seq_length, batch_size in super()._stream_eval_inputs(eval_dataloader, standard_batch_size):
            if self.train_config.use_ref_model:
                ref_log_probs = self.compute_reference_log_probs(models, step_inputs, seq_length, batch_size)
                if ref_log_probs is not None:
                    for i in range(len(ref_log_probs)):
                        step_inputs[i]["reference_chosen_logps"] = ref_log_probs[i]["chosen_logps"]
                        step_inputs[i]["reference_rejected_logps"] = ref_log_probs[i]["rejected_logps"]
            yield step_inputs, seq_length, batch_size
