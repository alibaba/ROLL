from typing import Dict, Any, List, Optional
import torch
import numpy as np
from ray.util.timer import _Timer
from codetiming import Timer

from roll.utils.functionals import masked_mean, reduce_metrics


class MetricsManager:
    """
    Organizes metrics for PPO pipeline
    """

    def __init__(self):
        self.metrics = {}
        self.domain_metrics = {}
        self.timers = {}

    def add_metric(self, name: str, value: Any) -> None:
        self.metrics[name] = value

    def add_metrics(self, metrics_dict: Dict[str, Any]) -> None:
        self.metrics.update(metrics_dict)

    def add_reduced_metrics(self, metrics_dict: Dict[str, Any], prefix: str = "", reduce_func=np.mean) -> None:
        reduced = reduce_metrics(metrics_dict, reduce_func=reduce_func)
        if prefix:
            reduced = {f"{prefix}/{k}": v for k, v in reduced.items()}
        self.metrics.update(reduced)

    def add_domain_metrics(self, domain: str, metrics_dict: Dict[str, Any]) -> None:
        if not metrics_dict:
            return

        if domain not in self.domain_metrics:
            self.domain_metrics[domain] = {}

        self.domain_metrics[domain].update(metrics_dict)

    def get_metrics(self) -> Dict[str, Any]:
        all_metrics = self.metrics.copy()

        for domain, domain_metrics in self.domain_metrics.items():
            for key, value in domain_metrics.items():
                all_metrics[f"{domain}/{key}"] = value

        return all_metrics

    def clear_metrics(self) -> None:
        self.metrics = {}
        self.domain_metrics = {}

    def add_system_metrics(
        self, global_step: int, batch_size: int, resource_manager=None, actor_infer=None, actor_train=None
    ) -> None:
        self.metrics["system/global_step"] = global_step
        self.metrics["system/batch_size"] = batch_size
        self.metrics["system/samples"] = (global_step + 1) * batch_size
        for name, timer in self.timers.items():
            if hasattr(timer, "mean_throughput"):
                self.metrics[f"system/{name}/tps"] = timer.mean_throughput
                if hasattr(timer, "mean"):
                    self.metrics[f"system/time/{name}_mean"] = timer.mean

                if resource_manager and name == "tps":
                    self.metrics["system/tps_gpu"] = timer.mean_throughput / resource_manager.num_gpus

                if actor_infer and name == "actor_infer":
                    self.metrics["system/actor_infer/tps_gpu"] = timer.mean_throughput / actor_infer.world_size
                    self.metrics["system/actor_infer/tps_dp"] = timer.mean_throughput / actor_infer.dp_size

                if actor_infer and name == "actor_infer_response":
                    self.metrics["system/actor_infer/response/tps_gpu"] = (
                        timer.mean_throughput / actor_infer.world_size
                    )
                    self.metrics["system/actor_infer/response/tps_dp"] = timer.mean_throughput / actor_infer.dp_size

                if actor_train and name == "actor_train":
                    self.metrics["system/actor_train/tps_gpu"] = timer.mean_throughput / actor_train.world_size
                    self.metrics["system/actor_train/tps_dp"] = timer.mean_throughput / actor_train.dp_size

    def add_timer_metrics(self, timer_dict: Dict[str, Timer]) -> None:
        for name, timer in timer_dict.items():
            if hasattr(timer, "last"):
                self.metrics[f"time/{name}"] = timer.last

    def add_token_metrics(self, batch, prefix: str = "token") -> Dict[str, Any]:
        response_mask = batch.batch["response_mask"][:, 1:].bool()
        prompt_mask = batch.batch["prompt_mask"].bool()

        max_response_length = batch.batch["responses"].shape[-1]
        prompt_length = prompt_mask.sum(-1).float()
        response_length = response_mask.sum(-1).float()
        sequence_score = batch.batch["scores"]

        max_score = 1
        min_score = 0
        correct_mask = sequence_score == max_score
        incorrect_mask = sequence_score == min_score

        metrics = {}

        prompt_length_max = torch.max(prompt_length).detach().item()
        prompt_length_min = torch.min(prompt_length).detach().item()
        prompt_length_mean = torch.mean(prompt_length).detach().item()

        metrics[f"{prefix}/prompt_length/mean"] = prompt_length_mean
        metrics[f"{prefix}/prompt_length/max"] = prompt_length_max
        metrics[f"{prefix}/prompt_length/min"] = prompt_length_min

        response_length_max = torch.max(response_length).detach().item()
        response_length_min = torch.min(response_length).detach().item()
        response_length_mean = torch.mean(response_length).detach().item()
        response_length_diff = response_length_max - response_length_min

        metrics[f"{prefix}/response_length/mean"] = response_length_mean
        metrics[f"{prefix}/response_length/max"] = response_length_max
        metrics[f"{prefix}/response_length/min"] = response_length_min
        metrics[f"{prefix}/response_length/diff"] = response_length_diff

        total_length = prompt_length + response_length
        total_length_max = torch.max(total_length).detach().item()
        total_length_min = torch.min(total_length).detach().item()
        total_length_mean = torch.mean(total_length).detach().item()
        total_length_diff = total_length_max - total_length_min

        metrics[f"{prefix}/total_length/mean"] = total_length_mean
        metrics[f"{prefix}/total_length/max"] = total_length_max
        metrics[f"{prefix}/total_length/min"] = total_length_min
        metrics[f"{prefix}/total_length/diff"] = total_length_diff

        try:
            metrics[f"{prefix}/total_response_length/clip"] = (
                torch.sum(response_length == max_response_length).detach().item()
            )
        except:
            pass

        try:
            metrics[f"{prefix}/right_response_length/clip"] = (
                torch.sum(response_length[correct_mask] == max_response_length).detach().item()
            )
            metrics[f"{prefix}/right_response_length/mean"] = (
                torch.mean(response_length[correct_mask & (response_length != max_response_length)]).detach().item()
            )
            metrics[f"{prefix}/right_response_length/max"] = (
                torch.max(response_length[correct_mask & (response_length != max_response_length)]).detach().item()
            )
            metrics[f"{prefix}/right_response_length/min"] = (
                torch.min(response_length[correct_mask & (response_length != max_response_length)]).detach().item()
            )
        except:
            pass

        try:
            metrics[f"{prefix}/error_response_length/clip"] = (
                torch.sum(response_length[incorrect_mask] == max_response_length).detach().item()
            )
            metrics[f"{prefix}/error_response_length/mean"] = (
                torch.mean(response_length[incorrect_mask & (response_length != max_response_length)]).detach().item()
            )
            metrics[f"{prefix}/error_response_length/max"] = (
                torch.max(response_length[incorrect_mask & (response_length != max_response_length)]).detach().item()
            )
            metrics[f"{prefix}/error_response_length/min"] = (
                torch.min(response_length[incorrect_mask & (response_length != max_response_length)]).detach().item()
            )
        except:
            pass
        self.add_metrics(metrics)
        return metrics

    def add_values_metrics(self, batch, prefix: str = "critic") -> Dict[str, Any]:
        metrics = {}

        sequence_score = batch.batch["scores"]
        sequence_reward = batch.batch["token_level_rewards"].sum(-1)
        sequence_reward_mean = batch.batch["token_level_rewards"].mean(-1)

        advantages = batch.batch["advantages"]
        prompt_mask = batch.batch["prompt_mask"].bool()
        response_mask = batch.batch["final_response_mask"].clone().bool()
        raw_advantages = batch.batch["raw_advantages"]
        returns = batch.batch["returns"]
        agg_entropy = batch.meta_info.get("agg_entropy", torch.tensor(0))

        max_score = 1
        min_score = 0

        correct_mask = sequence_score == max_score
        incorrect_mask = sequence_score == min_score

        metrics[f"{prefix}/entropy/mean"] = agg_entropy.item()
        metrics[f"{prefix}/correct/mean"] = (sequence_score == max_score).detach().float().mean().item()

        metrics[f"{prefix}/score_distribution/max_value"] = max_score
        metrics[f"{prefix}/score_distribution/min_value"] = min_score
        metrics[f"{prefix}/score_distribution/correct_samples_ratio"] = (
            (sequence_score == max_score).float().mean().item()
        )
        metrics[f"{prefix}/score_distribution/incorrect_samples_ratio"] = (
            (sequence_score == min_score).float().mean().item()
        )

        metrics[f"{prefix}/score/mean"] = torch.mean(sequence_score).detach().item()
        metrics[f"{prefix}/score/max"] = torch.max(sequence_score).detach().item()
        metrics[f"{prefix}/score/min"] = torch.min(sequence_score).detach().item()

        metrics[f"{prefix}/rewards/mean"] = torch.mean(sequence_reward).detach().item()
        metrics[f"{prefix}/rewards/max"] = torch.max(sequence_reward).detach().item()
        metrics[f"{prefix}/rewards/min"] = torch.min(sequence_reward).detach().item()
        metrics[f"{prefix}/token_level_rewards_mean/mean"] = torch.mean(sequence_reward_mean).detach().item()
        metrics[f"{prefix}/token_level_rewards_mean/max"] = torch.max(sequence_reward_mean).detach().item()
        metrics[f"{prefix}/token_level_rewards_mean/min"] = torch.min(sequence_reward_mean).detach().item()

        metrics[f"{prefix}/advantages/mean"] = masked_mean(advantages, response_mask).detach().item()
        if torch.any(response_mask):
            metrics[f"{prefix}/advantages/max"] = torch.max(advantages[response_mask]).detach().item()
            metrics[f"{prefix}/advantages/min"] = torch.min(advantages[response_mask]).detach().item()

        correct_mask_expanded = correct_mask.unsqueeze(-1).expand_as(response_mask)
        correct_response_mask = response_mask & correct_mask_expanded
        if torch.any(correct_response_mask):
            metrics[f"{prefix}/right_response/advantages/mean"] = (
                masked_mean(advantages, correct_response_mask).detach().item()
            )
            metrics[f"{prefix}/right_response/advantages/max"] = (
                torch.max(advantages[correct_response_mask]).detach().item()
            )
            metrics[f"{prefix}/right_response/advantages/min"] = (
                torch.min(advantages[correct_response_mask]).detach().item()
            )
            metrics[f"{prefix}/right_response/advantages/std"] = (
                torch.std(advantages[correct_response_mask]).detach().item()
            )

        incorrect_mask_expanded = incorrect_mask.unsqueeze(-1).expand_as(response_mask)
        incorrect_response_mask = response_mask & incorrect_mask_expanded
        if torch.any(incorrect_response_mask):
            metrics[f"{prefix}/error_response/advantages/mean"] = (
                masked_mean(advantages, incorrect_response_mask).detach().item()
            )
            metrics[f"{prefix}/error_response/advantages/max"] = (
                torch.max(advantages[incorrect_response_mask]).detach().item()
            )
            metrics[f"{prefix}/error_response/advantages/min"] = (
                torch.min(advantages[incorrect_response_mask]).detach().item()
            )
            metrics[f"{prefix}/error_response/advantages/std"] = (
                torch.std(advantages[incorrect_response_mask]).detach().item()
            )

        metrics[f"{prefix}/raw_advantages/mean"] = masked_mean(raw_advantages, response_mask).detach().item()
        if torch.any(response_mask):
            metrics[f"{prefix}/raw_advantages/max"] = torch.max(raw_advantages[response_mask]).detach().item()
            metrics[f"{prefix}/raw_advantages/min"] = torch.min(raw_advantages[response_mask]).detach().item()

        if torch.any(correct_response_mask):
            metrics[f"{prefix}/right_response/raw_advantages/mean"] = (
                masked_mean(raw_advantages, correct_response_mask).detach().item()
            )
            metrics[f"{prefix}/right_response/raw_advantages/max"] = (
                torch.max(raw_advantages[correct_response_mask]).detach().item()
            )
            metrics[f"{prefix}/right_response/raw_advantages/min"] = (
                torch.min(raw_advantages[correct_response_mask]).detach().item()
            )
            metrics[f"{prefix}/right_response/raw_advantages/std"] = (
                torch.std(raw_advantages[correct_response_mask]).detach().item()
            )

        if torch.any(incorrect_response_mask):
            metrics[f"{prefix}/error_response/raw_advantages/mean"] = (
                masked_mean(raw_advantages, incorrect_response_mask).detach().item()
            )
            metrics[f"{prefix}/error_response/raw_advantages/max"] = (
                torch.max(raw_advantages[incorrect_response_mask]).detach().item()
            )
            metrics[f"{prefix}/error_response/raw_advantages/min"] = (
                torch.min(raw_advantages[incorrect_response_mask]).detach().item()
            )
            metrics[f"{prefix}/error_response/raw_advantages/std"] = (
                torch.std(raw_advantages[incorrect_response_mask]).detach().item()
            )

        metrics[f"{prefix}/returns/mean"] = masked_mean(returns, response_mask).detach().item()
        if torch.any(response_mask):
            metrics[f"{prefix}/returns/max"] = torch.max(returns[response_mask]).detach().item()
            metrics[f"{prefix}/returns/min"] = torch.min(returns[response_mask]).detach().item()

        if "values" in batch.batch.keys():
            values = batch.batch["values"]
            metrics[f"{prefix}/values/mean"] = masked_mean(values, response_mask).detach().item()
            if torch.any(response_mask):
                metrics[f"{prefix}/values/max"] = torch.max(values[response_mask]).detach().item()
                metrics[f"{prefix}/values/min"] = torch.min(values[response_mask]).detach().item()

        self.add_metrics(metrics)
        return metrics

    def add_group_metrics(self, batch, n_sample: int, prefix: str = "group") -> Dict[str, Any]:
        if n_sample <= 1:
            return {}

        metrics = {}

        sequence_score = batch.batch["scores"]
        response_mask = batch.batch["response_mask"][:, 1:].bool()
        response_length = response_mask.sum(-1).float()
        advantages = batch.batch["advantages"]

        total_samples = sequence_score.shape[0]
        num_prompts = total_samples // n_sample

        grouped_scores = sequence_score.reshape(num_prompts, n_sample)
        grouped_response_length = response_length.reshape(num_prompts, n_sample)

        max_length_per_group = torch.max(grouped_response_length, dim=1)[0]
        min_length_per_group = torch.min(grouped_response_length, dim=1)[0]
        length_diff_per_group = max_length_per_group - min_length_per_group

        metrics[f"{prefix}/response_length_diff/mean"] = torch.mean(length_diff_per_group).item()
        metrics[f"{prefix}/response_length_diff/max"] = torch.max(length_diff_per_group).item()
        metrics[f"{prefix}/response_length_diff/min"] = torch.min(length_diff_per_group).item()

        max_score = 1
        min_score = 0

        correct_mask_grouped = grouped_scores == max_score
        incorrect_mask_grouped = grouped_scores == min_score

        correct_ratio_per_group = correct_mask_grouped.float().mean(dim=1)
        metrics[f"{prefix}/correct_ratio/mean"] = torch.mean(correct_ratio_per_group).item()
        metrics[f"{prefix}/correct_ratio/std"] = torch.std(correct_ratio_per_group).item()

        all_correct_groups = torch.sum(correct_mask_grouped.all(dim=1)).item()
        all_incorrect_groups = torch.sum(incorrect_mask_grouped.all(dim=1)).item()
        mixed_groups = num_prompts - all_correct_groups - all_incorrect_groups

        metrics[f"{prefix}/all_correct_groups_ratio"] = all_correct_groups / num_prompts
        metrics[f"{prefix}/all_incorrect_groups_ratio"] = all_incorrect_groups / num_prompts
        metrics[f"{prefix}/mixed_groups_ratio"] = mixed_groups / num_prompts

        if "advantages" in batch.batch:
            mean_adv_per_sample = masked_mean(advantages, response_mask, dim=1)
            grouped_advantages = mean_adv_per_sample.reshape(num_prompts, n_sample)

            max_adv_per_group = torch.max(grouped_advantages, dim=1)[0]
            min_adv_per_group = torch.min(grouped_advantages, dim=1)[0]
            adv_diff_per_group = max_adv_per_group - min_adv_per_group

            metrics[f"{prefix}/advantage_diff/mean"] = torch.mean(adv_diff_per_group).item()
            metrics[f"{prefix}/advantage_diff/max"] = torch.max(adv_diff_per_group).item()
            metrics[f"{prefix}/advantage_diff/min"] = torch.min(adv_diff_per_group).item()

            for group_idx in range(num_prompts):
                group_correct_mask = correct_mask_grouped[group_idx]
                group_incorrect_mask = incorrect_mask_grouped[group_idx]

                if torch.any(group_correct_mask) and torch.any(group_incorrect_mask):
                    correct_adv = grouped_advantages[group_idx, group_correct_mask]
                    incorrect_adv = grouped_advantages[group_idx, group_incorrect_mask]

                    if len(correct_adv) > 0 and len(incorrect_adv) > 0:
                        correct_adv_mean = torch.mean(correct_adv)
                        incorrect_adv_mean = torch.mean(incorrect_adv)

                        if "correct_incorrect_adv_diff" not in locals():
                            correct_incorrect_adv_diff = []

                        correct_incorrect_adv_diff.append(correct_adv_mean - incorrect_adv_mean)

            if "correct_incorrect_adv_diff" in locals() and len(correct_incorrect_adv_diff) > 0:
                correct_incorrect_adv_diff = torch.stack(correct_incorrect_adv_diff)
                metrics[f"{prefix}/correct_vs_incorrect_advantage_diff/mean"] = torch.mean(
                    correct_incorrect_adv_diff
                ).item()
                metrics[f"{prefix}/correct_vs_incorrect_advantage_diff/std"] = torch.std(
                    correct_incorrect_adv_diff
                ).item()

        self.add_metrics(metrics)
        return metrics

    def add_all_metrics(
        self, global_step, batch, n_sample=-1, resource_manager=None, actor_infer=None, actor_train=None
    ) -> None:
        batch_size = batch.batch.shape[0]
        # 添加system相关的指标
        self.add_system_metrics(
            global_step,
            batch_size,
            resource_manager=resource_manager,
            actor_infer=actor_infer,
            actor_train=actor_train,
        )
        # 添加token相关的指标
        self.add_token_metrics(batch)
        # 添加values相关的指标
        self.add_values_metrics(batch)

        if hasattr(batch, "meta_info") and "generation_config" in batch.meta_info:
            n_sample = batch.meta_info["generation_config"].get("num_return_sequences", 1)
        if n_sample > 1:
            self.add_group_metrics(batch, n_sample)

    def add_domain_all_metrics(self, global_step, batch_grouped: Dict[str, Any]) -> None:
        for domain, domain_batch in batch_grouped.items():
            original_metrics = self.metrics.copy()
            domain_metrics = self.add_values_metrics(batch=domain_batch)
            self.add_domain_metrics(domain, domain_metrics)

            token_metrics = self.add_token_metrics(batch=domain_batch)
            self.add_domain_metrics(domain, token_metrics)
            self.metrics = original_metrics
