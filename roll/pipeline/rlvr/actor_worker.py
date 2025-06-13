import numpy as np
import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.base_worker import ActorWorker as BaseActorWorker
from roll.utils.functionals import masked_mean, agg_loss, compute_approx_kl


class ActorWorker(BaseActorWorker):

    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        """
        loss func interface definition:
            data: DataProto, passed through from train_step
            output_tensor: torch.Tensor, output tensor from model.forward()
        """

        response_mask = data.batch["response_mask"][:, 1:].long()
        final_response_mask = data.batch.get("final_response_mask", response_mask)

        ref_log_probs = data.batch["ref_log_probs"]
        old_log_probs = data.batch["old_log_probs"]
        advantages = data.batch["advantages"]

        log_probs = self.strategy.op_compute_log_probs(
            logits=output_tensor, input_ids=data.batch["input_ids"], attention_mask=data.batch["response_mask"]
        )

        valid_samples = torch.any(final_response_mask > 0, dim=1).float()
        sample_weights = self.compute_sample_weights(data, response_mask)

        ratio = (log_probs - old_log_probs).exp()

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.pipeline_config.pg_clip, 1 + self.pipeline_config.pg_clip) * advantages
        loss = -torch.min(surr1, surr2)
        if self.pipeline_config.dual_clip_loss:
            dual_clip_loss = -torch.max(-loss, (1 + self.pipeline_config.pg_clip * 2) * advantages)
            loss = torch.where(advantages < 0, dual_clip_loss, loss)

        pg_loss_per_sample = masked_mean(loss, final_response_mask, dim=-1)
        weighted_pg_loss = (pg_loss_per_sample * sample_weights * valid_samples).sum() / (valid_samples.sum() + 1e-8)
        original_pg_loss = pg_loss_per_sample.sum() / (valid_samples.sum() + 1e-8)

        kl_loss = compute_approx_kl(
            log_probs=log_probs, log_probs_base=ref_log_probs, action_mask=final_response_mask, kl_penalty="k3"
        )
        kl_loss = masked_mean(kl_loss, mask=final_response_mask, dim=-1).mean()
        original_kl_loss = kl_loss.mean()

        approxkl = compute_approx_kl(
            log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="mse"
        )
        policykl = compute_approx_kl(
            log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="kl"
        )

        clipped_low = (ratio < 1 - self.pipeline_config.pg_clip).float()
        clipped_high = (ratio > 1 + self.pipeline_config.pg_clip).float()
        clipped = (clipped_low + clipped_high).float()

        entropy = self.strategy.op_compute_entropy(logits=output_tensor, attention_mask=data.batch["response_mask"])
        entropy_loss = agg_loss(
            loss_mat=entropy,
            loss_mask=data.batch["response_mask"][:, 1:],
            loss_agg_mode=self.pipeline_config.loss_agg_mode,
        )

        if self.pipeline_config.use_kl_loss:
            total_loss = weighted_pg_loss + kl_loss * self.pipeline_config.kl_loss_coef
        else:
            total_loss = weighted_pg_loss

        total_loss = total_loss * self.pipeline_config.rl_loss_coef

        if self.pipeline_config.entropy_loss_coef > 0:
            total_loss = total_loss - entropy_loss * self.pipeline_config.entropy_loss_coef

        metrics = {}
        if self.pipeline_config.sft_loss_coef > 0:
            logprobs_sum = log_probs.sum(-1)
            old_probs = old_log_probs.sum(-1).exp()
            scores = data.batch['scores']
            response_positive_mask = (scores > 0)
            response_negative_mask = (scores <= 0)
            total_sft_loss = -old_probs * logprobs_sum
            positive_sft_loss = masked_mean(total_sft_loss * scores, response_positive_mask)
            negative_tis_loss = 0
            if self.pipeline_config.use_topr_loss:
                clipped_ratio = torch.clamp((log_probs.detach().sum(-1) - old_log_probs.sum(-1)).exp(), 0 , 1)
                negative_tis_loss = masked_mean(clipped_ratio * total_sft_loss, response_negative_mask)
            sft_loss = positive_sft_loss + negative_tis_loss
            total_loss = total_loss + sft_loss * self.pipeline_config.sft_loss_coef
            metrics['actor/sft_loss'] = sft_loss.detach().item()
            metrics['actor/positive_sft_loss'] = positive_sft_loss.detach().item()
            metrics['actor/negative_tis_loss'] = negative_tis_loss.detach().item() if torch.is_tensor(negative_tis_loss) else negative_tis_loss

        pg_metrics = {
            "actor/ppo_ratio_high_clipfrac": clipped_high.mean().detach().item(),
            "actor/ppo_ratio_low_clipfrac": clipped_low.mean().detach().item(),
            "actor/ppo_ratio_clipfrac": clipped.mean().detach().item(),
            "actor/ratio_mean": masked_mean(ratio, response_mask, dim=-1).mean().detach().item(),
            "actor/ratio_max": torch.max(ratio * response_mask).detach().item(),
            "actor/ratio_min": torch.min(ratio * response_mask + (1 - response_mask) * 1e10).detach().item(),
            "actor/clipfrac": masked_mean(torch.lt(surr2, surr1).float(), response_mask, dim=-1).mean().detach().item(),
            "actor/pg_loss": original_pg_loss.detach().item(),
            "actor/weighted_pg_loss": weighted_pg_loss.detach().item(),
            "actor/kl_loss": original_kl_loss.detach().item(),
            "actor/total_loss": total_loss.detach().item(),
            "actor/approxkl": masked_mean(approxkl, response_mask, dim=-1).mean().detach().item(),
            "actor/policykl": masked_mean(policykl, response_mask, dim=-1).mean().detach().item(),
            "actor/valid_samples": valid_samples.sum().detach().item(),
            "actor/total_samples": float(valid_samples.size(0)),
            "actor/valid_sample_ratio": (valid_samples.sum() / valid_samples.size(0)).detach().item(),
            "actor/sample_weights_mean": sample_weights.mean().detach().item(),
            "actor/sample_weights_min": sample_weights.min().detach().item(),
            "actor/sample_weights_max": sample_weights.max().detach().item(),
            **metrics,
        }

        return total_loss, pg_metrics

    def compute_sample_weights(self, data: DataProto, response_mask: torch.Tensor):
        """
        Sample weights can be based on difficulty and length
        """
        batch_size = response_mask.shape[0]
        sample_weights = torch.ones(batch_size, device=response_mask.device)

        # 1. Difficulty-based weights - example: higher difficulty, higher weight
        if self.pipeline_config.difficulty_loss_weight and "difficulty" in data.non_tensor_batch:
            try:
                difficulty = data.non_tensor_batch["difficulty"]
                if isinstance(difficulty, np.ndarray):
                    difficulty = torch.tensor(difficulty, dtype=torch.float32, device=response_mask.device)
                elif not isinstance(difficulty, torch.Tensor):
                    difficulty = torch.tensor(difficulty, dtype=torch.float32, device=response_mask.device)
                norm_difficulty = torch.clamp(difficulty, 0.0, 1.0)
                difficulty_weights = 0.5 + 1.5 * norm_difficulty
                sample_weights = sample_weights * difficulty_weights
            except Exception as e:
                self.logger.warning(f"Skip difficulty weight calculation: {str(e)}")

        # 2. Length-based weights - example: longer length, smaller weight
        response_lengths = response_mask.sum(dim=1).float()
        if self.pipeline_config.length_loss_weight:
            # Similarly normalize length to [0.5, 2.0] range
            norm_lengths = (response_lengths - response_lengths.min()) / (
                    response_lengths.max() - response_lengths.min() + 1e-8
            )
            length_weights = 1.5 - norm_lengths
            sample_weights = sample_weights * length_weights

        if sample_weights.sum() > 0:
            sample_weights = sample_weights * (batch_size / (sample_weights.sum() + 1e-8))

        return sample_weights

