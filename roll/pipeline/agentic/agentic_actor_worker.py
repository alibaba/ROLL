import numpy as np
import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.base_worker import ActorWorker as BaseActorWorker
from roll.utils.functionals import masked_mean, agg_loss, compute_approx_kl
from roll.pipeline.agentic.utils import compute_segment_masked_mean
from roll.utils.train_infer_corrections import compute_train_infer_correction

class ActorWorker(BaseActorWorker):
    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        """
        loss func接口定义:
            data: DataProto, 由train_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """
        response_mask = data.batch["response_mask"][:, 1:].long()
        ref_log_probs = data.batch["ref_log_probs"]
        advantages = data.batch["advantages"]

        batch_num_tokens = data.meta_info['batch_num_tokens']
        global_valid_samples = data.meta_info['global_valid_samples']

        log_probs = self.strategy.op_compute_log_probs(
            logits=output_tensor, input_ids=data.batch["input_ids"], attention_mask=data.batch["response_mask"]
        )
        old_log_probs = self.get_old_log_probs_with_cache(data, log_probs)
        infer_log_probs = data.batch.get("infer_logprobs", old_log_probs)
        infer_log_probs = infer_log_probs if len(infer_log_probs) > 0 else old_log_probs

        train_infer_metric = {}
        if not self.pipeline_config.enable_old_logprobs_recompute:
            train_infer_is_weight, filter_mask, train_infer_metric = compute_train_infer_correction(
                cfg=self.pipeline_config.train_infer_correction,
                response_mask=response_mask,
                old_log_probs=old_log_probs,
                infer_log_probs=infer_log_probs,
                global_valid_samples=global_valid_samples['response_mask'],
                global_valid_tokens=batch_num_tokens['response_mask'],
            )

            # Apply filter mask to response_mask
            response_mask = response_mask.long() * filter_mask.long()
        else:
            train_infer_is_weight = data.batch['train_infer_is_weight']

        if self.pipeline_config.ratio_type == "segment":
            # 计算序列级别的 ratio：对每段连续的1分别计算 masked_mean，不连续的段不相乘
            log_ratio = log_probs - old_log_probs
            masked_log_ratio = compute_segment_masked_mean(log_ratio, response_mask)
            ratio = masked_log_ratio.exp()
        else:
            ratio = (log_probs - old_log_probs).exp()

        pg_clip_low = (
            self.pipeline_config.pg_clip_low
            if self.pipeline_config.use_pg_clip_range
            else self.pipeline_config.pg_clip
        )
        pg_clip_high = (
            self.pipeline_config.pg_clip_high
            if self.pipeline_config.use_pg_clip_range
            else self.pipeline_config.pg_clip
        )
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - pg_clip_low, 1 + pg_clip_high) * advantages
        pg_loss = -torch.min(surr1, surr2)
        if self.pipeline_config.dual_clip_loss:
            dual_clip_loss = -torch.max(-pg_loss, (1 + self.pipeline_config.pg_clip * 2) * advantages)
            pg_loss = torch.where(advantages < 0, dual_clip_loss, pg_loss)

        if self.pipeline_config.train_infer_correction.is_weight.enabled:
            pg_loss = pg_loss * train_infer_is_weight

        pg_loss = agg_loss(loss_mat=pg_loss, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.loss_agg_mode,
                           batch_num_tokens=batch_num_tokens['response_mask'], global_valid_samples=global_valid_samples['response_mask'])

        kl_loss = compute_approx_kl(
            log_probs=log_probs, log_probs_base=ref_log_probs, action_mask=response_mask, kl_penalty="k3"
        )
        kl_loss = agg_loss(loss_mat=kl_loss, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.loss_agg_mode,
                           batch_num_tokens=batch_num_tokens['response_mask'], global_valid_samples=global_valid_samples['response_mask'])

        approxkl = compute_approx_kl(
            log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="mse"
        )
        policykl = compute_approx_kl(
            log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="kl"
        )
        clipped_low = (ratio < 1 - pg_clip_low).float()
        clipped_high = (ratio > 1 + pg_clip_high).float()
        clipped = (clipped_low + clipped_high).float()

        if self.pipeline_config.use_kl_loss:
            total_loss = pg_loss + kl_loss * self.pipeline_config.kl_loss_coef
        else:
            total_loss = pg_loss
        if self.pipeline_config.entropy_loss_coef > 0:
            entropy = self.strategy.op_compute_entropy(
                logits=output_tensor, attention_mask=data.batch["response_mask"]
            )
            entropy_loss = agg_loss(
                loss_mat=entropy,
                loss_mask=response_mask,
                loss_agg_mode=self.pipeline_config.loss_agg_mode,
                batch_num_tokens=batch_num_tokens['response_mask'],
                global_valid_samples=global_valid_samples['response_mask'],
            )
            total_loss = total_loss - entropy_loss * self.pipeline_config.entropy_loss_coef

        pg_metrics = {
            "actor/ppo_ratio_high_clipfrac@sum": agg_loss(loss_mat=clipped_high,
                                                loss_mask=response_mask, loss_agg_mode='token-mean',
                                                batch_num_tokens=batch_num_tokens['response_mask'],).detach().item(),
            "actor/ppo_ratio_low_clipfrac@sum": agg_loss(loss_mat=clipped_low,
                                                loss_mask=response_mask, loss_agg_mode='token-mean',
                                                batch_num_tokens=batch_num_tokens['response_mask'],).detach().item(),
            "actor/ppo_ratio_clipfrac@sum": agg_loss(loss_mat=clipped,
                                                loss_mask=response_mask, loss_agg_mode='token-mean',
                                                batch_num_tokens=batch_num_tokens['response_mask'],).detach().item(),
            "actor/ratio_mean@sum": agg_loss(loss_mat=ratio,
                                                loss_mask=response_mask, loss_agg_mode='seq-mean-token-mean',
                                                global_valid_samples=global_valid_samples['response_mask'],).detach().item(),
            "actor/ratio_max@max": torch.max(ratio * response_mask).detach().item(),
            "actor/ratio_min@min": torch.min(ratio * response_mask + (1 - response_mask) * 1e10).detach().item(),
            "actor/clipfrac@sum": agg_loss(
                loss_mat=torch.lt(surr2, surr1).float(),
                loss_mask=response_mask,
                loss_agg_mode=self.pipeline_config.loss_agg_mode,
                batch_num_tokens=batch_num_tokens['response_mask'],
                global_valid_samples=global_valid_samples['response_mask'],
            ).detach().item(),
            "actor/pg_loss@sum": pg_loss.detach().item(),
            "actor/kl_loss@sum": kl_loss.detach().item(),
            "actor/total_loss@sum": total_loss.detach().item(),
            "actor/approxkl@sum": agg_loss(
                loss_mat=approxkl, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.loss_agg_mode,
                batch_num_tokens=batch_num_tokens['response_mask'], global_valid_samples=global_valid_samples['response_mask']
            ).detach().item(),
            "actor/policykl@sum": agg_loss(
                loss_mat=policykl, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.loss_agg_mode,
                batch_num_tokens=batch_num_tokens['response_mask'], global_valid_samples=global_valid_samples['response_mask']
            ).detach().item(),
            **train_infer_metric,
        }

        return total_loss, pg_metrics
