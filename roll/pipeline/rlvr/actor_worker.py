import numpy as np
import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.base_worker import ActorWorker as BaseActorWorker
from roll.utils.functionals import masked_mean, agg_loss, compute_approx_kl
from roll.utils.train_infer_corrections import compute_train_infer_correction

class ActorWorker(BaseActorWorker):

    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        """
        loss func接口定义:
            data: DataProto, 由train_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """
        response_mask = data.batch["response_mask"][:, 1:].long()
        final_response_mask = data.batch.get("final_response_mask", response_mask)
        ref_log_probs = data.batch["ref_log_probs"]
        advantages = data.batch["advantages"]

        batch_num_tokens = data.meta_info['batch_num_tokens']
        global_valid_samples = data.meta_info['global_valid_samples']
        if 'final_response_mask' not in batch_num_tokens:
            batch_num_tokens['final_response_mask'] = batch_num_tokens['response_mask']
            global_valid_samples['final_response_mask'] = global_valid_samples['response_mask']

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

            # Apply filter mask to both response_mask and final_response_mask
            response_mask = response_mask.long() * filter_mask.long()
            final_response_mask = final_response_mask.long() * filter_mask.long()
        else:
            train_infer_is_weight = data.batch['train_infer_is_weight']

        valid_samples = torch.any(final_response_mask > 0, dim=1).float()
        sample_weights = self.compute_sample_weights(data, response_mask)

        kl_loss = compute_approx_kl(
            log_probs=log_probs, log_probs_base=ref_log_probs, action_mask=final_response_mask, kl_penalty="k3"
        )
        kl_loss = agg_loss(loss_mat=kl_loss,
                        loss_mask=final_response_mask,
                        loss_agg_mode=self.pipeline_config.loss_agg_mode,
                        batch_num_tokens=batch_num_tokens['final_response_mask'],
                        global_valid_samples=global_valid_samples['final_response_mask'],)

        approxkl = compute_approx_kl(
            log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="mse"
        )
        policykl = compute_approx_kl(
            log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="kl"
        )

        if self.pipeline_config.importance_sampling == "token":
            ratio = (log_probs - old_log_probs).exp()
        elif self.pipeline_config.importance_sampling == "seq":
            log_ratio = log_probs - old_log_probs
            masked_log_ratio = masked_mean(log_ratio, final_response_mask, dim=-1)
            ratio = masked_log_ratio.exp().unsqueeze(-1).expand_as(log_ratio)

        pg_clip_low = self.pipeline_config.pg_clip_low if self.pipeline_config.use_pg_clip_range else self.pipeline_config.pg_clip
        pg_clip_high = self.pipeline_config.pg_clip_high if self.pipeline_config.use_pg_clip_range else self.pipeline_config.pg_clip
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - pg_clip_low, 1 + pg_clip_high) * advantages

        loss = -torch.min(surr1, surr2)

        if self.pipeline_config.dual_clip_loss:
            dual_clip_loss = -torch.max(-loss, (1 + self.pipeline_config.pg_clip * 2) * advantages)
            loss = torch.where(advantages < 0, dual_clip_loss, loss)

        if self.pipeline_config.train_infer_correction.is_weight.enabled:
            loss = loss * train_infer_is_weight

        weighted_pg_loss = agg_loss(loss_mat=loss, loss_mask=final_response_mask,
                                    loss_agg_mode=self.pipeline_config.loss_agg_mode,
                                    weights=sample_weights,
                                    batch_num_tokens=batch_num_tokens['final_response_mask'],
                                    global_valid_samples=global_valid_samples['final_response_mask'],)
        original_pg_loss = agg_loss(loss_mat=loss, loss_mask=final_response_mask,
                                    loss_agg_mode=self.pipeline_config.loss_agg_mode,
                                    batch_num_tokens=batch_num_tokens['final_response_mask'],
                                    global_valid_samples=global_valid_samples['final_response_mask'],)

        clipped_low = (ratio < 1 - pg_clip_low).float()
        clipped_high = (ratio > 1 + pg_clip_high).float()
        clipped = (clipped_low + clipped_high).float()

        if self.pipeline_config.use_kl_loss:
            total_loss = weighted_pg_loss + kl_loss * self.pipeline_config.kl_loss_coef
        else:
            total_loss = weighted_pg_loss

        total_loss = total_loss * self.pipeline_config.rl_loss_coef

        if self.pipeline_config.entropy_loss_coef > 0:
            entropy = self.strategy.op_compute_entropy(logits=output_tensor, attention_mask=data.batch["response_mask"])
            entropy_loss = agg_loss(
                loss_mat=entropy,
                loss_mask=data.batch["response_mask"][:, 1:],
                loss_agg_mode=self.pipeline_config.loss_agg_mode,
                batch_num_tokens=batch_num_tokens['response_mask'],
                global_valid_samples=global_valid_samples['response_mask'],
            )
            total_loss = total_loss - entropy_loss * self.pipeline_config.entropy_loss_coef

        loss_metric = {
            "actor/ppo_ratio_high_clipfrac@sum": agg_loss(loss_mat=clipped_high, loss_mask=final_response_mask,
                                loss_agg_mode='token-mean',
                                 batch_num_tokens=batch_num_tokens['final_response_mask']).detach().item(),
            "actor/ppo_ratio_low_clipfrac@sum": agg_loss(loss_mat=clipped_low, loss_mask=final_response_mask,
                                loss_agg_mode='token-mean',
                                 batch_num_tokens=batch_num_tokens['final_response_mask']).detach().item(),
            "actor/ppo_ratio_clipfrac@sum": agg_loss(loss_mat=clipped, loss_mask=final_response_mask,
                                loss_agg_mode='token-mean',
                                 batch_num_tokens=batch_num_tokens['final_response_mask']).detach().item(),
            "actor/ratio_mean@sum": agg_loss(loss_mat=ratio, loss_mask=response_mask,
                                loss_agg_mode='seq-mean-token-mean',
                                 global_valid_samples=global_valid_samples['response_mask']).detach().item(),
            "actor/ratio_max@max": torch.max(ratio * response_mask).detach().item(),
            "actor/ratio_min@min": torch.min(ratio * response_mask + (1 - response_mask) * 1e10).detach().item(),
            "actor/clipfrac@sum": agg_loss(loss_mat=torch.lt(surr2, surr1).float(), loss_mask=response_mask,
                                loss_agg_mode=self.pipeline_config.loss_agg_mode, batch_num_tokens=batch_num_tokens,
                                       global_valid_samples=global_valid_samples['response_mask']).detach().item(),
        }

        pg_metrics = {
            "actor/pg_loss@sum": original_pg_loss.detach().item(),
            "actor/weighted_pg_loss@sum": weighted_pg_loss.detach().item(),
            "actor/kl_loss@sum": kl_loss.detach().item(),
            "actor/total_loss@sum": total_loss.detach().item(),
            "actor/approxkl@sum": agg_loss(loss_mat=approxkl, loss_mask=response_mask,
                                       loss_agg_mode=self.pipeline_config.loss_agg_mode,
                                       batch_num_tokens=batch_num_tokens,
                                        global_valid_samples=global_valid_samples['response_mask'],).detach().item(),
            "actor/policykl@sum": agg_loss(loss_mat=policykl, loss_mask=response_mask,
                                       loss_agg_mode=self.pipeline_config.loss_agg_mode,
                                       batch_num_tokens=batch_num_tokens,
                                        global_valid_samples=global_valid_samples['response_mask'],).detach().item(),
            "actor/valid_samples@sum": valid_samples.sum().detach().item(),
            "actor/total_samples@sum": float(valid_samples.size(0)),
            "actor/valid_sample_ratio@sum": (valid_samples.sum() / global_valid_samples['response_mask']).detach().item(),
            "actor/sample_weights_mean@mean": sample_weights.mean().detach().item(),
            "actor/sample_weights_min@min": sample_weights.min().detach().item(),
            "actor/sample_weights_max@max": sample_weights.max().detach().item(),
            **loss_metric,
            **train_infer_metric,
        }

        return total_loss, pg_metrics

    def compute_sample_weights(self, data: DataProto, response_mask: torch.Tensor):
        """
        可以基于难度和长度的样本权重
        """
        batch_size = response_mask.shape[0]
        sample_weights = torch.ones(batch_size, device=response_mask.device)

        # 1. 基于难度的权重 - 例如：难度越高，权重越大
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
                self.logger.warning(f"跳过difficulty权重计算：{str(e)}")

        # 2. 基于长度的权重 - 例如：长度越长，权重越小
        response_lengths = response_mask.sum(dim=1).float()
        if self.pipeline_config.length_loss_weight:
            # 同样归一化长度到[0.5, 2.0]范围
            norm_lengths = (response_lengths - response_lengths.min()) / (
                    response_lengths.max() - response_lengths.min() + 1e-8
            )
            length_weights = 1.5 - norm_lengths
            sample_weights = sample_weights * length_weights

        if sample_weights.sum() > 0:
            sample_weights = sample_weights * (batch_size / (sample_weights.sum() + 1e-8))

        return sample_weights

