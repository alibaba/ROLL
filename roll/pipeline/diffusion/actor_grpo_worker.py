"""FlowGRPO diffusion actor worker.

Implements FlowGRPO forward_and_backward and log-probs forward.
"""

from typing import Optional

import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.diffusion.actor_diffusion_worker import ActorDiffusionWorker
from roll.utils.functionals import append_to_dict


class ActorGRPOWorker(ActorDiffusionWorker):
    """FlowGRPO diffusion actor worker."""

    def forward_and_backward(
        self,
        data: DataProto,
        model_ctx: dict,
        gradient_accumulation_steps: int,
        loss_scale: Optional[float],
        scaler,
    ) -> dict:
        """FlowGRPO step-wise forward + per-step backward.

        Iterates over SDE trajectory steps, calling self.strategy for each
        step's noise prediction and log-prob computation, then delegates
        per-step loss to _loss_func and immediately calls backward to
        release the computation graph.
        """
        assert "all_timesteps" in data.batch.keys(), "missing all_timesteps"
        assert data.batch["all_timesteps"].ndim == 2, f"all_timesteps.ndim={data.batch['all_timesteps'].ndim}"
        num_diffusion_steps = int(data.batch["all_timesteps"].shape[1])
        assert num_diffusion_steps > 0, f"num_diffusion_steps={num_diffusion_steps}"
        num_train_diffusion_steps = int(
            data.meta_info.get("num_train_diffusion_steps", num_diffusion_steps)
        )
        num_train_diffusion_steps = max(1, min(num_train_diffusion_steps, num_diffusion_steps))

        metrics: dict = {}
        for diffusion_step in range(num_train_diffusion_steps):
            step_data = self._build_step_training_data(data, diffusion_step, num_diffusion_steps)
            step_latents = step_data.batch["all_latents"][:, 0]
            step_timesteps = step_data.batch["all_timesteps"][:, 0]
            img_shapes = step_data.meta_info["img_shapes"]
            model_output = self.strategy.forward_diffusion_model_step(step_latents, step_timesteps, img_shapes, model_ctx)

            model_output = model_output.unsqueeze(1)

            step_log_probs = self.strategy.compute_diffusion_step_log_probs(
                model_output=model_output,
                all_latents=step_data.batch["all_latents"],
                all_timesteps=step_data.batch["all_timesteps"],
            )

            loss, loss_reduced = self._loss_func(step_data, step_log_probs)
            append_to_dict(metrics, loss_reduced)

            if loss_scale is not None:
                loss *= loss_scale
            loss = loss / gradient_accumulation_steps / num_train_diffusion_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        return metrics

    def _loss_func(self, data: DataProto, step_log_probs: torch.Tensor):
        """FlowGRPO clipped PG loss."""
        log_probs = step_log_probs

        old_log_probs = data.batch["old_log_probs"]
        advantages = data.batch["advantages"]
        if "flow_loss_mask" in data.batch.keys():
            loss_mask = data.batch["flow_loss_mask"]
        else:
            loss_mask = data.batch["response_mask"][:, 1:].long()

        pg_clip = self.pipeline_config.pg_clip
        advantage_clip = self.pipeline_config.advantage_clip

        if advantage_clip is not None:
            advantages = torch.clamp(advantages, min=-advantage_clip, max=advantage_clip)
        ratio = (log_probs - old_log_probs).exp()
        unclipped = ratio * advantages
        clipped = ratio.clamp(1 - pg_clip, 1 + pg_clip) * advantages
        pg_loss_mat = -torch.min(unclipped, clipped)
        denom = loss_mask.sum().clamp_min(1.0)
        pg_loss = (pg_loss_mat * loss_mask).sum() / denom

        total_loss = pg_loss
        metrics = {
            "actor/pg_loss@sum": float(pg_loss.detach().cpu().item()),
            "actor/total_loss@sum": float(total_loss.detach().cpu().item()),
            "actor/ratio_mean@mean": float(((ratio * loss_mask).sum() / denom).detach().cpu().item()),
        }
        return total_loss, metrics

    def forward_func_log_probs(self, data: DataProto, output_tensor: torch.Tensor):
        """Compute diffusion step log-probs from model output."""
        log_probs = self.strategy.compute_diffusion_step_log_probs(
            model_output=output_tensor,
            all_latents=data.batch["all_latents"],
            all_timesteps=data.batch["all_timesteps"],
        )
        return (
            torch.tensor(0.0, device=log_probs.device),
            {"log_probs": log_probs.detach()},
        )

    def _build_step_training_data(self, data: DataProto, step_idx: int, num_steps: int) -> DataProto:
        """Slice a single diffusion step from the full trajectory batch."""
        default_step_keys = ("old_log_probs", "advantages", "flow_loss_mask")
        step_indexed_keys = tuple(data.meta_info.get("step_indexed_keys", default_step_keys))
        pass_through_keys = tuple(data.meta_info.get("pass_through_keys", ()))

        all_latents = data.batch["all_latents"]

        step_tensors = {"all_latents": all_latents[:, step_idx : step_idx + 2]}
        for key in step_indexed_keys:
            tensor = data.batch[key]
            assert tensor.ndim >= 2, f"{key}.ndim={tensor.ndim} < 2"
            assert tensor.shape[1] == num_steps, f"{key}.shape[1]={tensor.shape[1]} != {num_steps}"
            step_tensors[key] = tensor[:, step_idx : step_idx + 1]

        if "all_timesteps" in data.batch.keys():
            all_timesteps = data.batch["all_timesteps"]
            assert all_timesteps.ndim == 2 and all_timesteps.shape[1] == num_steps, \
                f"all_timesteps shape={tuple(all_timesteps.shape)}"
            step_tensors["all_timesteps"] = all_timesteps[:, step_idx : step_idx + 1]

        for key in pass_through_keys:
            if key in data.batch.keys():
                step_tensors[key] = data.batch[key]

        result = DataProto.from_dict(
            tensors=step_tensors,
            meta_info=dict(data.meta_info),
        )
        result.non_tensor_batch = data.non_tensor_batch
        return result

    
