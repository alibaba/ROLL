"""DiffNFT diffusion actor worker.

Implements DiffNFT (forward-process NFT) forward_and_backward. Unlike
FlowGRPO, which replays the reverse sampling trajectory, DiffNFT takes the
final rollout latent (x0) and the optimality probability (stored in the
``advantages`` field by ``_prepare_diffnft_training_batch``), re-samples
timesteps and noise on the forward process, and optimizes a positive/negative
x0 reconstruction loss via a triple forward over the live adapter
(``default``), the frozen EMA shadow adapter (``ema_lora``, the rollout
policy) and optionally the base model (reference KL).
"""

from typing import Optional, Sequence, Tuple, Union

import torch

from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.diffusion.actor_diffusion_worker import ActorDiffusionWorker
from roll.utils.context_managers import state_offload_manger
from roll.utils.functionals import append_to_dict
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType

logger = get_logger()

_TIMESTEP_UNIT_SCALE = 1000.0


class ActorNFTWorker(ActorDiffusionWorker):
    """DiffNFT diffusion actor worker."""

    # DiffNFT is off-policy: the rollout must sample under the EMA shadow adapter
    # (created via copy_adapter below), not the live "default" adapter, so its
    # weights are what model_update broadcasts to the infer side.
    rollout_adapter_name = "ema_lora"

    def forward_and_backward(
        self,
        data: DataProto,
        model_ctx: dict,
        gradient_accumulation_steps: int,
        loss_scale: Optional[float],
        scaler,
    ) -> dict:
        """DiffNFT per-timestep triple forward + per-timestep backward.

        For each resolved training timestep: forward-noise x0, combine the
        live and EMA predictions into positive/negative targets, compute the
        reward-weighted x0 reconstruction loss, then backward immediately to
        release the computation graph (same pattern as ActorGRPOWorker).
        """
        assert "all_latents" in data.batch.keys(), "missing all_latents"
        assert "advantages" in data.batch.keys(), "missing advantages"

        # DiffNFT only uses the final rollout latent as x0 (ODE rollout keeps a
        # length-1 trajectory; take the last entry to stay explicit).
        x0 = data.batch["all_latents"][:, -1].float()
        batch_size = int(x0.shape[0])

        # The advantages field carries the optimality probability r in [0, 1].
        # [B] (or [B, 1], squeezed) is shared across timesteps; [B, K] provides
        # one value per training timestep, indexed inside the loop (ref-aligned).
        reward_prob = data.batch["advantages"].float()
        if reward_prob.ndim == 2 and int(reward_prob.shape[1]) == 1:
            reward_prob = reward_prob[:, 0]
        if reward_prob.ndim not in (1, 2):
            raise ValueError(f"reward_prob must be [B] or [B, K], got shape={tuple(reward_prob.shape)}")
        if int(reward_prob.shape[0]) != batch_size:
            raise ValueError(f"reward_prob batch={reward_prob.shape[0]} != all_latents batch={batch_size}")

        img_shapes = data.meta_info["img_shapes"]

        beta = float(self.pipeline_config.nft_beta)
        adv_clip_max = float(self.pipeline_config.adv_clip_max)
        ref_kl_coef = float(self.pipeline_config.ref_kl_coef)
        use_adaptive_weight = bool(self.pipeline_config.use_adaptive_weight)

        timesteps = self._resolve_training_timesteps(data, batch_size=batch_size, device=x0.device)
        if timesteps.ndim != 2 or int(timesteps.shape[0]) != batch_size or int(timesteps.shape[1]) <= 0:
            raise RuntimeError(
                "DiffNFT expected per-sample training timesteps shaped [B, K], "
                f"got {tuple(timesteps.shape)} for batch={batch_size}"
            )
        num_timesteps = int(timesteps.shape[1])

        metrics: dict = {}
        total_step_loss = 0.0
        total_pos_loss = 0.0
        total_neg_loss = 0.0
        total_ref_kl = 0.0
        total_old_deviate = 0.0
        for idx in range(num_timesteps):
            # Per-sample timestep column (ref-aligned): each sample in the
            # batch trains on its own t, reducing batch-level loss variance.
            t_k = timesteps[:, idx]
            # Forward noising: xt = (1 - t) * x0 + t * noise
            noise = torch.randn_like(x0)
            t_expanded = t_k.view(batch_size, *([1] * (x0.ndim - 1)))
            xt = (1.0 - t_expanded) * x0 + t_expanded * noise
            t_k_batch = t_k * _TIMESTEP_UNIT_SCALE

            # Triple forward: live (grad) / EMA shadow (old policy) / base (ref, optional)
            self.strategy.set_active_adapter("default")
            new_pred = self.strategy.forward_diffusion_model_step(xt, t_k_batch, img_shapes, model_ctx)

            ref_pred = None
            with torch.no_grad():
                self.strategy.set_active_adapter("ema_lora")
                old_pred = self.strategy.forward_diffusion_model_step(xt, t_k_batch, img_shapes, model_ctx)
                if ref_kl_coef > 0.0:
                    with self.strategy.lora_adapters_disabled():
                        ref_pred = self.strategy.forward_diffusion_model_step(xt, t_k_batch, img_shapes, model_ctx)

            new_pred = new_pred.float()
            old_pred = old_pred.float()
            old_deviate = ((new_pred.detach() - old_pred.detach()) ** 2).mean()

            # Positive/negative targets anchored at the EMA policy, mixed by beta
            positive = beta * new_pred + (1.0 - beta) * old_pred
            negative = (1.0 + beta) * old_pred - beta * new_pred
            x0_pos = xt - t_expanded * positive
            x0_neg = xt - t_expanded * negative

            pos_sq = (x0_pos - x0) ** 2
            neg_sq = (x0_neg - x0) ** 2
            if reward_prob.ndim == 1:
                reward_prob_k = reward_prob
            else:
                if int(reward_prob.shape[1]) <= idx:
                    raise ValueError(
                        f"reward_prob shape={tuple(reward_prob.shape)} does not cover timestep index {idx}"
                    )
                reward_prob_k = reward_prob[:, idx]
            r = reward_prob_k.view(batch_size, *([1] * (x0.ndim - 1)))
            if use_adaptive_weight:
                # Adaptive weighting: normalize per-sample error magnitudes so
                # that samples contribute comparable loss scales
                sample_dims = tuple(range(1, x0.ndim))
                with torch.no_grad():
                    pos_scale = (
                        (x0_pos.detach().double() - x0.double())
                        .abs()
                        .mean(dim=sample_dims, keepdim=True)
                        .clamp(min=1e-5)
                    ).to(dtype=x0_pos.dtype)
                    neg_scale = (
                        (x0_neg.detach().double() - x0.double())
                        .abs()
                        .mean(dim=sample_dims, keepdim=True)
                        .clamp(min=1e-5)
                    ).to(dtype=x0_neg.dtype)
                pos_loss = (r * pos_sq / pos_scale).mean()
                neg_loss = ((1.0 - r) * neg_sq / neg_scale).mean()
            else:
                pos_loss = (r * pos_sq).mean()
                neg_loss = ((1.0 - r) * neg_sq).mean()

            step_loss = (pos_loss + neg_loss) / beta * adv_clip_max
            if ref_pred is not None:
                ref_kl = ((new_pred - ref_pred.float()) ** 2).mean()
                step_loss = step_loss + ref_kl_coef * ref_kl
                total_ref_kl += float(ref_kl.detach().cpu().item())

            total_step_loss += float(step_loss.detach().cpu().item())
            total_pos_loss += float(pos_loss.detach().cpu().item())
            total_neg_loss += float(neg_loss.detach().cpu().item())
            total_old_deviate += float(old_deviate.detach().cpu().item())

            # Switch back to "default" before backward so that gradient
            # checkpointing recomputation uses the same adapter (and same
            # requires_grad state) as the original new_pred forward.
            self.strategy.set_active_adapter("default")
            if loss_scale is not None:
                step_loss = step_loss * loss_scale
            step_loss = step_loss / gradient_accumulation_steps / num_timesteps
            if scaler is not None:
                scaler.scale(step_loss).backward()
            else:
                step_loss.backward()

        # Record metrics once after the loop, averaged over timesteps (matching ref).
        append_to_dict(metrics, {
            "actor/nft_loss@sum": total_step_loss / num_timesteps,
            "actor/nft_pos_loss@mean": total_pos_loss / num_timesteps,
            "actor/nft_neg_loss@mean": total_neg_loss / num_timesteps,
            "actor/nft_ref_kl@mean": total_ref_kl / num_timesteps,
            "actor/nft_old_deviate@mean": total_old_deviate / num_timesteps,
            "actor/old_deviate": total_old_deviate / num_timesteps,
            "actor/nft_num_timesteps@mean": float(num_timesteps),
        })
        return metrics

    def forward_func_log_probs(self, data: DataProto, output_tensor: torch.Tensor):
        """DiffNFT does not replay rollout log-probs (FlowGRPO-only path)."""
        raise RuntimeError("DiffNFT does not replay rollout log-probs; use ActorGRPOWorker for FlowGRPO.")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def preflight_nft_ops(self):
        """Verify strategy adapter ops and seed the EMA shadow adapter.

        Called once by the pipeline at startup: shadow (ema_lora) is
        initialized to the current live (default) weights, and one EMA update
        is applied to smoke-test the update path before training starts.

        base_worker.initialize() ends with offload_states(), which rebinds
        every param's .data to an empty CPU tensor while the host backend
        holds the real shards. Adapter ops touch param data, so model params
        must be reloaded first — the same load/offload contract as
        train_step's state_offload_manger.
        """
        for op_name in (
            "set_active_adapter",
            "copy_adapter",
            "ema_update_lora",
            "lora_adapters_disabled",
            "invalidate_offloaded_model_params",
        ):
            if not hasattr(self.strategy, op_name):
                raise RuntimeError(f"DiffNFT requires strategy.{op_name}(), got {type(self.strategy).__name__}")
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/preflight_nft",
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            self.strategy.copy_adapter("default", "ema_lora")
            self.strategy.ema_update_lora(self.pipeline_config.get_ema_decay(0))
            # Adapter seeding mutates on-GPU params; drop the backend's stale
            # host copies so the manager's exit offload re-puts fresh values
            # (the write-once fast path would silently discard the EMA init).
            self.strategy.invalidate_offloaded_model_params()
        self.logger.info("DiffNFT preflight_nft_ops passed")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def copy_adapter(self, source: str = "default", target: str = "ema_lora"):
        """Copy the source adapter weights into the target adapter."""
        self.strategy.copy_adapter(source, target)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def ema_update_adapter(self, decay: float):
        """EMA-update the shadow adapter: shadow = decay * shadow + (1 - decay) * live."""
        self.strategy.ema_update_lora(decay)

    def _resolve_training_timesteps(self, data: DataProto, *, batch_size: int, device: torch.device) -> torch.Tensor:
        """Resolve per-sample training timesteps shaped [B, K] in flow-matching units [0, 1].

        Reuses the rollout timestep schedule (ref-aligned): a [start, end]
        window fraction slices the schedule and keeps every timestep in the
        window, while a scalar fraction randomly subsamples ``n * fraction``
        timesteps per row. With shuffling enabled each row is permuted
        independently, so every sample in the batch trains on its own t.
        """
        fraction = self.pipeline_config.training_timestep_fraction
        is_window_fraction = self._is_timestep_window_fraction(fraction)
        start, end = self._normalize_timestep_fraction(fraction)
        if end <= 0.0:
            raise ValueError(f"training_timestep_fraction end must be > 0, got ({start}, {end})")

        rollout_timesteps = data.batch["all_timesteps"]
        timesteps = self._as_timestep_matrix(rollout_timesteps, batch_size=batch_size, device=device)
        # A trailing ~0 timestep corresponds to the finished latent; nothing to train
        if timesteps.shape[1] > 1 and bool((timesteps[:, -1].abs() < 1e-8).all().item()):
            timesteps = timesteps[:, :-1]

        n = int(timesteps.shape[1])
        if is_window_fraction and n > 0 and (start > 0.0 or end < 1.0):
            eff_start = int(n * start)
            eff_end = min(int(n * end), n)
            timesteps = timesteps[:, eff_start:eff_end] if eff_start < eff_end else timesteps[:, :0]
        if timesteps.numel() == 0:
            raise RuntimeError(
                "DiffNFT timestep fraction removed all rollout timesteps; "
                f"fraction=({start}, {end}) rollout_shape={tuple(rollout_timesteps.shape)}"
            )

        n = int(timesteps.shape[1])
        num_train = n if is_window_fraction else max(1, int(n * end))
        num_train = min(num_train, n)
        if bool(self.pipeline_config.shuffle_train_timesteps) and n > 1:
            rows = []
            for row in timesteps:
                rows.append(row[torch.randperm(n, device=device)[:num_train]])
            timesteps = torch.stack(rows, dim=0)
        else:
            timesteps = timesteps[:, :num_train]

        min_t = self.pipeline_config.min_train_timestep
        max_t = self.pipeline_config.max_train_timestep
        if min_t is not None or max_t is not None:
            timesteps = self._filter_timestep_rows(
                timesteps,
                min_timestep=None if min_t is None else float(min_t),
                max_timestep=None if max_t is None else float(max_t),
            )
        if timesteps.numel() == 0:
            raise RuntimeError("DiffNFT min/max_train_timestep removed all training timesteps.")
        return timesteps

    @staticmethod
    def _as_timestep_matrix(rollout_timesteps: torch.Tensor, *, batch_size: int, device: torch.device) -> torch.Tensor:
        """Normalize rollout timesteps of shape [T], [1, T] or [B, T] to a [B, T] matrix."""
        timesteps = rollout_timesteps.detach().to(device=device, dtype=torch.float32)
        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(0).expand(batch_size, -1).clone()
        elif timesteps.ndim == 2:
            if int(timesteps.shape[0]) == batch_size:
                timesteps = timesteps.clone()
            elif int(timesteps.shape[0]) == 1:
                timesteps = timesteps.expand(batch_size, -1).clone()
            else:
                raise ValueError(
                    "DiffNFT rollout timesteps must have shape [T], [1, T], or [B, T]; "
                    f"got {tuple(timesteps.shape)} for batch={batch_size}"
                )
        else:
            raise ValueError(f"DiffNFT rollout timesteps must be 1D or 2D, got {tuple(timesteps.shape)}")
        return timesteps / _TIMESTEP_UNIT_SCALE

    @staticmethod
    def _filter_timestep_rows(
        timesteps: torch.Tensor,
        *,
        min_timestep: Optional[float],
        max_timestep: Optional[float],
    ) -> torch.Tensor:
        """Apply min/max bounds per row, truncating all rows to the shortest count."""
        rows = []
        counts = []
        for row in timesteps:
            mask = torch.ones_like(row, dtype=torch.bool)
            if min_timestep is not None:
                mask &= row >= float(min_timestep)
            if max_timestep is not None:
                mask &= row <= float(max_timestep)
            filtered = row[mask]
            rows.append(filtered)
            counts.append(int(filtered.numel()))
        min_count = min(counts) if counts else 0
        if min_count <= 0:
            return timesteps[:, :0]
        if len(set(counts)) > 1:
            logger.warning(
                "DiffNFT timestep bounds produced uneven per-sample counts; truncating to shortest count=%s counts=%s",
                min_count,
                counts[:16],
            )
        return torch.stack([row[:min_count] for row in rows], dim=0)

    @staticmethod
    def _is_timestep_window_fraction(value: Union[float, Sequence[float]]) -> bool:
        return (
            not isinstance(value, (str, bytes))
            and hasattr(value, "__len__")
            and hasattr(value, "__getitem__")
        )

    @staticmethod
    def _normalize_timestep_fraction(value: Union[float, Sequence[float]]) -> Tuple[float, float]:
        """Normalize training_timestep_fraction to a (start, end) range in [0, 1]."""
        if ActorNFTWorker._is_timestep_window_fraction(value):
            if len(value) != 2:
                raise ValueError(f"training_timestep_fraction sequence must have 2 values, got {len(value)}")
            start, end = float(value[0]), float(value[1])
        else:
            start, end = 0.0, float(value)
        if not (0.0 <= start <= 1.0 and 0.0 <= end <= 1.0):
            raise ValueError(f"training_timestep_fraction values must be in [0, 1], got ({start}, {end})")
        if start > end:
            raise ValueError(f"training_timestep_fraction start must be <= end, got ({start}, {end})")
        return start, end
