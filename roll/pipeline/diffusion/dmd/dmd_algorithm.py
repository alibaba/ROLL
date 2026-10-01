"""DMD algorithms and flow-matching scheduler."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from roll.pipeline.diffusion.dmd.dmd_config import DMDGANHeadConfig, DMDScoreTimestepConfig, DMDSelfForcingConfig

DMD_KEY_PROMPTS = "prompts"
DMD_KEY_NOISY_LATENT = "noisy_latent"
DMD_KEY_NOISY_LATENT_CA = "noisy_latent_ca"
DMD_KEY_TIMESTEP = "timestep"
DMD_KEY_TIMESTEP_CA = "timestep_ca"
DMD_KEY_PRED_X0 = "pred_x0"
DMD_KEY_PRED_FAKE_IMAGE = "pred_fake_image"
DMD_KEY_PRED_REAL_IMAGE = "pred_real_image"
DMD_KEY_PRED_REAL_COND_CA = "pred_real_cond_ca"
DMD_KEY_PRED_REAL_UNCOND_CA = "pred_real_uncond_ca"
DMD_KEY_GENERATED = "generated"
DMD_KEY_GAN_INPUT_GRADIENT = "gan_input_gradient"
DMD_KEY_REAL_LATENT = "real_latent"
DMD_KEY_REAL_PROMPTS = "real_prompts"
DMD_KEY_DENOISED_TIMESTEP_FROM = "denoised_timestep_from"
DMD_KEY_DENOISED_TIMESTEP_TO = "denoised_timestep_to"

DMD_PROMPT_TENSOR_PREFIX = "dmd_prompt_tensor_"
DMD_NEGATIVE_PROMPT_TENSOR_PREFIX = "dmd_negative_prompt_tensor_"
DMD_REAL_PROMPT_TENSOR_PREFIX = "dmd_real_prompt_tensor_"
DMD_GENERATOR_LOSS_EPS = 1e-6


class DMDResidualConv3DHead(nn.Module):
    """Classify coarse fake-score video features with a residual Conv3D head."""

    def __init__(self, feature_dim: int, head_config: DMDGANHeadConfig) -> None:
        super().__init__()
        hidden_dim = head_config.hidden_dim
        self.input_projection = nn.Conv3d(feature_dim, hidden_dim, kernel_size=1)
        self.residual_block = nn.Sequential(
            nn.GroupNorm(1, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.GroupNorm(1, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1),
        )
        self.output_projection = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one real/fake logit per diffusion sample."""
        # Bound classifier cost while retaining coarse temporal and spatial structure.
        features = F.adaptive_avg_pool3d(
            features,
            tuple(min(size, limit) for size, limit in zip(features.shape[2:], (4, 8, 8))),
        ).float()
        features = self.input_projection(features)
        features = features + self.residual_block(features)
        return self.output_projection(features.mean(dim=(2, 3, 4))).flatten()


def latent_shape_from_vae_config(
    *,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    vae_config: Any,
) -> list[int]:
    """Convert DMD sample dimensions to canonical ``[B,F,C,H,W]`` VAE latents."""
    spatial_scale = getattr(vae_config, "scale_factor_spatial", None)
    if spatial_scale is None:
        spatial_scale = 2 ** len(vae_config.temperal_downsample)
    spatial_scale = int(spatial_scale)
    temporal_scale = int(getattr(vae_config, "scale_factor_temporal", 1))
    if height % spatial_scale != 0 or width % spatial_scale != 0:
        raise ValueError(f"sample height and width must be divisible by VAE spatial scale {spatial_scale}")
    if (num_frames - 1) % temporal_scale != 0:
        raise ValueError(f"sample num_frames must satisfy (num_frames - 1) % {temporal_scale} == 0")
    return [
        batch_size,
        (num_frames - 1) // temporal_scale + 1,
        int(vae_config.z_dim),
        height // spatial_scale,
        width // spatial_scale,
    ]


def compute_dmd_fake_score_loss(
    flow_pred: torch.Tensor,
    clean_latents: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Compute DMD fake-score flow-matching loss."""
    target_flow = noise.float() - clean_latents.float()
    return F.mse_loss(flow_pred.float(), target_flow)


def compute_dmd_generator_loss(
    generated_latents: torch.Tensor,
    fake_pred_x0: torch.Tensor,
    real_pred_x0: torch.Tensor,
    gradient_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute DMD generator distillation loss and normalized gradient magnitude."""
    reduce_dims = tuple(range(1, generated_latents.ndim))
    generated_latents_float = generated_latents.float()
    gradient = fake_pred_x0.float() - real_pred_x0.float()
    normalizer = torch.abs(generated_latents.detach().float() - real_pred_x0.detach().float()).mean(
        dim=reduce_dims,
        keepdim=True,
    )
    gradient = torch.nan_to_num(
        gradient / normalizer.clamp_min(DMD_GENERATOR_LOSS_EPS),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if gradient_mask is not None:
        loss = 0.5 * F.mse_loss(
            generated_latents_float[gradient_mask],
            (generated_latents_float - gradient).detach()[gradient_mask],
            reduction="mean",
        )
    else:
        loss = 0.5 * F.mse_loss(
            generated_latents_float,
            (generated_latents_float - gradient).detach(),
            reduction="mean",
        )
    return loss, torch.mean(torch.abs(gradient))


def compute_ddmd_generator_loss(
    generated_latents: torch.Tensor,
    fake_pred_x0: torch.Tensor,
    real_pred_x0: torch.Tensor,
    real_cond_pred_x0: torch.Tensor,
    real_uncond_pred_x0: torch.Tensor,
    guidance_scale: float,
    gradient_scale: float,
    gradient_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute decoupled distribution-matching and CFG-augmentation gradients."""
    reduce_dims = tuple(range(1, generated_latents.ndim))
    generated_latents_float = generated_latents.float()
    distribution_gradient = (fake_pred_x0.float() - real_pred_x0.float()) / torch.abs(
        generated_latents.detach().float() - real_pred_x0.detach().float()
    ).mean(dim=reduce_dims, keepdim=True).clamp_min(DMD_GENERATOR_LOSS_EPS)
    cfg_augmentation_gradient = (guidance_scale - 1.0) * (
        real_uncond_pred_x0.float() - real_cond_pred_x0.float()
    ) / torch.abs(
        generated_latents.detach().float() - real_cond_pred_x0.detach().float()
    ).mean(dim=reduce_dims, keepdim=True).clamp_min(DMD_GENERATOR_LOSS_EPS)
    distribution_gradient = torch.nan_to_num(distribution_gradient, nan=0.0, posinf=0.0, neginf=0.0)
    cfg_augmentation_gradient = torch.nan_to_num(
        cfg_augmentation_gradient,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    gradient = gradient_scale * (distribution_gradient + cfg_augmentation_gradient)
    target = (generated_latents_float - gradient).detach()
    if gradient_mask is not None:
        loss = 0.5 * F.mse_loss(generated_latents_float[gradient_mask], target[gradient_mask])
    else:
        loss = 0.5 * F.mse_loss(generated_latents_float, target)
    return (
        loss,
        torch.mean(torch.abs(distribution_gradient)),
        torch.mean(torch.abs(cfg_augmentation_gradient)),
    )


def compute_dmd_gan_generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Compute the non-saturating DMD2 generator GAN objective."""
    return F.softplus(-fake_logits.float()).mean()


def compute_dmd_gan_classification_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute the logistic DMD2 fake-score classification objective."""
    return F.softplus(-real_logits.float()).mean() + F.softplus(fake_logits.float()).mean()


def compute_dmd_gan_generator_surrogate(
    generated_latents: torch.Tensor,
    input_gradient: torch.Tensor,
    gradient_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn the remote fake-score classifier input gradient into an exact local VJP surrogate."""
    gradient = input_gradient.detach().float()
    if gradient_mask is not None:
        mask = gradient_mask.reshape(*gradient_mask.shape, *([1] * (generated_latents.ndim - 2)))
        gradient = gradient * mask
    loss = torch.sum(generated_latents.float() * gradient) / generated_latents.shape[0]
    return loss, torch.mean(torch.abs(gradient))


def generate_self_forcing_sample(
    *,
    model: torch.nn.Module,
    scheduler: DMDFlowMatchScheduler,
    config: DMDSelfForcingConfig,
    denoising_step_list: torch.Tensor,
    noise: torch.Tensor,
    prompt: Mapping[str, Any],
    noise_generator: torch.Generator,
    exit_step_generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, int | None, int | None]:
    """Generate an AR Self-Forcing sample through a model's stable AR interface."""
    latent_shape = list(noise.shape)
    batch_size, num_frames, _, _, _ = latent_shape
    frames_per_block = int(config.num_frame_per_block)
    if config.independent_first_frame:
        if (num_frames - 1) % frames_per_block != 0:
            raise ValueError("self_forcing.independent_first_frame requires divisible trailing frames")
        block_sizes = [1] + [frames_per_block] * ((num_frames - 1) // frames_per_block)
    else:
        if num_frames % frames_per_block != 0:
            raise ValueError("num_frames must be divisible by self_forcing.num_frame_per_block")
        block_sizes = [frames_per_block] * (num_frames // frames_per_block)

    ar_state = model.create_ar_state(
        latent_shape=latent_shape,
        dtype=noise.dtype,
        device=noise.device,
    )
    exit_flags = sample_denoising_indices(
        len(block_sizes),
        len(denoising_step_list),
        noise.device,
        generator=exit_step_generator,
        last_step_only=bool(config.last_step_only),
    )
    gradient_start = max(0, num_frames - int(config.num_training_frames))
    generated_blocks: list[torch.Tensor] = []
    frame_start = 0

    def predict_x0(noisy_latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        flow_pred = model.forward_ar(
            noisy_image_or_video=noisy_latents,
            prompt=prompt,
            timestep=timestep,
            ar_state=ar_state,
            frame_start=frame_start,
        )
        return scheduler.x0_from_flow_pred(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_latents.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

    for block_index, block_size in enumerate(block_sizes):
        noisy_latents = noise[:, frame_start: frame_start + block_size]
        denoised_latents = noisy_latents
        for step_index, current_timestep in enumerate(denoising_step_list):
            exit_step = exit_flags[0] if config.same_step_across_blocks else exit_flags[block_index]
            timestep = torch.full(
                [batch_size, block_size],
                current_timestep.item(),
                device=noise.device,
                dtype=denoising_step_list.dtype,
            )
            if step_index == exit_step:
                block_end = frame_start + block_size
                if block_end <= gradient_start:
                    with torch.no_grad():
                        denoised_latents = predict_x0(noisy_latents, timestep)
                else:
                    denoised_latents = predict_x0(noisy_latents, timestep)
                break

            with torch.no_grad():
                denoised_latents = predict_x0(noisy_latents, timestep)
                next_timestep = denoising_step_list[step_index + 1]
                step_noise = torch.randn(
                    denoised_latents.shape,
                    device=denoised_latents.device,
                    dtype=denoised_latents.dtype,
                    generator=noise_generator,
                )
                noisy_latents = scheduler.add_noise(
                    denoised_latents.flatten(0, 1),
                    step_noise.flatten(0, 1),
                    torch.full(
                        [batch_size * block_size],
                        next_timestep.item(),
                        device=noise.device,
                        dtype=denoising_step_list.dtype,
                    ),
                ).unflatten(0, denoised_latents.shape[:2])

        generated_blocks.append(denoised_latents)
        context_timestep = torch.full_like(timestep, int(config.context_noise))
        context_noise = torch.randn(
            denoised_latents.shape,
            device=denoised_latents.device,
            dtype=denoised_latents.dtype,
            generator=noise_generator,
        )
        context_latent = scheduler.add_noise(
            denoised_latents.flatten(0, 1),
            context_noise.flatten(0, 1),
            context_timestep.flatten(0, 1),
        ).unflatten(0, denoised_latents.shape[:2])
        with torch.no_grad():
            model.forward_ar(
                noisy_image_or_video=context_latent,
                prompt=prompt,
                timestep=context_timestep,
                ar_state=ar_state,
                frame_start=frame_start,
            )
        frame_start += block_size

    generated_latents = torch.cat(generated_blocks, dim=1)
    gradient_mask = torch.zeros(
        generated_latents.shape[:2],
        device=generated_latents.device,
        dtype=torch.bool,
    )
    gradient_mask[:, gradient_start:] = True
    if not config.same_step_across_blocks:
        return generated_latents.to(noise.dtype), gradient_mask, None, None
    timestep_from, timestep_to = get_score_timestep_window(
        scheduler,
        denoising_step_list,
        exit_flags[0],
        noise.device,
    )
    return generated_latents.to(noise.dtype), gradient_mask, timestep_from, timestep_to


class DMDFlowMatchScheduler:
    """Flow-match scheduler with the training semantics used by DMD."""

    def __init__(
        self,
        num_inference_steps: int = 100,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_min: float = 0.0,
        extra_one_step: bool = False,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_min = sigma_min
        self.extra_one_step = extra_one_step
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps: int = 100,
        device: torch.device | None = None,
    ) -> None:
        """Create active flow-match timesteps and sigmas."""
        if self.extra_one_step:
            sigmas = torch.linspace(1.0, self.sigma_min, num_inference_steps + 1, device=device)[:-1]
        else:
            sigmas = torch.linspace(1.0, self.sigma_min, num_inference_steps, device=device)

        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.sigmas = sigmas
        self.timesteps = sigmas * self.num_train_timesteps

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Add flow-match noise: ``x_t = (1 - sigma_t) * x_0 + sigma_t * noise``."""
        sigma = self.sigma(timestep, sample_ndim=original_samples.ndim).to(
            device=noise.device,
            dtype=self.sigmas.dtype,
        )
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def sigma(self, timestep: torch.Tensor, sample_ndim: int = 4) -> torch.Tensor:
        """Find sigma(t) for active flow-match timesteps."""
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        sigmas = self.sigmas.to(device=timestep.device, dtype=torch.float32)
        timesteps = self.timesteps.to(device=timestep.device, dtype=torch.float32)
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.float().unsqueeze(1)).abs(),
            dim=1,
        )
        return sigmas[timestep_id].reshape(-1, *([1] * (sample_ndim - 1)))

    def x0_from_flow_pred(
        self,
        flow_pred: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Convert flow prediction to x0: ``x0 = xt - sigma * flow``."""
        original_dtype = flow_pred.dtype
        return (xt.float() - self.sigma(timestep, sample_ndim=xt.ndim) * flow_pred.float()).to(original_dtype)


def sample_denoising_indices(
    num_samples: int,
    num_denoising_steps: int,
    device: torch.device,
    *,
    generator: torch.Generator,
    last_step_only: bool = False,
) -> list[int]:
    """Sample denoising indices consistently across distributed model shards."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        indices = torch.empty(num_samples, dtype=torch.long, device=device)
    elif last_step_only:
        indices = torch.full((num_samples,), num_denoising_steps - 1, dtype=torch.long, device=device)
    else:
        indices = torch.randint(
            low=0,
            high=num_denoising_steps,
            size=(num_samples,),
            device=device,
            generator=generator,
        )
    if dist.is_initialized():
        dist.broadcast(indices, src=0)
    return indices.tolist()


def get_score_timestep_window(
    scheduler: DMDFlowMatchScheduler,
    denoising_step_list: torch.Tensor,
    denoising_index: int,
    device: torch.device,
) -> tuple[int, int]:
    """Map one generator denoising index to its score-query timestep window."""
    active_timesteps = scheduler.timesteps.to(device)
    exit_timestep = denoising_step_list[denoising_index].to(device)
    num_train_timesteps = scheduler.num_train_timesteps
    timestep_from = num_train_timesteps - torch.argmin(
        (active_timesteps - exit_timestep).abs(), dim=0
    ).item()
    timestep_from = max(0, min(num_train_timesteps, timestep_from))
    if denoising_index == len(denoising_step_list) - 1:
        return timestep_from, 0
    next_exit_timestep = denoising_step_list[denoising_index + 1].to(device)
    timestep_to = num_train_timesteps - torch.argmin(
        (active_timesteps - next_exit_timestep).abs(), dim=0
    ).item()
    timestep_to = max(0, min(num_train_timesteps, timestep_to))
    return timestep_from, min(timestep_from, timestep_to)


def shift_dmd_timesteps(
    timestep: torch.Tensor,
    num_train_timestep: int,
    shift: float,
) -> torch.Tensor:
    """Apply the flow-matching timestep shift and valid training range."""
    if shift > 1:
        normalized_timestep = timestep / num_train_timestep
        timestep = shift * normalized_timestep / (1 + (shift - 1) * normalized_timestep) * num_train_timestep
    return timestep.clamp(int(0.02 * num_train_timestep), int(0.98 * num_train_timestep))


def sample_dmd_timesteps(
    timestep_config: DMDScoreTimestepConfig,
    device: torch.device,
    denoised_timestep_from: int | torch.Tensor | None,
    denoised_timestep_to: int | torch.Tensor | None,
    batch_size: int,
    num_frames: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample DMD score timesteps for generator/fake-score updates."""
    num_train_timestep = int(timestep_config.num_train_timestep)
    min_timesteps = torch.full(
        (batch_size,),
        int(timestep_config.min_score_timestep),
        device=device,
        dtype=torch.long,
    )
    if timestep_config.ts_schedule and denoised_timestep_to is not None:
        min_timesteps = torch.as_tensor(denoised_timestep_to, device=device, dtype=torch.long).flatten()
        if min_timesteps.numel() == 1:
            min_timesteps = min_timesteps.expand(batch_size)

    max_timesteps = torch.full((batch_size,), num_train_timestep, device=device, dtype=torch.long)
    if timestep_config.ts_schedule_max and denoised_timestep_from is not None:
        max_timesteps = torch.as_tensor(denoised_timestep_from, device=device, dtype=torch.long).flatten()
        if max_timesteps.numel() == 1:
            max_timesteps = max_timesteps.expand(batch_size)

    if min_timesteps.numel() != batch_size or max_timesteps.numel() != batch_size:
        raise ValueError("DMD timestep bounds must contain one value per batch sample")
    if torch.any(min_timesteps >= max_timesteps):
        raise ValueError("DMD timestep sampling requires min_timestep < max_timestep for every sample")

    if torch.all(min_timesteps == min_timesteps[0]) and torch.all(max_timesteps == max_timesteps[0]):
        timestep = torch.randint(
            int(min_timesteps[0].item()),
            int(max_timesteps[0].item()),
            [batch_size, 1],
            device=device,
            dtype=torch.long,
            generator=generator,
        )
    else:
        timestep = torch.stack(
            [
                torch.randint(
                    int(min_timestep.item()),
                    int(max_timestep.item()),
                    (1,),
                    device=device,
                    dtype=torch.long,
                    generator=generator,
                )
                for min_timestep, max_timestep in zip(min_timesteps, max_timesteps)
            ]
        )
    timestep = timestep.repeat(1, num_frames)
    return shift_dmd_timesteps(timestep, num_train_timestep, float(timestep_config.sample_shift))
