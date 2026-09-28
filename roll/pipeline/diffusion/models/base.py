"""Base abstractions for diffusion model adapters.

This module defines:

- DiffusionAdapter: Model operations shared between trainer and inference.
  Encapsulates forward pass, diffusion loop, log-prob replay, and prompt encoding.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Literal

import torch


EncodedPrompt = Mapping[str, Any]


@dataclass(frozen=True)
class DiffusionPrediction:
    """Normalized single-step predictions exposed by diffusion model adapters.

    Adapters populate only the fields defined by their model parameterization.
    Scheduler-derived transition statistics such as log-probabilities, means,
    and variances do not belong to this model prediction contract.

    Attributes:
        flow_pred: Flow-matching or rectified-flow velocity.
        pred_x0: Estimated clean latent at timestep zero.
        noise_pred: Predicted diffusion noise (epsilon parameterization).
        score_pred: Predicted score of the noisy sample distribution.
        variance_pred: Learned variance output produced by models that predict it.
    """

    flow_pred: torch.Tensor | None = None
    pred_x0: torch.Tensor | None = None
    noise_pred: torch.Tensor | None = None
    score_pred: torch.Tensor | None = None
    variance_pred: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Input data structures
# ---------------------------------------------------------------------------


@dataclass
class DiffuseInput:
    """Base class for diffusion model input parameters.

    Subclasses define model-specific fields. Provides a ``from_dict``
    classmethod for convenient construction from dictionaries.
    """

    @classmethod
    def from_dict(cls, data: dict) -> "DiffuseInput":
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in field_names})


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------


@dataclass
class DiffuseOutput:
    """Output of a full diffusion generation loop.

    Field naming reflects what is actually collected: only the SDE window steps
    are recorded, NOT the full diffusion trajectory. Hence no "all_" prefix.

    Attributes:
        latents: Recorded latent trajectory within the SDE window. Shape [B, K+1, S, C]
            where K is the number of SDE steps actually recorded. This is NOT
            necessarily the final denoised latent when the SDE window is smaller
            than the full diffusion trajectory.
        final_latents: Final denoised latent after the full diffusion loop. Shape
            [B, S, C]. Use this for image decoding.
        log_probs: Per-step log probabilities within the SDE window. Shape [B, K].
            May be zeros if sde_type="ode" or logprobs=False.
        timesteps: Timestep values corresponding to each recorded step. Shape [B, K].
    """

    latents: torch.Tensor
    final_latents: torch.Tensor
    log_probs: torch.Tensor
    timesteps: torch.Tensor


# ---------------------------------------------------------------------------
# DiffusionAdapter — model operations (forward, diffuse, replay)
# ---------------------------------------------------------------------------


class DiffusionAdapter(ABC):
    """Base class for diffusion model operations shared between trainer and inference.

    Subclasses implement model-specific forward pass and diffusion loop. The same
    adapter class can be instantiated on both the FSDP2 training strategy side
    (with a wrapped transformer) and the vllm_omni inference pipeline side (with
    a bare transformer).

    The adapter does NOT own model construction or weight loading — it receives
    already-constructed components (transformer, scheduler, text_encoder, vae) and
    operates on them. This keeps the adapter lightweight and decoupled from framework-
    specific initialization (FSDP2, vllm_omni CuMemAllocator, etc.).
    """

    name: str = "base"

    # Number of prefix tokens (system message + user header) to drop from text encoder
    # hidden states. Set by the pipeline from the dataset's system_prompt. Default 0
    # means no dropping (e.g. when there is no system prompt).
    prompt_template_encode_start_idx: int = 0

    def __init__(
        self,
        transformer: torch.nn.Module,
        scheduler: Any | None = None,
    ) -> None:
        """Bind an adapter to a backend-built transformer."""
        self.transformer = transformer
        self.scheduler = scheduler

    def forward_step(
        self,
        *,
        latents: torch.Tensor,
        prompt: EncodedPrompt,
        timestep: torch.Tensor,
        negative_prompt: EncodedPrompt | None = None,
        guidance_scale: float = 0.0,
        **model_kwargs: Any,
    ) -> DiffusionPrediction:
        """Execute one normalized model prediction with optional CFG.

        This is the atomic model operation shared by prediction-space training
        and full diffusion sampling. Concrete adapters translate the canonical
        latent, prompt, and timestep inputs into model-native arguments, then
        normalize the raw model output into ``DiffusionPrediction``.

        Args:
            latents: Current noisy latents.
            prompt: Encoded positive prompt conditioning.
            timestep: Diffusion timestep for each sample.
            negative_prompt: Encoded negative prompt conditioning used by CFG.
            guidance_scale: Classifier-free guidance scale.
            **model_kwargs: Additional model-specific forward arguments.

        Returns:
            The normalized model prediction for this diffusion step.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support single-step prediction")

    def diffuse(self, input: DiffuseInput) -> DiffuseOutput:
        """Run the full diffusion sampling loop (generation mode).

        Iterates over all timesteps, applying the scheduler's SDE/ODE integration.
        Within the SDE window, stochastic noise is injected and log-probs are recorded.
        Outside the window, the step degenerates to a deterministic ODE step.

        Args:
            input: DiffuseInput subclass with model-specific fields populated.

        Returns:
            DiffuseOutput containing the trajectory and log-probs.
        """
        raise NotImplementedError

    @abstractmethod
    def encode_prompt(
        self,
        *,
        prompt_encoder: Any,
        prompt_inputs: Mapping[str, Any],
    ) -> EncodedPrompt:
        """Encode model-native prompt inputs into conditioning tensors.

        Args:
            prompt_encoder: Backend-built prompt encoder components.
            prompt_inputs: Model-specific prompt inputs interpreted by the adapter.

        Returns:
            Model-specific conditioning consumed by this adapter's model calls.
        """
        ...

    def load_frozen_weights(self) -> None:
        """Reload frozen components (text_encoder, VAE) from disk.

        Called after sleep_level=2 wake-up where non-trainable weights were discarded.
        Default implementation is a no-op for adapters without frozen components.
        """

    def reinit_non_persistent_states(self) -> None:
        """Recompute non-persistent states (e.g., RoPE frequencies) after memory discard.

        Called after sleep_level=2 wake-up. Default is a no-op.
        """

    def compute_weights_hash(self) -> dict[str, str]:
        """Compute stable hashes for each major component's weights (for diagnostics).

        Returns:
            Dict mapping component name to hash string.
        """
        return {}
