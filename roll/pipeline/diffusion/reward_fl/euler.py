from diffusers import FlowMatchEulerDiscreteScheduler
from torch import Tensor

import numpy as np
import torch


def unsqueeze_to_ndim(in_tensor: Tensor, tgt_n_dim: int):
    if in_tensor.ndim > tgt_n_dim:
        return in_tensor
    if in_tensor.ndim < tgt_n_dim:
        in_tensor = in_tensor[(...,) + (None,) * (tgt_n_dim - in_tensor.ndim)]
    return in_tensor


def get_timesteps(num_steps, max_steps: int = 1000):
    return np.linspace(max_steps, 0, num_steps + 1, dtype=np.float32)


def timestep_shift(timesteps, shift: float = 1.0):
    return shift * timesteps / (1 + (shift - 1) * timesteps)


class EulerScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(
        self,
        num_train_timesteps: int,
        shift: float = 1.0,
        device: torch.device | str = "cuda",
        **kwargs
    ) -> None:
        super().__init__(num_train_timesteps=num_train_timesteps, shift=shift,**kwargs)
        self.init_noise_sigma = 1.0
        self.num_train_timesteps = num_train_timesteps
        self._shift = shift
        self.init_noise_sigma = 1.0
        self.device = device
        self.training = True
        self.set_timesteps(num_inference_steps=num_train_timesteps)
    

    def set_shift(self, shift: float = 1.0):
        self.sigmas = self.timesteps_ori / self.num_train_timesteps
        self.sigmas = timestep_shift(self.sigmas, shift=shift)
        self.timesteps = self.sigmas * self.num_train_timesteps
        self._shift = shift
    

    def set_timesteps(
        self, num_inference_steps: int, device: torch.device | str | int | None = None
    ):
        timesteps = get_timesteps(
            num_steps=num_inference_steps, max_steps=self.num_train_timesteps
        )
        self.timesteps = torch.from_numpy(timesteps).to(
            dtype=torch.float32, device=device or self.device
        )
        self.timesteps_ori = self.timesteps.clone()
        self.set_shift(self._shift)
        self._step_index = None
        self._begin_index = None
    
    
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        sample: torch.FloatTensor,
        **kwargs,
    ) -> tuple:
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )
        if self.step_index is None:
            self._init_step_index(timestep)
        sigma = unsqueeze_to_ndim(self.sigmas[self.step_index], sample.ndim).to(sample.device)
        sigma_next = unsqueeze_to_ndim(self.sigmas[self.step_index + 1], sample.ndim).to(sample.device)
        x_t_next = sample + (sigma_next - sigma) * model_output
        self._step_index += 1
        return x_t_next