import numpy as np
import torch
from tensordict import TensorDict

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto


class ImageRewardWorker(Worker):
    """Simple image-level reward worker with RLVR-compatible DataProto outputs."""

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)

    def _compute_image_score(self, image_tensor: torch.Tensor, ground_truth):
        # Minimal deterministic score for first-phase integration.
        base_score = image_tensor.float().mean()
        if ground_truth is None:
            return base_score

        if isinstance(ground_truth, (float, int)):
            return -torch.abs(base_score - torch.tensor(float(ground_truth), device=base_score.device))

        # fallback: hash-like stable scalar for non-numeric GT.
        gt_scalar = float(abs(hash(str(ground_truth))) % 1000) / 1000.0
        return -torch.abs(base_score - torch.tensor(gt_scalar, device=base_score.device))

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        if "responses" not in data.batch.keys():
            raise ValueError("ImageRewardWorker expects `responses` tensor in batch.")

        responses = data.batch["responses"]
        batch_size = responses.shape[0]

        if "all_timesteps" in data.batch.keys():
            step_k = data.batch["all_timesteps"].shape[1]
            token_level_rewards = torch.zeros((batch_size, step_k), dtype=torch.float16, device=responses.device)
        else:
            token_level_rewards = torch.zeros((batch_size, 1), dtype=torch.float16, device=responses.device)

        ground_truth = data.non_tensor_batch.get("ground_truth", np.array([None] * batch_size, dtype=object))
        scores = []
        for idx in range(batch_size):
            score = self._compute_image_score(responses[idx], ground_truth[idx] if idx < len(ground_truth) else None)
            scores.append(score)

        scores_tensor = torch.stack(scores).to(dtype=torch.float16)

        output = DataProto(
            batch=TensorDict(
                {
                    "token_level_rewards": token_level_rewards,
                    "response_level_rewards": scores_tensor,
                    "scores": scores_tensor,
                },
                batch_size=[batch_size],
            ),
            meta_info={"metrics": {"flowgrpo/reward_mean": scores_tensor.float().mean().item()}},
        )

        self.logger.debug(
            "flowgrpo/reward_worker: batch=%s score_mean=%.6f score_std=%.6f",
            batch_size,
            scores_tensor.float().mean().item(),
            scores_tensor.float().std(unbiased=False).item(),
        )
        return output
