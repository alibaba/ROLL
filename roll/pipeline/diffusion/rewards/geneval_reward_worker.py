import json
import os
from typing import Dict

import numpy as np
import torch
from tensordict import TensorDict

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.diffusion.rewards.dump_mixin import RewardDumpMixin
from roll.pipeline.diffusion.rewards.geneval_scorer import GenevalScorer
from roll.platforms import current_platform


class GenevalRewardWorker(RewardDumpMixin, Worker):
    # Fixed layout inside the geneval asset bundle published on MOS.
    DETECTOR_CONFIG_RELPATH = "mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py"
    DETECTOR_CKPT_RELPATH = "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.pth"
    CLIP_CKPT_RELPATH = "vit_large_patch14_clip_224.openai/open_clip_pytorch_model.bin"

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.scorer = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)

        from openlm_hub import repo_download

        mos_uri = self.worker_config.model_args.model_name_or_path
        if not mos_uri:
            raise ValueError(
                "GenevalRewardWorker requires model_args.model_name_or_path set to the geneval asset MOS URI."
            )
        local_dir = repo_download(mos_uri)

        self.scorer = GenevalScorer(
            detector_ckpt_path=os.path.join(local_dir, self.DETECTOR_CKPT_RELPATH),
            detector_config_path=os.path.join(local_dir, self.DETECTOR_CONFIG_RELPATH),
            device=current_platform.device_type,
            clip_ckpt_path=os.path.join(local_dir, self.CLIP_CKPT_RELPATH),
        )

    def _tensor_to_pil_image(self, image_tensor: torch.Tensor):
        return self.tensor_to_pil_image(image_tensor)

    @staticmethod
    def _parse_metadata(ground_truth) -> Dict:
        """Deserialize a ground_truth entry into a geneval metadata dict.

        The offline-converted dataset stores geneval metadata as a JSON string
        in the flat ``ground_truth`` field (schema-aligned with other diffusion
        datasets); dicts are accepted as-is for compatibility.
        """
        if isinstance(ground_truth, dict):
            return ground_truth
        if isinstance(ground_truth, str):
            metadata = json.loads(ground_truth)
            if not isinstance(metadata, dict):
                raise ValueError(f"Geneval ground_truth JSON must decode to a dict, got {type(metadata)}")
            return metadata
        raise ValueError(f"Geneval ground_truth must be a dict or JSON string, got {type(ground_truth)}")

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        if "ground_truth" not in data.non_tensor_batch:
            raise ValueError("GenevalRewardWorker requires non_tensor_batch['ground_truth'].")
        if "id" not in data.non_tensor_batch:
            raise ValueError("GenevalRewardWorker requires non_tensor_batch['id'].")
        if "responses" not in data.batch.keys():
            raise ValueError("GenevalRewardWorker expects `responses` tensor in batch.")

        sample_ids = data.non_tensor_batch["id"].tolist()
        # Prefer unique per-sample IDs injected by rollout loop; fall back to prompt-level "id"
        unique_sample_ids = data.non_tensor_batch.get("sample_id")
        if unique_sample_ids is not None:
            unique_sample_ids = unique_sample_ids.tolist() if hasattr(unique_sample_ids, "tolist") else list(unique_sample_ids)
        else:
            unique_sample_ids = sample_ids
        metadatas = [self._parse_metadata(gt) for gt in data.non_tensor_batch["ground_truth"].tolist()]
        images = [self._tensor_to_pil_image(image_tensor) for image_tensor in data.batch["responses"]]
        # The continuous geneval score is the response reward; the strict/binary
        # verdicts are kept only for logging.
        scores, _, strict_rewards, _, _, details = self.scorer.score(
            images=images,
            metadatas=metadatas,
            only_strict=False,
        )

        response_rewards = scores
        response_rewards_tensor = torch.tensor(response_rewards, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        token_steps = data.batch["all_timesteps"].shape[1] if "all_timesteps" in data.batch.keys() else 1
        token_level_rewards = torch.zeros((len(response_rewards), token_steps), dtype=torch.float32)

        details_array = np.asarray(details, dtype=object)

        # --- Dump step output if configured ---
        self.maybe_dump_step_output(
            data=data,
            sample_ids=unique_sample_ids,
            images=images,
            reward_scores=response_rewards,
            extra_columns={
                "geneval_score": scores,
                "geneval_strict_reward": strict_rewards,
                "geneval_detail": details,
                "geneval_metadata": metadatas,
            },
        )

        # Metrics are intentionally not emitted here: the pipeline aggregates
        # per-task metrics by ``tag`` from the batch, so nothing needs to travel
        # through meta_info (which the scheduler would blindly re-aggregate).
        output = DataProto(
            batch=TensorDict(
                {
                    "token_level_rewards": token_level_rewards,
                    "response_level_rewards": response_rewards_tensor,
                    "scores": scores_tensor,
                },
                batch_size=[len(response_rewards)],
            ),
            non_tensor_batch={
                "geneval_details": details_array,
            },
        )
        return output
