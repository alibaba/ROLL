import os.path
import shutil
import subprocess
from datetime import datetime
from multiprocessing import Pool
from typing import List, Callable

from codetiming import Timer
import torch

from roll.agentic.utils import dump_frames_as_gif
from roll.utils.logging import get_logger

logger = get_logger()


def dump_rollout_render(save_dir, step, frames: List[List], env_ids: List, tags: List, episode_scores: List):
    with Timer(name="dump", logger=None) as timer:
        try:
            local_save_dir = f'/tmp/rollout_render/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
            os.makedirs(local_save_dir, exist_ok=True)
            os.makedirs(save_dir, exist_ok=True)

            args_list = [
                (os.path.join(local_save_dir, f"{step}", f"{env_id}_{tag}_{episode_score:.1f}.gif"), frame_list)
                for frame_list, env_id, tag, episode_score in zip(frames, env_ids, tags, episode_scores)
                if len(frame_list) > 0
            ]
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            with Pool(processes=16) as pool:
                pool.starmap(dump_frames_as_gif, args_list)

            rar_file_path = os.path.join(
                "/tmp", f'rollout_render_{datetime.now().strftime("%Y%m%d-%H%M%S")}_{step}.zip'
            )
            command = ["zip", "-rq", rar_file_path, local_save_dir]
            subprocess.run(command, check=True)
            shutil.move(rar_file_path, save_dir)
            shutil.rmtree(local_save_dir, ignore_errors=True)
        except Exception as e:
            logger.error(f"dump rollout render failed: {e}")
    logger.info(f"dump_rollout_render_cost: {timer.last}")

def get_score_normalize_fn(rn_cfg) -> Callable:
    grouping, method = rn_cfg.grouping, rn_cfg.method
    if method == "mean_std":
        norm_func = lambda x: (
            (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
            if x.std(dim=-1, keepdim=True).abs().max() > 1e-6
            else torch.zeros_like(x)
        )  # stable to bf16 than x.std()
    elif method == "mean":
        norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True))
    elif method == "asym_clip":
        norm_func = lambda x: (
            (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
            if x.std(dim=-1, keepdim=True).abs().max() > 1e-6
            else torch.zeros_like(x)
        ).clamp(min=-1, max=3)
    elif method == "identity":
        norm_func = lambda x: x
    else:
        raise ValueError(f"Invalid normalization method: {method}")

    return norm_func