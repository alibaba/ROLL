#!/usr/bin/env python3
"""Integration check for FSDP2 TRL critic HF checkpoint export.

This script intentionally exercises the real ROLL path:
Cluster -> CriticWorker -> FSDP2Strategy -> save_checkpoint -> reload critic
from the exported HF directory. It compares critic values on a fixed batch
before and after reload.
"""

import argparse
import copy
import glob
import os
import shutil

import ray
import torch
from dacite import from_dict
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from tensordict import TensorDict

from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.initialize import init
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.base_worker import CriticWorker
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.utils.offload_states import OffloadStateType
from roll.utils.logging import get_logger


logger = get_logger()
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "examples/docs_examples/example_ppo.yaml")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to an RLVR YAML config.",
    )
    parser.add_argument(
        "--work-dir",
        default="/tmp/roll_fsdp2_critic_ckpt_test",
        help="Local working directory for temporary checkpoint output.",
    )
    parser.add_argument("--checkpoint-step", type=int, default=1)
    parser.add_argument("--critic-gpus", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--param-perturb-delta", type=float, default=1e-3)
    parser.add_argument("--min-perturb-diff", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument(
        "--prompt",
        action="append",
        default=[
            "Solve 1+1.",
            "What is the capital of France?",
            "Write a short proof that 2 is even.",
            "Compute 3 * 7.",
        ],
        help="Prompt text. Can be repeated; prompts are cycled to fill the critic DP world size.",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Do not delete --work-dir before running.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> RLVRConfig:
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
    return from_dict(data_class=RLVRConfig, data=OmegaConf.to_container(cfg, resolve=True))


def prepare_config(config: RLVRConfig, output_dir: str, critic_gpus: int) -> RLVRConfig:
    config.track_with = "stdout"
    config.tracker_kwargs = {}
    config.output_dir = output_dir
    config.logging_dir = os.path.join(output_dir, "logs")
    config.resume_from_checkpoint = False
    config.save_steps = 1
    config.max_steps = max(config.max_steps, 2)
    config.num_nodes = 1
    config.num_gpus_per_node = critic_gpus

    # Keep all checkpoint artifacts local and deterministic for this test.
    config.checkpoint_config = {}
    config.critic.checkpoint_config = {}
    config.critic.training_args.output_dir = output_dir
    config.critic.device_mapping = list(range(critic_gpus))
    config.critic.num_gpus_per_worker = 1
    config.critic.world_size = critic_gpus
    config.critic.strategy_args.strategy_config["async_save_ckpt"] = False
    config.critic.strategy_args.strategy_config["offload_policy"] = False
    config.critic.strategy_args.strategy_config["fsdp_size"] = critic_gpus
    return config


class CheckpointTestCriticWorker(CriticWorker):
    def perturb_all_params(self, delta: float):
        self.strategy.load_states(include=[OffloadStateType.model_params])
        model = self.strategy.unwrap_model()
        num_perturbed = 0
        with torch.no_grad():
            for param in model.parameters():
                if param.dtype.is_floating_point:
                    param.add_(delta)
                    num_perturbed += param.numel()
        self.strategy.offload_states(include=[OffloadStateType.model_params])
        self.logger.info("Perturbed %s floating point parameters by %s", num_perturbed, delta)
        return num_perturbed


class CriticOnlyPipeline(BasePipeline):
    def __init__(self, pipeline_config: RLVRConfig, cluster_name: str = "critic"):
        super().__init__(pipeline_config)
        self.checkpoint_clusters = []
        self.model_update_groups = []
        self.critics = []
        self.critic = self.create_critic(cluster_name=cluster_name, worker_config=self.pipeline_config.critic)
        self.set_checkpoint_clusters(self.critic)

    def create_critic(self, cluster_name: str, worker_config):
        critic = Cluster(
            name=cluster_name,
            worker_cls=CheckpointTestCriticWorker,
            resource_manager=self.resource_manager,
            worker_config=worker_config,
        )
        critic.initialize(pipeline_config=self.pipeline_config, blocking=True)
        self.critics.append(critic)
        return critic

    def shutdown(self):
        for critic in getattr(self, "critics", []):
            for worker in getattr(critic, "workers", []):
                ray.kill(worker, no_restart=True)


def build_fixed_batch(config: RLVRConfig, batch_size: int, seq_len: int, prompts: list[str]) -> DataProto:
    tokenizer = default_tokenizer_provider(config.critic.model_args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = [prompts[i % len(prompts)] for i in range(batch_size)]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)

    batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=[batch_size],
    )
    return DataProto(
        batch=batch,
        meta_info={
            "global_step": 0,
            "is_offload_states": True,
            "loss_mask_keys": [],
        },
    )


def compute_values(critic: Cluster, batch: DataProto) -> torch.Tensor:
    values = critic.compute_values(copy.deepcopy(batch), blocking=True)
    return values.batch["values"].detach().cpu()


def perturb_all_params(critic: Cluster, delta: float):
    counts = ray.get([worker.perturb_all_params.remote(delta) for worker in critic.workers])
    if not all(count > 0 for count in counts):
        raise AssertionError(f"Some critic workers did not perturb any parameters: {counts}")
    return counts


def save_critic_checkpoint(pipeline: CriticOnlyPipeline, step: int):
    pipeline.state.log_history.append({"integration/test": 1.0})
    pipeline.do_checkpoint(global_step=step, is_last_step=True)


def find_hf_critic_dir(output_dir: str, step: int) -> str:
    pattern = os.path.join(output_dir, "critic-*", f"checkpoint-{step}", "critic")
    candidates = sorted(glob.glob(pattern))
    candidates = [
        path
        for path in candidates
        if os.path.exists(os.path.join(path, "value_head.safetensors"))
        and (
            os.path.exists(os.path.join(path, "model.safetensors.index.json"))
            or os.path.exists(os.path.join(path, "model.safetensors"))
            or os.path.exists(os.path.join(path, "pytorch_model.bin"))
        )
    ]
    if not candidates:
        raise FileNotFoundError(f"No exported critic HF checkpoint found under {pattern}")
    return candidates[0]


def assert_required_files(ckpt_dir: str):
    required = ["value_head.safetensors"]
    missing = [name for name in required if not os.path.exists(os.path.join(ckpt_dir, name))]
    if missing:
        raise FileNotFoundError(f"Missing required checkpoint files in {ckpt_dir}: {missing}")
    if not os.path.exists(os.path.join(ckpt_dir, "config.json")):
        logger.warning("config.json is missing from %s; reload may fail if config cannot be inferred.", ckpt_dir)


def main():
    args = parse_args()
    work_dir = os.path.abspath(args.work_dir)
    if os.path.exists(work_dir) and not args.keep_work_dir:
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    init()

    base_config = prepare_config(load_config(args.config), os.path.join(work_dir, "run_a"), args.critic_gpus)
    batch_size = base_config.critic.world_size
    fixed_batch = build_fixed_batch(base_config, batch_size, args.seq_len, args.prompt)

    pipeline = CriticOnlyPipeline(base_config)
    try:
        values_initial = compute_values(pipeline.critic, fixed_batch)
        perturb_all_params(pipeline.critic, args.param_perturb_delta)
        values_before_save = compute_values(pipeline.critic, fixed_batch)
        perturb_diff = (values_initial - values_before_save).abs().max().item()
        if perturb_diff <= args.min_perturb_diff:
            raise AssertionError(
                "Parameter perturbation did not change critic values enough: "
                f"max_abs_diff={perturb_diff}, min_perturb_diff={args.min_perturb_diff}"
            )
        save_critic_checkpoint(pipeline, args.checkpoint_step)
        ckpt_dir = find_hf_critic_dir(base_config.output_dir, args.checkpoint_step)
        assert_required_files(ckpt_dir)
        logger.info("Reloading critic from exported checkpoint: %s", ckpt_dir)

        reload_worker_config = copy.deepcopy(base_config.critic)
        reload_worker_config.model_args.model_name_or_path = ckpt_dir
        reload_critic = pipeline.create_critic(cluster_name="critic_reload", worker_config=reload_worker_config)
        values_after = compute_values(reload_critic, fixed_batch)
    finally:
        pipeline.shutdown()

    torch.testing.assert_close(values_before_save, values_after, rtol=args.rtol, atol=args.atol)
    max_abs_diff = (values_before_save - values_after).abs().max().item()
    pass_message = (
        "PASS: FSDP2 critic checkpoint reload values match. "
        f"ckpt_dir={ckpt_dir} "
        f"values_shape={tuple(values_before_save.shape)} "
        f"perturb_delta={args.param_perturb_delta} "
        f"initial_to_saved_max_abs_diff={perturb_diff} "
        f"max_abs_diff={max_abs_diff} "
        f"rtol={args.rtol} atol={args.atol}"
    )
    logger.info(pass_message)
    print(pass_message)


if __name__ == "__main__":
    main()
