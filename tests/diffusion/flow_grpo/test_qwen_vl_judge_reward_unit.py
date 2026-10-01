import asyncio
import itertools
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import ray
import torch
import torchvision
from PIL import Image

from roll.configs import GeneratingArguments, ModelArguments
from roll.configs.worker_config import StrategyArguments, WorkerConfig
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.generate_scheduler import RolloutContext
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.pipeline.diffusion.rewards.qwen_vl_judge_reward_worker import QwenVLJudgeRewardWorker
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.logging import get_logger

_REWARD_MODEL_PATH = Path("/tmp/models/Qwen/Qwen3-VL-8B-Instruct/")
_ROLLOUT_OUTPUT_PATH = Path("./output/rollout_output.json")
logger = get_logger()


def _torch_dtype(dtype_str: str) -> torch.dtype:
    if not dtype_str.startswith("torch."):
        raise ValueError(f"unsupported tensor dtype string: {dtype_str}")
    dtype_name = dtype_str.split(".", maxsplit=1)[1]
    if not hasattr(torch, dtype_name):
        raise ValueError(f"unknown torch dtype: {dtype_str}")
    return getattr(torch, dtype_name)


def _random_tensor_from_schema(schema: dict) -> torch.Tensor:
    shape = tuple(schema["shape"])
    dtype = _torch_dtype(schema["dtype"])

    if dtype.is_floating_point:
        return torch.randn(shape, dtype=torch.float32).to(dtype)
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        return torch.randint(low=0, high=32, size=shape, dtype=dtype)
    if dtype == torch.bool:
        return torch.randint(low=0, high=2, size=shape, dtype=torch.int32).bool()
    raise ValueError(f"unsupported dtype for random generation: {dtype}")


def _restore_value_from_schema(schema_or_value):
    if isinstance(schema_or_value, dict):
        value_type = schema_or_value.get("type")
        if value_type == "torch.Tensor":
            return _random_tensor_from_schema(schema_or_value)
        if value_type == "numpy.ndarray(object)":
            items = schema_or_value.get("items", [])
            restored_items = [_restore_value_from_schema(v) for v in items]
            arr = np.empty(len(restored_items), dtype=object)
            arr[:] = restored_items
            return arr
        if value_type == "numpy.ndarray":
            shape = tuple(schema_or_value["shape"])
            dtype = np.dtype(schema_or_value["dtype"])
            if np.issubdtype(dtype, np.floating):
                return np.random.randn(*shape).astype(dtype)
            if np.issubdtype(dtype, np.integer):
                return np.random.randint(0, 16, size=shape, dtype=dtype)
            return np.zeros(shape, dtype=dtype)
        if value_type == "tuple":
            return tuple(_restore_value_from_schema(v) for v in schema_or_value.get("items", []))
        return {k: _restore_value_from_schema(v) for k, v in schema_or_value.items()}
    if isinstance(schema_or_value, list):
        return [_restore_value_from_schema(v) for v in schema_or_value]
    return schema_or_value


def _load_rollout_dataproto(rollout_output_path: Path) -> tuple[DataProto, dict]:
    with rollout_output_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    batch_schema = payload.get("batch") or {}
    tensors = {}
    for key, value in batch_schema.items():
        if isinstance(value, dict) and value.get("type") == "torch.Tensor":
            tensors[key] = _random_tensor_from_schema(value)

    non_tensor_schema = payload.get("non_tensor_batch") or {}
    non_tensors = {}
    for key, value in non_tensor_schema.items():
        restored = _restore_value_from_schema(value)
        if isinstance(restored, np.ndarray):
            non_tensors[key] = restored
        elif isinstance(restored, list):
            arr = np.empty(len(restored), dtype=object)
            arr[:] = restored
            non_tensors[key] = arr

    meta_info = _restore_value_from_schema(payload.get("meta_info", {}))

    data = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)
    return data, payload


def _candidate_response_images(payload: dict, output_dir: Path) -> list[Path]:
    candidates = []
    for raw in payload.get("images", []):
        p = Path(raw)
        if p.exists():
            candidates.append(p)

    candidates.extend(sorted(output_dir.glob("response_*.png")))
    unique = []
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return unique


def _load_image_tensor(path: Path, height: int, width: int, channels: int, dtype: torch.dtype) -> torch.Tensor:
    with Image.open(path) as img:
        if channels == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        img = img.resize((width, height), resample=Image.BILINEAR)
        tensor = torchvision.transforms.functional.pil_to_tensor(img).float() / 255.0

    if channels == 1 and tensor.ndim == 3 and tensor.shape[0] != 1:
        tensor = tensor.mean(dim=0, keepdim=True)
    if channels == 3 and tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
    if channels == 3 and tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)

    return tensor.to(dtype=dtype)


def _inject_response_images(data: DataProto, image_paths: list[Path]):
    assert "responses" in data.batch.keys(), "rollout schema must contain batch['responses']"
    responses = torch.as_tensor(data.batch["responses"]).detach().clone()
    assert responses.ndim == 4, f"expected responses shape [B,C,H,W], got {tuple(responses.shape)}"
    bsz, channels, height, width = responses.shape
    assert len(image_paths) > 0, "response image list must be non-empty"

    for i in range(bsz):
        img_path = image_paths[i % len(image_paths)]
        responses[i] = _load_image_tensor(
            path=img_path,
            height=height,
            width=width,
            channels=channels,
            dtype=responses.dtype,
        )

    data.batch["responses"] = responses


class _NoOpTimer:
    @contextmanager
    def track(self):
        yield


class _FakeScheduler:
    def __init__(self, reward_worker, domain: str, meta_info: dict):
        self.reward_timer = {domain: _NoOpTimer()}
        self.reward_worker_iters = {domain: itertools.cycle([reward_worker])}
        self.meta_info = meta_info
        self.pipeline_config = SimpleNamespace(
            prompt_length=128,
            is_num_return_sequences_expand=True,
        )
        self.is_val = False
        self.sequence_length = 128


async def _run_context_compute_rewards(req: DataProto, domain: str, reward_worker):
    fake_scheduler = _FakeScheduler(reward_worker=reward_worker, domain=domain, meta_info=req.meta_info)
    context = RolloutContext(scheduler=fake_scheduler, prompt_id=0, meta_info=req.meta_info)
    context._in_do_generate_and_reward = True
    return await context.compute_rewards(req=req, domain=domain)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="requires at least one GPU")
def test_qwen_vl_judge_reward_compute_rewards_shape_from_rollout_output():
    pytest.importorskip("vllm")

    if not _REWARD_MODEL_PATH.exists():
        pytest.skip(f"reward model path not found: {_REWARD_MODEL_PATH}")
    if not _ROLLOUT_OUTPUT_PATH.exists():
        pytest.skip(f"rollout output file not found: {_ROLLOUT_OUTPUT_PATH}")

    req, payload = _load_rollout_dataproto(_ROLLOUT_OUTPUT_PATH)

    image_paths = _candidate_response_images(payload=payload, output_dir=_ROLLOUT_OUTPUT_PATH.parent)
    if not image_paths:
        pytest.skip("no response_*.png found under ./output and no valid payload images in rollout_output.json")

    _inject_response_images(req, image_paths=image_paths)

    if "domain" not in req.non_tensor_batch:
        pytest.skip("rollout_output.json missing non_tensor_batch['domain']")
    if "id" not in req.non_tensor_batch or "ground_truth" not in req.non_tensor_batch:
        pytest.skip("rollout_output.json must contain non_tensor_batch['id'] and ['ground_truth']")

    domain = str(req.non_tensor_batch["domain"][0])
    assert domain, "domain should not be empty"

    started_ray = False
    if not ray.is_initialized():
        ray.init(namespace=RAY_NAMESPACE, ignore_reinit_error=True, log_to_driver=True)
        started_ray = True

    reward_cluster = None
    resource_manager = None
    try:
        worker_config = WorkerConfig(
            name="reward_worker",
            worker_cls="roll.pipeline.diffusion.rewards.qwen_vl_judge_reward_worker.QwenVLJudgeRewardWorker",
            model_args=ModelArguments(
                model_name_or_path=str(_REWARD_MODEL_PATH),
            ),
            generating_args=GeneratingArguments(
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=64,
                num_return_sequences=1,
            ),
            strategy_args=StrategyArguments(
                strategy_name="vllm",
                strategy_config={
                    "tensor_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "enforce_eager": True,
                    "load_format": "auto",
                },
            ),
            device_mapping="[0]",
        )
        setattr(worker_config, "judge_model_type", "inference")

        resource_manager = ResourceManager(num_gpus_per_node=1, num_nodes=1)
        reward_cluster = Cluster(
            name="test_qwen_vl_judge_reward_worker",
            worker_cls=QwenVLJudgeRewardWorker,
            resource_manager=resource_manager,
            worker_config=worker_config,
        )

        pipeline_config = SimpleNamespace(
            seed=42,
            resume_from_checkpoint=False,
            is_actor_infer_colocated=False,
        )
        reward_cluster.initialize(pipeline_config=pipeline_config, blocking=True)
        reward_worker = reward_cluster.workers[0]

        rewards = asyncio.run(_run_context_compute_rewards(req=req, domain=domain, reward_worker=reward_worker))

        bsz = req.batch.batch_size[0]
        rollout_steps = req.batch["all_timesteps"].shape[1]

        assert isinstance(rewards, DataProto), "compute_rewards should return DataProto"
        assert "token_level_rewards" in rewards.batch.keys()
        assert "response_level_rewards" in rewards.batch.keys()
        assert "scores" in rewards.batch.keys()

        assert list(rewards.batch["token_level_rewards"].shape) == [bsz, rollout_steps]
        assert list(rewards.batch["response_level_rewards"].shape) == [bsz]
        assert list(rewards.batch["scores"].shape) == [bsz]

        assert torch.isfinite(rewards.batch["token_level_rewards"].float()).all()
        assert torch.isfinite(rewards.batch["response_level_rewards"].float()).all()
        assert torch.isfinite(rewards.batch["scores"].float()).all()

        metrics = rewards.meta_info.get("metrics", {})
        assert "flowgrpo/reward_mean" in metrics
        assert np.isfinite(float(metrics["flowgrpo/reward_mean"]))

        reward_values = rewards.batch["response_level_rewards"].detach().cpu().float().tolist()
        ground_truths = req.non_tensor_batch["ground_truth"].tolist()
        if "identified_words" not in rewards.non_tensor_batch:
            logger.error("compute_rewards output missing required non_tensor_batch['identified_words']")
            raise RuntimeError("missing non_tensor_batch['identified_words'] in rewards")
        identified_words = rewards.non_tensor_batch["identified_words"].tolist()
        if len(identified_words) != len(ground_truths):
            logger.error(
                "identified_words length mismatch: identified_words=%s ground_truths=%s",
                len(identified_words),
                len(ground_truths),
            )
            raise RuntimeError("identified_words length mismatch")
        reward_payload = {
            str(gt): {
                "reward": float(rv),
                "identified word": str(iw),
            }
            for gt, rv, iw in zip(ground_truths, reward_values, identified_words)
        }
        reward_output_path = _ROLLOUT_OUTPUT_PATH.parent / "reward.json"
        with reward_output_path.open("w", encoding="utf-8") as f:
            json.dump(reward_payload, f, ensure_ascii=False, indent=2)
    finally:
        if reward_cluster is not None:
            for worker in reward_cluster.workers:
                ray.kill(worker)
        if resource_manager is not None:
            resource_manager.destroy_placement_group()
        if started_ray and ray.is_initialized():
            ray.shutdown()
