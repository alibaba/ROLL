import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import ray
import torch
from PIL import Image

from roll.configs import ModelArguments
from roll.configs.base_config import RouterArguments
from roll.configs.data_args import DataArguments
from roll.configs.worker_config import StrategyArguments, WorkerConfig
from roll.distributed.executor.cluster import Cluster
from roll.distributed.executor.model_update_group import ModelUpdateGroup
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.generate_scheduler import DynamicSamplingScheduler
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.datasets.collator import DataCollatorForDiffusion
from roll.pipeline.base_worker import InferWorker
from roll.pipeline.rlvr.rlvr_config import RewardConfig, RewardFilterConfig
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.cuda_ipc_utils import MultiprocessingSerializer
from roll.utils.functionals import reduce_metrics
from roll.utils.logging import get_logger
from roll.utils.metrics.metrics_manager import MetricsManager
from roll.utils.send_recv_utils import named_tensors_from_bucket, serialize_named_weights


logger = get_logger()

VLLM_OMNI_TEST_MODEL_PATH = "/tmp/models/Qwen/Qwen-Image"
CUSTOM_PIPELINE_QUALNAME = "roll.pipeline.diffusion.models.qwen_image.vllm_omni_qwen_image_adapter.QwenImagePipelineWithLogProb"
_TOKENIZER_MAX_LENGTH = 1024


def _compute_encode_start_idx(tokenizer, system_prompt):
    """Compute the number of prefix tokens (system + user header) before user content."""
    from roll.datasets.chat_template import get_chat_template
    template_func = get_chat_template("native", tokenizer)
    text_a = template_func([{"role": "system", "content": system_prompt}, {"role": "user", "content": "A"}])
    text_b = template_func([{"role": "system", "content": system_prompt}, {"role": "user", "content": "B"}])
    ids_a = tokenizer([text_a])["input_ids"][0]
    ids_b = tokenizer([text_b])["input_ids"][0]
    min_len = min(len(ids_a), len(ids_b))
    for i in range(min_len):
        if ids_a[i] != ids_b[i]:
            return i
    return min_len


_SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:"
)
# Will be set dynamically in tests after tokenizer is loaded
_ENCODE_START_IDX = 34


def _build_verl_parity_sampling_params(encode_start_idx):
    return {
        "guidance_scale": 4.0,
        "height": 512,
        "width": 512,
        "num_inference_steps": 10,
        "max_sequence_length": _TOKENIZER_MAX_LENGTH + encode_start_idx,
        "max_new_tokens": 1,
        "extra_args": {
            "logprobs": True,
            "noise_level": 1.0,
            "sde_type": "sde",
            "sde_window_size": 2,
            "sde_window_range": [0, 5],
        },
    }

_REQUIRED_OUTPUT_KEYS = {
    "all_latents",
    "rollout_log_probs",
    "all_timesteps",
    "prompt_embeds",
    "prompt_embeds_mask",
    "negative_prompt_embeds",
    "negative_prompt_embeds_mask",
    "token_level_rewards",
    "scores",
}
_REAL_OCR_DATASET_PATH = "data/ocr/train.json"


def _get_storage_path() -> Path:
    return Path(os.getenv("STORAGE_PATH", "./output"))


def _to_schema_metadata(value):
    if isinstance(value, torch.Tensor):
        t = value.detach().cpu()
        return {
            "type": "torch.Tensor",
            "shape": list(t.shape),
            "dtype": str(t.dtype),
        }
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return {
                "type": "numpy.ndarray(object)",
                "shape": list(value.shape),
                "items": [_to_schema_metadata(v) for v in value.tolist()],
            }
        return {
            "type": "numpy.ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {k: _to_schema_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_schema_metadata(v) for v in value]
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_to_schema_metadata(v) for v in value]}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": type(value).__name__, "repr": repr(value)}


def _save_generate_output_schema(generate_output: DataProto):
    storage_path = _get_storage_path()
    storage_path.mkdir(parents=True, exist_ok=True)

    image_paths = []
    responses_tensor = None
    if generate_output.batch is not None and "responses" in generate_output.batch.keys():
        responses_tensor = torch.as_tensor(generate_output.batch["responses"]).detach().cpu()
    elif "responses" in generate_output.meta_info:
        responses = generate_output.meta_info["responses"]
        response_items = responses if isinstance(responses, list) else [responses]
        response_tensors = [torch.as_tensor(resp).detach().cpu() for resp in response_items]
        if len(response_tensors) > 0:
            responses_tensor = torch.stack(response_tensors, dim=0)

    if responses_tensor is not None:
        if responses_tensor.ndim == 3:
            responses_tensor = responses_tensor.unsqueeze(0)
        image_idx = 0
        for i in range(responses_tensor.shape[0]):
            img = responses_tensor[i]
            if img.ndim != 3:
                continue
            if img.shape[0] in (1, 3):
                img = img.permute(1, 2, 0)
            img = img.float()
            if img.numel() == 0:
                continue
            min_v = float(img.min())
            max_v = float(img.max())
            if max_v <= 1.0 and min_v >= 0.0:
                img = img * 255.0
            elif max_v > min_v:
                img = (img - min_v) / (max_v - min_v) * 255.0
            else:
                img = torch.zeros_like(img)
            img_np = img.clamp(0, 255).to(torch.uint8).numpy()
            if img_np.ndim == 3 and img_np.shape[-1] == 1:
                img_np = img_np[..., 0]
            if img_np.ndim == 3 and img_np.shape[-1] > 3:
                img_np = img_np[..., :3]
            img_path = storage_path / f"response_{image_idx}.png"
            Image.fromarray(img_np).save(img_path)
            image_paths.append(str(img_path))
            image_idx += 1

    payload = {
        "meta_info": _to_schema_metadata(generate_output.meta_info),
        "batch": _to_schema_metadata(dict(generate_output.batch.items()) if generate_output.batch is not None else None),
        "non_tensor_batch": _to_schema_metadata(generate_output.non_tensor_batch),
        "images": image_paths,
    }
    output_json = storage_path / "rollout_output.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("[test] saved generate_output schema to %s", output_json)


@dataclass
class _DummyRolloutMockConfig:
    enable: bool = False
    mode: str = "dump"
    dump_dir: str = "/tmp/rollout_mock_dumps"


@dataclass
class _MinimalPipelineConfig:
    seed: int = 42
    sequence_length: int = 128
    val_sequence_length: int = 128
    prompt_length: int = _TOKENIZER_MAX_LENGTH + _ENCODE_START_IDX
    async_generation_ratio: float = 0.0
    max_running_requests: int = 2
    is_num_return_sequences_expand: bool = True
    is_use_additional_prompts: bool = False
    max_additional_running_prompts: int = 0
    num_gpus_per_node: int = 1
    user_defined_rollout_loop_cls: str = "roll.pipeline.diffusion.rollout_loop.DiffusionRolloutLoop"
    generate_opt_level: int = 1
    rpc_timeout: int = 3600
    global_template: str | None = None
    validation: object | None = None
    rollout_mock: _DummyRolloutMockConfig = field(default_factory=_DummyRolloutMockConfig)


@ray.remote
class DummyRewardWorker:
    async def compute_rewards(self, data: DataProto):
        bsz = data.batch.batch_size[0]
        device = data.batch.device
        rollout_steps = 1
        if "all_timesteps" in data.batch.keys():
            rollout_steps = data.batch["all_timesteps"].shape[1]
        rewards = torch.zeros((bsz,), dtype=torch.float32, device=device)
        token_level_rewards = torch.zeros((bsz, rollout_steps), dtype=torch.float32, device=device)
        scores = torch.zeros((bsz,), dtype=torch.float32, device=device)
        return DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": rewards,
                "scores": scores,
            },
            meta_info={"metrics": {"reward/dummy": 0.0}},
        )


class _DummyCluster:
    def __init__(self, workers):
        self.workers = workers


try:
    from roll.third_party.vllm_omni.worker import VllmOmniColocateWorkerExtension as _VllmOmniWorkerExtBase
except Exception:
    _VllmOmniWorkerExtBase = object


class VllmOmniDebugWeightUpdateExtension(_VllmOmniWorkerExtBase):
    """Debug extension to make weight-update verification deterministic in tests."""

    _DEBUG_KEYWORDS = ("model", "module", "pipeline", "transform", "runner", "engine", "worker", "diffusion")

    def _iter_candidate_objects(self, root, max_depth=3):
        queue = [("self", root, 0)]
        seen = set()
        while queue:
            path, obj, depth = queue.pop(0)
            oid = id(obj)
            if oid in seen:
                continue
            seen.add(oid)
            yield path, obj
            if depth >= max_depth:
                continue
            try:
                obj_vars = vars(obj)
            except Exception:
                obj_vars = {}
            for attr, child in obj_vars.items():
                if child is None:
                    continue
                if not any(k in attr.lower() for k in self._DEBUG_KEYWORDS):
                    continue
                if isinstance(child, (str, bytes, int, float, bool)):
                    continue
                queue.append((f"{path}.{attr}", child, depth + 1))

    def _find_best_param_source(self):
        best = None
        for path, obj in self._iter_candidate_objects(self):
            named = None
            if hasattr(obj, "named_parameters"):
                try:
                    named = list(obj.named_parameters())
                except Exception:
                    named = None
            if not named:
                continue
            if best is None or len(named) > len(best["named"]):
                best = {"path": path, "obj": obj, "named": named}
        if best is None:
            raise RuntimeError(
                "[weight-update-debug] cannot locate any object exposing named_parameters() under vllm_omni worker extension"
            )
        return best

    def roll_debug_pick_weight_target(self, max_numel: int = 2048):
        src = self._find_best_param_source()
        logger.info(
            "[weight-update-debug] selected parameter source path=%s param_count=%s",
            src["path"],
            len(src["named"]),
        )
        chosen = None
        for name, tensor in src["named"]:
            if not torch.is_tensor(tensor):
                continue
            if tensor.numel() == 0 or tensor.numel() > max_numel:
                continue
            if not tensor.dtype.is_floating_point:
                continue
            chosen = (name, tensor.detach().clone())
            break
        if chosen is None:
            raise RuntimeError(
                "[weight-update-debug] no suitable floating parameter found; "
                f"source_path={src['path']} param_count={len(src['named'])} max_numel={max_numel}"
            )
        name, tensor = chosen
        self._roll_debug_param_source = src["obj"]
        self._roll_debug_param_source_path = src["path"]
        self._roll_debug_target_name = name
        logger.info(
            "[weight-update-debug] target name=%s shape=%s dtype=%s",
            name,
            list(tensor.shape),
            str(tensor.dtype),
        )
        return {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "before": tensor.cpu(),
            "source_path": src["path"],
        }

    def roll_debug_get_weight(self, name: str):
        source_obj = getattr(self, "_roll_debug_param_source", None)
        source_path = getattr(self, "_roll_debug_param_source_path", "unknown")
        if source_obj is None:
            raise RuntimeError("[weight-update-debug] no cached parameter source, call roll_debug_pick_weight_target first")
        named = dict(source_obj.named_parameters())
        if name not in named:
            raise KeyError(
                f"[weight-update-debug] parameter {name} not found in cached source {source_path}; "
                f"available_count={len(named)}"
            )
        return named[name].detach().cpu().clone()

    def update_parameter_in_bucket(self, serialized_named_tensors, is_lora: bool = False):
        result = super().update_parameter_in_bucket(serialized_named_tensors, is_lora=is_lora)
        bucket_with_meta = MultiprocessingSerializer.deserialize(serialized_named_tensors[self.rank])
        named_params = list(named_tensors_from_bucket(**bucket_with_meta))
        self._roll_debug_last_received = {name: tensor.detach().cpu().clone() for name, tensor in named_params}
        logger.info(
            "[weight-update-debug] rank=%s received %s tensor(s): %s",
            self.rank,
            len(named_params),
            [name for name, _ in named_params],
        )
        return result

    def roll_debug_get_last_received(self, name: str):
        received = getattr(self, "_roll_debug_last_received", None)
        if received is None:
            raise RuntimeError("[weight-update-debug] no received tensor cache found")
        if name not in received:
            raise KeyError(f"[weight-update-debug] tensor {name} not found in received cache: {list(received.keys())}")
        return received[name]


class DebugInferWorkerForModelUpdate(InferWorker):
    async def _debug_collective_rpc(self, method: str, *args, **kwargs):
        assert hasattr(self, "strategy"), "[weight-update-debug] infer worker has no strategy"
        model = getattr(self.strategy, "model", None)
        assert model is not None, "[weight-update-debug] infer strategy model is None"
        engine = getattr(model, "_engine", None)
        assert engine is not None, "[weight-update-debug] infer strategy model has no _engine"
        assert hasattr(engine, "collective_rpc"), "[weight-update-debug] _engine has no collective_rpc"
        logger.info("[weight-update-debug] infer collective_rpc method=%s", method)
        return await engine.collective_rpc(method=method, args=args, kwargs=kwargs)

    async def debug_pick_weight_target(self):
        return await self._debug_collective_rpc("roll_debug_pick_weight_target")

    async def debug_get_weight(self, name: str):
        return await self._debug_collective_rpc("roll_debug_get_weight", name)

    async def debug_get_last_received(self, name: str):
        return await self._debug_collective_rpc("roll_debug_get_last_received", name)

    async def debug_strategy_is_model_in_gpu(self):
        assert hasattr(self, "strategy"), "[offload-test] infer worker has no strategy"
        return bool(getattr(self.strategy, "is_model_in_gpu", False))

    async def debug_set_actor_infer_colocated(self, value: bool):
        assert self.pipeline_config is not None, "[offload-test] pipeline_config is None"
        setattr(self.pipeline_config, "is_actor_infer_colocated", bool(value))
        return bool(getattr(self.pipeline_config, "is_actor_infer_colocated", False))

    async def debug_direct_collective_sleep(self, level: int = 1):
        return await self._debug_collective_rpc("sleep", level)


class DummyActorTrainModelUpdateWorker(Worker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config, *args, **kwargs):
        super().initialize(pipeline_config=pipeline_config, *args, **kwargs)
        self._infer_workers = None
        self._model_update_name = None
        self._dummy_param_name = None
        self._dummy_tensor = None
        logger.info("[dummy-train] initialized rank=%s", self.rank)

    def setup_model_update(self, infer_cluster, model_update_name: str):
        self._infer_workers = infer_cluster.workers
        self._model_update_name = model_update_name
        logger.info(
            "[dummy-train] setup_model_update done model_update_name=%s infer_worker_count=%s",
            model_update_name,
            len(self._infer_workers),
        )

    def set_dummy_update_payload(self, parameter_name: str, tensor: torch.Tensor):
        self._dummy_param_name = parameter_name
        self._dummy_tensor = tensor.detach().cpu().clone()
        logger.info(
            "[dummy-train] payload configured param=%s shape=%s dtype=%s",
            parameter_name,
            list(self._dummy_tensor.shape),
            str(self._dummy_tensor.dtype),
        )

    def start_model_update(self, model_update_name: str):
        assert self._infer_workers is not None, "[dummy-train] infer workers not configured, setup_model_update missing"
        assert self._model_update_name == model_update_name, (
            f"[dummy-train] unexpected model_update_name={model_update_name}, expected={self._model_update_name}"
        )
        assert self._dummy_param_name is not None, "[dummy-train] dummy parameter name is not configured"
        assert self._dummy_tensor is not None, "[dummy-train] dummy tensor is not configured"

        device_tensor = self._dummy_tensor.to(torch.cuda.current_device())
        serialized = serialize_named_weights(
            named_weights=[(self._dummy_param_name, device_tensor)],
            infer_strategy="vllm_omni",
        )
        logger.info(
            "[dummy-train] sending tensor param=%s shape=%s dtype=%s to infer workers=%s",
            self._dummy_param_name,
            list(device_tensor.shape),
            str(device_tensor.dtype),
            len(self._infer_workers),
        )
        refs = [worker.update_parameter_in_bucket.remote([serialized], is_lora=False) for worker in self._infer_workers]
        ray.get(refs)
        logger.info("[dummy-train] model_update send complete")
        return DataProto(meta_info={"metrics": {"dummy/model_update_sent": 1.0}})


class DummyModelUpdatePipeline:
    def __init__(self, pipeline_config):
        self.pipeline_config = pipeline_config
        self.model_update_groups = []

    def set_model_update_pair(self, src_cluster, tgt_cluster, frequency=1):
        logger.info(
            "[dummy-pipeline] set_model_update_pair src=%s tgt=%s frequency=%s",
            src_cluster.cluster_name,
            tgt_cluster.cluster_name,
            frequency,
        )
        self.model_update_groups.append(
            ModelUpdateGroup(
                src_cluster=src_cluster,
                tgt_cluster=tgt_cluster,
                frequency=frequency,
                pipeline_config=self.pipeline_config,
            )
        )

    def model_update(self, global_step):
        logger.info("[dummy-pipeline] model_update begin global_step=%s groups=%s", global_step, len(self.model_update_groups))
        metrics = {}
        for group_idx, model_update_group in enumerate(self.model_update_groups):
            logger.info("[dummy-pipeline] triggering group_idx=%s", group_idx)
            metrics.update(model_update_group.model_update(global_step))
            model_update_group.tgt_cluster.process_weights_after_loading()
        logger.info("[dummy-pipeline] model_update done metrics=%s", metrics)
        return metrics


def _prepare_dataset_by_qwen_image_rollout_only_build_dataset(tokenizer):
    try:
        from roll.pipeline.diffusion.flow_grpo.qwen_image_rollout_only_pipeline import QwenImageRolloutOnlyPipeline
    except (ImportError, ModuleNotFoundError):
        pytest.skip("QwenImageRolloutOnlyPipeline has been removed; use DiffusionPipeline data adapter for dataset building.")

    data_args = DataArguments(
        template="native",
        file_name=_REAL_OCR_DATASET_PATH,
        messages="prompt",
        preprocessing_num_workers=1,
    )
    pipeline_config = _MinimalPipelineConfig()
    pipeline_config.validation = SimpleNamespace(data_args=data_args)
    pipeline_config.rewards = {
        "domain_single": RewardConfig(query_filter_config=RewardFilterConfig(type="no_filter")),
    }

    # Reuse the exact rollout-only dataset builder path without bootstrapping full pipeline runtime.
    dataset_builder = object.__new__(QwenImageRolloutOnlyPipeline)
    dataset_builder.pipeline_config = pipeline_config
    dataset_builder.tokenizer = tokenizer
    dataset = QwenImageRolloutOnlyPipeline._build_dataset(dataset_builder)
    take_n = min(5, len(dataset))
    dataset = dataset.select(range(take_n))

    assert len(dataset) > 0, "dataset should not be empty after qwen_image rollout-only _build_dataset"
    for required_key in [
        "domain",
        "id",
        "ground_truth",
        "prompt_ids",
        "prompt_mask",
        "negative_prompt_ids",
        "negative_prompt_mask",
        "encode_start_idx",
    ]:
        assert required_key in dataset.column_names, f"missing required dataset key: {required_key}"

    return dataset, pipeline_config


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="requires at least one GPU")
def test_vllm_omni_scheduler_dataset_preprocess_and_generate_output_contract():
    logger.info("[test] start vllm_omni scheduler e2e test")
    pytest.importorskip("vllm_omni")

    if not os.path.exists(VLLM_OMNI_TEST_MODEL_PATH):
        pytest.skip(f"model path not found: {VLLM_OMNI_TEST_MODEL_PATH}")
    if not os.path.exists(_REAL_OCR_DATASET_PATH):
        pytest.skip(f"dataset path not found: {_REAL_OCR_DATASET_PATH}")

    from transformers import AutoTokenizer

    logger.info("[test] initializing ray if needed")
    started_ray = False
    resource_manager = None
    if not ray.is_initialized():
        ray.init(namespace=RAY_NAMESPACE, ignore_reinit_error=True, log_to_driver=True)
        started_ray = True

    try:
        tokenizer_path = os.path.join(VLLM_OMNI_TEST_MODEL_PATH, "tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)

        # Compute encode_start_idx dynamically from the tokenizer
        global _ENCODE_START_IDX
        _ENCODE_START_IDX = _compute_encode_start_idx(tokenizer, _SYSTEM_PROMPT)
        logger.info("[test] computed encode_start_idx=%d", _ENCODE_START_IDX)

        dataset, pipeline_config = _prepare_dataset_by_qwen_image_rollout_only_build_dataset(tokenizer=tokenizer)

        logger.info("[test] creating actor_infer cluster with vllm_omni strategy")
        worker_config = WorkerConfig(
            name="actor_infer",
            worker_cls="roll.pipeline.base_worker.InferWorker",
            model_args=ModelArguments(
                model_name_or_path=VLLM_OMNI_TEST_MODEL_PATH,
                dtype="bf16",
            ),
            strategy_args=StrategyArguments(
                strategy_name="vllm_omni",
                strategy_config={
                    "tensor_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "trust_remote_code": True,
                    "compilation_config": {
                        "cudagraph_mode": "FULL_AND_PIECEWISE",
                    },
                    "vllm_omni": {
                        "custom_pipeline": CUSTOM_PIPELINE_QUALNAME,
                    },
                },
            ),
            device_mapping="[0]",
        )

        resource_manager = ResourceManager(num_gpus_per_node=1, num_nodes=1)
        actor_cluster = Cluster(
            name="test_vllm_omni_scheduler_infer",
            worker_cls=InferWorker,
            resource_manager=resource_manager,
            worker_config=worker_config,
        )

        pipeline_boot = SimpleNamespace(seed=42, resume_from_checkpoint=False, is_actor_infer_colocated=False)
        actor_cluster.initialize(pipeline_config=pipeline_boot, blocking=True)

        pipeline_config.router_args = RouterArguments(router_name="PromptAffinityRouter", router_config={}, max_running_requests=2)
        domain_single = dataset.filter(lambda x: x["domain"] == "domain_single")
        assert len(domain_single) > 0, "domain_single dataset is empty"

        reward_clusters = {
            "domain_single": _DummyCluster([DummyRewardWorker.remote()]),
        }

        logger.info("[test] creating 1 generate scheduler")
        generate_schedulers = {
            "domain_single": ray.remote(DynamicSamplingScheduler).remote(
                pipeline_config=pipeline_config,
                actor_cluster=actor_cluster,
                reward_clusters={"domain_single": reward_clusters["domain_single"]},
                dataset=domain_single,
                collect_fn_cls=DataCollatorForDiffusion,
                collect_fn_kwargs={"max_length": pipeline_config.prompt_length},
            ),
        }

        logger.info("[test] initialize schedulers")
        ray.get([scheduler.initialize.remote() for scheduler in generate_schedulers.values()])

        logger.info("[test] validating router_client strategy_name")
        # for domain, scheduler in generate_schedulers.items():
        #     strategy_name = ray.get(scheduler.router_client.strategy_name)
        #     logger.info("[test] scheduler %s strategy_name=%s", domain, strategy_name)
        #     assert strategy_name == "vllm_omni", f"{domain} expected vllm_omni, got {strategy_name}"

        metrics_mgr = MetricsManager()
        batch = DataProto(
            meta_info={
                "global_step": 0,
                "collect_unfinished": False,
                "is_training": True,
                "generation_config": {
                    "num_return_sequences": 1,
                    **_build_verl_parity_sampling_params(_ENCODE_START_IDX),
                },
            }
        )

        logger.info("[test] launch one get_batch for single domain")
        scheduler_refs = {}
        domain_batch_size = {
            "domain_single": len(domain_single),
        }
        logger.info(
            "[test] domain batch size: domain_single=%s total=%s",
            domain_batch_size["domain_single"],
            domain_batch_size["domain_single"],
        )
        for domain, scheduler in generate_schedulers.items():
            scheduler_refs[domain] = scheduler.get_batch.remote(
                data=batch,
                global_step=0,
                batch_size=domain_batch_size[domain],
            )

        domain_batches = {}
        for domain, scheduler_ref in scheduler_refs.items():
            logger.info("[test] waiting get_batch result domain=%s", domain)
            domain_batch = ray.get(scheduler_ref, timeout=pipeline_config.rpc_timeout)
            metrics_mgr.add_domain_metrics(domain, reduce_metrics(domain_batch.meta_info.pop("metrics", {})))
            domain_batches[domain] = domain_batch

            missing_keys = _REQUIRED_OUTPUT_KEYS - set(domain_batch.batch.keys())
            assert not missing_keys, f"domain {domain} missing output keys: {missing_keys}"

            for key in _REQUIRED_OUTPUT_KEYS | {"responses"}:
                assert key in domain_batch.batch.keys(), f"{key} should be in batch"
                assert domain_batch.batch[key].shape[0] == len(domain_batch), (
                    f"{key} batch size mismatch: {domain_batch.batch[key].shape[0]} != {len(domain_batch)}"
                )

            # for key in _ROUTER_OUTPUT_KEYS:
            #     assert key not in domain_batch.meta_info, f"{key} should not be in meta_info after router postprocess"
            #     if key in domain_batch.batch.keys():
            #         assert domain_batch.batch[key].shape[0] == len(domain_batch), (
            #             f"{key} batch size mismatch: {domain_batch.batch[key].shape[0]} != {len(domain_batch)}"
            #         )
            #     elif key in domain_batch.non_tensor_batch:
            #         assert domain_batch.non_tensor_batch[key].shape[0] == len(domain_batch), (
            #             f"{key} non_tensor_batch size mismatch: {domain_batch.non_tensor_batch[key].shape[0]} != {len(domain_batch)}"
            #         )

            rollout_log_probs = domain_batch.batch["rollout_log_probs"]
            all_timesteps = torch.as_tensor(domain_batch.batch["all_timesteps"])
            all_latents = torch.as_tensor(domain_batch.batch["all_latents"])
            assert all_timesteps.shape[1] == rollout_log_probs.shape[1], "all_timesteps K mismatch"
            assert all_latents.shape[1] == rollout_log_probs.shape[1] + 1, "all_latents K+1 mismatch"
            

        logger.info("[test] concat domain batches")
        generate_output = DataProto.concat([domain_batch for domain_batch in domain_batches.values()])
        expected_total = len(dataset)
        assert len(generate_output) == expected_total, (
            f"expected all prompts processed once, got {len(generate_output)} != {expected_total}"
        )

        missing_keys = _REQUIRED_OUTPUT_KEYS - set(generate_output.batch.keys())
        assert not missing_keys, f"generate_output missing output keys: {missing_keys}"

        assert "id" in generate_output.non_tensor_batch, "generate_output should keep non_tensor_batch['id']"
        assert "ground_truth" in generate_output.non_tensor_batch, "generate_output should keep non_tensor_batch['ground_truth']"
        assert "domain" in generate_output.non_tensor_batch, "generate_output should keep non_tensor_batch['domain']"

        expected_pairs = sorted((row["id"], row["ground_truth"]) for row in domain_single)
        actual_pairs = sorted(
            zip(
                generate_output.non_tensor_batch["id"].tolist(),
                generate_output.non_tensor_batch["ground_truth"].tolist(),
            )
        )
        assert actual_pairs == expected_pairs, "id-ground_truth pairs in generate_output must match _build_dataset output exactly"
        assert set(generate_output.non_tensor_batch["domain"].tolist()) == {"domain_single"}

        logger.info("[test] success: generate_output contains required vllm_omni keys")
        _save_generate_output_schema(generate_output=generate_output)

        logger.info("[test] shutdown schedulers")
        ray.get([scheduler.shutdown.remote() for scheduler in generate_schedulers.values()])

    finally:
        logger.info("[test] cleanup resources")
        if resource_manager is not None:
            resource_manager.destroy_placement_group()
        if started_ray:
            ray.shutdown()


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="requires at least one GPU")
def test_vllm_omni_model_update_applies_dummy_weight_tensor():
    logger.info("[weight-update-test] start")
    pytest.importorskip("vllm_omni")

    if not os.path.exists(VLLM_OMNI_TEST_MODEL_PATH):
        pytest.skip(f"model path not found: {VLLM_OMNI_TEST_MODEL_PATH}")

    def _unwrap_collective_result(value, stage: str):
        cur = value
        depth = 0
        while isinstance(cur, list):
            assert len(cur) > 0, f"[weight-update-test] {stage} returned empty list at depth={depth}"
            cur = cur[0]
            depth += 1
            if depth > 8:
                raise RuntimeError(f"[weight-update-test] {stage} collective result nesting is too deep: {type(value)}")
        logger.info("[weight-update-test] %s unwrapped at depth=%s final_type=%s", stage, depth, type(cur).__name__)
        return cur

    started_ray = False
    resource_manager = None
    actor_infer = None
    actor_train = None

    logger.info("[weight-update-test] init ray if needed")
    if not ray.is_initialized():
        ray.init(namespace=RAY_NAMESPACE, ignore_reinit_error=True, log_to_driver=True)
        started_ray = True

    try:
        pipeline_boot = SimpleNamespace(seed=42, resume_from_checkpoint=False, is_actor_infer_colocated=False)
        device_mapping = "[0]"

        logger.info("[weight-update-test] create actor_infer cluster with debug extension")
        infer_worker_config = WorkerConfig(
            name="actor_infer",
            worker_cls="tests.diffusion.flow_grpo.test_vllm_omni_scheduler_e2e.DebugInferWorkerForModelUpdate",
            model_args=ModelArguments(
                model_name_or_path=VLLM_OMNI_TEST_MODEL_PATH,
                dtype="bf16",
            ),
            strategy_args=StrategyArguments(
                strategy_name="vllm_omni",
                strategy_config={
                    "tensor_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "trust_remote_code": True,
                    "compilation_config": {"cudagraph_mode": "FULL_AND_PIECEWISE"},
                    "worker_extension_cls": (
                        "tests.diffusion.flow_grpo.test_vllm_omni_scheduler_e2e."
                        "VllmOmniDebugWeightUpdateExtension"
                    ),
                    "vllm_omni": {
                        "custom_pipeline": CUSTOM_PIPELINE_QUALNAME,
                    },
                },
            ),
            device_mapping=device_mapping,
        )

        resource_manager = ResourceManager(num_gpus_per_node=1, num_nodes=1)
        actor_infer = Cluster(
            name="test_vllm_omni_model_update_infer",
            worker_cls="tests.diffusion.flow_grpo.test_vllm_omni_scheduler_e2e.DebugInferWorkerForModelUpdate",
            resource_manager=resource_manager,
            worker_config=infer_worker_config,
        )
        actor_infer.initialize(pipeline_config=pipeline_boot, blocking=True)

        logger.info("[weight-update-test] create dummy actor_train cluster")
        train_worker_config = WorkerConfig(
            name="actor_train_dummy",
            worker_cls="tests.diffusion.flow_grpo.test_vllm_omni_scheduler_e2e.DummyActorTrainModelUpdateWorker",
            model_args=ModelArguments(model_name_or_path=None),
            strategy_args=StrategyArguments(strategy_name="mock_infer", strategy_config={}),
            model_update_frequency=1,
            device_mapping=device_mapping,
        )
        actor_train = Cluster(
            name="test_vllm_omni_model_update_train_dummy",
            worker_cls="tests.diffusion.flow_grpo.test_vllm_omni_scheduler_e2e.DummyActorTrainModelUpdateWorker",
            resource_manager=resource_manager,
            worker_config=train_worker_config,
        )
        actor_train.initialize(pipeline_config=pipeline_boot, blocking=True)

        logger.info("[weight-update-test] pick a real target parameter from vllm_omni worker")
        target_raw = ray.get(actor_infer.workers[0].debug_pick_weight_target.remote())
        target = _unwrap_collective_result(target_raw, "debug_pick_weight_target")
        assert isinstance(target, dict), f"[weight-update-test] target payload should be dict, got {type(target)}"
        target_name = target["name"]
        target_before = torch.as_tensor(target["before"]).detach().cpu()
        target_shape = list(target_before.shape)
        logger.info(
            "[weight-update-test] selected target name=%s shape=%s dtype=%s source=%s",
            target_name,
            target_shape,
            str(target_before.dtype),
            target.get("source_path"),
        )

        dummy_tensor = torch.full(target_shape, 0.03125, dtype=target_before.dtype).cpu()
        logger.info(
            "[weight-update-test] configure dummy payload param=%s shape=%s dtype=%s",
            target_name,
            target_shape,
            str(dummy_tensor.dtype),
        )
        ray.get(actor_train.workers[0].set_dummy_update_payload.remote(target_name, dummy_tensor))

        logger.info("[weight-update-test] trigger pipeline.set_model_update_pair + pipeline.model_update")
        dummy_pipeline_cfg = SimpleNamespace(actor_train=SimpleNamespace(model_update_frequency=1))
        dummy_pipeline = DummyModelUpdatePipeline(dummy_pipeline_cfg)
        dummy_pipeline.set_model_update_pair(
            src_cluster=actor_train,
            tgt_cluster=actor_infer,
            frequency=dummy_pipeline_cfg.actor_train.model_update_frequency,
        )
        update_metrics = dummy_pipeline.model_update(global_step=0)
        logger.info("[weight-update-test] model_update metrics=%s", update_metrics)
        assert update_metrics, "[weight-update-test] model_update returned empty metrics"

        logger.info("[weight-update-test] fetch infer-side received payload tensor")
        received_raw = ray.get(actor_infer.workers[0].debug_get_last_received.remote(target_name))
        received = torch.as_tensor(_unwrap_collective_result(received_raw, "debug_get_last_received")).detach().cpu()
        assert received.shape == dummy_tensor.shape, (
            f"[weight-update-test] received shape mismatch {received.shape} vs {dummy_tensor.shape}"
        )
        assert torch.allclose(received, dummy_tensor, atol=0.0, rtol=0.0), (
            "[weight-update-test] infer worker received tensor does not match dummy tensor exactly"
        )

        logger.info("[weight-update-test] fetch infer-side model parameter after update")
        after_raw = ray.get(actor_infer.workers[0].debug_get_weight.remote(target_name))
        target_after = torch.as_tensor(_unwrap_collective_result(after_raw, "debug_get_weight")).detach().cpu()
        assert target_after.shape == dummy_tensor.shape, (
            f"[weight-update-test] updated weight shape mismatch {target_after.shape} vs {dummy_tensor.shape}"
        )
        assert torch.allclose(target_after, dummy_tensor, atol=1e-5, rtol=1e-5), (
            "[weight-update-test] updated vllm_omni model weight does not equal dummy tensor"
        )
        logger.info("[weight-update-test] success: vllm_omni model weight equals dummy tensor")

    finally:
        logger.info("[weight-update-test] cleanup resources")
        if resource_manager is not None:
            resource_manager.destroy_placement_group()
        if started_ray:
            ray.shutdown()


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="requires at least one GPU")
def test_vllm_omni_offload_states_rpc_returns_and_updates_strategy_state():
    logger.info("[offload-test] start")
    pytest.importorskip("vllm_omni")

    if not os.path.exists(VLLM_OMNI_TEST_MODEL_PATH):
        pytest.skip(f"model path not found: {VLLM_OMNI_TEST_MODEL_PATH}")

    def _unwrap_collective_result(value, stage: str):
        cur = value
        depth = 0
        while isinstance(cur, list):
            assert len(cur) > 0, f"[offload-test] {stage} returned empty list at depth={depth}"
            cur = cur[0]
            depth += 1
            if depth > 8:
                raise RuntimeError(f"[offload-test] {stage} collective result nesting is too deep: {type(value)}")
        return cur

    started_ray = False
    resource_manager = None
    actor_infer = None
    prev_roll_rpc_timeout = os.environ.get("roll_RPC_TIMEOUT")

    logger.info("[offload-test] init ray if needed")
    if not ray.is_initialized():
        ray.init(namespace=RAY_NAMESPACE, ignore_reinit_error=True, log_to_driver=True)
        started_ray = True

    try:
        # Keep cluster-level blocking calls bounded.
        os.environ["roll_RPC_TIMEOUT"] = "300"
        pipeline_boot = SimpleNamespace(seed=42, resume_from_checkpoint=False, is_actor_infer_colocated=False)
        device_mapping = "[0]"

        infer_worker_config = WorkerConfig(
            name="actor_infer",
            worker_cls="tests.diffusion.flow_grpo.test_vllm_omni_scheduler_e2e.DebugInferWorkerForModelUpdate",
            model_args=ModelArguments(
                model_name_or_path=VLLM_OMNI_TEST_MODEL_PATH,
                dtype="bf16",
            ),
            strategy_args=StrategyArguments(
                strategy_name="vllm_omni",
                strategy_config={
                    "tensor_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "trust_remote_code": True,
                    "compilation_config": {"cudagraph_mode": "FULL_AND_PIECEWISE"},
                    "vllm_omni": {
                        "custom_pipeline": CUSTOM_PIPELINE_QUALNAME,
                    },
                },
            ),
            device_mapping=device_mapping,
        )

        resource_manager = ResourceManager(num_gpus_per_node=1, num_nodes=1)
        actor_infer = Cluster(
            name="test_vllm_omni_offload_infer",
            worker_cls="tests.diffusion.flow_grpo.test_vllm_omni_scheduler_e2e.DebugInferWorkerForModelUpdate",
            resource_manager=resource_manager,
            worker_config=infer_worker_config,
        )
        actor_infer.initialize(pipeline_config=pipeline_boot, blocking=True)

        in_gpu = ray.get(actor_infer.workers[0].debug_strategy_is_model_in_gpu.remote())
        assert in_gpu is True, "[offload-test] strategy should be loaded after initialize"

        colocated = ray.get(actor_infer.workers[0].debug_set_actor_infer_colocated.remote(True))
        assert colocated is True, "[offload-test] failed to set is_actor_infer_colocated=True"

        # Main assertion: cluster offload path should return and flip strategy state to offloaded.
        actor_infer.offload_states(blocking=True)
        in_gpu_after_offload = ray.get(actor_infer.workers[0].debug_strategy_is_model_in_gpu.remote())
        assert in_gpu_after_offload is False, "[offload-test] strategy state should be offloaded after offload_states"

        # Validate load->offload cycle remains functional.
        actor_infer.load_states(blocking=True)
        in_gpu_after_load = ray.get(actor_infer.workers[0].debug_strategy_is_model_in_gpu.remote())
        assert in_gpu_after_load is True, "[offload-test] strategy should be loaded after load_states"

        # Control-path sanity: direct collective sleep RPC should also return True.
        direct_sleep_raw = ray.get(actor_infer.workers[0].debug_direct_collective_sleep.remote(1))
        direct_sleep_result = _unwrap_collective_result(direct_sleep_raw, "debug_direct_collective_sleep")
        assert bool(direct_sleep_result) is True, "[offload-test] direct collective sleep RPC should return truthy"

    finally:
        if prev_roll_rpc_timeout is None:
            os.environ.pop("roll_RPC_TIMEOUT", None)
        else:
            os.environ["roll_RPC_TIMEOUT"] = prev_roll_rpc_timeout
        logger.info("[offload-test] cleanup resources")
        if resource_manager is not None:
            resource_manager.destroy_placement_group()
        if started_ray:
            ray.shutdown()
