from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import ray

from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.router import RouterManager
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.tinker_backend_runtime import types
from roll.pipeline.tinker_backend_runtime.data_adapter import (
    _model_input_to_token_ids,
    prepared_batch_to_dataproto,
)
from roll.utils.functionals import reduce_metrics
from roll.utils.logging import get_logger


logger = get_logger()


class ROLLRuntimeBackend(BasePipeline):
    """Runtime-local ROLL primitive wrapper for Tinker backend actions.

    This class intentionally does not expose FastAPI, database models, futures,
    or any SDK-facing service. The external control plane is tinker-backend;
    this object only owns local ROLL clusters and primitive operators.
    """

    def __init__(self, pipeline_config: AgenticConfig):
        super().__init__(pipeline_config)
        self.pipeline_config: AgenticConfig
        self._training_backend_enabled = os.environ.get("TINKER_ENABLE_TRAINING_BACKEND", "1") != "0"
        if self._training_backend_enabled and getattr(self.pipeline_config.actor_train.training_args, "max_steps", -1) <= 0:
            self.pipeline_config.actor_train.training_args.max_steps = max(1, self.pipeline_config.max_steps)
        self._ensure_actor_infer_loads_weights()
        self._active_model_ids: set[str] = set()

        self.actor_train: Any | None = None
        if self._training_backend_enabled:
            self.actor_train = Cluster(
                name=self.pipeline_config.actor_train.name,
                worker_cls=self.pipeline_config.actor_train.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.actor_train,
            )

        self.actor_infer: Any = Cluster(
            name=self.pipeline_config.actor_infer.name,
            worker_cls=self.pipeline_config.actor_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )
        if self.actor_train is not None:
            self.download_models(self.actor_train, self.actor_infer)
        else:
            self.download_models(self.actor_infer)

        self.tokenizer = default_tokenizer_provider(
            model_args=self.pipeline_config.actor_infer.model_args
        )

        if self.actor_train is not None:
            self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=True)
            self.set_checkpoint_clusters(self.actor_train)

        use_policy_model: bool = (
            self.pipeline_config.train_env_manager.llm_proxy.proxy_type in {"policy", "tinker_policy"}
        )
        self.router_manager: Any | None = None
        self.router_client: Any | None = None
        if use_policy_model:
            self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=True)
            self._initialize_rollout_router()
            if self.actor_train is not None:
                self.set_model_update_pair(
                    src_cluster=self.actor_train,
                    tgt_cluster=self.actor_infer,
                    frequency=self.pipeline_config.actor_train.model_update_frequency,
                )
        else:
            logger.warning(
                "ROLLRuntimeBackend actor_infer is not initialized because llm_proxy.proxy_type=%s; "
                "runtime sampling and publish_to_sampler require policy/tinker_policy.",
                self.pipeline_config.train_env_manager.llm_proxy.proxy_type,
            )

        if self.actor_train is None:
            logger.info("ROLLRuntimeBackend started in rollout-only mode; training actions are disabled")

        self.reference: Any | None = None
        if getattr(self.pipeline_config, "enable_reference", False):
            self.reference = Cluster(
                name=self.pipeline_config.reference.name,
                worker_cls=self.pipeline_config.reference.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.reference,
            )
            self.download_models(self.reference)
            self.reference.initialize(pipeline_config=self.pipeline_config, blocking=True)
            self.reference.offload_states(blocking=True)
            logger.info(
                "ROLLRuntimeBackend reference cluster initialized and offloaded; device_mapping=%s",
                self.pipeline_config.reference.device_mapping,
            )

        logger.info("ROLLRuntimeBackend initialized")

    def _initialize_rollout_router(self) -> None:
        max_concurrency = max(2, int(self.pipeline_config.max_running_requests) + 1)
        self.router_manager = ray.remote(RouterManager).options(
            max_concurrency=max_concurrency,
        ).remote(
            actor_cluster=self.actor_infer,
            router_args=self.pipeline_config.router_args,
            num_gpus_per_node=self.pipeline_config.num_gpus_per_node,
        )
        ray.get(self.router_manager.initialize.remote())
        self.router_client = RouterManager.create_client_sync(self.router_manager)
        logger.info(
            "ROLL native rollout router initialized; router=%s, dp_size=%s, max_running_requests=%s",
            self.pipeline_config.router_args.router_name,
            self.actor_infer.world_size,
            max_concurrency - 1,
        )

    def _ensure_actor_infer_loads_weights(self) -> None:
        strategy_args = self.pipeline_config.actor_infer.strategy_args
        if strategy_args is None or strategy_args.strategy_name != "vllm":
            return
        strategy_config = strategy_args.strategy_config
        if strategy_config.get("load_format", "dummy") == "dummy":
            logger.warning(
                "Overriding actor_infer vLLM load_format=dummy to auto because runtime sampling "
                "needs real model weights before the first model_update"
            )
            strategy_config["load_format"] = "auto"

    def close(self) -> None:
        try:
            if self.router_manager is not None:
                ray.get(self.router_manager.shutdown.remote())
        except Exception as exc:
            logger.warning("Failed to shut down ROLL rollout router cleanly: %s", exc)
        finally:
            self.router_client = None
            self.router_manager = None
        try:
            ray.shutdown()
        except Exception:
            pass

    def _pad_token_id(self) -> int:
        if self.tokenizer.pad_token_id is not None:
            return self.tokenizer.pad_token_id
        if self.tokenizer.eos_token_id is not None:
            return self.tokenizer.eos_token_id
        return 0

    def _require_training_backend(self) -> None:
        if self.actor_train is None:
            raise NotImplementedError(
                "Tinker training backend is disabled for this process. "
                "Unset TINKER_ENABLE_TRAINING_BACKEND=0 to enable training actions."
            )

    @staticmethod
    def _tensor_data_payload(values: list[float], *, dtype: str, shape: list[int]) -> dict:
        return {"data": values, "dtype": dtype, "shape": shape}

    def _prepared_model_pass_to_dataproto(self, prepared: types.PreparedModelPassBatch) -> DataProto:
        data = prepared_batch_to_dataproto(
            prepared,
            sequence_length=self.pipeline_config.sequence_length,
            pad_token_id=self._pad_token_id(),
        )
        data.meta_info["is_offload_states"] = False
        return data

    @staticmethod
    def _checkpoint_dir_from_path(checkpoint_path: str | Path) -> str:
        checkpoint_path = str(checkpoint_path)
        if checkpoint_path.endswith(".tar.gz"):
            checkpoint_path = checkpoint_path[:-len(".tar.gz")]
        return checkpoint_path

    def _build_forward_outputs(
        self,
        prepared: types.PreparedModelPassBatch,
        output: DataProto,
    ) -> dict[str, types.ForwardBackwardOutput]:
        log_probs = output.batch["tinker_target_logprobs"]
        prompt_lengths = output.non_tensor_batch["tinker_prompt_lengths"]
        target_lengths = output.non_tensor_batch["tinker_target_lengths"]
        results: dict[str, types.ForwardBackwardOutput] = {}

        for request_id, _, start_idx, end_idx in prepared.request_batch_slices:
            loss_fn_outputs = []
            for row_idx in range(start_idx, end_idx):
                prompt_len = int(prompt_lengths[row_idx])
                target_len = int(target_lengths[row_idx])
                start = max(prompt_len - 1, 0)
                row_values = log_probs[row_idx][start : start + target_len].detach().cpu().tolist()
                loss_fn_outputs.append(
                    {
                        "logprobs": self._tensor_data_payload(
                            [float(v) for v in row_values],
                            dtype="float32",
                            shape=[target_len],
                        )
                    }
                )
            results[request_id] = types.ForwardBackwardOutput(
                loss_fn_output_type="logprobs",
                loss_fn_outputs=loss_fn_outputs,
                metrics={},
            )
        return results

    def _build_backward_outputs(
        self,
        prepared: types.PreparedModelPassBatch,
        output: DataProto,
    ) -> dict[str, types.ForwardBackwardOutput]:
        example_losses = output.batch["tinker_example_losses"]
        reduced_metrics = reduce_metrics(output.meta_info.get("metrics", {}))
        results: dict[str, types.ForwardBackwardOutput] = {}

        for request_id, _, start_idx, end_idx in prepared.request_batch_slices:
            loss_fn_outputs = []
            for row_idx in range(start_idx, end_idx):
                loss_value = float(example_losses[row_idx].detach().cpu().item())
                loss_fn_outputs.append(
                    {
                        "loss": self._tensor_data_payload(
                            [loss_value],
                            dtype="float32",
                            shape=[1],
                        )
                    }
                )
            request_losses = example_losses[start_idx:end_idx]
            request_metrics = {"tinker_ce_loss:mean": float(request_losses.mean().detach().cpu().item())}
            if reduced_metrics:
                request_metrics.update({f"{k}:mean": float(v) for k, v in reduced_metrics.items()})
            results[request_id] = types.ForwardBackwardOutput(
                loss_fn_output_type="cross_entropy",
                loss_fn_outputs=loss_fn_outputs,
                metrics=request_metrics,
            )
        return results

    def _publish_actor_train_to_infer(self) -> None:
        if not self.model_update_groups:
            raise RuntimeError(
                "actor_infer model update path is not initialized; "
                "publish_to_sampler requires llm_proxy.proxy_type='policy'"
            )
        self.model_update(global_step=0)

    def sample(self, prepared: types.PreparedSampleBatch) -> Dict[str, types.SampleOutput]:
        if prepared.request_batch_slices and self._is_score_only_batch(prepared):
            return self._sample_score_only(prepared)
        return self._sample_generate(prepared)

    @staticmethod
    def _is_score_only_batch(prepared: types.PreparedSampleBatch) -> bool:
        env_ids = prepared.all_env_ids or []
        all_envless = all(eid is None for eid in env_ids)
        any_prompt_logprobs = any(plp for *_, plp in prepared.request_batch_slices)
        return all_envless and any_prompt_logprobs

    def _sample_generate(self, prepared: types.PreparedSampleBatch) -> Dict[str, types.SampleOutput]:
        results: Dict[str, types.SampleOutput] = {}
        for request_id, _model_id, start_idx, end_idx, _plp in prepared.request_batch_slices:
            prompt_ids = _model_input_to_token_ids(prepared.all_model_inputs[start_idx])
            num_samples = max(1, end_idx - start_idx)
            sequences: list[types.GeneratedSequence] = []
            for sample_idx in range(start_idx, end_idx):
                env_id = prepared.all_env_ids[sample_idx] if prepared.all_env_ids else None
                routed_request_id = (
                    request_id
                    if num_samples == 1
                    else f"{request_id}:{sample_idx - start_idx}"
                )
                sequences.extend(
                    self._generate_one(
                        prompt_ids,
                        prepared.all_sampling_params[sample_idx],
                        1,
                        request_id=routed_request_id,
                        env_id=env_id,
                    )
                )
            results[request_id] = types.SampleOutput(
                sequences=sequences,
                prompt_token_ids=prompt_ids,
                prompt_logprobs=None,
            )
        return results

    def _generate_one(
        self,
        prompt_ids: list[int],
        sampling_params: types.SamplingParams,
        num_samples: int,
        *,
        request_id: str,
        env_id: str | None,
    ) -> list[types.GeneratedSequence]:
        if self.router_client is None:
            raise RuntimeError("ROLL native rollout router is not initialized")
        data = self._build_score_batch([("sample", prompt_ids)])
        generation_config = self.actor_infer.worker_config.generating_args.to_dict()
        generation_config.update(
            {
                "max_new_tokens": sampling_params.max_tokens,
                "temperature": sampling_params.temperature,
                "top_p": sampling_params.top_p,
                "top_k": sampling_params.top_k,
                "seed": sampling_params.seed,
                "num_return_sequences": num_samples,
                "stop_strings": sampling_params.stop_strings or generation_config.get("stop_strings", []),
                "stop_token_ids": sampling_params.stop_tokens or generation_config.get("stop_token_ids", []),
                "logprobs": 0,
            }
        )
        data.meta_info["generation_config"] = generation_config
        output = self.router_client.generate_request_sync(
            data,
            request_id=request_id,
            uid=env_id or request_id,
        )
        if output is None:
            raise RuntimeError("ROLL native rollout router stopped before generation completed")
        responses = output.meta_info.get("output_token_ids") or []
        finish_reasons = output.meta_info.get("finish_reasons") or []
        infer_logprobs = output.meta_info.get("output_logprobs") or []

        sequences: list[types.GeneratedSequence] = []
        max_tokens = int(sampling_params.max_tokens)
        for row_idx, row in enumerate(responses[:num_samples]):
            tokens = [int(token) for token in row]
            finish_reason = str(finish_reasons[row_idx]) if row_idx < len(finish_reasons) else "stop"
            if finish_reason == "abort":
                raise RuntimeError(f"ROLL native rollout request {request_id} was aborted")
            logprobs = []
            if row_idx < len(infer_logprobs):
                logprobs = [float(value) for value in infer_logprobs[row_idx][: len(tokens)]]
            if not logprobs:
                logprobs = [0.0 for _ in tokens]
            sequences.append(
                types.GeneratedSequence(
                    stop_reason="length" if finish_reason == "length" or len(tokens) >= max_tokens else "stop",
                    tokens=tokens,
                    output_token_ids=tokens,
                    logprobs=logprobs,
                )
            )
        if len(sequences) != num_samples:
            raise RuntimeError(
                f"ROLL native rollout returned {len(sequences)} sequences, expected {num_samples}"
            )
        return sequences

    def _sample_score_only(
        self, prepared: types.PreparedSampleBatch
    ) -> Dict[str, types.SampleOutput]:
        ref_available = getattr(self, "reference", None) is not None
        ref_entries: list[tuple[str, list[int]]] = []
        pol_entries: list[tuple[str, list[int]]] = []
        for request_id, model_id, start_idx, _end_idx, _plp in prepared.request_batch_slices:
            prompt_ids = _model_input_to_token_ids(prepared.all_model_inputs[start_idx])
            checkpoint_id = prepared.all_checkpoint_ids[start_idx] if prepared.all_checkpoint_ids else ""
            is_base = (not model_id) and (not checkpoint_id)
            if is_base and ref_available:
                ref_entries.append((request_id, prompt_ids))
            else:
                pol_entries.append((request_id, prompt_ids))

        results: Dict[str, types.SampleOutput] = {}
        if ref_entries:
            ref_data = self._build_score_batch(ref_entries)
            ref_prompt_logprobs = self._score_with_reference(ref_data, ref_entries)
            self._fill_score_results(results, ref_entries, ref_prompt_logprobs)
        if pol_entries:
            pol_data = self._build_score_batch(pol_entries)
            pol_prompt_logprobs = self._score_with_actor_infer(pol_data, pol_entries)
            self._fill_score_results(results, pol_entries, pol_prompt_logprobs)
        return results

    def _build_score_batch(self, entries: list[tuple[str, list[int]]]) -> DataProto:
        import torch
        from tensordict import TensorDict

        pad_token_id = self._pad_token_id()
        max_len = max(len(ids) for _, ids in entries)
        batch_size = len(entries)
        input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        position_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        for row_idx, (_request_id, ids) in enumerate(entries):
            n_tokens = len(ids)
            input_ids[row_idx, :n_tokens] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row_idx, :n_tokens] = 1
            position_ids[row_idx, :n_tokens] = torch.arange(n_tokens, dtype=torch.long)

        return DataProto(
            batch=TensorDict(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "response_mask": attention_mask.clone(),
                },
                batch_size=batch_size,
            )
        )

    @staticmethod
    def _fill_score_results(
        results: Dict[str, types.SampleOutput],
        entries: list[tuple[str, list[int]]],
        prompt_logprobs_per_request: list[list[float | None] | None],
    ) -> None:
        for row_idx, (request_id, ids) in enumerate(entries):
            results[request_id] = types.SampleOutput(
                sequences=[types.GeneratedSequence(stop_reason="length", tokens=[], output_token_ids=[], logprobs=[])],
                prompt_token_ids=ids,
                prompt_logprobs=prompt_logprobs_per_request[row_idx],
            )

    def _score_with_reference(
        self,
        data: DataProto,
        request_entries: list[tuple[str, list[int]]],
    ) -> list[list[float | None] | None]:
        import torch
        from tensordict import TensorDict

        ref_devices = set(self.pipeline_config.reference.device_mapping or [])
        candidates = [("actor_train", self.actor_train), ("actor_infer", self.actor_infer)]
        overlapping: list[tuple[str, Any]] = []
        for name, cluster in candidates:
            if cluster is None:
                continue
            devices = set(getattr(cluster.worker_config, "device_mapping", None) or [])
            if ref_devices & devices:
                overlapping.append((name, cluster))

        batch_size = len(request_entries)
        try:
            ref_dp = max(1, int(self.reference.dp_size))
        except Exception:
            ref_dp = 1
        pad_count = (ref_dp - (batch_size % ref_dp)) % ref_dp
        if pad_count > 0:
            pad_rows = data.batch[:1].expand(pad_count, *data.batch.batch_size[1:])
            padded = TensorDict(
                {key: torch.cat([data.batch[key], pad_rows[key]], dim=0) for key in data.batch.keys()},
                batch_size=batch_size + pad_count,
            )
            data = DataProto(batch=padded, meta_info=dict(data.meta_info))

        data.meta_info.setdefault("loss_mask_keys", [])
        awake_before = {name: self._cluster_is_awake(cluster) for name, cluster in overlapping}
        for name, cluster in overlapping:
            if awake_before[name]:
                cluster.offload_states(blocking=True)

        try:
            ref_out = self.reference.compute_log_probs(data, blocking=True)
        finally:
            for name, cluster in overlapping:
                if awake_before[name]:
                    cluster.load_states(blocking=True)

        log_probs = ref_out.batch["log_probs"].cpu()
        prompt_logprobs_per_request: list[list[float | None] | None] = []
        for row_idx, (_request_id, ids) in enumerate(request_entries):
            n_tokens = len(ids)
            row: list[float | None] = [None]
            for token_idx in range(n_tokens - 1):
                row.append(float(log_probs[row_idx, token_idx].item()))
            prompt_logprobs_per_request.append(row)
        return prompt_logprobs_per_request

    def _score_with_actor_infer(
        self,
        data: DataProto,
        request_entries: list[tuple[str, list[int]]],
    ) -> list[list[float | None] | None]:
        out = self.actor_infer.compute_prompt_logprobs(data, blocking=True)
        prompt_logprobs_array = out.non_tensor_batch["prompt_logprobs"]
        prompt_logprobs_per_request: list[list[float | None] | None] = []
        for row_idx in range(len(request_entries)):
            row = prompt_logprobs_array[row_idx] if row_idx < len(prompt_logprobs_array) else None
            prompt_logprobs_per_request.append(list(row) if row is not None else None)
        return prompt_logprobs_per_request

    def _cluster_is_awake(self, cluster: Any) -> bool:
        try:
            return bool(cluster.is_model_in_gpu(blocking=True))
        except Exception as exc:
            logger.warning("cluster.is_model_in_gpu failed (%s); assuming awake", exc)
            return True

    @property
    def metrics(self) -> types.EngineMetrics:
        return types.EngineMetrics()

    def has_model(self, model_id: str, *_args, **_kwargs) -> bool:
        if self.actor_train is None:
            return False
        return model_id in self._active_model_ids

    def create_model(self, model_id: str, lora_config: types.LoraConfig) -> None:
        self._require_training_backend()
        if lora_config.rank > 0:
            logger.warning(
                "ROLLRuntimeBackend.create_model currently reuses one shared actor_train model; "
                "LoRA adapter isolation is not implemented yet. model_id=%s rank=%s",
                model_id,
                lora_config.rank,
            )
        self._active_model_ids.add(model_id)

    def delete_model(self, model_id: str, *_args, **_kwargs) -> None:
        self._require_training_backend()
        self._active_model_ids.discard(model_id)

    def forward_backward(self, prepared: types.PreparedModelPassBatch) -> dict:
        self._require_training_backend()
        unsupported = sorted({loss_fn for loss_fn in prepared.all_loss_fns if loss_fn != "cross_entropy"})
        if unsupported:
            raise NotImplementedError(
                "ROLLRuntimeBackend.forward_backward currently only supports loss_fn='cross_entropy'; "
                f"got unsupported values: {unsupported}"
            )
        data = self._prepared_model_pass_to_dataproto(prepared)
        refs = self.actor_train.forward_backward_accumulate_tinker(data, blocking=False)
        output = DataProto.materialize_concat(data_refs=refs)
        return self._build_backward_outputs(prepared, output)

    def forward(self, prepared: types.PreparedModelPassBatch) -> dict:
        self._require_training_backend()
        unsupported = sorted({loss_fn for loss_fn in prepared.all_loss_fns if loss_fn != "cross_entropy"})
        if unsupported:
            raise NotImplementedError(
                "ROLLRuntimeBackend.forward currently only supports loss_fn='cross_entropy'; "
                f"got unsupported values: {unsupported}"
            )
        data = self._prepared_model_pass_to_dataproto(prepared)
        refs = self.actor_train.forward_tinker(data, blocking=False)
        output = DataProto.materialize_concat(data_refs=refs)
        return self._build_forward_outputs(prepared, output)

    def optim_step(self, model_id: str, request_data: types.OptimStepInput) -> types.OptimStepOutput:
        self._require_training_backend()
        refs = self.actor_train.apply_optim_step_tinker(request_data.adam_params.model_dump(), blocking=False)
        output = DataProto.materialize_concat(data_refs=refs)
        reduced_metrics = reduce_metrics(output.meta_info.get("metrics", {}))
        scalar_metrics = {
            key: float(value) if not isinstance(value, (int, float)) else value
            for key, value in reduced_metrics.items()
        }
        return types.OptimStepOutput(metrics=scalar_metrics)

    def save_checkpoint(self, checkpoint_path: str | Path, model_id: str) -> None:
        self._require_training_backend()
        checkpoint_dir = self._checkpoint_dir_from_path(checkpoint_path)
        checkpoint_id = Path(checkpoint_dir).name
        refs = self.actor_train.save_checkpoint_tinker(checkpoint_dir, checkpoint_id, blocking=False)
        DataProto.materialize_concat(data_refs=refs)

    def load_checkpoint(self, checkpoint_path: str | Path, model_id: str, load_optimizer: bool = True) -> None:
        self._require_training_backend()
        checkpoint_dir = self._checkpoint_dir_from_path(checkpoint_path)
        refs = self.actor_train.load_checkpoint_tinker(
            checkpoint_dir,
            load_optimizer=load_optimizer,
            blocking=False,
        )
        DataProto.materialize_concat(data_refs=refs)

    def save_sampler_checkpoint(self, _output_path: str | Path, _model_id: str, persist: bool = True) -> None:
        self._require_training_backend()
        if persist:
            logger.warning(
                "ROLLRuntimeBackend.save_sampler_checkpoint publishes latest actor_train weights "
                "to actor_infer but does not persist a reusable historical sampler artifact yet."
            )
        self._publish_actor_train_to_infer()
