import asyncio
import copy
import inspect
import os
import time
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Sequence

from roll.utils.logging import get_logger


logger = get_logger()


class CustomAsyncOmni:
    """Adapter that exposes ROLL-compatible APIs on top of vllm_omni.AsyncOmni."""

    def __init__(self, async_omni: Any):
        self._engine = async_omni
        self._runtime_logged = False

    @staticmethod
    def _get_stage_type(stage_cfg: Any) -> str:
        if isinstance(stage_cfg, dict):
            return stage_cfg.get("stage_type", "llm")
        return getattr(stage_cfg, "stage_type", "llm")

    @staticmethod
    def _normalize_stage_configs(stage_configs: Any) -> List[Any]:
        if stage_configs is None:
            return []
        if isinstance(stage_configs, list):
            return stage_configs
        if isinstance(stage_configs, Sequence) and not isinstance(stage_configs, (str, bytes)):
            return list(stage_configs)
        return []

    def _describe_runtime(self) -> Dict[str, Any]:
        stage_configs = self._normalize_stage_configs(getattr(self._engine, "stage_configs", None))

        engine = getattr(self._engine, "engine", None)
        stage_clients = getattr(engine, "stage_clients", None)
        if not isinstance(stage_clients, list):
            stage_clients = []

        return {
            "engine_type": type(self._engine).__name__,
            "num_stages": len(stage_configs),
            "stage_types": [self._get_stage_type(stage_cfg) for stage_cfg in stage_configs],
            "stage_classes": [type(stage_client).__name__ for stage_client in stage_clients],
            "stage_ids": list(range(len(stage_configs))),
        }

    def _log_runtime_topology(self) -> None:
        if self._runtime_logged:
            return
        self._runtime_logged = True
        runtime = self._describe_runtime()
        logger.info("CustomAsyncOmni runtime topology: %s", runtime)

    def _stage_summary(self, stage_id: int) -> Dict[str, Any]:
        engine = getattr(self._engine, "engine", None)
        stage_clients = getattr(engine, "stage_clients", None)
        stage_client = stage_clients[stage_id] if isinstance(stage_clients, list) and stage_id < len(stage_clients) else None
        stage_configs = self._normalize_stage_configs(getattr(self._engine, "stage_configs", None))
        stage_cfg = stage_configs[stage_id] if stage_id < len(stage_configs) else None
        return {
            "stage_id": stage_id,
            "stage_index": stage_id,
            "stage_type": self._get_stage_type(stage_cfg),
            "stage_class": type(stage_client).__name__ if stage_client is not None else None,
        }

    def _get_diffusion_stage_id(self) -> int:
        self._log_runtime_topology()
        stage_configs = self._normalize_stage_configs(getattr(self._engine, "stage_configs", None))
        if not stage_configs:
            raise RuntimeError(
                "CustomAsyncOmni requires AsyncOmni.stage_configs for Qwen-Image control; "
                f"runtime={self._describe_runtime()}"
            )
        diffusion_stages = [
            stage_id
            for stage_id, stage_cfg in enumerate(stage_configs)
            if self._get_stage_type(stage_cfg) == "diffusion"
        ]

        if len(diffusion_stages) != 1:
            raise RuntimeError(
                "CustomAsyncOmni requires exactly one diffusion stage for Qwen-Image control; "
                f"runtime={self._describe_runtime()}"
            )
        stage_id = diffusion_stages[0]
        logger.info("CustomAsyncOmni selected diffusion stage: %s", self._stage_summary(stage_id))
        return stage_id

    def _require_collective_rpc(self) -> None:
        if not hasattr(self._engine, "collective_rpc"):
            raise RuntimeError(
                "CustomAsyncOmni requires AsyncOmni.collective_rpc for Qwen-Image control; "
                f"engine_type={type(self._engine).__name__}"
            )

    async def _run_diffusion_stage_rpc(
        self,
        method: str,
        *args,
        **kwargs,
    ) -> Any:
        stage_id = self._get_diffusion_stage_id()
        self._require_collective_rpc()
        logger.info(
            "CustomAsyncOmni diffusion stage RPC: stage_id=%s method=%s args=%s kwargs=%s",
            stage_id,
            method,
            args,
            kwargs,
        )
        start_time = time.perf_counter()
        # Optional timeout guard for diagnosing hangs in AsyncOmni.collective_rpc.
        # Set ROLL_VLLM_OMNI_RPC_TIMEOUT_S to a positive float (seconds) to enable.
        rpc_timeout_s = os.environ.get("ROLL_VLLM_OMNI_RPC_TIMEOUT_S")
        timeout = float(rpc_timeout_s) if rpc_timeout_s not in (None, "") else None
        try:
            result = await self._engine.collective_rpc(
                method,
                timeout=timeout,
                args=args,
                kwargs=kwargs,
                stage_ids=[stage_id],
            )
        except asyncio.TimeoutError:
            logger.error(
                "CustomAsyncOmni diffusion stage RPC timeout: stage_id=%s method=%s timeout_s=%s elapsed=%.3fs",
                stage_id,
                method,
                timeout,
                time.perf_counter() - start_time,
            )
            raise
        except Exception:
            logger.exception(
                "CustomAsyncOmni diffusion stage RPC failed: stage_id=%s method=%s elapsed=%.3fs",
                stage_id,
                method,
                time.perf_counter() - start_time,
            )
            raise
        logger.info(
            "CustomAsyncOmni diffusion stage RPC complete: stage_id=%s method=%s elapsed=%.3fs result_type=%s result_len=%s",
            stage_id,
            method,
            time.perf_counter() - start_time,
            type(result).__name__,
            len(result) if hasattr(result, "__len__") else None,
        )
        return result

    @staticmethod
    def _coerce_worker_bool(result: Any) -> bool:
        if isinstance(result, list):
            return all(CustomAsyncOmni._coerce_worker_bool(item) for item in result)
        return bool(result)

    async def _run_engine_control(self, method: str, *args, **kwargs) -> Any:
        fn = getattr(self._engine, method, None)
        if fn is None or not callable(fn):
            raise RuntimeError(
                "CustomAsyncOmni direct engine control method is unavailable: "
                f"method={method} engine_type={type(self._engine).__name__}"
            )
        start_time = time.perf_counter()
        logger.info(
            "CustomAsyncOmni direct engine control: method=%s args=%s kwargs=%s",
            method,
            args,
            kwargs,
        )
        try:
            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
        except Exception:
            logger.exception(
                "CustomAsyncOmni direct engine control failed: method=%s elapsed=%.3fs",
                method,
                time.perf_counter() - start_time,
            )
            raise
        logger.info(
            "CustomAsyncOmni direct engine control complete: method=%s elapsed=%.3fs result_type=%s",
            method,
            time.perf_counter() - start_time,
            type(result).__name__,
        )
        return result

    async def get_tokenizer(self):
        if hasattr(self._engine, "get_tokenizer"):
            return await self._engine.get_tokenizer()
        return None

    def _convert_sampling_params(self, sampling_params: Dict[str, Any]) -> Sequence[Any] | None:
        default = getattr(self._engine, "default_sampling_params_list", None)
        if not default:
            return None

        converted = []
        for sp in default:
            if is_dataclass(sp):
                sp_new = type(sp)(**asdict(sp))
                for k, v in sampling_params.items():
                    if hasattr(sp_new, k):
                        setattr(sp_new, k, v)
                converted.append(sp_new)
                continue

            # vllm SamplingParams has clone() in current versions.
            if hasattr(sp, "clone"):
                sp_new = sp.clone()
            else:
                sp_new = copy.deepcopy(sp)
            for k, v in sampling_params.items():
                if hasattr(sp_new, k):
                    setattr(sp_new, k, v)
            converted.append(sp_new)

        return converted

    async def generate_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request_id = str(payload.get("rid", uuid.uuid4().hex))
        prompt = payload.get("multi_modal_data", payload.get("input_ids"))
        sampling_params = self._convert_sampling_params(payload.get("sampling_params", {}))

        # Wrap flat token-id list into the structured format expected by diffusion pipelines.
        # The pipeline's forward() reads prompt_ids from req.prompts[0]["prompt_ids"].
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            prompt = [{"prompt_ids": prompt}]
        elif isinstance(prompt, dict) and "prompt_ids" in prompt:
            prompt = [prompt]

        final_output = None
        async for output in self._engine.generate(
            prompt=prompt,
            request_id=request_id,
            sampling_params_list=sampling_params,
        ):
            final_output = output

        if final_output is None:
            logger.warning("CustomAsyncOmni.generate_request debug: request_id=%s final_output is None", request_id)
            return {"finish_reasons": ["abort"], "output_token_ids": []}

        custom_output = getattr(final_output, "custom_output", None)
        request_output = getattr(final_output, "request_output", None)
        outputs = getattr(final_output, "outputs", None)

        # When patched_generate_batch takes the single-result branch it wraps the
        # engine result inside request_output without duplicating _custom_output
        # onto the outer object.  Recover the custom_output from the inner object.
        if not custom_output and request_output is not None:
            inner_custom_output = getattr(request_output, "custom_output", None)
            if isinstance(inner_custom_output, dict) and inner_custom_output:
                custom_output = inner_custom_output

        if isinstance(custom_output, dict) and custom_output:
            return custom_output

        if isinstance(request_output, dict):
            return request_output

        # Text-generation fallback format (same keys as vllm_strategy path).
        logger.warning(
            "CustomAsyncOmni.generate_request debug: request_id=%s falling back to text output format",
            request_id,
        )
        output_token_ids, finish_reasons, output_logprobs = [], [], []
        for completion_output in getattr(final_output, "outputs", []):
            token_ids = getattr(completion_output, "token_ids", None)
            if token_ids is None:
                continue
            output_token_ids.append(token_ids)
            finish_reason = getattr(completion_output, "finish_reason", None) or "abort"
            finish_reasons.append(finish_reason)
            logprobs = getattr(completion_output, "logprobs", None)
            if logprobs is not None:
                output_logprobs.append(logprobs)

        return {
            "output_token_ids": output_token_ids,
            "finish_reasons": finish_reasons if finish_reasons else ["stop"],
            "output_logprobs": output_logprobs,
        }

    async def abort_request(self, request_id: str):
        if hasattr(self._engine, "abort"):
            await self._engine.abort(request_id)

    async def abort_requests(self, request_ids: Iterable[str]):
        for request_id in request_ids:
            await self.abort_request(request_id)

    async def abort_all_requests(self):
        if hasattr(self._engine, "collective_rpc"):
            try:
                await self._engine.collective_rpc(method="abort_all_requests")
            except Exception:
                logger.warning("collective_rpc abort_all_requests is not available in current vllm_omni runtime")

    async def reset_prefix_cache(self):
        if not hasattr(self._engine, "reset_prefix_cache"):
            return True
        result = self._engine.reset_prefix_cache()
        if inspect.isawaitable(result):
            return await result
        return result

    async def load_states(self):
        await self.wake_up()

    async def offload_states(self, level: int = 1):
        # Keep semantic parity with vllm strategy; offload resets cache before sleeping.
        # logger.info("CustomAsyncOmni offload_states: level=%s direct engine sleep", level)
        if hasattr(self._engine, "reset_prefix_cache"):
            logger.info("CustomAsyncOmni offload_states: reset_prefix_cache before sleep")
            await self._engine.reset_prefix_cache()
        await self.sleep(level)

    async def sleep(self, level: int = 1):
        logger.info("CustomAsyncOmni sleep: level=%s direct engine sleep", level)
        await self._run_engine_control("sleep", level)

    async def wake_up(self):
        logger.info("CustomAsyncOmni wake_up: direct engine wake_up tags=None (restore all)")
        await self._run_engine_control("wake_up", None)

    async def process_weights_after_loading(self):
        logger.info("CustomAsyncOmni process_weights_after_loading: dispatch to diffusion stage")
        await self._run_diffusion_stage_rpc("process_weights_after_loading")

    async def setup_collective_group(self, *args, **kwargs):
        logger.info(
            "CustomAsyncOmni setup_collective_group: args_len=%s kwargs_keys=%s",
            len(args),
            sorted(kwargs.keys()),
        )
        await self._run_diffusion_stage_rpc("setup_collective_group", *args, **kwargs)

    async def broadcast_parameter(self, *args, **kwargs):
        logger.info(
            "CustomAsyncOmni broadcast_parameter: args_len=%s kwargs_keys=%s",
            len(args),
            sorted(kwargs.keys()),
        )
        await self._run_diffusion_stage_rpc("broadcast_parameter", *args, **kwargs)

    async def update_parameter_in_bucket(self, serialized_named_tensors, is_lora: bool = False):
        logger.info(
            "CustomAsyncOmni update_parameter_in_bucket: num_buckets=%s is_lora=%s",
            len(serialized_named_tensors) if hasattr(serialized_named_tensors, "__len__") else None,
            is_lora,
        )
        await self._run_diffusion_stage_rpc("update_parameter_in_bucket", serialized_named_tensors, is_lora=is_lora)

    async def add_lora(self, lora_request):
        # Keep ROLL compatibility: strategy passes peft_config dict.
        if isinstance(lora_request, dict):
            logger.info(
                "CustomAsyncOmni add_lora: dispatch custom_add_lora with config_keys=%s",
                sorted(lora_request.keys()),
            )
            result = await self._run_diffusion_stage_rpc("custom_add_lora", lora_request)
            # collective_rpc may return nested lists [[dict]] or [dict];
            # unwrap until we reach the dict (all workers return identical info).
            while isinstance(result, list) and result:
                result = result[0]
            logger.info("CustomAsyncOmni add_lora: lora_info=%s", result)
            return result
        # AsyncOmni native path (LoRARequest).
        logger.info("CustomAsyncOmni add_lora: dispatch native AsyncOmni.add_lora for %s", type(lora_request).__name__)
        return await self._engine.add_lora(lora_request)

    async def set_global_steps(self, global_step: int):
        logger.info("CustomAsyncOmni set_global_steps: global_step=%s", global_step)
        await self._run_diffusion_stage_rpc("set_global_steps", global_step)

    async def set_ema_decay(self, ema_decay: float):
        logger.info("CustomAsyncOmni set_ema_decay: ema_decay=%s", ema_decay)
        await self._run_diffusion_stage_rpc("set_ema_decay", ema_decay)

    async def forward_step(self, **kwargs):
        """Single-step diffusion forward dispatched to diffusion worker pipeline.forward_step."""
        logger.info(
            "CustomAsyncOmni forward_step: kwargs_keys=%s",
            sorted(kwargs.keys()),
        )
        results = await self._run_diffusion_stage_rpc("forward_step", **kwargs)
        # collective_rpc returns a list of results from each worker; take the first.
        if isinstance(results, list) and len(results) > 0:
            return results[0]
        return results
