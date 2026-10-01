import asyncio
import copy
import gc
import os
import socket
import time
from typing import Any, Callable, Dict, List, Optional, Tuple


def _find_free_port() -> int:
    """Return an available TCP port on the local machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

import torch
from transformers import set_seed

from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.strategy import InferenceStrategy
from roll.platforms import current_platform
from roll.third_party.vllm_omni import create_async_llm_omni
from roll.datasets.collator import collate_fn_to_dict_list
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType


logger = get_logger()


def create_sampling_params_for_vllm_omni(gen_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Build sampling params for vllm_omni from generation config.

    Align with helper conventions in other strategies:
    - expose `n` for router credit accounting (from num_return_sequences)
    - carry common stop/max fields when available
    - preserve diffusion-specific fields (e.g. height/width/extra_args)
    - filter out None values to let vllm_omni use its own defaults
    """
    sampling_params = {k: v for k, v in copy.deepcopy(gen_kwargs).items() if v is not None}

    if "num_return_sequences" in gen_kwargs:
        sampling_params["n"] = gen_kwargs["num_return_sequences"]
    if "max_new_tokens" in gen_kwargs:
        sampling_params.setdefault("max_tokens", gen_kwargs["max_new_tokens"])
    if "eos_token_id" in gen_kwargs:
        sampling_params.setdefault("stop_token_ids", gen_kwargs["eos_token_id"])
    if "stop_strings" in gen_kwargs:
        sampling_params.setdefault("stop", gen_kwargs["stop_strings"])
    if "include_stop_str_in_output" in gen_kwargs:
        sampling_params.setdefault("include_stop_str_in_output", gen_kwargs["include_stop_str_in_output"])

    return sampling_params


class VllmOmniStrategy(InferenceStrategy):
    strategy_name = "vllm_omni"

    def __init__(self, worker: Worker):
        super().__init__(worker)
        self.sleep_level = 1
        self.is_model_in_gpu = False
        self._request_seed = 0
        self._roll_lora_request_info = None

    def _compute_worker_seed(self) -> int:
        # Keep parity with VERL vllm_omni async server semantics:
        # engine/request seed = rollout_seed + replica_rank.
        base_seed = int(getattr(self.worker.pipeline_config, "seed", 0) or 0)
        replica_rank = int(getattr(self.worker, "rank", 0) or 0)
        return base_seed + replica_rank

    async def initialize(self, model_provider):
        self._request_seed = self._compute_worker_seed()
        set_seed(seed=self._request_seed)
        omni_config = copy.deepcopy(self.worker_config.strategy_args.strategy_config)
        self.sleep_level = omni_config.pop("sleep_level", 1)
        custom_pipeline = None

        # custom_pipeline = omni_config.pop("custom_pipeline", None)
        vllm_omni_cfg = omni_config.pop("vllm_omni", None)
        if isinstance(vllm_omni_cfg, dict):
            custom_pipeline = vllm_omni_cfg.get("custom_pipeline", None)
        if custom_pipeline is not None:
            omni_config["custom_pipeline_args"] = {"pipeline_class": custom_pipeline}
            # Explicitly indicate custom pipeline mode for clarity.
            omni_config["diffusion_load_format"] = "custom_pipeline"
            logger.info("vllm_omni custom pipeline enabled: %s", custom_pipeline)

        if self.worker_config.model_args.dtype == "fp32":
            dtype = "float32"
        elif self.worker_config.model_args.dtype == "fp16":
            dtype = "float16"
        else:
            dtype = "bfloat16"

        omni_config.update(
            {
                "model": self.worker_config.model_args.model_name_or_path,
                "dtype": dtype,
                "seed": self._request_seed,
                "master_port": _find_free_port(),
            }
        )

        os.environ.setdefault("VLLM_OMNI_CACHE_ROOT", os.path.join(os.path.expanduser("~"), ".cache", "vllm_omni"))
        logger.info("vllm_omni initialize with config: %s", omni_config)
        logger.info(
            "vllm_omni initialize debug: model_type=%s, custom_pipeline=%s, worker_rank=%s",
            getattr(self.worker_config.model_args, "model_type", None),
            custom_pipeline,
            getattr(self.worker, "rank", None),
        )

        self.model = await create_async_llm_omni(
            resource_placement_groups=self.worker_config.resource_placement_groups,
            **omni_config,
        )
        logger.info(
            "vllm_omni initialize debug: engine_type=%s, wrapped_engine_type=%s",
            type(self.model).__name__,
            type(getattr(self.model, "_engine", None)).__name__ if hasattr(self.model, "_engine") else None,
        )

        # For diffusion pipelines on vllm_omni==0.18.0, tokenizer construction is
        # owned by ROLL's data adapter instead of the engine lifecycle.
        self.tokenizer = None

        self.is_model_in_gpu = True

    def _normalize_output(
        self,
        output: Dict[str, Any],
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        required = [
            "responses",
            "rollout_log_probs",
            "all_timesteps",
            "all_latents",
            "prompt_embeds",
            "prompt_embeds_mask",
        ]
        missing = [k for k in required if k not in output]
        if missing:
            raise ValueError(f"vllm_omni output missing required keys: {missing}")

        optional_negative_keys = ("negative_prompt_embeds", "negative_prompt_embeds_mask")
        negative_keys_present = [key in output for key in optional_negative_keys]
        if any(negative_keys_present) and not all(negative_keys_present):
            raise ValueError(
                "vllm_omni output must contain both negative_prompt_embeds and "
                "negative_prompt_embeds_mask, or neither"
            )

        if all(negative_keys_present):
            negative_values_present = [output[key] is not None for key in optional_negative_keys]
            if any(negative_values_present) and not all(negative_values_present):
                raise ValueError(
                    "vllm_omni output negative_prompt_embeds and negative_prompt_embeds_mask "
                    "must both be non-None, or both be None"
                )
            if not any(negative_values_present):
                for key in optional_negative_keys:
                    del output[key]
                negative_keys_present = [False, False]

        guidance_scale = float((sampling_params or {}).get("guidance_scale", 1.0))
        if guidance_scale > 1.0 and not all(negative_keys_present):
            raise ValueError(
                "vllm_omni output missing negative prompt tensors while guidance_scale > 1.0"
            )

        # Contract check: K/K+1 requirement.
        tensor_keys = required + [key for key in optional_negative_keys if key in output]
        for key in tensor_keys:
            if output[key] is None:
                continue
            if not isinstance(output[key], torch.Tensor):
                output[key] = torch.as_tensor(output[key])

        rollout_log_probs = output["rollout_log_probs"]
        all_timesteps = output["all_timesteps"]
        all_latents = output["all_latents"]

        extra_args = (sampling_params or {}).get("extra_args") or {}
        sde_type = extra_args.get("sde_type", "sde")

        k = rollout_log_probs.shape[1]
        if all_timesteps.shape[1] != k:
            raise ValueError(
                f"Invalid rollout contract: all_timesteps.shape[1]={all_timesteps.shape[1]} != K={k}"
            )
        if sde_type == "ode":
            if all_latents.shape[1] != 1:
                raise ValueError(
                    "Invalid ODE rollout contract: "
                    f"all_latents.shape[1]={all_latents.shape[1]}, "
                    "expected final-only=1"
                )
        elif all_latents.shape[1] != k + 1:
            raise ValueError(
                f"Invalid rollout contract: all_latents.shape[1]={all_latents.shape[1]} != K+1={k + 1}"
            )

        logger.debug(
            "rollout_omni/normalize_output: sde_type=%s responses=%s, log_probs=%s, timesteps=%s, latents=%s",
            sde_type,
            tuple(output["responses"].shape),
            tuple(rollout_log_probs.shape),
            tuple(all_timesteps.shape),
            tuple(all_latents.shape),
        )
        return output

    async def generate(self, batch: DataProto, generation_config):
        raise NotImplementedError(
            "VllmOmniStrategy.generate token fallback path is unsupported. "
            "Use scheduler/router generate_request with diffusion custom outputs."
        )

    async def generate_request(self, payload: Dict):
        if not hasattr(self.model, "generate_request"):
            raise NotImplementedError("vllm_omni engine must implement `generate_request` for scheduler path")
        payload = copy.deepcopy(payload)
        sampling_params = payload.get("sampling_params")
        if sampling_params is None:
            sampling_params = {}
            payload["sampling_params"] = sampling_params

        # Inject LoRA request into sampling_params so vLLM applies the adapter during rollout.
        lora_info = getattr(self, "_roll_lora_request_info", None)
        if lora_info is not None:
            from vllm_omni.lora.request import LoRARequest
            lora_request = LoRARequest(
                lora_name=lora_info["lora_name"],
                lora_int_id=lora_info["lora_int_id"],
                lora_path=lora_info["lora_path"],
            )
            sampling_params["lora_request"] = lora_request
            logger.info(
                "vllm_omni generate_request: injected lora_request into sampling_params "
                "lora_int_id=%s lora_path=%s",
                lora_info["lora_int_id"],
                lora_info["lora_path"],
            )

        logger.info(
            "vllm_omni generate_request debug: rid=%s payload_keys=%s sampling_param_keys=%s",
            payload.get("rid"),
            sorted(payload.keys()),
            sorted(sampling_params.keys()) if isinstance(sampling_params, dict) else type(sampling_params).__name__,
        )
        output = await self.model.generate_request(payload)
        logger.info(
            "vllm_omni generate_request debug: rid=%s output_type=%s output_keys=%s",
            payload.get("rid"),
            type(output).__name__,
            sorted(output.keys()) if isinstance(output, dict) else None,
        )
        normalized = self._normalize_output(output, sampling_params=sampling_params)
        return normalized

    async def abort_requests(self, request_ids):
        if hasattr(self.model, "abort_request"):
            for req_id in request_ids:
                await self.model.abort_request(req_id)
            return
        if hasattr(self.model, "abort_requests"):
            await self.model.abort_requests(request_ids)

    async def load_states(self, *args, **kwargs):
        if not self.is_model_in_gpu and hasattr(self.model, "load_states"):
            await self.model.load_states()
            self.is_model_in_gpu = True

    async def offload_states(self, include=None, non_blocking=False):
        logger.info(
            "vllm_omni offload_states enter: worker_rank=%s include=%s non_blocking=%s is_model_in_gpu=%s sleep_level=%s colocated=%s has_offload=%s has_sleep=%s",
            getattr(self.worker, "rank", None),
            include,
            non_blocking,
            self.is_model_in_gpu,
            self.sleep_level,
            getattr(self.worker.pipeline_config, "is_actor_infer_colocated", None),
            hasattr(self.model, "offload_states"),
            hasattr(self.model, "sleep"),
        )
        if include is None or OffloadStateType.model_params in include:
            if self.is_model_in_gpu and hasattr(self.model, "sleep") and self.worker.pipeline_config.is_actor_infer_colocated:
                if hasattr(self.model, "offload_states"):
                    await self.model.offload_states(level=self.sleep_level)
                else:
                    await self.model.sleep(level=self.sleep_level)
                self.is_model_in_gpu = False
        gc.collect()
        logger.info(
            "vllm_omni offload_states cache_clear: phase=before_empty_cache worker_rank=%s",
            getattr(self.worker, "rank", None),
        )
        cache_t0 = time.perf_counter()
        current_platform.empty_cache()
        logger.info(
            "vllm_omni offload_states cache_clear: phase=after_empty_cache worker_rank=%s elapsed_s=%.3f",
            getattr(self.worker, "rank", None),
            time.perf_counter() - cache_t0,
        )
        logger.info(
            "vllm_omni offload_states exit: worker_rank=%s is_model_in_gpu=%s",
            getattr(self.worker, "rank", None),
            self.is_model_in_gpu,
        )

    async def process_weights_after_loading(self, *args, **kwargs):
        if hasattr(self.model, "process_weights_after_loading"):
            await self.model.process_weights_after_loading()
            # model_update paths can load/refresh weights on engine side without
            # going through strategy.load_states(); keep strategy state in sync.
            self.is_model_in_gpu = True

    async def setup_collective_group(self, *args, **kwargs):
        if hasattr(self.model, "setup_collective_group"):
            await self.model.setup_collective_group(*args, **kwargs)

    async def broadcast_parameter(self, *args, **kwargs):
        if hasattr(self.model, "broadcast_parameter"):
            await self.model.broadcast_parameter(*args, **kwargs)
            # In colocated model_update flow, parameter broadcast may implicitly
            # wake/load model weights in worker extension.
            self.is_model_in_gpu = True

    async def update_parameter_in_bucket(self, serialized_named_tensors, is_lora=False):
        if hasattr(self.model, "update_parameter_in_bucket"):
            await self.model.update_parameter_in_bucket(serialized_named_tensors, is_lora=is_lora)
            # update_parameter_in_bucket -> worker.load_weights may wake/load
            # weights internally; synchronize strategy-level loaded flag.
            self.is_model_in_gpu = True

    async def add_lora(self, peft_config):
        if hasattr(self.model, "add_lora"):
            lora_info = await self.model.add_lora(peft_config)
            if isinstance(lora_info, dict) and "lora_int_id" in lora_info:
                self._roll_lora_request_info = lora_info
                logger.info(
                    "vllm_omni add_lora: stored lora_request_info=%s",
                    lora_info,
                )
            return lora_info

    # async def wake_up(self):
    #     if hasattr(self.model, "wake_up"):
    #         await self.model.wake_up()
    #     self.is_model_in_gpu = True

    # async def sleep(self):
    #     if hasattr(self.model, "sleep"):
    #         await self.model.sleep(level=self.sleep_level)
    #     self.is_model_in_gpu = False

    async def abort_all_requests(self):
        if hasattr(self.model, "abort_all_requests"):
            await self.model.abort_all_requests()

    async def set_global_steps(self, global_step: int):
        if hasattr(self.model, "set_global_steps"):
            await self.model.set_global_steps(global_step)

    async def set_ema_decay(self, ema_decay: float):
        if hasattr(self.model, "set_ema_decay"):
            await self.model.set_ema_decay(ema_decay)
    
    async def forward_step(
        self,
        batch: DataProto,
    ) -> Dict[str, torch.Tensor]:
        """Diffusion inference forward step, calling forward_step for each diffusion timestep."""
        batch_size = batch.batch.batch_size[0]
        micro_batch_size = batch.meta_info["micro_batch_size"]
        num_microbatches = max(batch_size // micro_batch_size, 1)
        micro_batches = batch.chunk(chunks=num_microbatches)
    
        result = []
        for data in micro_batches:
            model_output = await self._forward_diffusion_model(data)
            result.append({"noise_pred": model_output})
    
        return collate_fn_to_dict_list(result)
    
    async def _forward_diffusion_model(self, data: DataProto) -> torch.Tensor:
        """Stepwise diffusion forward, calling forward_step per timestep.
    
        Processes all_latents [B, K+1, S, C] step by step, dispatching each
        diffusion step to pipeline.forward_step via the vllm_omni engine RPC.
        Returns stacked noise predictions [B, K, S, C].
        """
        all_latents = data.batch["all_latents"]        # [B, K+1, S, C]
        all_timesteps = data.batch["all_timesteps"]    # [B, K]
        prompt_embeds = data.batch["prompt_embeds"]
        prompt_embeds_mask = data.batch["prompt_embeds_mask"]
        negative_prompt_embeds = data.batch.get("negative_prompt_embeds", None)
        negative_prompt_embeds_mask = data.batch.get("negative_prompt_embeds_mask", None)
    
        guidance_scale = float(data.meta_info.get("guidance_scale", 1.0))
    
        if all_latents.ndim != 4:
            raise ValueError(
                f"vllm_omni forward_step invalid all_latents rank: "
                f"got={tuple(all_latents.shape)} expected_rank=4"
            )
        if all_timesteps.ndim != 2:
            raise ValueError(
                f"vllm_omni forward_step invalid all_timesteps rank: "
                f"got={tuple(all_timesteps.shape)} expected_rank=2"
            )
    
        cur_latents = all_latents[:, :-1]  # [B, K, S, C]
        bsz, num_steps, seq_len, channels = cur_latents.shape
    
        if all_timesteps.shape[1] != num_steps:
            raise ValueError(
                f"vllm_omni forward_step mismatched diffusion horizon: "
                f"all_latents.shape[1]={all_latents.shape[1]}, "
                f"all_timesteps.shape[1]={all_timesteps.shape[1]}, expected K+1 vs K"
            )
        if (negative_prompt_embeds is None) != (negative_prompt_embeds_mask is None):
            raise ValueError(
                "vllm_omni forward_step negative prompt tensors must appear together"
            )
        if guidance_scale > 1.0 and negative_prompt_embeds is None:
            raise ValueError(
                "vllm_omni forward_step missing negative prompt tensors while guidance_scale > 1.0"
            )
    
        # Resolve img_shapes from meta_info
        img_shapes = data.meta_info.get("img_shapes")
    
        # Resolve txt_seq_lens
        txt_seq_lens = data.meta_info.get("txt_seq_lens")
        if txt_seq_lens is None:
            txt_seq_lens = prompt_embeds_mask.sum(dim=-1).tolist()
    
        negative_txt_seq_lens = data.meta_info.get("negative_txt_seq_lens")
        if negative_txt_seq_lens is None and isinstance(negative_prompt_embeds_mask, torch.Tensor):
            negative_txt_seq_lens = negative_prompt_embeds_mask.sum(dim=-1).tolist()
    
        # Guidance embedding
        guidance = None
        if data.meta_info.get("guidance_embeds", False):
            guidance = torch.full(
                [1], guidance_scale, dtype=torch.float32, device=cur_latents.device
            )
    
        logger.info(
            "vllm_omni forward_step: bsz=%s num_steps=%s seq_len=%s channels=%s "
            "guidance_scale=%s has_negative=%s worker_rank=%s",
            bsz, num_steps, seq_len, channels, guidance_scale,
            negative_prompt_embeds is not None,
            getattr(self.worker, "rank", None),
        )
    
        step_outputs = []
        for step in range(num_steps):
            step_latents = cur_latents[:, step]        # [B, S, C]
            step_timesteps = all_timesteps[:, step]    # [B]
    
            noise_pred = await self.model.forward_step(
                latents=step_latents,
                timestep=step_timesteps,
                guidance=guidance,
                prompt_embeds_mask=prompt_embeds_mask,
                prompt_embeds=prompt_embeds,
                guidance_scale=guidance_scale,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_txt_seq_lens=negative_txt_seq_lens,
            )
            step_outputs.append(noise_pred)
    
        stacked = torch.stack(step_outputs, dim=1)  # [B, K, S, C]
        logger.debug(
            "vllm_omni forward_step complete: output_shape=%s worker_rank=%s",
            tuple(stacked.shape),
            getattr(self.worker, "rank", None),
        )
        return stacked
