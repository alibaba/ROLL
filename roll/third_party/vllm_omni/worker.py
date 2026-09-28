import fcntl
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional, Tuple

import torch
from torch.distributed.tensor import DTensor
from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension
from vllm_omni.lora.request import LoRARequest

from roll.utils.collective import collective
from roll.utils.constants import DIFFUSION_LORA_ADAPTER_STAGING_DIR
from roll.utils.cuda_ipc_utils import MultiprocessingSerializer
from roll.utils.logging import get_logger
from roll.utils.send_recv_utils import monkey_patch_torch_reductions, named_tensors_from_bucket


logger = get_logger()


def _normalize_peft_config(peft_config: dict) -> dict:
    """Recursively convert sets to sorted lists so the config is JSON-native.
    """
    def _convert(obj):
        if isinstance(obj, set):
            return sorted(obj, key=str)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(x) for x in obj]
        return obj
    return _convert(peft_config)


def _stable_lora_int_id(peft_config: dict) -> int:
    """Derive a deterministic adapter id; expects a normalized peft_config."""
    cfg_str = json.dumps(peft_config, sort_keys=True, default=str)
    return int(hashlib.sha256(cfg_str.encode("utf-8")).hexdigest(), 16) % 0x7FFFFFFF


class _WeightsResidency(Enum):
    """GPU residency of the worker's weights (sleep/wake state machine).

    AWAKE: weights on GPU, forward allowed.
    ASLEEP_LEVEL1: weights swapped to CPU by CuMem; content intact.
    ASLEEP_LEVEL2: GPU buffers discarded; content is garbage until reloaded.
    """

    AWAKE = "awake"
    ASLEEP_LEVEL1 = "asleep_level1"
    ASLEEP_LEVEL2 = "asleep_level2"


@dataclass
class _RollWorkerState:
    """All ROLL-specific mutable state of :class:`VllmOmniColocateWorkerExtension`.

    Centralized so the full state surface is visible at a glance; accessed via
    ``self._roll`` (created lazily on first use).

    LoRA publication transaction (cross-RPC ordering contract):
      1. ``set_global_steps`` provides ``global_step``;
      2. ``broadcast_parameter``/``update_parameter_in_bucket`` (is_lora=True)
         accumulate adapter tensors into ``pending_lora_tensors``;
      3. ``custom_add_lora`` writes them to ``<staging>/step<global_step>/``,
         activates the adapter, clears ``pending_lora_tensors`` and rotates
         ``prev_lora_dir`` (the previous step's directory is deleted).
    """

    # Weights residency state machine (see _WeightsResidency).
    residency: _WeightsResidency = _WeightsResidency.AWAKE

    # LoRA publication transaction (see class docstring above).
    global_step: Optional[int] = None
    pending_lora_tensors: dict = field(default_factory=dict)
    prev_lora_dir: Optional[str] = None
    is_lora: bool = False  # True once custom_add_lora has injected LoRA wrappers

    # EMA blending factor for incoming weight updates; <= 0 disables blending.
    ema_decay: float = 0.0

    # Keepalive only — never read. Holds the CuMem "weights" pool evicted by
    # the first LoRA injection; vLLM keeps pool refs alive to dodge a PyTorch
    # GC bug (pytorch#146431), so we do the same.
    cumem_evicted_pool_keepalive: object = None


class VllmOmniColocateWorkerExtension(CustomPipelineWorkerExtension):
    """ROLL worker extension for vllm_omni diffusion workers in colocate mode.

    All ROLL-specific mutable state lives in :class:`_RollWorkerState` (accessed
    via ``self._roll``); weight sleep/wake transitions are modeled by
    :class:`_WeightsResidency`.
    """

    def __new__(cls, **kwargs):
        return super().__new__(cls)

    @staticmethod
    def _adapt_qwen_image_param_name_for_pipeline(name: str) -> str:
        # Train-side diffusion module is QwenImageTransformer2DModel and emits names like
        # `transformer_blocks.*`, `img_in.*`, etc.
        # Infer-side module is QwenImagePipelineWithLogProb with child module `transformer`,
        # so corresponding runtime names are `transformer.<train_name>`.
        if name.startswith("transformer."):
            return name
        transformer_roots = (
            "transformer_blocks.",
            "img_in.",
            "txt_in.",
            "norm_out.",
            "proj_out.",
            "pos_embed.",
            "time_text_embed.",
            "image_rope_prepare.",
            "modulate_index_prepare.",
            "txt_norm.",
        )
        if name.startswith(transformer_roots):
            name = f"transformer.{name}"
        # vLLM's diffusion model uses `to_out` (direct Linear) while the
        # HuggingFace model uses `to_out.0` (first child of ModuleList).
        # vLLM's check_unexpected_modules uses rsplit(".", 1)[-1] which
        # extracts "0" instead of "to_out", causing validation failure.
        # See: https://github.com/vllm-project/vllm/issues/35734
        name = name.replace(".to_out.0.", ".to_out.")
        return name

    def _adapt_named_params_for_pipeline(self, named_params: list[tuple[str, torch.Tensor]]) -> list[tuple[str, torch.Tensor]]:
        adapted: list[tuple[str, torch.Tensor]] = []
        renamed = 0
        samples: list[tuple[str, str]] = []
        for name, tensor in named_params:
            new_name = self._adapt_qwen_image_param_name_for_pipeline(name)
            if new_name != name:
                renamed += 1
                if len(samples) < 5:
                    samples.append((name, new_name))
            adapted.append((new_name, tensor))

        if renamed > 0:
            logger.info(
                "vllm_omni worker control: op=adapt_weight_names rank=%s renamed=%s total=%s samples=%s",
                self.rank,
                renamed,
                len(named_params),
                samples,
            )
        return adapted

    def _cleanup_stale_lora_adapters(self):
        """Clean up leftover directories under /dev/shm/lora_adapter/ at startup."""
        base_dir = DIFFUSION_LORA_ADAPTER_STAGING_DIR
        if not os.path.isdir(base_dir):
            return
        for entry in os.listdir(base_dir):
            entry_path = os.path.join(base_dir, entry)
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path, ignore_errors=True)
        logger.info("Cleaned up stale LoRA adapter directories in %s", base_dir)

    @property
    def _roll(self) -> _RollWorkerState:
        """ROLL-specific worker state, created on first access (which also
        triggers the one-time stale LoRA adapter cleanup)."""
        state = self.__dict__.get("_roll_state")
        if state is None:
            state = self.__dict__["_roll_state"] = _RollWorkerState()
            self._cleanup_stale_lora_adapters()
        return state

    def _reload_frozen_weights_if_needed(self, prev_residency: _WeightsResidency):
        """After a level-2 wake, reload text_encoder/VAE from disk (they were discarded)."""
        if prev_residency is not _WeightsResidency.ASLEEP_LEVEL2:
            return
        pipeline = self._resolve_pipeline()
        if hasattr(pipeline, "load_non_dit_weights_from_disk"):
            logger.info(
                "vllm_omni worker control: op=reload_frozen_weights rank=%s",
                self.rank,
            )
            pipeline.load_non_dit_weights_from_disk()
        # Reinit RoPE pos_freqs/neg_freqs (complex tensors lost during level-2 sleep)
        if hasattr(pipeline, "reinit_pos_embed"):
            pipeline.reinit_pos_embed()

    def _reset_lora_buffers_after_wake(self, prev_residency: _WeightsResidency):
        """Deactivate adapters after a level-2 wake, whose LoRA buffer content is garbage.
        No-op for level-1 sleep or when already awake.
        """
        if prev_residency is not _WeightsResidency.ASLEEP_LEVEL2:
            return
        lora_manager = getattr(self, "lora_manager", None)
        if lora_manager is None or not getattr(lora_manager, "_lora_modules", None):
            return
        lora_manager._deactivate_all_adapters()
        logger.info("vllm_omni worker control: op=reset_lora_buffers_after_wake rank=%s", self.rank)

    def _ensure_weights_loaded(self):
        if self._roll.residency is _WeightsResidency.AWAKE:
            return
        prev_residency = self._roll.residency
        logger.info("vllm_omni worker wake weights before control op: rank=%s", self.rank)
        super().wake_up(None)
        # Mark awake right after a successful wake_up, BEFORE the reload
        # below: if the reload raises, a stale asleep state here would make
        # sleep() skip the real sleep and the next wake_up() double-map the
        # already mapped memory, crashing CuMemAllocator with CUDA invalid argument.
        self._roll.residency = _WeightsResidency.AWAKE
        self._reset_lora_buffers_after_wake(prev_residency)
        self._reload_frozen_weights_if_needed(prev_residency)

    def sleep(self, level: int = 1):
        residency_before = self._roll.residency
        if residency_before is not _WeightsResidency.AWAKE:
            logger.info(
                "vllm_omni worker control: op=sleep_skip rank=%s level=%s residency_before=%s",
                self.rank,
                level,
                residency_before.value,
            )
            return True
        logger.info(
            "vllm_omni worker control: op=sleep rank=%s level=%s residency_before=%s",
            self.rank,
            level,
            residency_before.value,
        )
        result = super().sleep(level)
        logger.info(
            "vllm_omni worker control: op=sleep_done rank=%s level=%s result_type=%s result=%s",
            self.rank,
            level,
            type(result).__name__,
            result,
        )
        self._roll.residency = (
            _WeightsResidency.ASLEEP_LEVEL2 if level >= 2 else _WeightsResidency.ASLEEP_LEVEL1
        )
        return result

    def wake_up(self, tags: list[str] | None = None):
        residency_before = self._roll.residency
        logger.info(
            "vllm_omni worker control: op=wake_up rank=%s tags=%s residency_before=%s",
            self.rank,
            tags,
            residency_before.value,
        )
        result = super().wake_up(tags)
        if tags is None or "weights" in tags:
            # Flag awake first for the same reason as _ensure_weights_loaded: keep
            # the state in sync with the allocator even if the reload below fails.
            self._roll.residency = _WeightsResidency.AWAKE
            self._reset_lora_buffers_after_wake(residency_before)
            self._reload_frozen_weights_if_needed(residency_before)
        return result

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        logger.info("vllm_omni worker control: op=load_weights rank=%s", self.rank)
        self._ensure_weights_loaded()
        pipeline = self._resolve_pipeline()
        if self._roll.is_lora:
            # LoRA mode (from step 1 on): the transformer layers are wrapped by
            # BaseLayerWithLoRA, so vLLM's native load_weights params_dict is
            # keyed by wrapped names and would KeyError on the incoming base
            # weight names. Route through the LoRA-aware base loader instead.
            base_sd = {}
            for name, weight in weights:
                if name.startswith("transformer."):
                    name = name[len("transformer."):]
                base_sd[name] = weight
            loaded_params = pipeline.load_transformer_base_weights(base_sd)
            logger.info(
                "vllm_omni worker control: op=load_base_weights_lora_aware rank=%s loaded=%s",
                self.rank, len(loaded_params),
            )
            return None
        result = super().load_weights(weights)
        return result

    def load_states(self):
        # Diffusion worker uses wake_up/sleep for load/offload semantics.
        if self._roll.residency is _WeightsResidency.AWAKE:
            return True
        logger.info("vllm_omni worker control: op=load_states rank=%s", self.rank)
        return self.wake_up(None)

    def offload_states(self, level: int = 1):
        logger.info("vllm_omni worker control: op=offload_states rank=%s level=%s", self.rank, level)
        return self.sleep(level)

    def setup_collective_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        group_rank = self.rank + rank_offset
        collective.init_collective_group(
            world_size,
            rank=group_rank,
            backend=backend,
            group_name=group_name,
            master_addr=master_address,
            master_port=master_port,
        )
        logger.info("vllm_omni setup_collective_group: %s rank=%s world_size=%s", group_name, group_rank, world_size)

    def broadcast_parameter(self, names, dtypes, shapes, group_name, is_lora: bool = False):
        logger.info(
            "vllm_omni worker control: op=broadcast_parameter rank=%s group=%s num_tensors=%s is_lora=%s",
            self.rank,
            group_name,
            len(names),
            is_lora,
        )
        weights_and_handles: list[tuple[str, torch.Tensor, object]] = []
        for name, dtype, shape in zip(names, dtypes, shapes):
            target_dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
            weight = torch.empty(shape, dtype=target_dtype, device=self.device)
            handle = collective.broadcast(tensor=weight, src_rank=0, group_name=group_name, async_op=True)
            weights_and_handles.append((name, weight, handle))

        def _weights_iter() -> Iterable[Tuple[str, torch.Tensor]]:
            for name, weight, handle in weights_and_handles:
                handle.wait()
                yield name, weight

        if is_lora:
            adapted = [
                (self._adapt_qwen_image_param_name_for_pipeline(name), weight)
                for name, weight in _weights_iter()
            ]
            self._roll.pending_lora_tensors.update(dict(adapted))
            logger.info(
                "Accumulated LoRA tensors: total=%s rank=%s",
                len(self._roll.pending_lora_tensors),
                self.rank,
            )
            return

        self.load_weights(self._apply_ema_to_weights(_weights_iter()))

    def update_parameter_in_bucket(self, serialized_named_tensors, is_lora: bool = False):
        logger.info(
            "vllm_omni worker control: op=update_parameter_in_bucket rank=%s num_buckets=%s is_lora=%s",
            self.rank,
            len(serialized_named_tensors) if hasattr(serialized_named_tensors, "__len__") else None,
            is_lora,
        )
        monkey_patch_torch_reductions()
        bucket_with_meta = MultiprocessingSerializer.deserialize(serialized_named_tensors[self.rank])
        named_params = list(named_tensors_from_bucket(**bucket_with_meta))
        named_params = self._adapt_named_params_for_pipeline(named_params)

        if is_lora:
            self._roll.pending_lora_tensors.update({name: weight for name, weight in named_params})
            logger.info(
                "Accumulated LoRA tensors: total=%s rank=%s",
                len(self._roll.pending_lora_tensors),
                self.rank,
            )
            return

        self.load_weights(self._apply_ema_to_weights(named_params))

    def process_weights_after_loading(self):
        # Diffusion worker loads weights directly into pipeline; no extra post-hook required for now.
        logger.info("vllm_omni worker control: op=process_weights_after_loading rank=%s", self.rank)
        return None

    def set_global_steps(self, global_step: int):
        logger.info("vllm_omni worker control: op=set_global_steps rank=%s global_step=%s", self.rank, global_step)
        self._roll.global_step = int(global_step)
        return None

    def set_ema_decay(self, ema_decay: float):
        """Set EMA decay factor for subsequent weight updates.

        When ``ema_decay > 0``, :meth:`broadcast_parameter` and
        :meth:`update_parameter_in_bucket` will blend received weights with
        the current model weights:

            θ_infer = decay * θ_infer + (1 - decay) * θ_train
        """
        logger.info("vllm_omni worker control: op=set_ema_decay rank=%s ema_decay=%s", self.rank, ema_decay)
        self._roll.ema_decay = float(ema_decay)
        return None

    def _resolve_pipeline(self) -> torch.nn.Module:
        """Locate the diffusion pipeline ``nn.Module`` from the worker hierarchy.

        In vllm-omni the pipeline lives at ``self.model_runner.pipeline``
        (see ``DiffusionWorker.init_lora_manager``).
        """
        for attr_chain in ("model_runner.pipeline", "pipeline", "worker.pipeline", "model_runner.model"):
            obj = self
            try:
                for part in attr_chain.split("."):
                    obj = getattr(obj, part)
                if isinstance(obj, torch.nn.Module):
                    return obj
            except AttributeError:
                continue
        raise RuntimeError("Cannot locate diffusion pipeline model on vllm_omni worker.")

    def forward_step(self, **kwargs):
        """Single-step diffusion forward via pipeline.forward_step."""
        self._ensure_weights_loaded()
        pipeline = self._resolve_pipeline()
        return pipeline.forward_step(**kwargs)

    def _apply_ema_to_weights(
        self,
        named_weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        """Wrap *named_weights* so each tensor is EMA-blended with the
        current pipeline parameter before being handed to ``load_weights``.

        If ``ema_decay`` ≤ 0 (the default), the original weights are
        yielded unchanged (zero overhead).
        """
        ema_decay = self._roll.ema_decay
        if ema_decay <= 0.0:
            yield from named_weights
            return

        try:
            pipeline = self._resolve_pipeline()
            param_lookup = dict(pipeline.named_parameters())
        except Exception:
            logger.warning("Cannot resolve pipeline for EMA blending, skipping")
            yield from named_weights
            return

        for name, new_weight in named_weights:
            adapted_name = self._adapt_qwen_image_param_name_for_pipeline(name)
            current_param = param_lookup.get(adapted_name) or param_lookup.get(name)
            if current_param is not None:
                cur = current_param.data
                # FSDP2 wraps parameters as DTensor; gather the full tensor before blending.
                if isinstance(cur, DTensor):
                    cur = cur.full_tensor()
                new_w = new_weight.to(device=cur.device, dtype=cur.dtype)
                # θ_infer = decay * θ_infer + (1 - decay) * θ_train
                blended = torch.lerp(new_w, cur, ema_decay)
                yield name, blended.to(device=new_weight.device, dtype=new_weight.dtype)
            else:
                yield name, new_weight

    def _save_lora_adapter_to_disk(self, lora_dir, peft_config, config_json):
        """Atomically write adapter_config.json + adapter_model.safetensors.

        Uses an fcntl.flock exclusive lock to protect concurrent writes from
        multiple workers on the same node. Uses temp file + os.rename for
        crash safety.
        """
        from safetensors.torch import save_file as safetensors_save_file

        os.makedirs(lora_dir, exist_ok=True)
        lock_path = os.path.join(lora_dir, ".lock")
        weight_file = os.path.join(lora_dir, "adapter_model.safetensors")
        config_file = os.path.join(lora_dir, "adapter_config.json")

        with open(lock_path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)

            if os.path.exists(weight_file):
                logger.info("LoRA adapter already on disk (written by another worker): %s", lora_dir)
                return

            with open(config_file, "w") as cf:
                cf.write(config_json)

            tensors = self._roll.pending_lora_tensors
            if not tensors:
                raise ValueError("No LoRA tensors to save. update_parameter_in_bucket must be called first.")

            # Log sample tensor names for debugging vLLM key format issues
            to_out_keys = [k for k in tensors if "to_out" in k]
            sample_keys = list(tensors.keys())[:10]
            logger.info(
                "LoRA tensor keys before save: total=%s rank=%s "
                "sample_keys=%s to_out_keys_count=%s to_out_sample=%s",
                len(tensors), self.rank, sample_keys,
                len(to_out_keys), to_out_keys[:4],
            )

            tmp_file = weight_file + ".tmp"
            safetensors_save_file(tensors, tmp_file)
            os.rename(tmp_file, weight_file)

            logger.info(
                "Saved LoRA adapter to disk: dir=%s num_tensors=%s rank=%s",
                lora_dir, len(tensors), self.rank,
            )

    def custom_add_lora(self, peft_config: dict) -> dict:
        # Normalize once at the entry (set → sorted list) so that hashing and
        # the on-disk adapter_config.json share the same JSON-native dict.
        peft_config = _normalize_peft_config(peft_config)

        # Set the LoRA flag BEFORE _ensure_weights_loaded(): it routes all
        # subsequent base weight loads (load_weights) through the LoRA-aware
        # loader, since vLLM's native path breaks once BaseLayerWithLoRA
        # wrappers exist. The transformer base itself is restored via the
        # model_update base weight stream, not from disk.
        self._roll.is_lora = True

        self._ensure_weights_loaded()

        if self._roll.global_step is None:
            raise ValueError("set_global_steps must be called before custom_add_lora")
        lora_dir = os.path.join(
            DIFFUSION_LORA_ADAPTER_STAGING_DIR,
            f"step{self._roll.global_step}",
        )

        peft_config_with_type = dict(peft_config)
        peft_config_with_type.setdefault("peft_type", "LORA")
        peft_config_with_type.setdefault("base_model_name_or_path", "")
        config_json = json.dumps(peft_config_with_type, indent=2, default=str)

        self._save_lora_adapter_to_disk(lora_dir, peft_config, config_json)

        lora_int_id = _stable_lora_int_id(peft_config)
        if hasattr(self, "remove_lora"):
            self.remove_lora(lora_int_id)
        lora_request = LoRARequest(
            lora_name=str(lora_int_id),
            lora_int_id=lora_int_id,
            lora_path=lora_dir,
        )
        result = self._add_lora_in_weights_pool(lora_request)

        prev_lora_dir = self._roll.prev_lora_dir
        if prev_lora_dir and prev_lora_dir != lora_dir:
            shutil.rmtree(prev_lora_dir, ignore_errors=True)
            logger.info("Cleaned up previous LoRA adapter: %s", prev_lora_dir)
        self._roll.prev_lora_dir = lora_dir

        self._roll.pending_lora_tensors = {}

        return {
            "lora_int_id": lora_int_id,
            "lora_path": lora_dir,
            "lora_name": str(lora_int_id),
        }

    def _add_lora_in_weights_pool(self, lora_request: LoRARequest):
        """Run add_lora inside the CuMem "weights" pool so the LoRA buffers
        are discarded on sleep()
        """
        enable_sleep = getattr(getattr(self, "od_config", None), "enable_sleep_mode", False)
        first_injection = not getattr(getattr(self, "lora_manager", None), "_lora_modules", None)
        if not enable_sleep or not first_injection:
            # Later calls reuse the existing wrappers and allocate no GPU
            # memory, so entering the pool again would only churn MemPools.
            return self.add_lora(lora_request)
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        # use_memory_pool(tag) creates a new MemPool and overwrites
        # allocator_and_pools[tag]; vLLM keeps those refs alive to dodge a
        # PyTorch GC bug (pytorch#146431), so keep the replaced entry (the
        # base-weights pool from load_model) alive ourselves.
        prev_pool = allocator.allocator_and_pools.get("weights")
        with allocator.use_memory_pool(tag="weights"):
            result = self.add_lora(lora_request)
        if prev_pool is not None:
            self._roll.cumem_evicted_pool_keepalive = prev_pool
        logger.info(
            "vllm_omni worker control: op=add_lora_in_weights_pool rank=%s (LoRA buffers tagged for sleep)",
            self.rank,
        )
        return result

    def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done: bool = False, use_shm: bool = False):
        # ROLL currently uses bucketed RPC-based model sync for vllm_omni.
        raise NotImplementedError(
            "update_weights_from_ipc is not wired in ROLL vllm_omni path yet. "
            "Use update_parameter_in_bucket/broadcast_parameter based sync."
        )

    def _update_weights(self, weights: list[tuple[str, torch.Tensor]], peft_config: dict, base_sync_done: bool):
        if peft_config and base_sync_done:
            return self.custom_add_lora(peft_config)
        self.load_weights(weights)
        return True

    def _get_zmq_handle(self) -> str:
        return f"ipc:///tmp/roll-vllm-omni-zmq-rank-{self.rank}.sock"
