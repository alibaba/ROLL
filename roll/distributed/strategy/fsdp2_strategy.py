import contextlib
import os
import random
from collections import defaultdict
from contextlib import nullcontext
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import ray
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from codetiming import Timer
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import optim
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm
from transformers import AutoConfig, get_scheduler, set_seed

from roll.datasets.collator import collate_fn_to_dict_list
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import (
    clear_fsdp2_init_context,
    default_processor_provider,
    default_tokenizer_provider,
    set_fsdp2_init_context,
)
from roll.platforms import current_platform
from roll.third_party.fsdp2.model_update import FSDP2WeightUpdater
from roll.utils.checkpoint_manager import CheckpointManager, download_model
from roll.utils.collective import collective
from roll.utils.context_parallel import get_ulysses_group, set_upg_manager
from roll.utils.context_parallel.autograd_gather import ulysses_gather
from roll.utils.context_parallel.rmpad_ulysses import (
    gather_outputs_and_unpad,
    ulysses_pad_and_slice_inputs,
    ulysses_pad_inputs,
)
from roll.utils.fsdp_utils import (
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    get_init_weight_context_manager,
    get_shard_placement_fn,
)
from roll.utils.functionals import append_to_dict, log_probs_from_logits
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType

logger = get_logger()


def _parse_dtype(dtype):
    if dtype is None:
        return None

    if isinstance(dtype, torch.dtype):
        return dtype

    if isinstance(dtype, str):
        dtype_lower = dtype.lower()
        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "half": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
            "float": torch.float32,
            "fp64": torch.float64,
            "float64": torch.float64,
        }

        if dtype_lower in dtype_map:
            return dtype_map[dtype_lower]
        else:
            if hasattr(torch, dtype):
                return getattr(torch, dtype)
            else:
                raise ValueError(
                    f"Unsupported dtype string: '{dtype}'. " f"Supported values: {list(dtype_map.keys())}"
                )

    return dtype


def create_device_mesh_with_ulysses(world_size: int, fsdp_size: int):
    """
    Create device mesh for FSDP.
    """

    # Default to global sharding (1D mesh) if fsdp_size is not explicitly set for HSDP
    if fsdp_size <= 1 or fsdp_size >= world_size:
        mesh_shape = (world_size,)
        mesh_dim_names = ["fsdp"]
    else:
        # HSDP Case: Shard within fsdp_size group, Replicate across the rest
        # PyTorch fully_shard shards on the LAST dimension (inner) and replicates on outer dimensions.
        # Example: world=8, fsdp=4. We want 2 replicas of 4-way sharding.
        # Mesh: (2, 4). Replicate on dim 0 (2), Shard on dim 1 (4).
        ddp_size = world_size // fsdp_size
        mesh_shape = (ddp_size, fsdp_size)
        mesh_dim_names = ["ddp", "fsdp"]

    return init_device_mesh(
        current_platform.device_type,
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_dim_names,
    )


class FSDP2StrategyBase(InferenceStrategy):
    def __init__(self, worker: Worker):
        super().__init__(worker)
        self.cpu_offload_enabled: bool = False
        if not hasattr(self, "checkpoint_manager") or self.checkpoint_manager is None:
            checkpoint_config = getattr(self.worker_config, "checkpoint_config", None)
            self.checkpoint_manager = CheckpointManager(checkpoint_config=checkpoint_config)
        self._model_update_device_buffer: Optional[torch.Tensor] = None
        self.weight_updaters = {}

    def _get_dp_rank(self) -> int:
        rank_info = getattr(self.worker, "rank_info", None)
        if rank_info is not None and getattr(rank_info, "dp_rank", None) is not None:
            return rank_info.dp_rank
        return dist.get_rank()

    def _build_checkpoint_paths(
        self,
        base_dir: str,
        world_size: Optional[int] = None,
        dp_rank: Optional[int] = None,
    ):
        world_size = world_size or dist.get_world_size()
        dp_rank = dp_rank if dp_rank is not None else self._get_dp_rank()
        suffix = f"world_size_{world_size}_rank_{dp_rank}.pt"
        model_path = os.path.join(base_dir, f"model_{suffix}")
        optim_path = os.path.join(base_dir, f"optim_{suffix}")
        extra_path = os.path.join(base_dir, f"extra_state_{suffix}")
        return model_path, optim_path, extra_path

    @staticmethod
    def _get_dcp_checkpoint_dir(base_dir: str) -> str:
        return os.path.join(base_dir, "dcp")

    def _get_dcp_state_dict_options(self, full_state_dict: bool = False) -> StateDictOptions:
        # Always use cpu_offload=True for DCP to avoid OOM during load/save
        # independent of training offload configuration.
        return StateDictOptions(
            full_state_dict=full_state_dict,
            cpu_offload=True,
        )

    def _save_checkpoint_with_dcp(self, checkpoint_dir: str, is_last_step: bool):
        state_dict = {
            **self.model.state_dict(),
        }

        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None:
            state_dict["optimizer"] = optimizer

        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None:
            state_dict["scheduler"] = scheduler

        rng_state = self.get_rng_state()
        state_dict["rng_state"] = rng_state

        if not self.async_save_strategy or is_last_step:
            if self.checkpoint_future is not None:
                self.checkpoint_future.result()
                self.checkpoint_future = None
            dcp.save(
                state_dict=state_dict,
                checkpoint_id=checkpoint_dir,
            )
        else:
            if self.checkpoint_future is not None:
                self.checkpoint_future.result()
            self.checkpoint_future = dcp.async_save(
                state_dict=state_dict,
                checkpoint_id=checkpoint_dir,
            )

    def _load_checkpoint_with_dcp(self, checkpoint_dir: str):
        state_dict = {
            **self.model.state_dict(),
        }

        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None:
            state_dict["optimizer"] = optimizer

        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None:
            state_dict["scheduler"] = scheduler

        state_dict["rng_state"] = {}

        dcp.load(
            state_dict=state_dict,
            checkpoint_id=checkpoint_dir,
        )

        if "rng_state" in state_dict and state_dict["rng_state"]:
            self.load_rng_state(state_dict["rng_state"])

        info = self.model.load_state_dict(state_dict, strict=False)
        missing_keys = info.missing_keys
        unexpected_keys = info.unexpected_keys

        filtered_unexpected_keys = [
            key for key in unexpected_keys if key not in ("optimizer", "scheduler", "rng_state")
        ]

        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        if filtered_unexpected_keys:
            logger.warning(f"Unexpected keys: {filtered_unexpected_keys}")

    def _load_checkpoint_from_legacy_shards(
        self,
        load_dir: str,
        world_size: int,
        dp_rank: int,
        optimizer,
    ):
        model_path, optim_path, _ = self._build_checkpoint_paths(
            load_dir,
            world_size=world_size,
            dp_rank=dp_rank,
        )

        model_state_dict = self._load_torch_file(model_path, required=True)
        optimizer_state_dict = self._load_torch_file(optim_path, required=optimizer is not None)

        if not model_state_dict:
            logger.warning("Empty model state dict loaded from %s, skipping model restore", model_path)
            return

        first_param = next(iter(model_state_dict.values()))
        if isinstance(first_param, DTensor):
            self.model.load_state_dict(model_state_dict, assign=True)
        else:
            meta_sharded_sd = self.model.state_dict()
            sharded_sd = {}
            for param_name, full_tensor in model_state_dict.items():
                if param_name in meta_sharded_sd:
                    sharded_meta_param = meta_sharded_sd[param_name]
                    if isinstance(sharded_meta_param, DTensor):
                        # Respect the DTensor's device (CPU for offload_policy=True)
                        target_device = sharded_meta_param.device
                        sharded_tensor = distribute_tensor(
                            full_tensor.to(target_device),
                            sharded_meta_param.device_mesh,
                            sharded_meta_param.placements,
                        )
                        sharded_sd[param_name] = torch.nn.Parameter(sharded_tensor)
                    else:
                        sharded_sd[param_name] = torch.nn.Parameter(full_tensor)
                else:
                    sharded_sd[param_name] = torch.nn.Parameter(full_tensor)
            self.model.load_state_dict(sharded_sd, assign=True)

        if optimizer_state_dict is not None and optimizer is not None:
            optimizer.load_state_dict(optimizer_state_dict)

    def _load_extra_state_dict(self, base_dir: str, world_size: int, dp_rank: int):
        _, _, extra_state_path = self._build_checkpoint_paths(
            base_dir,
            world_size=world_size,
            dp_rank=dp_rank,
        )

        if os.path.exists(extra_state_path):
            return torch.load(extra_state_path, map_location="cpu", weights_only=False)

        return {}

    def save_checkpoint(self, save_dir, global_step, ckpt_id, tag="checkpoint", local_state_path=None, **kwargs):
        """
        Save the sharded (DTensor) checkpoint as well as HF-compatible full weights.
        In FSDP, all ranks should coordinate:
        1. All ranks save their sharded checkpoints (model/optim/extra state) to the same directory
        2. Only rank 0 saves the full HuggingFace-compatible model
        """
        logger.info(f"save_dir: {save_dir}")
        if local_state_path is None:
            local_state_path = save_dir

        is_last_step = kwargs.get("is_last_step", None)

        if is_last_step is None:
            if self.worker_config.training_args.max_steps is not None:
                is_last_step = global_step == self.worker_config.training_args.max_steps - 1
            else:
                # If max_steps is not set, we consider all steps as the last step in case of hang for async saving
                is_last_step = True

        # PumpkinComment:
        # Why we need to wait here and also in save_dcp? Because if not, easy to hang in LoRA
        # Not sure why, but keep the logic here for now.
        if self.async_save_strategy and self.checkpoint_future is not None:
            logger.info("Waiting for previous async checkpoint to complete...")
            self.checkpoint_future.result()
            self.checkpoint_future = None

        os.makedirs(save_dir, exist_ok=True)

        with Timer("load", logger=None) as load_timer:
            self.load_states()

        dcp_checkpoint_dir = self._get_dcp_checkpoint_dir(save_dir)
        os.makedirs(dcp_checkpoint_dir, exist_ok=True)

        with Timer("dcp_save", logger=None) as dcp_timer:
            self._save_checkpoint_with_dcp(checkpoint_dir=dcp_checkpoint_dir, is_last_step=is_last_step)

        # PumpkinComment:
        # If DCP save is async, uploading (which may copy+delete the local dir) must not start
        # until the async save has fully finished writing checkpoint shards.
        dcp_save_future = self.checkpoint_future if (self.async_save_strategy and not is_last_step) else None

        with Timer("hf_save", logger=None) as hf_timer:
            full_state_options = self._get_dcp_state_dict_options(full_state_dict=True)
            full_model_state = get_model_state_dict(
                model=self.model,
                options=full_state_options,
            )

            if dist.get_rank() == 0:
                underlying_model = self.unwrap_model()
                underlying_model.save_pretrained(
                    save_dir,
                    state_dict=full_model_state,
                    safe_serialization=True,
                )
                self.tokenizer.save_pretrained(save_dir)
                if getattr(self, "processor", None):
                    self.processor.save_pretrained(save_dir)

        checkpoint_config = getattr(self.worker_config, "checkpoint_config", None) or {}
        async_upload = checkpoint_config.get("async_upload", True)
        keep_local_file = checkpoint_config.get("keep_local_file", False)
        if dcp_save_future is not None and async_upload:

            def _on_dcp_done(fut):
                print("[DEBUG] Enter Callback for DCP save")
                try:
                    fut.result()
                except Exception:
                    logger.error(f"Async DCP save failed for ckpt_id={ckpt_id}, skip upload.")
                    return

                self.thread_executor.submit(
                    self.checkpoint_manager.upload,
                    ckpt_id=ckpt_id,
                    local_state_path=local_state_path,
                    keep_local_file=keep_local_file,
                )

            dcp_save_future.add_done_callback(_on_dcp_done)
        else:
            # If async_upload=False, block until DCP async save completes, then upload.
            if dcp_save_future is not None:
                dcp_save_future.result()

            if async_upload:
                self.thread_executor.submit(
                    self.checkpoint_manager.upload,
                    ckpt_id=ckpt_id,
                    local_state_path=local_state_path,
                    keep_local_file=keep_local_file,
                )
            else:
                self.checkpoint_manager.upload(
                    ckpt_id=ckpt_id,
                    local_state_path=local_state_path,
                    keep_local_file=keep_local_file,
                )

        return {
            "load": load_timer.last,
            "dcp_save": dcp_timer.last,
            "hf_save": hf_timer.last,
        }

    def _load_torch_file(self, path: str, required: bool = True):
        if os.path.exists(path):
            return torch.load(path, map_location="cpu", weights_only=False)
        if required:
            raise FileNotFoundError(f"Missing checkpoint shard: {path}")
        logger.warning(f"Optional checkpoint shard missing, skipping: {path}")
        return None

    def load_checkpoint(self, load_dir, tag="checkpoint", **kwargs):
        """
        Load checkpoint from a shared directory where all ranks' sharded checkpoints are stored.

        In FSDP, synchronize the load_dir across all ranks to ensure they load from the same location.
        """
        logger.info(f"load_dir: {load_dir}")

        dcp_checkpoint_dir = self._get_dcp_checkpoint_dir(load_dir)
        used_dcp = False
        if os.path.isdir(dcp_checkpoint_dir):
            if dist.is_initialized():
                dist.barrier()

            self._load_checkpoint_with_dcp(
                checkpoint_dir=dcp_checkpoint_dir,
            )
            used_dcp = True
            logger.info(f"Loaded DCP checkpoint from {dcp_checkpoint_dir}")
            if dist.is_initialized():
                dist.barrier()
            return

    @staticmethod
    def get_rng_state():
        rng_state = {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }
        return rng_state

    @staticmethod
    def load_rng_state(rng_state):
        torch.set_rng_state(rng_state["cpu"])
        torch.cuda.set_rng_state(rng_state["cuda"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])

    def _copy_weight_to_param(self, param: torch.nn.Parameter, weight: torch.Tensor):
        """
        Copy a full (replicated) tensor onto a possibly-sharded FSDP2 parameter.
        Handles DTensor placement to keep shards consistent across ranks.
        """

        target = param.data if hasattr(param, "data") else param
        source = weight.data if hasattr(weight, "data") else weight
        source = source.detach()

        if isinstance(source, DTensor):
            if isinstance(target, DTensor):
                same_mesh = source.device_mesh == target.device_mesh
                same_place = source.placements == target.placements
                if same_mesh and same_place:
                    target.copy_(source)
                    return
            source = source.full_tensor()

        if isinstance(target, DTensor):
            sharded = distribute_tensor(
                source.to(target.device),
                target.device_mesh,
                target.placements,
            )
            target.copy_(sharded)
        else:
            target.copy_(source.to(target.device))

    def _gather_full_tensor(self, param: torch.nn.Parameter) -> torch.Tensor:
        tensor = param.data if hasattr(param, "data") else param
        if isinstance(tensor, DTensor):
            original_device = tensor.device
            if original_device.type == "cpu" and current_platform.device_type == "cuda" and torch.cuda.is_available():
                tensor = tensor.to(current_platform.device_type)
            tensor = tensor.full_tensor()
            if original_device.type == "cpu":
                tensor = tensor.cpu()
            # full_tensor() already returns a new tensor from all-gather
            return tensor.detach()
        # For non-DTensor (e.g., LoRA params that aren't sharded), we need to clone
        # to avoid modifying the original parameter during bucket packing
        return tensor.detach().clone()

    def _move_optimizer_states(self, device: torch.device, non_blocking: bool = False):
        optimizer = getattr(self, "optimizer", None)
        if optimizer is None:
            return
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device, non_blocking=non_blocking)

    def _get_broadcast_tensor(self, weight_cpu: torch.Tensor) -> torch.Tensor:
        """
        Reuse buffer to avoid allocating new memory.
        """
        if current_platform.device_type == "cpu":
            return weight_cpu
        numel = weight_cpu.numel()
        dtype = weight_cpu.dtype
        buffer = self._model_update_device_buffer
        if buffer is None or buffer.numel() < numel or buffer.dtype != dtype:
            buffer = torch.empty(numel, dtype=dtype, device=current_platform.device_type)
            self._model_update_device_buffer = buffer
        device_view = buffer[:numel].view(weight_cpu.shape)
        device_view.copy_(weight_cpu, non_blocking=True)
        return device_view

    def get_data_input(self, batch: DataProto):
        """Ensure Ulysses/context-parallel ranks receive identical data."""

        def broadcast_obj(obj, group):
            obj_list = [obj if dist.get_rank(group) == 0 else None]
            src_rank = dist.get_process_group_ranks(group)[0]
            dist.broadcast_object_list(obj_list, src=src_rank, group=group)
            return obj_list[0]

        if getattr(self.worker.rank_info, "cp_size", 1) <= 1:
            return batch

        broadcast_non_tensor_batch = batch.meta_info.get("_broadcast_non_tensor_batch", False)
        if broadcast_non_tensor_batch:
            tmp_batch = broadcast_obj(batch, get_ulysses_group())
            batch.batch = tmp_batch.batch
            batch.non_tensor_batch = tmp_batch.non_tensor_batch
        else:
            batch.batch = broadcast_obj(batch.batch, get_ulysses_group())
        return batch

    def _prepare_fsdp2_model(
        self,
        model_provider,
        *,
        is_trainable: bool,
        default_model_dtype: torch.dtype,
        warmup_collective: bool = False,
    ):

        set_seed(seed=self.worker.pipeline_config.seed)

        if not torch.distributed.is_initialized():
            if current_platform.device_type != "cpu":
                backends_str = f"cpu:gloo,{current_platform.device_type}:{current_platform.communication_backend}"
            else:
                backends_str = current_platform.communication_backend
            torch.distributed.init_process_group(backend=backends_str)

        if warmup_collective:
            dist.all_reduce(torch.zeros(1).to(current_platform.device_type))

        if self.worker_config.strategy_args.strategy_config.get("apply_tiled_mlp", False):
            from roll.third_party.fsdp2.tiled_mlp import apply_tiled_mlp_monkey_patch

            apply_tiled_mlp_monkey_patch(
                num_shards=self.worker_config.strategy_args.strategy_config.get("tiled_num_shards", 4),
                model_type=self.worker_config.strategy_args.strategy_config.get("model_type", None),
            )

        world_size = torch.distributed.get_world_size()
        global_rank = torch.distributed.get_rank()

        cp_size = self.worker_config.model_args.ulysses_size
        if cp_size > 1:
            if current_platform.apply_ulysses_patch() is not None:
                set_upg_manager(
                    ulysses_size=cp_size,
                    rank=global_rank,
                    world_size=world_size,
                )
            else:
                cp_size = 1

        if self.worker_config.model_args.ulysses_size != cp_size:
            # PumpkinComment: Fallback if something goes wrong with CP
            logger.warning(
                f"ulysses_size in config ({self.worker_config.model_args.ulysses_size}) is not equal to cp_size ({cp_size}), using cp_size instead"
            )
            self.worker_config.strategy_args.strategy_config["fsdp_size"] = (
                self.worker_config.strategy_args.strategy_config["fsdp_size"]
                * self.worker_config.model_args.ulysses_size
            )
            self.worker_config.model_args.ulysses_size = cp_size

        self.worker.rank_info.dp_rank = global_rank // cp_size
        self.worker.rank_info.dp_size = world_size // cp_size
        self.worker.rank_info.cp_rank = global_rank % cp_size
        self.worker.rank_info.cp_size = cp_size

        if cp_size > 1 and global_rank == 0:
            logger.debug(f"FSDP2 CP(Ulysses) enabled: cp_size={cp_size}, dp_size={self.worker.rank_info.dp_size}")

        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.processor = default_processor_provider(model_args=self.worker_config.model_args)

        torch_dtype = self.worker_config.strategy_args.strategy_config.get("param_dtype", default_model_dtype)
        torch_dtype = _parse_dtype(torch_dtype)
        self.worker_config.model_args.compute_dtype = torch_dtype

        fsdp_size = self.worker_config.strategy_args.strategy_config.get("fsdp_size", 1)
        if cp_size > 1 and (fsdp_size <= 1 or fsdp_size >= world_size):
            fsdp_size = world_size // cp_size
            self.worker_config.strategy_args.strategy_config["fsdp_size"] = fsdp_size
            if global_rank == 0:
                logger.info(f"CP enabled: auto-setting fsdp_size={fsdp_size} so ddp_size==cp_size for hybrid sharding")
        elif fsdp_size != world_size:
            logger.warning(f"fsdp_size {fsdp_size} is not equal to world_size {world_size}, using world_size instead")
            fsdp_size = world_size

        self.device_mesh = create_device_mesh_with_ulysses(world_size=world_size, fsdp_size=fsdp_size)

        model_name_or_path = download_model(self.worker_config.model_args.model_name_or_path)
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            **self.worker_config.model_args.model_config_kwargs,
        )

        self._validate_ulysses_compat(config, cp_size)

        use_meta_tensor = not getattr(config, "tie_word_embeddings", False)
        init_context = get_init_weight_context_manager(
            use_meta_tensor=use_meta_tensor,
            mesh=self.device_mesh,
        )

        set_fsdp2_init_context(init_context)
        try:
            model = model_provider(
                tokenizer=self.tokenizer,
                model_args=self.worker_config.model_args,
                is_trainable=is_trainable,
            )
        finally:
            clear_fsdp2_init_context()

        self.is_lora = self.worker_config.model_args.lora_target is not None

        return model, torch_dtype, cp_size

    @staticmethod
    def _validate_ulysses_compat(config, cp_size: int):
        try:
            num_attention_heads, num_key_value_heads = (
                config.num_attention_heads,
                config.num_key_value_heads,
            )
        except AttributeError:
            num_attention_heads, num_key_value_heads = (
                config.text_config.num_attention_heads,
                config.text_config.num_key_value_heads,
            )

        assert (
            num_attention_heads % cp_size == 0
        ), f"num_attention_heads {num_attention_heads} must be divisible by ulysses_size {cp_size}"
        assert num_key_value_heads % cp_size == 0 or cp_size % num_key_value_heads == 0, (
            f"num_key_value_heads {num_key_value_heads} must be divisible by ulysses_size "
            f"{cp_size}or vise versa. Upon ulysses_size % num_key_value_heads == 0,"
            f"kv heads are repeated to ensure correctness."
        )

    def load_states(self, include=None, non_blocking=False):
        if not self.cpu_offload_enabled:
            if include is None or OffloadStateType.model_params in include:
                device = current_platform.current_device()
                self.model.to(device, non_blocking=non_blocking)
            # When cpu_offload is disabled, always keep optimizer states on GPU
            self._move_optimizer_states(current_platform.current_device(), non_blocking=non_blocking)
        else:
            # When cpu_offload is enabled, only load optimizer states if requested
            if include is None or OffloadStateType.optimizer_states in include:
                self._move_optimizer_states(
                    current_platform.current_device(),
                    non_blocking=non_blocking,
                )

    def offload_states(self, include=None, non_blocking=False):
        """ "
        PumpkinComment:

        If CPUOFFloadPolicy is True: Every thing about offload /load model param is built from FSDP2.
        If CPUOFFloadPolicy is False: The model param in on GPU, we need to mvoe the optimizer to GPU as well.

        Therefore, we actually could leave model param. offload/onload logic to FSDP2 during training
        But here, I maintain mannual support and compatible with FSDP2 CPUOFFloadPolicy for other offload logic.
        """
        if not self.cpu_offload_enabled:
            if include is None or OffloadStateType.model_params in include:
                self.model.to("cpu", non_blocking=non_blocking)
                if current_platform.device_type == "cuda":
                    torch.cuda.empty_cache()
            # When cpu_offload is disabled, optimizer states should stay on GPU
            # Only offload optimizer states if cpu_offload is enabled
        else:
            # When cpu_offload is enabled, offload optimizer states
            if include is None or OffloadStateType.optimizer_states in include:
                self._move_optimizer_states(torch.device("cpu"), non_blocking=non_blocking)


class FSDP2InferStrategy(FSDP2StrategyBase):
    strategy_name = "fsdp2_infer"

    def __init__(self, worker: Worker):
        super().__init__(worker)
        self.device_mesh = None
        self.fsdp_config = None

    def initialize(self, model_provider):
        model, torch_dtype, _ = self._prepare_fsdp2_model(
            model_provider,
            is_trainable=False,
            default_model_dtype=torch.bfloat16,
        )

        self.setup_fsdp2_configuration()
        self.initialize_fsdp2_model(model)

        dist.barrier()

    def setup_fsdp2_configuration(self):
        """Setup FSDP-2 configuration"""
        # ckpt strategy
        async_save_strategy = self.worker_config.strategy_args.strategy_config.get("async_save_ckpt", True)
        self.async_save_strategy = async_save_strategy
        if self.async_save_strategy:
            self.checkpoint_future = None

        # Get mixed precision settings from config
        param_dtype = self.worker_config.strategy_args.strategy_config.get("param_dtype", torch.bfloat16)
        reduce_dtype = self.worker_config.strategy_args.strategy_config.get("reduce_dtype", torch.float32)

        # Convert string dtype specifications to torch.dtype
        param_dtype = _parse_dtype(param_dtype)
        reduce_dtype = _parse_dtype(reduce_dtype)
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype

        mixed_precision = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cast_forward_inputs=True,
        )

        # Reshard after forward setting (FSDP2 uses this instead of sharding_strategy)
        # FULL_SHARD: reshard_after_forward=True
        # SHARD_GRAD_OP: reshard_after_forward=False
        # HYBRID_SHARD: reshard_after_forward=True with a 2D device mesh
        # HYBRID_SHARD_ZERO2: reshard_after_forward=False with a 2D device mesh
        # If None, True for submodules, False for root module
        reshard_after_forward = self.worker_config.strategy_args.strategy_config.get("reshard_after_forward", None)

        offload_policy_cfg = self.worker_config.strategy_args.strategy_config.get("offload_policy", False)
        self.cpu_offload_enabled = bool(offload_policy_cfg)
        offload_policy = None
        if self.cpu_offload_enabled:
            offload_policy = CPUOffloadPolicy(
                pin_memory=True,
            )

        # Store configuration for fully_shard()
        self.fsdp_config = {
            "mesh": self.device_mesh,
            "reshard_after_forward": reshard_after_forward,
            "mp_policy": mixed_precision,
            "offload_policy": offload_policy,
            "shard_placement_fn": get_shard_placement_fn(
                fsdp_size=self.worker_config.strategy_args.strategy_config.get("fsdp_size", 1)
            ),
        }

    def initialize_fsdp2_model(self, model):
        offload_policy = self.fsdp_config["offload_policy"]
        full_state = model.state_dict()
        apply_fsdp2(
            model,
            self.fsdp_config,
            self.worker_config.strategy_args.strategy_config,
            self.is_lora,
        )

        fsdp2_load_full_state_dict(
            model,
            full_state,
            self.device_mesh,
            offload_policy,
        )

        self.model = model

    def forward_step(
        self,
        batch: DataProto,
        forward_func: Callable[
            [DataProto, torch.Tensor],
            Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        ],
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        batch_size = batch.batch.batch_size[0]
        micro_batch_size = batch.meta_info["micro_batch_size"]
        num_microbatches = max(batch_size // micro_batch_size, 1)
        micro_batches = batch.chunk(chunks=num_microbatches)

        cp_size = self.worker.rank_info.cp_size
        batch_num_tokens = self._get_batch_num_tokens(batch)
        batch.meta_info['batch_num_tokens'] = {k: v // cp_size for k, v in batch_num_tokens.items()}
        global_valid_tokens = self._get_global_valid_samples(batch)
        batch.meta_info['global_valid_samples'] = {k: v // cp_size for k, v in global_valid_tokens.items()}

        loss_scale = num_microbatches * self.worker.rank_info.dp_size

        disable_adapter = batch.meta_info.get("disable_adapter", False)
        adapter_context = self.unwrap_model().disable_adapter() if disable_adapter else nullcontext()
        losses_reduced = []

        with adapter_context:
            for data in micro_batches:
                with torch.autocast(
                    device_type=current_platform.device_type,
                    dtype=self.param_dtype,
                ):
                    input_ids = data.batch["input_ids"]
                    attention_mask = data.batch["attention_mask"]
                    position_ids = data.batch["position_ids"]
                    forward_args = data.meta_info.get("forward_args", {})
                    if position_ids.dim() == 3:
                        # qwen-vl mrope-style 3D position_ids stored in DataProto as (bsz, C, seqlen)
                        # transpose to (C, bsz, seqlen) for model forward.
                        position_ids = position_ids.transpose(0, 1)  # (bsz, C, seqlen) -> (C, bsz, seqlen)
                    if "multi_modal_inputs" in data.non_tensor_batch:
                        multi_modal_inputs = data.non_tensor_batch["multi_modal_inputs"]
                        multi_modal_data = defaultdict(list)
                        # mm inputs of some samples would be empty to allow text and mm mixed data
                        for sample_mm_inputs in multi_modal_inputs:
                            for key in sample_mm_inputs.keys():
                                multi_modal_data[key].append(sample_mm_inputs[key])
                        for key in multi_modal_data.keys():
                            assert key not in forward_args
                            # DataProto.to('cuda') in upper frame not work for non_tensor_batch
                            forward_args[key] = torch.concat(multi_modal_data[key], dim=0).to(input_ids.device)
                        forward_args.update({"force_vit_image": True})

                    logits = self._fsdp2_forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        forward_args=forward_args,
                    )

                    loss, loss_reduced = forward_func(data, logits)
                    if self.worker_config.apply_loss_scale:
                        loss *= loss_scale
                losses_reduced.append(loss_reduced)

        results = collate_fn_to_dict_list(losses_reduced)
        return results

    def get_feature_on_cp_rank(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
    ):
        """Get features for specific context parallel rank"""
        seqlens_in_batch = input_ids.size(1)
        assert (
            seqlens_in_batch % self.worker.rank_info.cp_size == 0
        ), f"input_length={seqlens_in_batch} not divisible by cp_size={self.worker.rank_info.cp_size}"
        cp_middle_rank_len = seqlens_in_batch // self.worker.rank_info.cp_size
        padded_input_ids = input_ids
        result = {}
        start_index = cp_middle_rank_len * self.worker.rank_info.cp_rank
        end_index = cp_middle_rank_len * (self.worker.rank_info.cp_rank + 1)
        result["input_ids"] = padded_input_ids[:, start_index:end_index]
        if attention_mask is not None:
            result["attention_mask"] = attention_mask[:, start_index:end_index]
        if position_ids is not None:
            if position_ids.dim() == 3:
                result["position_ids"] = position_ids[:, :, start_index:end_index]
            else:
                result["position_ids"] = position_ids[:, start_index:end_index]
        return result

    def _fsdp2_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        forward_args: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        cp_size = self.worker.rank_info.cp_size
        cp_rank = self.worker.rank_info.cp_rank

        # PumpkinComment:
        # - do NOT slice padded tensors first (would reintroduce imbalance)
        # - unpad to token stream, pad-to-multiple-of-cp, slice equally, run model with attn_mask=None
        # - gather outputs and unpad, then pad back to original (bs, seqlen) so downstream remains unchanged
        if cp_size > 1:
            underlying = self.unwrap_model()
            model_type = getattr(getattr(underlying, "config", None), "model_type", "") or ""
            is_vlm = getattr(getattr(underlying, "config", None), "vision_config", None) is not None
            is_supported_vlm = is_vlm and model_type in ("qwen2_5_vl", "qwen3_vl")

            if not is_supported_vlm:
                features = self.get_feature_on_cp_rank(input_ids, attention_mask, position_ids)
                input_ids = features["input_ids"]
                attention_mask = features["attention_mask"]
                position_ids = features["position_ids"]

        # Ensure use_cache is False if not specified (matches HF strategy)
        if "use_cache" not in forward_args:
            forward_args["use_cache"] = False

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **forward_args,
        ).logits

    def generate(self, batch: DataProto, generation_config):
        if self.worker.rank_info.cp_size > 1:
            raise RuntimeError("FSDP2 generate() is not supported with CP(Ulysses) enabled yet. ")
        input_ids = batch.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = batch.batch["attention_mask"]  # left-padded attention_mask

        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **generation_config,
        )

        return output

    def unwrap_model(self):
        if hasattr(self.model, "module"):
            return self.model.module
        return self.model

    def broadcast_parameter(
        self,
        model_update_name,
        src_pp_rank,
        dtype,
        shape,
        parameter_name,
        is_lora=False,
    ):
        if model_update_name not in self.model_update_comm_plan:
            self.model_update_comm_plan[model_update_name] = {}
        if src_pp_rank not in self.model_update_comm_plan[model_update_name]:
            self._setup_collective_group_impl(
                model_update_name=model_update_name,
                comm_plan=None,
                backend=None,
                mode="receiver",
            )
        comm_plan = self.model_update_comm_plan[model_update_name][src_pp_rank]
        weight = torch.empty(shape, dtype=dtype, device=current_platform.device_type)
        collective.broadcast(tensor=weight, src_rank=0, group_name=comm_plan["group_name"])
        param = self.model.get_parameter(parameter_name)
        self._copy_weight_to_param(param, weight)
        del weight

    def update_parameter(
        self,
        model_update_name,
        parameter_name,
        weight,
        ranks_in_worker,
        is_lora: bool = False,
    ):
        # TODO: Update in bucket
        param = self.model.get_parameter(parameter_name)
        self._copy_weight_to_param(param, weight)
        del weight

    def op_compute_log_probs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """
        input_ids [[p, p, r, r, r, 0, 0]] p: prompt, r: response, 0: pad
        response_mask [[0, 0, 1, 1, 1, 0, 0]]
        """
        # Create labels from FULL input_ids (shifted by 1)
        labels: torch.Tensor = input_ids[:, 1:].clone()
        labels[attention_mask[:, 1:] == 0] = 0  # avoid invalid token id

        if self.worker.rank_info.cp_size > 1:
            # For CP: slice the shifted labels to match the sharded logits
            # logits are sharded across sequence dimension by Ulysses
            labels = torch.cat([labels, torch.zeros_like(labels[:, :1])], dim=1)
            labels = self.get_feature_on_cp_rank(labels)["input_ids"]

            # Compute log_probs for this CP rank
            log_probs = log_probs_from_logits(logits, labels)

            log_probs = ulysses_gather(
                log_probs,
                gather_dim=1,
                group=get_ulysses_group(),
                grad_scaler=True,
            )

            # Apply mask using FULL attention_mask and handle the shift
            log_probs = log_probs[:, :-1] * attention_mask[:, 1:]
        else:
            # Non-CP path: original logic
            labels = torch.cat([labels, torch.zeros_like(labels[:, :1])], dim=1)
            log_probs = log_probs_from_logits(logits, labels)
            log_probs = log_probs[:, :-1] * attention_mask[:, 1:]

        return log_probs

    def op_compute_entropy(self, logits: torch.Tensor, attention_mask: torch.Tensor):
        from roll.utils.functionals import entropy_from_logits

        entropy = entropy_from_logits(logits)
        if self.worker.rank_info.cp_size > 1:
            entropy = ulysses_gather(
                entropy,
                gather_dim=1,
                group=get_ulysses_group(),
                grad_scaler=True,
            )
        entropy = entropy[:, :-1] * attention_mask[:, 1:]
        return entropy


class FSDP2TrainStrategy(FSDP2InferStrategy, TrainStrategy):
    strategy_name = "fsdp2_train"

    def initialize(self, model_provider):
        model, torch_dtype, _ = self._prepare_fsdp2_model(
            model_provider,
            is_trainable=True,
            default_model_dtype=torch.float32,
            warmup_collective=True,
        )

        logger.info(f"max steps pipeline {self.worker_config.training_args.max_steps}")
        self.worker_config.training_args.max_steps = (
            self.worker_config.training_args.max_steps // self.worker.rank_info.dp_size
        )
        logger.info(f"max steps worker train {self.worker_config.training_args.max_steps}")

        # Setup FSDP-2 configuration
        self.setup_fsdp2_configuration()

        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Initialize FSDP-2 model
        self.initialize_fsdp2_model(model)

        # In-case of LoRA
        trainable_params = (param for param in self.model.parameters() if param.requires_grad)
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.worker_config.training_args.learning_rate,
            betas=(
                self.worker_config.training_args.adam_beta1,
                self.worker_config.training_args.adam_beta2,
            ),
            weight_decay=self.worker_config.training_args.weight_decay,
        )

        self.scheduler = get_scheduler(
            self.worker_config.training_args.lr_scheduler_type,
            self.optimizer,
            num_warmup_steps=self.worker_config.training_args.get_warmup_steps(
                self.worker_config.training_args.max_steps
            ),
            num_training_steps=self.worker_config.training_args.max_steps,
        )

        dist.barrier()

    def _grad_accumulation_context(self):
        set_sync_fn = getattr(self.model, "set_requires_gradient_sync", None)
        if callable(set_sync_fn):
            return self._requires_grad_sync_context(set_sync_fn)

        no_sync_method = getattr(self.model, "no_sync", None)
        if callable(no_sync_method):
            return no_sync_method()

        return contextlib.nullcontext()

    @contextlib.contextmanager
    def _requires_grad_sync_context(self, set_sync_fn):
        set_sync_fn(False)
        try:
            yield
        finally:
            set_sync_fn(True)

    def _clip_grad_norm(self, max_norm: float):
        if not self.cpu_offload_enabled:
            grad_norm = clip_grad_norm_(
                self.model.parameters(),
                max_norm=max_norm,
            )
        else:
            grad_norm = self._clip_grad_norm_cpu_offload(max_norm)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        return grad_norm

    def _clip_grad_norm_cpu_offload(self, max_norm: float):
        """
        Mirror VERL's fsdp2_clip_grad_norm_:
        1. operate on local gradients
        2. move norm scalar to GPU (avoid CPU DTensor collectives)

        Reference: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py#L566
        Related discussion: https://github.com/volcengine/verl/pull/1026#discussion_r2064879123
        """
        parameters = list(self.model.parameters())
        grads = [p.grad for p in parameters if getattr(p, "grad", None) is not None]
        if not grads:
            device = current_platform.current_device()
            return torch.zeros(1, device=device)

        total_norm = _get_total_norm(
            grads,
            norm_type=2.0,
            error_if_nonfinite=False,
            foreach=None,
        )
        total_norm = total_norm.to(current_platform.current_device(), non_blocking=True)
        _clip_grads_with_norm_(
            parameters,
            max_norm=max_norm,
            total_norm=total_norm,
            foreach=None,
        )
        return total_norm

    def train_step(
        self,
        batch: DataProto,
        loss_func: Callable[
            [DataProto, torch.Tensor],
            Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        ],
        no_sync: bool = False,
    ):
        """
        Comment:
        no_sync: Usually, the inner step already handle no-sync, but leave this option for user if want other accumulation logic
        """
        self.model.train()
        mini_batch_size = self.worker_config.training_args.per_device_train_batch_size
        data_iter = batch.make_iterator(mini_batch_size=mini_batch_size, epochs=1)
        mini_steps = batch.batch.batch_size[0] // self.worker_config.training_args.per_device_train_batch_size

        cp_size = self.worker.rank_info.cp_size
        batch_num_tokens = self._get_batch_num_tokens(batch)
        batch.meta_info['batch_num_tokens'] = {k: v // cp_size for k, v in batch_num_tokens.items()}
        global_valid_tokens = self._get_global_valid_samples(batch)
        batch.meta_info['global_valid_samples'] = {k: v // cp_size for k, v in global_valid_tokens.items()}
        loss_scale = mini_steps * self.worker.rank_info.dp_size
        batch.meta_info['micro_batch_size'] = mini_batch_size

        gradient_accumulation_steps = self.worker_config.training_args.gradient_accumulation_steps

        metrics = {}
        cp_size = max(self.worker.rank_info.cp_size, 1)

        for step in range(mini_steps):
            data: DataProto = next(data_iter)
            input_ids = data.batch["input_ids"]
            attention_mask = data.batch["attention_mask"]
            position_ids = data.batch["position_ids"]
            forward_args = data.meta_info.get("forward_args", {})
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)  # (bsz, C, seqlen) -> (C, bsz, seqlen)
            if "multi_modal_inputs" in data.non_tensor_batch:
                multi_modal_inputs = data.non_tensor_batch["multi_modal_inputs"]
                multi_modal_data = defaultdict(list)
                for sample_mm_inputs in multi_modal_inputs:
                    for key in sample_mm_inputs.keys():
                        multi_modal_data[key].append(sample_mm_inputs[key])
                for key in multi_modal_data.keys():
                    assert key not in forward_args
                    forward_args[key] = torch.concat(multi_modal_data[key], dim=0).to(input_ids.device)
                forward_args.update({"force_vit_image": True})

            sync_boundary = ((step + 1) % gradient_accumulation_steps == 0 or (step + 1 == mini_steps)) and not no_sync

            # PumpkinComment:
            # model.no_sync is replaced by model.set_requires_gradient_sync(False) in FSDP2
            # but also add support for model.no_sync for compatibility
            sync_context = (
                self._grad_accumulation_context() if not sync_boundary and not no_sync else contextlib.nullcontext()
            )

            with sync_context:
                logits = self._fsdp2_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    forward_args=forward_args,
                )

                loss, loss_reduced = loss_func(data, logits)
                append_to_dict(metrics, loss_reduced)

                if self.worker_config.apply_loss_scale:
                    loss *= loss_scale

                loss = loss / gradient_accumulation_steps

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

            if sync_boundary:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                grad_norm = self._clip_grad_norm(
                    max_norm=self.worker.pipeline_config.max_grad_norm,
                )
                metrics[f"{self.worker_config.name}/grad_norm"] = grad_norm.item()

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if not torch.isfinite(grad_norm):
                        logger.warning(f"WARN: rank {dist.get_rank()} grad_norm is not finite: {grad_norm}")
                    else:
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

        torch.cuda.empty_cache()
        return metrics

    def setup_model_update(self, infer_cluster, model_update_name: str):
        assert model_update_name not in self.weight_updaters
        is_lora = self.worker_config.model_args.lora_target is not None
        self.weight_updaters[model_update_name] = FSDP2WeightUpdater(
            pipeline_config=self.worker.pipeline_config,
            infer_cluster=infer_cluster,
            worker_config=self.worker_config,
            model_update_name=model_update_name,
            model=self.unwrap_model(),
            is_lora=is_lora,
        )

    def model_update(self, model_update_name: str):
        return self.weight_updaters[model_update_name].model_update()
