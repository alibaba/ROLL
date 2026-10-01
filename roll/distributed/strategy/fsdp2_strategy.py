import contextlib
import os
import random
import types
from collections import defaultdict
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Set, Tuple
from abc import abstractmethod, ABC

import accelerate
import numpy as np
import ray
import transformers
from transformers import AutoConfig, get_scheduler, set_seed
from codetiming import Timer
from packaging import version
import torch
from torch import optim
from torch import Tensor
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, distribute_tensor, distribute_module, Shard
from torch.distributed.tensor.parallel import parallelize_module, ParallelStyle
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm

from roll.datasets.collator import collate_fn_to_dict_list
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.store.base import relink_dtensor_local, shrink_cuda_storage
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import (
    clear_fsdp2_init_context,
    default_processor_provider,
    default_tokenizer_provider,
    set_fsdp2_init_context,
)
from roll.platforms import current_platform
from roll.third_party.fsdp2.checkpoint import (
    async_save_dtensor,
    supports_passthrough_staging,
    tensor_storage_key,
)
from roll.third_party.fsdp2.model_update import FSDP2WeightUpdater
from roll.utils.checkpoint_manager import CheckpointManager, download_model
from roll.utils.collective import collective
from roll.utils.context_parallel import get_ulysses_group, set_upg_manager
from roll.utils.context_parallel.autograd_gather import ulysses_gather
from roll.utils.fsdp_utils import (
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    get_init_weight_context_manager,
    get_shard_placement_fn,
    get_shard_placement_fn_ep,
    iter_fsdp_params,
    _permute,
    _unpermute,
    register_experts_forward_in_ExpertsInterface,
    set_use_grouped_mm,
)
from roll.utils.functionals import append_to_dict, log_probs_from_logits, parse_dtype
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType, clear_memory
from roll.utils.constants import IGNORE_INDEX, RNG_STATE_DIR

logger = get_logger()


def _patch_peft_enable_adapters():
    from peft.tuners.tuners_utils import BaseTunerLayer
    if hasattr(BaseTunerLayer, "original_enable_adapters"):
        return

    def enable_adapters_keep_requires_grad(self, enabled: bool) -> None:
        if enabled:
            self.set_adapter(self.active_adapters)
            self._disable_adapters = False
        else:
            self._disable_adapters = True

    BaseTunerLayer.original_enable_adapters = BaseTunerLayer.enable_adapters
    BaseTunerLayer.enable_adapters = enable_adapters_keep_requires_grad
    logger.info("Patched PEFT BaseTunerLayer.enable_adapters to keep requires_grad=True in FSDP2 LoRA inference.")


def create_device_mesh_with_ep(world_size: int, fsdp_size: int, efsdp_size: int, ep_size: int):
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

    device_mesh = init_device_mesh(
        current_platform.device_type,
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_dim_names,
    )

    # Create device mesh for MoE
    if ep_size > 1:
        moe_model_size = efsdp_size * ep_size
        if moe_model_size == world_size:
            moe_mesh_shape = (efsdp_size, ep_size)
            moe_mesh_dim_names = ("efsdp", "ep")
        else:
            eddp_size = world_size // moe_model_size
            moe_mesh_shape = (eddp_size, efsdp_size, ep_size)
            moe_mesh_dim_names = ["eddp", "efsdp", "ep"]

        moe_device_mesh = init_device_mesh(
            current_platform.device_type,
            mesh_shape=moe_mesh_shape,
            mesh_dim_names=moe_mesh_dim_names,
        )
    else:
        moe_device_mesh = None

    return device_mesh, moe_device_mesh


class BaseExpertParallel(ParallelStyle, ABC):
    """
    Mirror torchtitan's BaseExpertParallel.
    Reference: https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/expert_parallel.py #L30 #v0.2.2
    """
    @abstractmethod
    def _partition_fn(self, name: str, mod: torch.nn.Module, device_mesh: DeviceMesh) -> None:
        ...

    @abstractmethod
    def _token_dispatch(
        self, mod: torch.nn.Module, inputs: tuple, device_mesh: DeviceMesh
    ) -> tuple[Tensor, Tensor]:
        ...

    @abstractmethod
    def _token_combine(
        self, mod: torch.nn.Module, routed_output: Tensor, device_mesh: DeviceMesh
    ) -> Tensor:
        ...


class ExpertParallel(BaseExpertParallel):
    """
    Mirror torchtitan's BaseExpertParallel.
    Reference: https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/expert_parallel.py #L89 #v0.2.2
    """
    def __init__(self, num_experts: int):
        super().__init__()
        self.input_splits = None
        self.output_splits = None
        self.input_shape = None
        self.permuted_indices = None
        self.num_experts = num_experts

    def _partition_fn(self, name: str, mod: torch.nn.Module, device_mesh: DeviceMesh) -> None:
        for param_name, param in mod.named_parameters(recurse=False):
            dist_param = torch.nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(param_name, dist_param)

    def _token_dispatch(
        self, mod: torch.nn.Module, inputs: tuple, device_mesh: DeviceMesh
    ) -> tuple[Tensor, Tensor]:
        # Preprocess
        x, selected_experts_indices, top_scores = inputs
        self.x = x
        self.top_scores = top_scores
        self.bs_slen, self.dim = x.shape
        self.top_k = top_scores.shape[-1]
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )
        self.token_indices_experts_sorted = token_indices_experts_sorted = torch.argsort(
            selected_experts_indices.view(-1), stable=True
        )
        top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]
        # shape (bs*slen*top_k, dim)
        routed_input = x[token_indices_experts_sorted // self.top_k]

        # annotate module input placements/sharding with input_layouts
        ep_degree = device_mesh.shape[0]
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree

        # generate the input splits and output splits for all-to-all
        with torch.no_grad():
            num_tokens_per_expert_group = all_to_all_single(
                num_tokens_per_expert,
                None,
                None,
                group=device_mesh.get_group(),
            )
            # Need to wait explicitly because it is used by a triton kernel later
            # which doesn't realize that AsyncCollectiveTensor needs unwrapping
            num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
                num_tokens_per_expert_group
            )
            input_splits = (
                num_tokens_per_expert.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=True)
            )
            # NOTE: this would incur a device-to-host sync
            output_splits = (
                num_tokens_per_expert_group.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=False)
            )
            self.input_splits = input_splits.tolist()
            self.output_splits = output_splits.tolist()

        # perform all-to-all
        routed_input = all_to_all_single_autograd(
            routed_input,
            self.output_splits,
            self.input_splits,
            device_mesh.get_group(),
        )

        # NOTE: After this all-to-all, the routed input is put on proper EP rank.
        # However, the num_tokens_per_expert_group is not of the final target format
        # [#tokens for local expert 0, #tokens for local expert 1, ...]
        # Rather, it is of the format
        # [#tokens for local expert 0 from EP rank 0, #tokens for local expert 1 from EP rank 0, ...,
        #  #tokens for local expert 0 from EP rank 1, #tokens for local expert 1 from EP rank 1, ...]
        # We need to perform another shuffle to get the correct layout, via the _permute function
        # below, which also does padding to make sure the number of tokens each expert gets locally
        # is a multiple of TOKEN_GROUP_ALIGN_SIZE_M.
        # Note that this will create side effects when wrapping the for-loop implementation
        # of GroupedExperts, as it does not need padding.

        (
            self.input_shape,
            routed_input,
            self.permuted_indices,
            num_tokens_per_expert_group,
        ) = _permute(
            routed_input, num_tokens_per_expert_group, ep_degree, num_local_experts
        )

        return routed_input, num_tokens_per_expert_group

    def _token_combine(
        self, mod: torch.nn.Module, routed_output: Tensor, device_mesh: DeviceMesh
    ) -> Tensor:
        routed_output = _unpermute(
            routed_output, self.input_shape, self.permuted_indices
        )

        routed_output = all_to_all_single_autograd(
            routed_output,
            self.input_splits,
            self.output_splits,
            device_mesh.get_group(),
        )

        # Postprocess
        routed_output_unsorted = torch.zeros(
            (self.bs_slen * self.top_k, self.dim),
            dtype=routed_output.dtype,
            device=routed_output.device,
        )
        routed_output_unsorted[self.token_indices_experts_sorted] = routed_output
        routed_output_unsorted = routed_output_unsorted.reshape(
            -1, self.top_k, self.dim
        )
        out_experts = (
            torch.bmm(
                self.top_scores.reshape(-1, 1, self.top_k),
                routed_output_unsorted.to(self.top_scores.dtype),
            )
            .to(self.x.dtype)
            .squeeze(1)
        )

        # Do cloning here to ensure that experts return a not-view tensor.
        # View tensor + in-place modification later can break pre-backward hook of fsdp.
        # For example, in qwen3.5, in-place modification is applied to the output of experts: expert_output += shared_expert_output
        return out_experts.clone()

    def _apply(self, module: torch.nn.Module, device_mesh: DeviceMesh) -> torch.nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            # pyrefly: ignore [bad-argument-type]
            input_fn=self._token_dispatch,
            # pyrefly: ignore [bad-argument-type]
            output_fn=self._token_combine,
        )


class FSDP2StrategyBase(InferenceStrategy):
    def __init__(self, worker: Worker):
        super().__init__(worker)
        self.cpu_offload_enabled: bool = False
        if not hasattr(self, "checkpoint_manager") or self.checkpoint_manager is None:
            checkpoint_config = getattr(self.worker_config, "checkpoint_config", None)
            self.checkpoint_manager = CheckpointManager(
                checkpoint_config=checkpoint_config, register=True
            )
        self._model_update_device_buffer: Optional[torch.Tensor] = None
        self.weight_updaters = {}
        self._dcp_process_group: Optional[dist.ProcessGroup] = None
        self._offload_backend = None
        self._offload_keys: list = []          # [(backend_key, (dtype, dedup_tag)), ...] one flat key per param group
        self._offload_param_groups = None      # (dtype, dedup_tag) -> [Parameter], rebuilt at each re-put
        self._replication_groups: dict = {}    # dedup_tag ("ddp"/"eddp") -> ProcessGroup, resolved lazily

    def _get_dcp_process_group(self) -> Optional[dist.ProcessGroup]:
        if self._dcp_process_group is None:
            self._dcp_process_group = dist.new_group(backend="gloo", group_desc="roll_dcp_checkpoint_pg")
        return self._dcp_process_group

    def _get_dp_rank(self) -> int:
        rank_info = getattr(self.worker, "rank_info", None)
        if rank_info is not None and getattr(rank_info, "dp_rank", None) is not None:
            return rank_info.dp_rank
        return dist.get_rank()

    def _get_offload_backend(self):
        """Get or initialize the offload backend (lazy, based on config).

        Pure FSDP (1D mesh): every rank holds a distinct shard, nothing to
        dedup; dp_group=None keeps barrier a no-op. HSDP: dense shards are
        replicated across the ddp mesh dim, so pass that group for DP
        deduplication (dedup-aware backends store 1/ddp_size per rank). The
        constructor group is only the default; param puts pass their own group
        per key (dense over ddp, MoE expert over eddp).
        """
        if self._offload_backend is None:
            from roll.distributed.store.factory import get_offload_backend
            dp_rank, dp_group = 0, None
            if self.device_mesh is not None and "ddp" in self.device_mesh.mesh_dim_names:
                ddp_mesh = self.device_mesh["ddp"]
                dp_rank = ddp_mesh.get_local_rank()
                dp_group = ddp_mesh.get_group()
            self._offload_backend = get_offload_backend(self.worker, dp_rank=dp_rank, dp_group=dp_group)
        return self._offload_backend

    @staticmethod
    def _param_dedup_tag(param) -> Optional[str]:
        """Replication group a param's local shard is duplicated over: 'ddp'
        for HSDP dense params, 'eddp' for MoE expert params (they live on the
        MoE mesh and replicate along eddp, a different rank set from ddp), or
        None when the shard is unique to this rank."""
        d = param.data
        if not isinstance(d, DTensor):
            return None
        dim_names = d.device_mesh.mesh_dim_names or ()
        if "ddp" in dim_names:
            return "ddp"
        if "eddp" in dim_names:
            return "eddp"
        return None

    def _replication_group(self, tag: Optional[str]):
        """Resolve a dedup tag to its ProcessGroup (False = store per-rank)."""
        if tag is None:
            return False
        if tag not in self._replication_groups:
            if tag == "ddp":
                self._replication_groups[tag] = self.device_mesh["ddp"].get_group()
            else:
                self._replication_groups[tag] = self.moe_device_mesh["eddp"].get_group()
        return self._replication_groups[tag]

    def _reset_fsdp_sharded_params(self):
        """After a reload, re-derive FSDPParam._sharded_param_data (the all-gather
        copy-in source) from the new _local_tensor; it still views the old buffer."""
        for fsdp_param in iter_fsdp_params(self.model):
            fsdp_param.reset_sharded_param()

    def _cleanup_offloaded_model_params(self):
        """Clean up stale model parameters from backend after gradient updates."""
        if not self._offload_keys:
            return

        # A direct async checkpoint reads these CPU store buffers. Do not
        # recycle them until the background writer has finished.
        if self.checkpoint_future is not None and self._checkpoint_reads_offload_store:
            self.checkpoint_future.result()
            self._checkpoint_reads_offload_store = False

        backend = self._get_offload_backend()
        for key, (_dtype, tag) in self._offload_keys:
            backend.delete_tensors(key, replicated_over=self._replication_group(tag))

        self._offload_keys = []

    def invalidate_offloaded_model_params(self):
        """Drop cached host copies of offloaded model params.

        Call after mutating params on-GPU outside of train_step (e.g. DiffNFT
        preflight adapter seeding) so the next offload_states() re-puts the
        updated values instead of taking the write-once fast path in
        _offload_model_params_to_backend (which would keep the stale host
        copies and lose the update on the next load_states).
        """
        self._cleanup_offloaded_model_params()

    def _offload_model_params_to_backend(self):
        """Group params by (dtype, replication group) and offload each group as
        one flat backend key. put_tensors records DTensor meta and detaches the
        params from device memory (plain .data rebind / DTensor _local_tensor
        swap + storage shrink), so GPU memory is actually released. HSDP dense
        groups dedup over the ddp group and MoE expert groups over the eddp
        group (every member holds an exact replica); pure-FSDP and MoE-mesh
        groups whose mesh has no eddp dim store independently.
        one flat backend key. put_tensors records DTensor meta and detaches the
        params from device memory (plain .data rebind / DTensor _local_tensor
        swap + storage shrink), so GPU memory is actually released. HSDP dense
        groups dedup over the ddp group and MoE expert groups over the eddp
        group (every member holds an exact replica); pure-FSDP and MoE-mesh
        groups whose mesh has no eddp dim store independently.

        Write-once: keys survive reloads and are only deleted after optimizer
        updates (_cleanup_offloaded_model_params), so when keys still exist the
        host copies are current — just free GPU memory, no re-put."""
        if self._offload_keys:
            for params in self._offload_param_groups.values():
                for p in params:
                    if isinstance(p, DTensor):
                        old_local = p._local_tensor
                        relink_dtensor_local(p, None)
                        # shrink old storage: _sharded_param_data still views it
                        shrink_cuda_storage(old_local)
                    else:
                        p.data = torch.empty(0, dtype=p.dtype, device="cpu")
            return

        backend = self._get_offload_backend()

        # param set may change between cycles (e.g. LoRA merge/unmerge)
        groups = {}
        for _, p in self.model.named_parameters():
            groups.setdefault((p.dtype, self._param_dedup_tag(p)), []).append(p)
        self._offload_param_groups = groups

        key_prefix = f"fsdp2_{self.worker.cluster_name}"
        self._offload_keys = []
        for (dtype, tag), params in self._offload_param_groups.items():
            key = f"{key_prefix}_params_{str(dtype).split('.')[-1]}_{tag or 'nodedup'}"
            backend.put_tensors(key, params, replicated_over=self._replication_group(tag))
            self._offload_keys.append((key, (dtype, tag)))

    def _reload_model_params_from_backend(self):
        """One flat H2D transfer per group; the backend rebuilds the params (see
        get_tensors). Keys are kept: train_step deletes them after the update."""
        backend = self._get_offload_backend()
        for key, (dtype, tag) in self._offload_keys:
            backend.get_tensors(key, self._offload_param_groups[(dtype, tag)],
                                device=current_platform.device_type, replicated_over=self._replication_group(tag))
        self._reset_fsdp_sharded_params()

    def _can_checkpoint_from_store(self, asynchronous: bool) -> bool:
        """Use the zero-copy checkpoint path only for a local, full CPU store."""
        if self.cpu_offload_enabled or not getattr(self.worker.pipeline_config, "is_offload_states", False):
            return False
        backend = self._get_offload_backend()
        if not backend.supports_direct_checkpoint:
            return False
        return not asynchronous or supports_passthrough_staging()

    def _build_checkpoint_state_from_store(
        self,
    ) -> Tuple[Dict[str, Tensor], Set[Tuple[int, int]]]:
        """Build the model state dict as views over existing CPU store buffers."""
        if not self._offload_keys:
            raise RuntimeError("Direct checkpoint requires model parameters in the offload store")

        try:
            named_parameters = self.model.named_parameters(remove_duplicate=False)
        except TypeError:  # torch <= 2.0 compatibility
            named_parameters = self.model.named_parameters()

        names_by_param: Dict[int, List[str]] = defaultdict(list)
        for name, param in named_parameters:
            names_by_param[id(param)].append(name)

        state_dict = self.model.state_dict()
        backend = self._get_offload_backend()
        passthrough_storages: Set[Tuple[int, int]] = set()
        replaced_params: Set[int] = set()

        for key, (dtype, tag) in self._offload_keys:
            flat, tensor_meta = backend.get_state(key)
            storage_key = tensor_storage_key(flat)
            if storage_key is not None:
                passthrough_storages.add(storage_key)

            params = self._offload_param_groups[(dtype, tag)]
            if len(params) != len(tensor_meta):
                raise RuntimeError(
                    f"Offload metadata mismatch for '{key}': {len(params)} params, "
                    f"{len(tensor_meta)} metadata entries"
                )

            offset = 0
            for param, (_, _, local_shape) in zip(params, tensor_meta):
                numel = local_shape.numel()
                local = flat.narrow(0, offset, numel).view(local_shape)
                value = DTensor(local, param._spec, requires_grad=False) if isinstance(param, DTensor) else local

                matched = False
                for name in names_by_param.get(id(param), []):
                    candidates = [name]
                    if name.startswith("pretrained_model."):
                        candidates.append(name.removeprefix("pretrained_model."))
                    for candidate in candidates:
                        if candidate in state_dict:
                            state_dict[candidate] = value
                            matched = True
                if not matched:
                    raise RuntimeError(f"Offloaded parameter has no state_dict entry: {names_by_param.get(id(param))}")
                replaced_params.add(id(param))
                offset += numel

            if offset != flat.numel():
                raise RuntimeError(
                    f"Offload flat-buffer size mismatch for '{key}': consumed {offset}, stored {flat.numel()}"
                )

        missing = [names for param_id, names in names_by_param.items() if param_id not in replaced_params]
        if missing:
            raise RuntimeError(f"Parameters missing from offload checkpoint state: {missing[:8]}")
        return state_dict, passthrough_storages

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

    @staticmethod
    def _is_diffusers_model(model) -> bool:
        try:
            from diffusers.models.modeling_utils import ModelMixin
        except Exception:
            return False
        return isinstance(model, ModelMixin)

    @contextlib.contextmanager
    def _patch_diffusers_state_dict(self, model, full_model_state: Dict[str, torch.Tensor]):
        original_state_dict = model.state_dict
        def patched_state_dict(self, *args, **kwargs):
            return full_model_state
        model.state_dict = types.MethodType(patched_state_dict, model)
        try:
            yield
        finally:
            model.state_dict = original_state_dict

    def _save_checkpoint_with_dcp(
        self,
        checkpoint_dir: str,
        is_last_step: bool,
        model_state_dict: Optional[Dict[str, Tensor]] = None,
        passthrough_storages: Optional[Set[Tuple[int, int]]] = None,
    ):
        state_dict = {
            **(model_state_dict if model_state_dict is not None else self.model.state_dict()),
        }

        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None:
            state_dict["optimizer"] = optimizer

        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None:
            state_dict["scheduler"] = scheduler

        dcp_process_group = self._get_dcp_process_group()

        if not self.async_save_strategy or is_last_step:
            if self.checkpoint_future is not None:
                self.checkpoint_future.result()
                self.checkpoint_future = None
                self._checkpoint_reads_offload_store = False
            dcp.save(
                state_dict=state_dict,
                checkpoint_id=checkpoint_dir,
                process_group=dcp_process_group,
            )
            return

        if self.checkpoint_future is not None:
            self.checkpoint_future.result()
        self.checkpoint_future = async_save_dtensor(
            state_dict=state_dict,
            checkpoint_id=checkpoint_dir,
            process_group=dcp_process_group,
            passthrough_storages=passthrough_storages,
        )
        self._checkpoint_reads_offload_store = bool(passthrough_storages)

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

        dcp_process_group = self._get_dcp_process_group()

        dcp.load(
            state_dict=state_dict,
            checkpoint_id=checkpoint_dir,
            process_group=dcp_process_group,
        )

        info = self.model.load_state_dict(state_dict, strict=False)
        missing_keys = info.missing_keys
        unexpected_keys = info.unexpected_keys

        filtered_unexpected_keys = [
            key for key in unexpected_keys if key not in ("optimizer", "scheduler")
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

    @staticmethod
    def _is_trl_value_head_model(model: torch.nn.Module) -> bool:
        return hasattr(model, "pretrained_model") and hasattr(model, "v_head")

    @staticmethod
    def _with_module_state_dict(model: torch.nn.Module, fn: Callable):
        original_state_dict = model.__dict__.get("state_dict", None)
        had_instance_state_dict = "state_dict" in model.__dict__

        model.state_dict = torch.nn.Module.state_dict.__get__(model, type(model))
        try:
            return fn()
        finally:
            if had_instance_state_dict:
                model.state_dict = original_state_dict
            else:
                delattr(model, "state_dict")

    def _get_hf_full_model_state_dict(self, model: torch.nn.Module, options: StateDictOptions):
        if not self._is_trl_value_head_model(model):
            return get_model_state_dict(model=model, options=options), None

        # TRL value-head wrappers override state_dict() and strip the
        # `pretrained_model.` prefix. DCP resolves keys against nn.Module FQNs,
        # so use the standard module state_dict while exporting full weights.
        state_dict = self._with_module_state_dict(
            model,
            lambda: get_model_state_dict(model=model, options=options),
        )
        return self._split_trl_value_head_state_dict(state_dict)

    @staticmethod
    def _split_trl_value_head_state_dict(state_dict: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        full_model_state = {}
        value_head_state = {
            name: tensor
            for name, tensor in state_dict.items()
            if name.startswith("v_head.")
        }
        for name, tensor in state_dict.items():
            if name.startswith("v_head."):
                continue
            if name.startswith("pretrained_model."):
                full_model_state[name.removeprefix("pretrained_model.")] = tensor
            else:
                full_model_state[name] = tensor
        return full_model_state, value_head_state

    @staticmethod
    def _save_trl_value_head_state_dict(value_head_state: Dict[str, Tensor], save_dir: str):
        from safetensors.torch import save_file

        if not value_head_state:
            raise RuntimeError("No v_head.* keys found when saving TRL value head checkpoint.")

        # Keep the v_head. prefix: ROLL's non-resume initialization loads this
        # file with the wrapper's default load_state_dict(strict=False).
        save_file(
            {name: tensor.detach().cpu().contiguous() for name, tensor in value_head_state.items()},
            os.path.join(save_dir, "value_head.safetensors"),
        )

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
            self._checkpoint_reads_offload_store = False

        os.makedirs(save_dir, exist_ok=True)

        direct_store_checkpoint = (
            not self.save_only_model
            and self._can_checkpoint_from_store(
                asynchronous=self.async_save_strategy and not is_last_step,
            )
        )
        direct_offload_time = 0.0

        with Timer("load", logger=None) as load_timer:
            self.load_states()

        with Timer("hf_save", logger=None) as hf_timer:
            full_state_options = self._get_dcp_state_dict_options(full_state_dict=True)
            underlying_model = self.unwrap_model()
            full_model_state, vhead_params = self._get_hf_full_model_state_dict(
                model=underlying_model,
                options=full_state_options,
            )

            if dist.get_rank() == 0:
                if self._is_diffusers_model(underlying_model):
                    logger.info("is diffusers model")
                    with self._patch_diffusers_state_dict(underlying_model, full_model_state):
                        underlying_model.save_pretrained(
                            save_dir,
                            safe_serialization=True
                        )
                else:
                    underlying_model.save_pretrained(
                        save_dir,
                        state_dict=full_model_state,
                        safe_serialization=True,
                    )
                if vhead_params is not None:
                    self._save_trl_value_head_state_dict(vhead_params, save_dir)
                self.tokenizer.save_pretrained(save_dir)
                if getattr(self, "processor", None):
                    self.processor.save_pretrained(save_dir)
            del full_model_state, vhead_params


        clear_memory(clear_host_memory=True)
        if dist.is_initialized():
            dist.barrier()

        dcp_save_time = 0
        dcp_save_future = None
        if not self.save_only_model:
            dcp_checkpoint_dir = self._get_dcp_checkpoint_dir(save_dir)
            os.makedirs(dcp_checkpoint_dir, exist_ok=True)

            self._save_rank_rng_state(save_dir)
            model_state_dict = None
            passthrough_storages = None
            if direct_store_checkpoint:
                with Timer("store_offload", logger=None) as store_offload_timer:
                    self.offload_states(include=[OffloadStateType.model_params])
                direct_offload_time = store_offload_timer.last
                model_state_dict, passthrough_storages = self._build_checkpoint_state_from_store()
                logger.info(
                    "Saving DCP directly from CPU offload buffers: keys=%d storages=%d",
                    len(self._offload_keys),
                    len(passthrough_storages),
                )

            with Timer("dcp_save", logger=None) as dcp_timer:
                self._save_checkpoint_with_dcp(
                    checkpoint_dir=dcp_checkpoint_dir,
                    is_last_step=is_last_step,
                    model_state_dict=model_state_dict,
                    passthrough_storages=passthrough_storages,
                )
            dcp_save_time = dcp_timer.last

            # PumpkinComment:
            # If DCP save is async, uploading (which may copy+delete the local dir) must not start
            # until the async save has fully finished writing checkpoint shards.
            dcp_save_future = self.checkpoint_future if (self.async_save_strategy and not is_last_step) else None

        checkpoint_config = getattr(self.worker_config, "checkpoint_config", None) or {}
        async_upload = checkpoint_config.get("async_upload", True) and not is_last_step
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

            if not self.save_only_model:
                clear_memory(clear_host_memory=True)
                if dist.is_initialized():
                    dist.barrier()

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

        # When cpu_offload is enabled, restore optimizer states back to CPU after saving.
        # Without this, optimizer.step() will fail due to device mismatch (optimizer on GPU, grads on CPU).
        with Timer("offload", logger=None) as offload_timer:
            if self.cpu_offload_enabled:
                self.offload_states()

        return {
            "load": load_timer.last,
            "offload": direct_offload_time + offload_timer.last,
            "dcp_save": dcp_save_time,
            "hf_save": hf_timer.last,
        }

    def _load_torch_file(self, path: str, required: bool = True):
        if os.path.exists(path):
            return torch.load(path, map_location="cpu", weights_only=False)
        if required:
            raise FileNotFoundError(f"Missing checkpoint shard: {path}")
        logger.warning(f"Optional checkpoint shard missing, skipping: {path}")
        return None

    def _save_rank_rng_state(self, save_dir: str) -> None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        rng_path = os.path.join(save_dir, RNG_STATE_DIR, f"rng_state_{rank}.pth")
        os.makedirs(os.path.dirname(rng_path), exist_ok=True)
        torch.save(self.get_rng_state(), rng_path)
        logger.info("Saved rank-specific RNG state: rank=%s path=%s", rank, rng_path)

    def _load_rank_rng_state(self, load_dir: str) -> None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        rng_path = os.path.join(load_dir, RNG_STATE_DIR, f"rng_state_{rank}.pth")
        if not os.path.exists(rng_path):
            logger.warning(
                "Rank-specific RNG state not found; RNG was not restored: rank=%s path=%s",
                rank,
                rng_path,
            )
            return

        rng_state = torch.load(rng_path, weights_only=False)
        self.load_rng_state(rng_state)
        logger.info("Restored rank-specific RNG state: rank=%s path=%s", rank, rng_path)

    @staticmethod
    def get_rng_state():
        rng_state = {
            "cpu": torch.get_rng_state(),
            "device": current_platform.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }
        return rng_state

    @staticmethod
    def load_rng_state(rng_state):
        torch.set_rng_state(rng_state["cpu"])
        current_platform.set_rng_state(rng_state["device"])
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
            if original_device.type == "cpu" and current_platform.device_type != "cpu":
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
                if key == "step":
                    continue
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

    def _prepare_fsdp2_model(
        self,
        model_provider,
        *,
        is_trainable: bool,
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

        init_tokenizer_processor = self.worker_config.strategy_args.strategy_config.get(
            "init_tokenizer_processor", True
        )
        if init_tokenizer_processor:
            self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
            self.processor = default_processor_provider(model_args=self.worker_config.model_args)
        else:
            self.tokenizer = None
            self.processor = None

        fsdp_size = self.worker_config.strategy_args.strategy_config.get("fsdp_size", 1)
        assert fsdp_size >= cp_size, "fsdp size should be greater than cp size."

        # Get expert parallel config
        efsdp_size = self.worker_config.strategy_args.strategy_config.get("efsdp_size", 1)
        ep_size = self.worker_config.strategy_args.strategy_config.get("ep_size", 1)
        moe_use_grouped_mm = self.worker_config.strategy_args.strategy_config.get("moe_use_grouped_mm", False)
        self.ep_enabled = ep_size > 1

        # Check efsdp, ep size
        if self.ep_enabled:
            assert version.parse(transformers.__version__) >= version.parse("5.2.0")

            moe_model_size = efsdp_size * ep_size
            assert moe_model_size > 0 and moe_model_size <= world_size and world_size % moe_model_size == 0, \
                f"efsdp_size * ep_size {moe_model_size} must be in [1, world_size({world_size})], and divisible by world_size {world_size}"

        if moe_use_grouped_mm and self.ep_enabled:
            set_use_grouped_mm(True)

        # Patch ExpertsInterface to offer more options of forward function in transformers's MoeExperts
        if self.ep_enabled:
            register_experts_forward_in_ExpertsInterface()

        # Create device mesh
        self.device_mesh, self.moe_device_mesh = create_device_mesh_with_ep(world_size=world_size, fsdp_size=fsdp_size, efsdp_size=efsdp_size, ep_size=ep_size)

        model_name_or_path = download_model(self.worker_config.model_args.model_name_or_path)
        try:
            config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                **self.worker_config.model_args.model_config_kwargs,
            )
        except (OSError, ValueError):
            if init_tokenizer_processor:
                raise
            config = None
            logger.info(
                "FSDP2 skips Transformers AutoConfig because tokenizer/processor initialization is disabled. "
                "The model_provider is expected to return a fully constructed nn.Module."
            )

        if config is not None:
            self._validate_ulysses_compat(config, cp_size)
            use_meta_tensor = not getattr(config, "tie_word_embeddings", False) and not self.ep_enabled
        else:
            if cp_size > 1:
                raise NotImplementedError(
                    "FSDP2 provider-built non-Transformers models do not support Ulysses context parallelism."
                )
            use_meta_tensor = False
        # accelerate v1.7.0 don't support _is_hf_initialized which is needed by use_meta_tensor
        if version.parse(accelerate.__version__) == version.parse("1.7.0"):
            use_meta_tensor = False
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

        if self.ep_enabled or moe_use_grouped_mm:
            if moe_use_grouped_mm:
                assert version.parse(transformers.__version__) >= version.parse("5.2.0"), (
                    "moe_use_grouped_mm requires transformers>=5.2.0"
                )
            layers = self._get_layers(model)
            layers[0].mlp.experts.config._experts_implementation = "ep" if self.ep_enabled else "grouped_mm"

        return model

    def _get_layers(self, model):
        model = getattr(model, "pretrained_model", model)  # TRL AutoModelForCausalLMWithValueHead
        if self.is_lora: # PeftModel
            base_model = model.base_model.model.model
        else:
            base_model = model.model
        
        if hasattr(base_model, "layers"):
            return base_model.layers
        else: # multi-modal model
            return base_model.language_model.layers

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
        """Load states from offload backend."""
        if self.cpu_offload_enabled:
            # FSDP CPU offload policy handles model params; only optimizer states need management
            if include is None or OffloadStateType.optimizer_states in include:
                self._move_optimizer_states(current_platform.current_device(), non_blocking=non_blocking)
            return

        if include is None or OffloadStateType.model_params in include:
            if self._offload_keys:
                self._reload_model_params_from_backend()
            else:
                # Simple device move (no prior backend offload)
                device = current_platform.current_device()
                self.model.to(device, non_blocking=non_blocking)

        if include is None or OffloadStateType.optimizer_states in include:
            self._move_optimizer_states(current_platform.current_device(), non_blocking=non_blocking)

    def offload_states(self, include=None, non_blocking=False):
        """Offload states to backend."""
        if self.cpu_offload_enabled:
            # FSDP CPU offload policy handles model params; only optimizer states need management
            if include is None or OffloadStateType.optimizer_states in include:
                self._move_optimizer_states(torch.device("cpu"), non_blocking=non_blocking)
            return

        if include is None or OffloadStateType.model_params in include:
            self._offload_model_params_to_backend()
            clear_memory()
            self._log_offload_summary()

        if include is None or OffloadStateType.optimizer_states in include:
            self._move_optimizer_states(torch.device("cpu"), non_blocking=non_blocking)


class FSDP2InferStrategy(FSDP2StrategyBase):
    strategy_name = "fsdp2_infer"

    def __init__(self, worker: Worker):
        super().__init__(worker)
        self.device_mesh = None
        self.moe_device_mesh = None
        self.fsdp_config = None

    def initialize(self, model_provider):
        model = self._prepare_fsdp2_model(
            model_provider,
            is_trainable=False,
        )

        self.setup_fsdp2_configuration(is_trainable=False)
        # Cast model parameters if users set param dtype
        logger.info(f"[FSDP2] Casting inference model parameters to {self.param_dtype}")
        model = model.to(dtype=self.param_dtype)

        # Initialize expert parallel model (must be before FSDP2 wrapping)
        if self.ep_enabled:
            self.apply_moe_ep(model)

        self.initialize_fsdp2_model(model)

        dist.barrier()

    def setup_fsdp2_configuration(self, is_trainable: bool = False):
        """Setup FSDP-2 configuration"""
        # ckpt strategy
        async_save_strategy = self.worker_config.strategy_args.strategy_config.get("async_save_ckpt", True)
        self.async_save_strategy = async_save_strategy
        self.checkpoint_future = None
        self._checkpoint_reads_offload_store = False
        self.save_only_model = self.worker_config.strategy_args.strategy_config.get("save_only_model", False)

        # Get mixed precision settings from config
        if is_trainable:
            enable_mix_precision = self.worker_config.strategy_args.strategy_config.get("enable_mix_precision", True)
            if enable_mix_precision:
                param_dtype = torch.bfloat16
                reduce_dtype = torch.float32
            else:
                param_dtype = torch.bfloat16
                reduce_dtype = torch.bfloat16
            param_dtype_from_config = self.worker_config.strategy_args.strategy_config.get("param_dtype", None)
            reduce_dtype_from_config = self.worker_config.strategy_args.strategy_config.get("reduce_dtype", None)
            if param_dtype_from_config is not None:
                logger.info(
                    f"[FSDP2] Override param_dtype from enable_mix_precision default: "
                    f"{param_dtype} -> {param_dtype_from_config}"
                )
                param_dtype = param_dtype_from_config
            if reduce_dtype_from_config is not None:
                logger.info(
                    f"[FSDP2] Override reduce_dtype from enable_mix_precision default: "
                    f"{reduce_dtype} -> {reduce_dtype_from_config}"
                )
                reduce_dtype = reduce_dtype_from_config

            # Convert string dtype specifications to torch.dtype
            param_dtype = parse_dtype(param_dtype)
            reduce_dtype = parse_dtype(reduce_dtype)
            self.param_dtype = param_dtype
            self.reduce_dtype = reduce_dtype

            mixed_precision = MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                cast_forward_inputs=True,
            )
        else:
            logger.info("[FSDP2] Do not use Mixed Precision for inference")
            mixed_precision = MixedPrecisionPolicy()
            param_dtype = torch.bfloat16
            param_dtype_from_config = self.worker_config.strategy_args.strategy_config.get("param_dtype", None)
            if param_dtype_from_config is not None:
                logger.info(
                    f"[FSDP2] Override param_dtype from inference default: "
                    f"{param_dtype} -> {param_dtype_from_config}"
                )
                param_dtype = param_dtype_from_config
            param_dtype = parse_dtype(param_dtype)
            self.param_dtype = param_dtype

        # Reshard after forward setting (FSDP2 uses this instead of sharding_strategy)
        # FULL_SHARD: reshard_after_forward=True
        # SHARD_GRAD_OP: reshard_after_forward=False
        # HYBRID_SHARD: reshard_after_forward=True with a 2D device mesh
        # HYBRID_SHARD_ZERO2: reshard_after_forward=False with a 2D device mesh
        # If None, True for submodules, False for root module
        reshard_after_forward = self.worker_config.strategy_args.strategy_config.get("reshard_after_forward", None)

        offload_policy_cfg = self.worker_config.strategy_args.strategy_config.get("offload_policy", False)
        self.cpu_offload_enabled = bool(offload_policy_cfg)

        # Perform reduce scatter during gradient accumulation.
        # Default to True, because it decrease cuda memory usage and increase training speed according to tests.
        self.reduce_scatter_during_grad_accumulation = bool(
            self.worker_config.strategy_args.strategy_config.get("reduce_scatter_during_grad_accumulation", True)
        )
        if is_trainable and self.reduce_scatter_during_grad_accumulation:
            logger.info("[FSDP2] reduce_scatter_during_grad_accumulation is ENABLED")

        offload_policy = None
        if self.cpu_offload_enabled:
            offload_policy = CPUOffloadPolicy(
                pin_memory=True,
            )

        # Store configuration for fully_shard()
        print(f"[DEBUG] fsdp_config: {self.worker_config.strategy_args.strategy_config.get('fsdp_size', 1)}")
        self.fsdp_config = {
            "mesh": self.device_mesh,
            "reshard_after_forward": reshard_after_forward,
            "mp_policy": mixed_precision,
            "offload_policy": offload_policy,
            "shard_placement_fn": get_shard_placement_fn(
                fsdp_size=self.worker_config.strategy_args.strategy_config.get("fsdp_size", 1)
            ),
        }
        self.moe_fsdp_config = self.fsdp_config.copy()
        if self.ep_enabled:
            self.moe_fsdp_config["mesh"] = self.moe_device_mesh["eddp", "efsdp"] if "eddp" in self.moe_device_mesh.mesh_dim_names else self.moe_device_mesh["efsdp"]
            self.moe_fsdp_config["shard_placement_fn"] = get_shard_placement_fn_ep(
                efsdp_size=self.worker_config.strategy_args.strategy_config.get("efsdp_size", 1),
            )

    def apply_moe_ep(self, model):
        layers = self._get_layers(model)

        num_experts = layers[0].mlp.experts.num_experts
        experts_plan = ExpertParallel(num_experts=num_experts)
        experts_mesh = self.moe_device_mesh["ep"]
        for layer in layers:
            if hasattr(layer.mlp, "experts"):
                parallelize_module(
                    module=layer.mlp.experts,
                    device_mesh=experts_mesh,
                    parallelize_plan=experts_plan,
                )

    def initialize_fsdp2_model(self, model):
        offload_policy = self.fsdp_config["offload_policy"]

        # Use nn.Module.state_dict() instead of model.state_dict() to get standard FQN keys
        # (e.g. 'pretrained_model.model.layers.0...') that match what PyTorch's
        # _iterate_valid_model_state() returns internally. Custom state_dict() methods
        # (like AutoModelForCausalLMWithValueHead's) may strip prefixes, causing key
        # mismatches in _broadcast_state_dict during FSDP2 initialization.
        full_state = torch.nn.Module.state_dict(model)
        apply_fsdp2(
            model,
            self.fsdp_config,
            self.moe_fsdp_config,
            self.worker_config.strategy_args.strategy_config,
            self.is_lora,
        )

        fsdp2_load_full_state_dict(
            model,
            full_state,
            self.device_mesh,
            offload_policy,
            self.ep_enabled,
        )

        self.model = model

    # Add torch.no_grad() to disable gradient calculation.
    # torch.no_grad() in pipeline cannot work because pipeline and worker are in different processes.
    @torch.no_grad()
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
        batch.meta_info["batch_num_tokens"] = {k: v // cp_size for k, v in batch_num_tokens.items()}
        global_valid_tokens = self._get_global_valid_samples(batch)
        batch.meta_info["global_valid_samples"] = {k: v // cp_size for k, v in global_valid_tokens.items()}

        loss_scale = num_microbatches * self.worker.rank_info.dp_size

        disable_adapter = batch.meta_info.get("disable_adapter", False)
        if disable_adapter:
            _patch_peft_enable_adapters()
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
                            mm_data = multi_modal_data[key]
                            # All mm fields are not padded in collator currently and some should be padded first
                            # pixel_values/pixel_values_videos for images/videos are concated for packing
                            # input_features for audios with shape `(bs, freqs, frames)` should be padded before concat
                            need_padding = any(t.shape[-1] != mm_data[0].shape[-1] for t in mm_data[1:])
                            if need_padding:  # input_features/feature_attention_mask
                                max_mm_len = max(t.shape[-1] for t in mm_data)
                                for i, t in enumerate(mm_data):
                                    mm_data[i] = torch.nn.functional.pad(t, (0, max_mm_len - t.shape[-1]), "constant", 0)
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

        lp_chunk_size = (
            getattr(self.worker_config, "logits_chunk_size", 2048)
            if getattr(self.worker_config, "use_logits_chunking", False)
            else 0
        )

        if self.worker.rank_info.cp_size > 1:
            # For CP: slice the shifted labels to match the sharded logits
            # logits are sharded across sequence dimension by Ulysses
            labels = torch.cat([labels, torch.zeros_like(labels[:, :1])], dim=1)
            labels = self.get_feature_on_cp_rank(labels)["input_ids"]

            # Compute log_probs for this CP rank
            log_probs = log_probs_from_logits(logits, labels, chunk_size=lp_chunk_size)

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
            log_probs = log_probs_from_logits(logits, labels, chunk_size=lp_chunk_size)
            log_probs = log_probs[:, :-1] * attention_mask[:, 1:]

        return log_probs

    def op_compute_entropy(self, logits: torch.Tensor, attention_mask: torch.Tensor):
        from roll.utils.functionals import entropy_from_logits

        ent_chunk_size = (
            getattr(self.worker_config, "logits_chunk_size", 2048)
            if getattr(self.worker_config, "use_logits_chunking", False)
            else 0
        )
        entropy = entropy_from_logits(logits, chunk_size=ent_chunk_size)
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
            self._load_rank_rng_state(load_dir)
            used_dcp = True
            logger.info(f"Loaded DCP checkpoint from {dcp_checkpoint_dir}")
            if dist.is_initialized():
                dist.barrier()
            return

    def initialize(self, model_provider):
        model = self._prepare_fsdp2_model(
            model_provider,
            is_trainable=True,
            warmup_collective=True,
        )

        logger.info(f"max steps pipeline {self.worker_config.training_args.max_steps}")
        self.worker_config.training_args.max_steps = (
            self.worker_config.training_args.max_steps // self.worker.rank_info.dp_size
        )
        logger.info(f"max steps worker train {self.worker_config.training_args.max_steps}")

        # Setup FSDP-2 configuration
        self.setup_fsdp2_configuration(is_trainable=True)

        # Cast model according to reduce dtype
        logger.info(f"[FSDP2] Casting trainable model parameters to reduce_dtype={self.reduce_dtype}")
        model = model.to(dtype=self.reduce_dtype)

        # Initialize expert parallel model
        if self.ep_enabled:
            self.apply_moe_ep(model)

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
            fused=self.worker_config.strategy_args.strategy_config.get("fused_optimizer", False),
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

    def clip_grad_norm(self, max_norm: float) -> torch.Tensor:
        """Clip gradients across FSDP, expert-parallel, and CPU-offload layouts."""
        if not self.cpu_offload_enabled and not self.ep_enabled:
            grad_norm = clip_grad_norm_(
                self.model.parameters(),
                max_norm=max_norm,
            )
        elif not self.cpu_offload_enabled:
            grad_norm = self._clip_grad_norm_with_ep(max_norm)
        elif not self.ep_enabled:
            grad_norm = self._clip_grad_norm_cpu_offload(max_norm)
        else:
            grad_norm = self._clip_grad_norm_with_ep(max_norm, cpu_offload_enabled=True)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        return grad_norm
    
    @torch.no_grad()
    def _clip_grad_norm_with_ep(self, max_norm: float, cpu_offload_enabled: bool = False):
        """
        Reference to torchtitan's _clip_grad_norm_with_ep:
        Reference: https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/utils.py #L479 #v0.2.2
        """
        parameters = list(self.model.parameters())
        ep_params = []
        non_ep_params = []
        ep_grads = []
        non_ep_grads = []

        for p in parameters:
            if p.grad is None:
                continue
            assert isinstance(p, DTensor) and isinstance(p.grad, DTensor)
            mesh_dim_names = p.device_mesh.mesh_dim_names
            assert mesh_dim_names is not None
            if "efsdp" in mesh_dim_names:
                # Divide the gradient of the moe parameter by ep to ensure that the gradient is calculated from data on a single rank,
                # keeping it aligned with fsdp.
                ep_size = self.worker_config.strategy_args.strategy_config.get("ep_size", 1)
                p.grad.div_(ep_size)
                ep_params.append(p)
                ep_grads.append(p.grad)
            else:
                non_ep_params.append(p)
                non_ep_grads.append(p.grad)

        # Either list can be empty depending on the parallelization strategy:
        # - In torchtitan with separate dense/sparse meshes, both lists are typically non-empty
        # - In autoparallel, all params may live on a single sparse mesh with "ep" dimension,
        #   so non_ep_grads would be empty
        # - In PP + EP setups, certain PP ranks may only own EP or non-EP layers
        ep_grads_total_norm = _get_total_norm(
            ep_grads,
            norm_type=2.0,
            error_if_nonfinite=False,
            foreach=None,
        )
        # get_total_norm returns tensor(0.) for empty list, which is a non-DTensor
        if isinstance(ep_grads_total_norm, DTensor):
            ep_grads_total_norm = ep_grads_total_norm.full_tensor()

        non_ep_grads_total_norm = _get_total_norm(
            non_ep_grads,
            norm_type=2.0,
            error_if_nonfinite=False,
            foreach=None,
        )
        # get_total_norm returns tensor(0.) for empty list, which is a non-DTensor
        if isinstance(non_ep_grads_total_norm, DTensor):
            non_ep_grads_total_norm = non_ep_grads_total_norm.full_tensor()

        # move norm scalar to GPU
        if cpu_offload_enabled:
            ep_grads_total_norm = ep_grads_total_norm.to(current_platform.current_device(), non_blocking=True)
            non_ep_grads_total_norm = non_ep_grads_total_norm.to(current_platform.current_device(), non_blocking=True)

        # Calculate total norm
        norm_type = 2.0
        total_norm = (
            ep_grads_total_norm**norm_type + non_ep_grads_total_norm**norm_type
        )
        total_norm **= 1.0 / norm_type

        # Do clipping
        _clip_grads_with_norm_(
            ep_params,
            max_norm=max_norm,
            total_norm=total_norm,
            foreach=None,
        )
        _clip_grads_with_norm_(
            non_ep_params,
            max_norm=max_norm,
            total_norm=total_norm,
            foreach=None,
        )

        return total_norm

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
        batch.meta_info["batch_num_tokens"] = {k: v // cp_size for k, v in batch_num_tokens.items()}
        global_valid_tokens = self._get_global_valid_samples(batch)
        batch.meta_info["global_valid_samples"] = {k: v // cp_size for k, v in global_valid_tokens.items()}
        loss_scale = mini_steps * self.worker.rank_info.dp_size
        batch.meta_info["micro_batch_size"] = mini_batch_size

        gradient_accumulation_steps = self.worker_config.training_args.gradient_accumulation_steps
        is_offload_optimizer_states_in_train_step = batch.meta_info.get("is_offload_optimizer_states_in_train_step", True)

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
            #
            # When reduce_scatter_during_grad_accumulation is enabled, sync every step to decrease cuda memory usage.
            # Reference: https://docs.pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.no_sync
            if self.reduce_scatter_during_grad_accumulation:
                sync_context = contextlib.nullcontext()
            else:
                sync_context = (
                    self._grad_accumulation_context() if not sync_boundary and not no_sync else contextlib.nullcontext()
                )

            with (
                sync_context,
                torch.autocast(
                    device_type=current_platform.device_type,
                    dtype=self.param_dtype,
                ),
            ):
                with getattr(self, "model_fwd_context", contextlib.nullcontext()):
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

                with getattr(self, "model_bwd_context", contextlib.nullcontext()):
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

            if sync_boundary:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                grad_norm = self.clip_grad_norm(
                    max_norm=self.worker.pipeline_config.max_grad_norm,
                )
                metrics[f"{self.worker_config.name}/grad_norm"] = grad_norm.item()

                # Lazy load optimizer states just before step (Megatron pattern)
                if not self.cpu_offload_enabled:
                    self.load_states(include=[OffloadStateType.optimizer_states], non_blocking=True)

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if not torch.isfinite(grad_norm):
                        logger.warning(f"WARN: rank {dist.get_rank()} grad_norm is not finite: {grad_norm}")
                    else:
                        self.optimizer.step()

                # Offload optimizer states after step, before scheduler.step (Megatron pattern)
                if not self.cpu_offload_enabled and is_offload_optimizer_states_in_train_step:
                    self.offload_states(include=[OffloadStateType.optimizer_states], non_blocking=True)
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                # Clean up old offloaded parameters after optimizer update
                self._cleanup_offloaded_model_params()

        # Log cuda memory
        max_memory_allocated = torch.cuda.memory.max_memory_allocated() / (1024 ** 3)
        max_memory_reserved = torch.cuda.memory.max_memory_reserved() / (1024 ** 3)
        metrics[f"system/max_memory_allocated@max"] = max_memory_allocated
        metrics[f"system/max_memory_reserved@max"] = max_memory_reserved
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
            adapter_name=self.worker.rollout_adapter_name,
        )

    def model_update(self, model_update_name: str):
        return self.weight_updaters[model_update_name].model_update()
    
    def get_labels_on_cp_rank(
        self,
        labels: torch.Tensor,
    ):
        """Get labels for specific context parallel rank"""
        seqlens_in_batch = labels.size(1)
        assert (
            seqlens_in_batch % self.worker.rank_info.cp_size == 0
        ), f"input_length={seqlens_in_batch} not divisible by cp_size={self.worker.rank_info.cp_size}"
        cp_middle_rank_len = seqlens_in_batch // self.worker.rank_info.cp_size
        padded_labels = labels
        start_index = cp_middle_rank_len * self.worker.rank_info.cp_rank
        end_index = cp_middle_rank_len * (self.worker.rank_info.cp_rank + 1)
        labels_this_cp_rank = padded_labels[:, start_index:end_index]
        return labels_this_cp_rank

    def op_compute_language_loss(self, logits: torch.Tensor, labels: torch.Tensor, batch_num_tokens: int):
        """
        Override for FSDP2 strategy: compute language loss from logits.

        In FSDP2 strategy with HuggingFace models, the model returns logits
        (not per-token loss like in Megatron strategy where labels are passed to the model).

        Note: DataCollatorForSFT already shifts labels (shift_feature=True by default),
        so logits and labels are already aligned. Do NOT shift again here.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            labels: Pre-shifted labels [batch_size, seq_len], already aligned with logits
            batch_num_tokens: Number of valid tokens for loss normalization

        Returns:
            loss: Scalar loss tensor
            metrics: Dict
        """
        cp_size = self.worker.rank_info.cp_size
        # Slice labels to match the sharded logits
        if cp_size > 1:
            labels = self.get_labels_on_cp_rank(labels)
            labels = labels.contiguous()
        
        # Compute per-token loss mask
        loss_mask = (labels != IGNORE_INDEX).float()
        loss_mask = loss_mask.view(-1).float()
        
        # Compute cross entropy loss per token (reduction='none' to get per-token losses)
        per_token_losses = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction='none',  # Get per-token loss
        )
        
        # Apply loss mask and sum
        losses_sum = torch.sum(per_token_losses.view(-1) * loss_mask)
        
        # All-reduce across CP ranks
        if cp_size > 1:
            loss_info = torch.cat([losses_sum.view(1)])
            dist.all_reduce(
                loss_info, op=dist.ReduceOp.SUM, group=get_ulysses_group()
            )
            losses_sum = loss_info[0]
        
        # Normalize by batch_num_tokens
        loss = losses_sum.clone() / batch_num_tokens  # clone to make sure loss is not a view
        
        metrics = {f"{self.worker_config.name}/loss@sum": loss.clone().detach().item()}

        return loss, metrics
