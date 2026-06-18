import copy
import dataclasses
from abc import ABC
from contextlib import contextmanager
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed.tensor import Shard, DTensor
import torch.nn.functional as F
from packaging import version
import transformers

from roll.models.model_providers import _is_moe_config
from roll.platforms import current_platform
from roll.utils.logging import get_logger

logger = get_logger()

try:
    from torch.distributed.device_mesh import DeviceMesh
except ImportError:
    DeviceMesh = None

# Optional Triton dependency: enables GPU-side fill_indices kernel for EP permutation.
# Falls back to CPU implementation transparently when triton is unavailable.
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

fully_shard_module = torch.distributed.fsdp._fully_shard._fully_shard

# TOKEN_GROUP_ALIGN_SIZE_M is set as 8 in torchtitan while quantization is not used
TOKEN_GROUP_ALIGN_SIZE_M = 8


use_grouped_mm = False

def set_use_grouped_mm(moe_use_grouped_mm):
    global use_grouped_mm
    use_grouped_mm = moe_use_grouped_mm


def get_use_grouped_mm():
    global use_grouped_mm
    return use_grouped_mm


@contextmanager
def maybe_patch_fsdp_module(model):
    if fully_shard_module is None:
        yield
        return

    orig_fsdp_module = fully_shard_module.FSDPModule

    class FSDPModuleABC(ABC, orig_fsdp_module):
        pass

    try:
        if isinstance(model, ABC):
            fully_shard_module.FSDPModule = FSDPModuleABC
        yield
    finally:
        fully_shard_module.FSDPModule = orig_fsdp_module


def get_init_weight_context_manager(use_meta_tensor=True, mesh: DeviceMesh = None):
    from accelerate import init_empty_weights

    cpu_init_weights = lambda: torch.device("cpu")
    if use_meta_tensor:
        if mesh is None:
            init_context = init_empty_weights if torch.distributed.get_rank() != 0 else cpu_init_weights
        else:
            init_context = init_empty_weights if mesh.get_coordinate()[-1] != 0 else cpu_init_weights
    else:
        init_context = cpu_init_weights
    return init_context


def get_shard_placement_fn(fsdp_size):
    """
    Choose the dimension that can divide fsdp_size to avoid padding
    Reference: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py

    """

    def shard_placement_fn(param):
        shape = list(param.shape)
        for i in range(len(shape)):
            if shape[i] % fsdp_size == 0:
                return Shard(i)
        return Shard(0)

    return shard_placement_fn


def get_shard_placement_fn_ep(efsdp_size):
    """
    Choose the dimension that can divide fsdp_size to avoid padding
    Reference: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py

    """

    def shard_placement_fn(param):
        if isinstance(param, DTensor):
            shape = list(param.data._local_tensor.shape)
        else:
            shape = list(param.shape)
        for i in range(len(shape)):
            if shape[i] % efsdp_size == 0:
                return Shard(i)
        return Shard(0)

    return shard_placement_fn


def _clone_mp_policy(mp_policy, **overrides):
    if mp_policy is None:
        return None

    if dataclasses.is_dataclass(mp_policy):
        return dataclasses.replace(mp_policy, **overrides)

    # Try reconstructing via constructor from common attributes.
    attrs = {}
    for k in ("param_dtype", "reduce_dtype", "output_dtype", "cast_forward_inputs"):
        if hasattr(mp_policy, k):
            attrs[k] = getattr(mp_policy, k)
    attrs.update(overrides)
    return mp_policy.__class__(**attrs)


def _fsdp_kwargs_for_module(fsdp_kwargs: dict, module: nn.Module) -> dict:
    """
    Allows overriding FSDP2 kwargs per module, e.g. disabling mp_policy.cast_forward_inputs
    for specific classes like VL blocks.
    """
    mp_policy = fsdp_kwargs.get("mp_policy", None)
    if mp_policy is None or not hasattr(mp_policy, "cast_forward_inputs"):
        return fsdp_kwargs

    attr_override = getattr(module, "_fsdp2_cast_forward_inputs", None)
    if attr_override is not None:
        desired = bool(attr_override)
    else:
        desired = False

    if desired == mp_policy.cast_forward_inputs:
        return fsdp_kwargs

    new_kwargs = dict(fsdp_kwargs)
    new_kwargs["mp_policy"] = _clone_mp_policy(mp_policy, cast_forward_inputs=desired)
    return new_kwargs


def apply_fsdp2(model, fsdp_kwargs, moe_fsdp_kwargs, config, is_lora=False):
    """
    model: AutoModelForCausalLM

    Reference: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
    and LoRA Patch: https://github.com/volcengine/verl/issues/3470

    """
    assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"

    model_cfg = getattr(model, "config", None)
    is_moe = _is_moe_config(model_cfg)
    apply_expert_patch = bool(config.get("apply_expert_patch", False))
    if version.parse(transformers.__version__) >= version.parse("5.2.0"):
        apply_expert_patch = False
        logger.warning("[apply_fsdp2] apply_expert_patch was set as False automatically, because transformers>=5.2.0 uses fused experts")

    if is_moe and apply_expert_patch:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

        from roll.third_party.fsdp2.qwen3_moe_patch import qwen3_moe_forward

        Qwen3MoeSparseMoeBlock.forward = qwen3_moe_forward
        print("[apply_fsdp2] Applied expert patch for Qwen3MoeSparseMoeBlock")

    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = config.get("wrap_policy", {}).get(
        "transformer_layer_cls_to_wrap",
        default_transformer_cls_names_to_wrap,
    )

    if fsdp_transformer_layer_cls_to_wrap is None:
        fsdp_transformer_layer_cls_to_wrap = []
    elif isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]
    else:
        fsdp_transformer_layer_cls_to_wrap = list(fsdp_transformer_layer_cls_to_wrap)

    assert len(fsdp_transformer_layer_cls_to_wrap) > 0 and all(
        layer_cls is not None for layer_cls in fsdp_transformer_layer_cls_to_wrap
    )

    wrap_embeddings = bool(config.get("wrap_policy", {}).get("wrap_embeddings", False))
    wrap_lm_output = bool(config.get("wrap_policy", {}).get("wrap_lm_output", False))

    def _get_embed_tokens(m: nn.Module):
        inner = getattr(m, "model", None)
        if inner is not None and hasattr(inner, "embed_tokens"):
            return getattr(inner, "embed_tokens")
        if hasattr(m, "embed_tokens"):
            return getattr(m, "embed_tokens")
        if hasattr(m, "get_input_embeddings"):
            return m.get_input_embeddings()
        return None

    def _already_fully_sharded(mod: nn.Module) -> bool:
        # `fully_shard()` mutates the module into an internal FSDPModule type. If so, do not re-apply.
        return fully_shard_module is not None and isinstance(mod, fully_shard_module.FSDPModule)

    lora_modules = []
    selected = []
    moe_modules = []
    for name, module in model.named_modules():
        if is_lora and (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            lora_modules.append(module)

        # PumpkinComment:
        #  (MoE): Do NOT FSDP-wrap individual experts by default.
        # Experts are invoked conditionally per-rank (based on routing),
        # so wrapping `experts.*` as separate FSDP modules can deadlock collectives when
        # different ranks activate different experts. Therefor we only wrap experts
        # if we apply the expert patch or expert parallelism is enabled.
        if is_moe and (config.get("apply_expert_patch", False) or config.get("ep_size", 1) > 1):
            moe_block = config.get("wrap_policy", {}).get("moe_experts", None)
            if isinstance(moe_block, str):
                moe_block = [moe_block]
            if moe_block is not None and module.__class__.__name__ in moe_block:
                moe_modules.append(module)
                print("[apply_fsdp2] Wrapped MoE expert module: ", name, module.__class__.__name__)

        # If `wrap_embeddings` is enabled, embeddings are handled explicitly below to avoid double wrapping.
        if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (
            (not wrap_embeddings)
            and isinstance(module, nn.Embedding)
            and (not getattr(getattr(model, "config", None), "tie_word_embeddings", True))
        ):
            selected.append((name, module))

    # PumpkinComment:
    # Avoid wrapping both a parent module and its child module with the same mesh.
    selected_names = [n for n, _ in selected]
    non_leaf = set()
    for n in selected_names:
        if not n:
            continue
        parts = n.split(".")
        for i in range(1, len(parts)):
            non_leaf.add(".".join(parts[:i]))

    modules = [m for n, m in selected if n not in non_leaf]

    wrapped_ids = set()

    def _wrap_once(mod: nn.Module, kwargs: dict):
        if mod is None:
            return
        if id(mod) in wrapped_ids:
            return
        if _already_fully_sharded(mod):
            wrapped_ids.add(id(mod))
            return
        with maybe_patch_fsdp_module(mod):
            fully_shard(mod, **kwargs)
        wrapped_ids.add(id(mod))

    # 1. Embeddings
    if wrap_embeddings:
        _wrap_once(_get_embed_tokens(model), fsdp_kwargs)

    # 2. LoRA Modules (Linear Layer)
    for idx, module in enumerate(lora_modules):
        _wrap_once(module, fsdp_kwargs)

    # 3. MoE
    for idx, module in enumerate(moe_modules):
        _wrap_once(module, moe_fsdp_kwargs)

    # 4. Transformers Layers
    for idx, module in enumerate(modules):
        _wrap_once(module, _fsdp_kwargs_for_module(fsdp_kwargs, module))

    # 5. LM Output
    if wrap_lm_output:
        _wrap_once(getattr(model, "lm_head", None), fsdp_kwargs)

    # Root wrap last for remaining modules. (FSDP2 will not reshard_after_forward for the root module.)
    root_kwargs = dict(fsdp_kwargs)
    root_kwargs["mp_policy"] = _clone_mp_policy(root_kwargs.get("mp_policy", None), cast_forward_inputs=False)
    _wrap_once(model, root_kwargs)


def fsdp2_load_full_state_dict(
    model: torch.nn.Module,
    full_state: dict,
    device_mesh=None,
    cpu_offload=None,
    ep_enabled=False,
):
    """
    Reference: https://github1s.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py

    Loads the full state dict (could be only on rank 0) into the sharded model. This is done by broadcasting the
    parameters from rank 0 to all other ranks. This function modifies the model in-place.

    Args:
        model (`torch.nn.Module`): The model to load the state dict into
        full_state (`dict`): The full state dict to load, can only be on rank 0
    """

    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    device_id = current_platform.current_device()

    if not ep_enabled:
        if dist.get_rank() == 0:
            model = model.to(device=device_id, non_blocking=True)
        else:
            model = model.to_empty(device=device_id)

        cpu_offload = cpu_offload is not None
        options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=cpu_offload,
            broadcast_from_rank0=True,
        )
        set_model_state_dict(model, full_state, options=options)
    else:
        model = model.to(device=device_id, non_blocking=True)

    # rotary_emb is not in state_dict, so we need to broadcast it manually
    for name, buf in model.named_buffers():
        dist.broadcast(buf, src=0)

    if cpu_offload:
        # Ensure model is on CPU but buffers are on GPU for FSDP2 CPU offload
        model.to("cpu", non_blocking=True)
        for buf in model.buffers():
            buf.data = buf.data.to(device_id)

def _permute(x, num_tokens_per_expert, ep_degree, num_local_experts):
    """
    Reference: https://github.com/pytorch/torchtitan/blob/73a0e6979dd10b6b1904098eb3c8f62c18ab87ce/torchtitan/models/moe/utils.py #42 #v0.2.2
    """
    global TOKEN_GROUP_ALIGN_SIZE_M
    x_padded_per_expert = x.shape[0] + num_local_experts * TOKEN_GROUP_ALIGN_SIZE_M
    padded_max_len = _round_up(x_padded_per_expert, TOKEN_GROUP_ALIGN_SIZE_M)
    with torch.no_grad():
        (permuted_indices, num_tokens_per_expert, _offsets,) = generate_permute_indices(
            num_tokens_per_expert,
            num_local_experts,
            ep_degree,
            padded_max_len,
            TOKEN_GROUP_ALIGN_SIZE_M,
        )

    x = torch.vstack((x, x.new_zeros((x.shape[-1]))))
    input_shape = x.shape
    x = x[permuted_indices, :]

    return input_shape, x, permuted_indices, num_tokens_per_expert


def _unpermute(out, input_shape, permuted_indices):
    """
    Reference: https://github.com/pytorch/torchtitan/blob/73a0e6979dd10b6b1904098eb3c8f62c18ab87ce/torchtitan/models/moe/utils.py #62 #v0.2.2
    """
    out_unpermuted = out.new_empty(input_shape)
    out_unpermuted[permuted_indices, :] = out
    out = out_unpermuted[:-1]
    return out

def _round_up(x: int, y: int) -> int:
    """
    Round up x to the nearest multiple of y.
    Reference: https://github.com/pytorch/torchtitan/blob/73a0e6979dd10b6b1904098eb3c8f62c18ab87ce/torchtitan/tools/utils.py #228 #v0.2.2
    """
    x_ceil_div_y = (x + y - 1) // y
    return x_ceil_div_y * y

def generate_permute_indices(
    tokens_per_expert_group: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
    alignment: int,
    use_cpu: bool = False,
):
    """
    Prepare permutation indices and the number of tokens for each expert.

    Args:
        tokens_per_expert_group: number of tokens for each expert from all ranks.
        experts_per_rank: number of experts per rank.
        num_ranks: number of ranks.
        max_len: maximum length of the output index vector.
        alignment: alignment for each returned element in `m_sizes` and padding min for zero token experts.
        use_cpu: whether to use CPU implementation.


    Returns:
        permuted_indices: Tensor of indices that map original token order to the expert-grouped order.
        m_sizes: aligned number of tokens for each expert (padded to alignment boundary).
        m_offsets: Cumulative sum of m_sizes. The exclusive ending position for each expert's tokens.

    Explanatory details:
        `tokens_per_expert_group` is of shape (num_ranks * experts_per_rank,), for example:
        From: |       rank 0      |       rank 1      |
        To:   | E0 | E1 | E2 | E3 | E0 | E1 | E2 | E3 |
              |  4 |  2 |  1 |  3 |  1 |  2 |  3 |  4 |

    Reference: https://github.com/pytorch/torchtitan/blob/73a0e6979dd10b6b1904098eb3c8f62c18ab87ce/torchtitan/models/moe/kernels.py #143 #v0.2.2
    """

    # prefix sum to get start index of each expert (parallel scan kernel in future?)
    start_index_values = (
        torch.cumsum(tokens_per_expert_group, 0) - tokens_per_expert_group
    )

    # total tokens for each expert (sum over ranks)
    total_tokens_per_expert = tokens_per_expert_group.view(num_ranks, -1).sum(0)

    # pad out empty experts to alignment requirement
    total_tokens_per_expert = torch.clamp_min(total_tokens_per_expert, alignment)

    # align the chunk sizes (cdiv)
    m_sizes = ((total_tokens_per_expert + alignment - 1) // alignment * alignment).to(
        torch.int32
    )

    # additional prefix sum to get write offset of each expert in permuted_indices
    # write offsets is per local expert, not global
    m_offsets = torch.cumsum(m_sizes, 0)
    write_offsets = m_offsets - m_sizes

    if use_cpu or not _TRITON_AVAILABLE:
        permuted_indices = fill_indices_cpu(
            tokens_per_expert_group,
            start_index_values,
            write_offsets,
            experts_per_rank,
            num_ranks,
            max_len,
        )
    else:
        permuted_indices = fill_indices_wrapper(
            tokens_per_expert_group,
            start_index_values,
            write_offsets,
            experts_per_rank,
            num_ranks,
            max_len,
        )

    return permuted_indices, m_sizes, m_offsets.to(torch.int32)

# ============================================================================
# Triton GPU implementation of fill_indices.
# Avoids hundreds of D2H .item() syncs per MoE layer that the CPU fallback
# incurs. The kernel and wrapper below are guarded by `_TRITON_AVAILABLE` and
# only defined when triton can be imported, so environments without triton
# (e.g. CPU-only debug runs) keep working through the CPU fallback path.
# Reference: https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/moe/kernels.py
# ============================================================================
if _TRITON_AVAILABLE:

    @triton.jit
    def _fill_indices_kernel(
        tokens_per_expert_group_ptr,
        start_index_values_ptr,
        write_offsets_ptr,
        output_ptr,
        experts_per_rank: tl.constexpr,
        num_ranks: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,  # Number of threads per block
    ):
        pid = tl.program_id(axis=0)
        num_programs = tl.num_programs(axis=0)

        # map programs (blocks) to the experts and loop (grid stride) if needed
        for expert_id in range(pid, experts_per_rank, num_programs):
            # read this expert's write offset
            write_offset = tl.load(write_offsets_ptr + expert_id)

            for r in range(num_ranks):
                # index into tokens_per_expert_group array
                i = r * experts_per_rank + expert_id

                # load start index and number of tokens for this expert-rank pair
                start_index = tl.load(start_index_values_ptr + i)
                length = tl.load(tokens_per_expert_group_ptr + i)

                # each thread in block processes tokens in parallel
                offsets = tl.arange(0, BLOCK_SIZE)

                # tokens are processed in chunks of BLOCK_SIZE
                for chunk_start in range(0, length, BLOCK_SIZE):
                    chunk_offsets = chunk_start + offsets
                    mask = chunk_offsets < length
                    values = start_index + chunk_offsets
                    dest_indices = write_offset + chunk_offsets
                    tl.store(output_ptr + dest_indices, values, mask=mask)

                # update write offset for next rank
                write_offset += length

def fill_indices_wrapper(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
    block_size: int = 128,
    max_blocks: int = 1024,  # cap on total number of blocks to launch
):
    """
    GPU implementation of fill_indices via a Triton kernel. All inputs must be
    on the same CUDA device; the returned tensor lives on that device too.

    Reference: https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/moe/kernels.py
    """
    assert _TRITON_AVAILABLE, "fill_indices_wrapper requires triton to be installed."
    # preallocate output on the same device as the inputs (GPU)
    permuted_indices = torch.full(
        (max_len,), -1, dtype=torch.int32, device=tokens_per_expert_group.device,
    )

    # one block per local expert (capped to avoid launching huge grids on
    # configurations with very many experts per rank)
    num_blocks = min(experts_per_rank, max_blocks)
    grid = (num_blocks,)

    _fill_indices_kernel[grid](
        tokens_per_expert_group,
        start_index_values,
        write_offsets,
        permuted_indices,
        experts_per_rank,
        num_ranks,
        BLOCK_SIZE=block_size,
    )
    return permuted_indices

def fill_indices_cpu(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
):
    """
    Reference: https://github.com/pytorch/torchtitan/blob/73a0e6979dd10b6b1904098eb3c8f62c18ab87ce/torchtitan/models/moe/kernels.py #106 #v0.2.2
    """
    # We need to preallocate the output - we ignore device and force it on cpu
    # device = tokens_per_expert_group.device
    permuted_indices = torch.full(
        (max_len,),
        -1,
        dtype=torch.int32,
    )  # device=device)
    # Fill the permuted indices
    # For each local expert
    for e in range(experts_per_rank):
        write_start = write_offsets[e].item()
        # For each remote rank
        for r in range(num_ranks):
            i = r * experts_per_rank + e
            start_index = start_index_values[i].item()
            length = tokens_per_expert_group[i].item()
            # Fill in the indices
            if length > 0:
                end_idx = min(write_start + length, max_len)
                permuted_indices[write_start:end_idx] = torch.arange(
                    start_index,
                    start_index + (end_idx - write_start),
                    dtype=torch.int32,
                    # device=device,
                )
            write_start += length
    return permuted_indices

def register_experts_forward_in_ExpertsInterface():
    from transformers.integrations.moe import ExpertsInterface
    ExpertsInterface.register("ep", ep_experts_forward)

def ep_experts_forward(
    self,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """
    Reference: https://github.com/pytorch/torchtitan/blob/73a0e6979dd10b6b1904098eb3c8f62c18ab87ce/torchtitan/models/moe/moe.py #L148 #v0.2.2
    """
    if isinstance(self.gate_up_proj, DTensor):
        # Convert parameters from DTensors to plain Tensors, to work with
        # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
        gate_up_proj = self.gate_up_proj.to_local()
        down_proj = self.down_proj.to_local()
    else:
        gate_up_proj = self.gate_up_proj
        down_proj = self.down_proj
    
    use_grouped_mm = get_use_grouped_mm()
    if use_grouped_mm:
        return _run_experts_grouped_mm(gate_up_proj, down_proj, x, num_tokens_per_expert, self.act_fn)
    else:
        return _run_experts_for_loop(gate_up_proj, down_proj, x, num_tokens_per_expert, self.act_fn)
    
def _run_experts_grouped_mm(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    act_fn: Callable
) -> torch.Tensor:
    """
    Reference: https://github.com/pytorch/torchtitan/blob/73a0e6979dd10b6b1904098eb3c8f62c18ab87ce/torchtitan/models/moe/moe.py #L113 #v0.2.2
    """
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    gate, up = torch._grouped_mm(
        x.bfloat16(), gate_up_proj.bfloat16().transpose(-2, -1), offs=offsets
    ).chunk(2, dim=-1)
    h = act_fn(gate) * up
    out = torch._grouped_mm(
        h, down_proj.bfloat16().transpose(-2, -1), offs=offsets
    ).type_as(x)

    return out

def _run_experts_for_loop(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    act_fn: Callable,
) -> torch.Tensor:
    """
    Reference: https://github.com/pytorch/torchtitan/blob/73a0e6979dd10b6b1904098eb3c8f62c18ab87ce/torchtitan/models/moe/moe.py #L78 #v0.2.2
    """
    # NOTE: this would incur a synchronization between device and host
    num_tokens_per_expert_list = num_tokens_per_expert.tolist()

    # side-effect code due to the usage of generate_permute_indices
    num_padding = x.shape[0] - sum(num_tokens_per_expert_list)

    # a tuple of tensors indexed by experts
    # each with shape (tokens_per_expert(varying), dim)
    x_splits = torch.split(
        x[: sum(num_tokens_per_expert_list)],
        split_size_or_sections=num_tokens_per_expert_list,
        dim=0,
    )
    out_experts_splits = []
    for expert_idx, x_expert in enumerate(x_splits):
        gate, up = nn.functional.linear(x_expert, gate_up_proj[expert_idx]).chunk(2, dim=-1)
        h = act_fn(gate) * up
        h = nn.functional.linear(h, down_proj[expert_idx])

        # h shape (tokens_per_expert(varying), dim)
        out_experts_splits.append(h)
    out = torch.cat(out_experts_splits, dim=0)

    # side-effect code due to the usage of generate_permute_indices
    out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))

    return out