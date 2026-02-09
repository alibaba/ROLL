import os

import torch
import torch.distributed as dist

from roll.utils.context_parallel.globals import get_ulysses_group, get_ulysses_size, set_upg_manager
from roll.utils.context_parallel.hf_flash_attention_patch import make_ulysses_flash_attention_forward


def _pad_to(x: torch.Tensor, target: int) -> torch.Tensor:
    if x.size(1) >= target:
        return x
    pad_len = target - x.size(1)
    pad = [0, 0] * x.ndim
    pad[2 * (x.ndim - 2) + 1] = pad_len
    return torch.nn.functional.pad(x, pad, value=0)


def _gather_seq_shards(x_local: torch.Tensor, lens: list[int], group) -> torch.Tensor:
    max_len = max(lens)
    x_pad = _pad_to(x_local, max_len)
    gathered = [torch.empty_like(x_pad) for _ in range(len(lens))]
    dist.all_gather(gathered, x_pad, group=group)
    parts = [g[:, :l] for g, l in zip(gathered, lens)]
    return torch.cat(parts, dim=1)


def original_forward(query_states, key_states, value_states, attention_mask, query_length, *args, **kwargs):
    # A head-wise function that depends on the full sequence length, so CP needs correct all-to-all.
    # Shape in/out: (bs, seqlen, heads, dim)
    assert query_states.size(1) == query_length
    global_mix = query_states.mean(dim=1, keepdim=True)  # (bs, 1, heads, dim)
    return query_states + global_mix


def main():
    backend = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, "This smoke test expects torchrun --nproc_per_node=2"

    # Use the full world as the CP group for simplicity.
    set_upg_manager(ulysses_size=world, rank=rank, world_size=world)
    group = get_ulysses_group()
    assert group is not None and get_ulysses_size() == world

    # Variable local lengths to simulate remove-padding imbalance.
    local_len = 2 + rank  # rank0=2, rank1=3 => total=5
    lens_t = torch.tensor([local_len], dtype=torch.int64)
    lens_list = [torch.zeros_like(lens_t) for _ in range(world)]
    dist.all_gather(lens_list, lens_t, group=group)
    lens = [int(x.item()) for x in lens_list]
    total_len = sum(lens)

    # Shapes
    bs, heads, dim = 1, 4, 2  # heads divisible by world

    torch.manual_seed(1234)
    q_local = torch.randn(bs, local_len, heads, dim)
    k_local = torch.randn(bs, local_len, heads, dim)
    v_local = torch.randn(bs, local_len, heads, dim)
    attn_mask_local = torch.ones(bs, local_len, dtype=torch.long)

    # Wrapped call (simulates patched HF hook)
    wrapped = make_ulysses_flash_attention_forward(original_forward)
    out_local = wrapped(q_local, k_local, v_local, attn_mask_local, local_len)

    # Baseline: run original on the *global* sequence (cp_size=1 semantics), then slice back to local.
    q_global = _gather_seq_shards(q_local, lens, group)
    k_global = _gather_seq_shards(k_local, lens, group)
    v_global = _gather_seq_shards(v_local, lens, group)
    attn_mask_global = _gather_seq_shards(attn_mask_local.unsqueeze(-1).to(q_local.dtype), lens, group).squeeze(-1)

    baseline_global = original_forward(q_global, k_global, v_global, attn_mask_global, total_len)

    start = sum(lens[:rank])
    end = start + local_len
    baseline_local = baseline_global[:, start:end]

    torch.testing.assert_close(out_local, baseline_local, rtol=0, atol=1e-6)

    if rank == 0:
        print("Ulysses wrapper equivalence smoke test passed.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
    main()
