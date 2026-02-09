import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from roll.utils.context_parallel.globals import set_upg_manager
from roll.utils.context_parallel.monkey_patch import apply_ulysses_patch, unapply_ulysses_patch


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _pad_to(x: torch.Tensor, target: int, *, dim: int = 1) -> torch.Tensor:
    if x.size(dim) >= target:
        return x
    pad_len = target - x.size(dim)
    pad = [0, 0] * x.ndim
    pad[2 * (x.ndim - 1 - dim) + 1] = pad_len
    return torch.nn.functional.pad(x, pad, value=0)


def _gather_seq_shards(x_local: torch.Tensor, lens: list[int], group) -> torch.Tensor:
    max_len = max(lens)
    x_pad = _pad_to(x_local, max_len, dim=1)
    gathered = [torch.empty_like(x_pad) for _ in range(len(lens))]
    dist.all_gather(gathered, x_pad, group=group)
    parts = [g[:, :l] for g, l in zip(gathered, lens)]
    return torch.cat(parts, dim=1)


def _worker_qwen3_hf_equivalence(rank: int, world_size: int, port: int, model_id: str) -> None:
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("flash_attn")

    if not torch.cuda.is_available():
        pytest.skip("Qwen3 HF + FlashAttention2 CP equivalence test requires CUDA")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need >= {world_size} CUDA devices, got {torch.cuda.device_count()}")

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, trust_remote_code=True)
        except Exception as e:
            pytest.skip(f"Tokenizer for {model_id} not available locally: {e}")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        except Exception as e:
            pytest.skip(f"Model for {model_id} not available locally (or FA2 unsupported): {e}")

        model.to(device)
        model.eval()

        patch_info = apply_ulysses_patch()
        if patch_info is None or (isinstance(patch_info, dict) and not patch_info.get("patched", True)):
            pytest.skip("Ulysses patch was not applied (no FlashAttention2 hook patched)")

        max_len = 64
        assert max_len % world_size == 0

        # One long "real-ish" prompt (tokenized by the real tokenizer).
        text = (
            "Explain Ulysses context parallelism in Transformers with FlashAttention2. "
            "Include a short example and mention sequence sharding, all-to-all, and why it preserves global attention. "
        )
        for _ in range(8):
            enc = tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=max_len,
                add_special_tokens=True,
            )
            if enc["input_ids"].size(1) >= max_len:
                break
            text = text + " Add more technical detail about rotary embeddings and KV heads."

        input_ids = enc["input_ids"][:, :max_len].to(device)
        # Important for equivalence: RoPE/position embedding is applied before the FA2 hook.
        position_ids = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            # Baseline: CP disabled (ulysses_size=1 means the patch is a no-op).
            set_upg_manager(ulysses_size=1, rank=rank, world_size=world_size)
            baseline = model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False,
            ).logits

            # CP: enable Ulysses group and run on local sequence shard, then gather to full logits.
            set_upg_manager(ulysses_size=world_size, rank=rank, world_size=world_size)

            local_len = max_len // world_size
            start = rank * local_len
            end = start + local_len

            input_ids_local = input_ids[:, start:end]
            position_ids_local = position_ids[:, start:end]

            logits_local = model(
                input_ids=input_ids_local,
                position_ids=position_ids_local,
                use_cache=False,
            ).logits

        group = dist.group.WORLD
        lens = [local_len for _ in range(world_size)]
        logits_cp_full = _gather_seq_shards(logits_local.float(), lens, group)
        baseline_full = baseline.float()

        if rank == 0:
            torch.testing.assert_close(logits_cp_full, baseline_full, rtol=2e-2, atol=2e-2)
    finally:
        try:
            unapply_ulysses_patch()
        except Exception:
            pass
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA + FlashAttention2")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires >= 2 CUDA devices for CP all-to-all")
def test_fsdp2_cp_qwen3_hf_logits_equivalence():
    world_size = 2
    port = _find_free_port()
    model_id = os.environ.get(
        "ROLL_TEST_QWEN3_MODEL_ID", "/home/dilixiati.dlxtmhte/.cache/openlm/hub/14ffd5928d24731fd670f04c645a5928"
    )
    mp.spawn(
        _worker_qwen3_hf_equivalence,
        args=(world_size, port, model_id),
        nprocs=world_size,
        join=True,
    )
