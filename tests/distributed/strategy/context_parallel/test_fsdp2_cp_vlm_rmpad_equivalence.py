import inspect
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from roll.models.model_providers import get_extra_data_provider, load_model
from roll.utils.context_parallel.globals import get_ulysses_group, set_upg_manager
from roll.utils.context_parallel.monkey_patch import apply_ulysses_patch, unapply_ulysses_patch
from roll.utils.context_parallel.rmpad_ulysses import gather_outputs_and_unpad, ulysses_pad_inputs

try:
    # Optional debugging capture utilities used elsewhere in tests.
    from tests.distributed.strategy.log_probs.layer_states_capture import is_enabled as _capture_is_enabled
    from tests.distributed.strategy.log_probs.layer_states_capture import save_tensor as _capture_save_tensor
except Exception:  # pragma: no cover

    def _capture_is_enabled() -> bool:
        return False

    def _capture_save_tensor(*_args, **_kwargs):
        return None


def _maybe_save_cp_gathered_tensors(
    *,
    rank: int,
    base_logits: torch.Tensor | None,
    cp_gathered_logits: torch.Tensor | None,
    attention_mask: torch.Tensor | None = None,
):
    """
    Opt-in persistence of gathered CP outputs to debug divergence.

    Enable either:
    - CP_GATHER_SAVE_DIR=/path (saves via torch.save to that directory), OR
    - LAYER_STATES_SAVE_DIR=... (uses layer_states_capture.save_tensor), plus CP_SAVE_GATHERED=1
      (handy when you already have layer-state capture configured).

    Notes:
    - We save only on rank0 by default to avoid duplicate files.
    - We also save a small per-token error map to quickly localize divergence.
    """
    if os.getenv("CP_SAVE_GATHERED", "0") != "1":
        return
    if rank != 0:
        return
    if base_logits is None or cp_gathered_logits is None:
        return

    with torch.no_grad():
        # (bs, seq, vocab) -> (bs, seq)
        err_absmax = (cp_gathered_logits.float() - base_logits.float()).abs().amax(dim=-1)
        if attention_mask is not None:
            err_absmax = err_absmax * attention_mask.to(err_absmax.dtype)

    save_dir = os.getenv("CP_GATHER_SAVE_DIR", "").strip()
    prefix = os.getenv("CP_GATHER_PREFIX", "cp_gather").strip() or "cp_gather"
    step = os.getenv("LAYER_STATES_STEP", "0")
    batch = os.getenv("LAYER_STATES_BATCH", "0")

    if save_dir:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(base_logits.detach().cpu(), out_dir / f"{prefix}_step{step}_batch{batch}_base_logits.pt")
        torch.save(
            cp_gathered_logits.detach().cpu(), out_dir / f"{prefix}_step{step}_batch{batch}_cp_gathered_logits.pt"
        )
        torch.save(err_absmax.detach().cpu(), out_dir / f"{prefix}_step{step}_batch{batch}_cp_vs_base_err_absmax.pt")
        return

    if _capture_is_enabled():
        _capture_save_tensor(base_logits, "base_logits", subdir="cp_gather")
        _capture_save_tensor(cp_gathered_logits, "cp_gathered_logits", subdir="cp_gather")
        _capture_save_tensor(err_absmax, "cp_vs_base_err_absmax", subdir="cp_gather")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _make_dummy_pil_image():
    PIL = pytest.importorskip("PIL")
    from PIL import Image

    # Deterministic small RGB image.
    w, h = 32, 32
    arr = torch.arange(w * h * 3, dtype=torch.uint8).reshape(h, w, 3).numpy()
    return Image.fromarray(arr, mode="RGB")


def _build_mm_batch(model_path: str, device: torch.device, max_len: int = 64):
    transformers = pytest.importorskip("transformers")
    from transformers import AutoProcessor, AutoTokenizer

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    img = _make_dummy_pil_image()
    # Qwen-VL style models require explicit vision placeholder tokens in the text stream
    # so that image/video features can be scattered into matching token positions.
    text = "<|vision_start|><|image_pad|><|vision_end|> Describe the image briefly."
    # Many VLM processors accept `text` + `images`; keep it simple and deterministic.
    #
    # Qwen3-VL is strict about multimodal token counts: if truncation clips placeholder tokens,
    # it raises an error. So we disable truncation and retry with a larger max_length if needed.
    last_err = None
    for trial_max_len in (max_len, 128, 256, 512):
        try:
            model_inputs = processor(
                text=[text],
                images=[img],
                return_tensors="pt",
                padding="max_length",
                truncation=False,
                max_length=trial_max_len,
            )
            max_len = trial_max_len
            break
        except ValueError as e:
            last_err = e
            continue
    else:
        raise last_err  # type: ignore[misc]
    model_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in dict(model_inputs).items()}

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]

    # Position ids: use existing ROLL provider (qwen2-vl) or default (others, incl qwen3-vl).
    extra_provider = get_extra_data_provider(model_path, processor=processor)
    extra_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "image_grid_thw": model_inputs.get("image_grid_thw"),
        "video_grid_thw": model_inputs.get("video_grid_thw"),
    }
    # `get_extra_data_provider()` returns providers with different signatures:
    # - Qwen2-VL-style provider expects image/video grid args
    # - default provider only accepts (input_ids, attention_mask)
    try:
        sig = inspect.signature(extra_provider)
        accepted = set(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in extra_kwargs.items() if k in accepted}
        extra = extra_provider(**filtered_kwargs)
    except Exception:
        # Best-effort fallback (handles unexpected kwargs TypeError).
        extra = extra_provider(input_ids=input_ids, attention_mask=attention_mask)
    position_ids = extra["position_ids"].to(device)
    # Match strategy behavior: (bs, C, seqlen) -> (C, bs, seqlen)
    if position_ids.dim() == 3:
        position_ids = position_ids.transpose(0, 1).contiguous()

    # Keep only tensors relevant for forward.
    mm_args = {}
    for k in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        if k in model_inputs and torch.is_tensor(model_inputs[k]):
            mm_args[k] = model_inputs[k]
    # Some VLMs have conditional vision tower paths; keep consistent with pipelines.
    mm_args["force_vit_image"] = True

    return input_ids, attention_mask, position_ids, mm_args


def _to_rmpad(input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor):
    pytest.importorskip("flash_attn")
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input

    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

    if position_ids.dim() == 3:
        position_ids_rmpad = (
            index_first_axis(
                rearrange(position_ids, "c b s ... -> (b s) c ..."),
                indices,
            )
            .transpose(0, 1)
            .unsqueeze(1)
        )  # (C, 1, total_nnz)
    else:
        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
            indices,
        ).transpose(
            0, 1
        )  # (1, total_nnz)

    def pad_back(x_rmpad: torch.Tensor) -> torch.Tensor:
        # x_rmpad: (1, total_nnz, ...)
        dense = pad_input(
            hidden_states=x_rmpad.squeeze(0).unsqueeze(-1),
            indices=indices,
            batch=input_ids.size(0),
            seqlen=input_ids.size(1),
        ).squeeze(-1)
        return dense

    return input_ids_rmpad, position_ids_rmpad, pad_back


def _worker_vlm_cp_equivalence(rank: int, world_size: int, port: int, model_path: str):
    pytest.importorskip("transformers")
    pytest.importorskip("flash_attn")

    if not torch.cuda.is_available():
        pytest.skip("VLM CP equivalence test requires CUDA")
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

        # Patch HF attention hooks for Ulysses.
        patch_info = apply_ulysses_patch()
        if patch_info is None or (isinstance(patch_info, dict) and not patch_info.get("patched", True)):
            pytest.skip("Ulysses patch was not applied (no FlashAttention2 hook patched)")

        # Load model via ROLL provider so our VLM CP decoder patch is exercised.
        from roll.configs.model_args import ModelArguments

        model_args = ModelArguments(
            model_name_or_path=model_path,
            attn_implementation="fa2",
            dtype="bf16",
            ulysses_size=world_size,  # install decoder slice patch; runtime CP size controlled by set_upg_manager
        )
        # Force each rank to keep weights on its own GPU.
        model_args.device_map = {"": rank}

        model = load_model(model_args=model_args, is_trainable=False)
        model.eval()

        input_ids, attention_mask, position_ids, mm_args = _build_mm_batch(model_path, device=device, max_len=256)
        input_ids_rmpad, position_ids_rmpad, pad_back = _to_rmpad(input_ids, attention_mask, position_ids)

        # Baseline: CP disabled (ulysses_size=1 semantics) on the same world_size job.
        set_upg_manager(ulysses_size=1, rank=rank, world_size=world_size)
        with torch.no_grad():
            out_base = model(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                use_cache=False,
                **mm_args,
            ).logits  # (1, total_nnz, vocab)
            dense_base = pad_back(out_base)

        # CP: use slice-after-embedding (pad-only here, slice in decoder patch).
        set_upg_manager(ulysses_size=world_size, rank=rank, world_size=world_size)
        input_ids_pad, pos_pad, pad_size = ulysses_pad_inputs(
            input_ids_rmpad,
            position_ids_rmpad,
            cp_size=world_size,
        )
        with torch.no_grad():
            out_local = model(
                input_ids=input_ids_pad,
                attention_mask=None,
                position_ids=pos_pad,
                use_cache=False,
                **mm_args,
            ).logits  # (1, local_tokens, vocab)

            out_full = gather_outputs_and_unpad(
                out_local,
                gather_dim=1,
                unpad_dim=1,
                padding_size=pad_size,
                group=get_ulysses_group(),
            )
            dense_cp = pad_back(out_full)

        _maybe_save_cp_gathered_tensors(
            rank=rank,
            base_logits=dense_base,
            cp_gathered_logits=dense_cp,
            attention_mask=attention_mask,
        )

        if rank == 0:
            mask = attention_mask.to(torch.bool)
            # Compare a small vocabulary slice to reduce memory pressure.
            dense_base_s = dense_base[..., :64].float()
            dense_cp_s = dense_cp[..., :64].float()
            torch.testing.assert_close(dense_cp_s[mask], dense_base_s[mask], rtol=3e-2, atol=3e-2)
    finally:
        try:
            unapply_ulysses_patch()
        except Exception:
            pass
        dist.destroy_process_group()


def _worker_vlm_cp_equivalence_nonrmpad(rank: int, world_size: int, port: int, model_path: str):
    pytest.importorskip("transformers")
    pytest.importorskip("flash_attn")

    if not torch.cuda.is_available():
        pytest.skip("VLM CP equivalence test requires CUDA")
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

        patch_info = apply_ulysses_patch()
        if patch_info is None or (isinstance(patch_info, dict) and not patch_info.get("patched", True)):
            pytest.skip("Ulysses patch was not applied (no FlashAttention2 hook patched)")

        from roll.configs.model_args import ModelArguments

        model_args = ModelArguments(
            model_name_or_path=model_path,
            attn_implementation="fa2",
            dtype="bf16",
            ulysses_size=world_size,
        )
        model_args.device_map = {"": rank}
        model = load_model(model_args=model_args, is_trainable=False)
        from tests.distributed.strategy.log_probs.apply_model_patch import apply_qwen3vl_patches

        if apply_qwen3vl_patches():
            print("Applied Qwen3VL layer states capture patches")
        model.eval()

        # Use a length divisible by world_size to match CP shard requirements.
        input_ids, attention_mask, position_ids, mm_args = _build_mm_batch(model_path, device=device, max_len=256)
        assert input_ids.size(1) % world_size == 0

        # Baseline (CP disabled) -> full logits.
        os.environ["LAYER_STATES_SAVE_DIR"] = "./cp_layerwise_out/base"
        os.environ["LAYER_STATES_PREFIX"] = "base"
        os.environ["LAYER_STATES_STEP"] = "0"
        os.environ["LAYER_STATES_BATCH"] = "0"
        if rank == 0:  # attach only one process to avoid chaos
            import debugpy

            debugpy.listen(("0.0.0.0", 5679))
            print("Waiting for debugger attach on 5678...")
            debugpy.wait_for_client()
            debugpy.breakpoint()  # or use breakpoint() after attach
        set_upg_manager(ulysses_size=1, rank=rank, world_size=world_size)
        with torch.no_grad():
            base_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
                **mm_args,
            )

            base_states = base_output.hidden_states
            base_layer_states = base_output.layer_states
            base = base_output.logits  # (bs, seq, vocab)

        # CP enabled -> decoder outputs local shard; gather to full for comparison.
        os.environ["LAYER_STATES_SAVE_DIR"] = "./cp_layerwise_out/cp"
        os.environ["LAYER_STATES_PREFIX"] = "cp"
        set_upg_manager(ulysses_size=world_size, rank=rank, world_size=world_size)
        with torch.no_grad():
            local_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
                **mm_args,
            )

            local_states = local_output.hidden_states
            local_layer_states = local_output.layer_states
            local = local_output.logits  # (bs, local_seq, vocab)

            # Sanity: ensure CP actually shards the sequence.
            assert dist.get_world_size(get_ulysses_group()) == world_size
            assert local.size(1) * world_size == input_ids.size(1), (
                f"Expected local_seq={input_ids.size(1)//world_size}, got local_seq={local.size(1)}. "
                "This usually means the VLM decoder slice-after-embedding patch did not take effect."
            )

            full = gather_outputs_and_unpad(
                local,
                gather_dim=1,
                unpad_dim=None,
                padding_size=0,
                group=get_ulysses_group(),
            )

        _maybe_save_cp_gathered_tensors(
            rank=rank,
            base_logits=base,
            cp_gathered_logits=full,
            attention_mask=attention_mask,
        )

        if rank == 0:
            mask = attention_mask.to(torch.bool)
            base_s = base.float()
            full_s = full.float()
            torch.testing.assert_close(full_s[mask], base_s[mask], rtol=3e-2, atol=3e-2)
    finally:
        try:
            unapply_ulysses_patch()
        except Exception:
            pass
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA + FlashAttention2")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires >= 2 CUDA devices for CP all-to-all")
@pytest.mark.parametrize(
    "env_key",
    [
        "ROLL_TEST_QWEN25VL_PATH",
        "ROLL_TEST_QWEN3VL_PATH",
    ],
)
def test_fsdp2_cp_vlm_rmpad_equivalence(env_key: str):
    model_path = os.environ.get(env_key)
    if not model_path:
        pytest.skip(f"Set {env_key} to a local model path to run this test.")
    if not os.path.exists(model_path):
        pytest.skip(f"{env_key}={model_path} does not exist on this machine.")

    world_size = 2
    port = _find_free_port()
    mp.spawn(
        _worker_vlm_cp_equivalence,
        args=(world_size, port, model_path),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA + FlashAttention2")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires >= 2 CUDA devices for CP all-to-all")
@pytest.mark.parametrize(
    "env_key",
    [
        "ROLL_TEST_QWEN25VL_PATH",
        "ROLL_TEST_QWEN3VL_PATH",
    ],
)
def test_fsdp2_cp_vlm_nonrmpad_equivalence(env_key: str):
    model_path = os.environ.get(env_key)
    if not model_path:
        pytest.skip(f"Set {env_key} to a local model path to run this test.")
    if not os.path.exists(model_path):
        pytest.skip(f"{env_key}={model_path} does not exist on this machine.")

    world_size = 2
    port = _find_free_port()
    mp.spawn(
        _worker_vlm_cp_equivalence_nonrmpad,
        args=(world_size, port, model_path),
        nprocs=world_size,
        join=True,
    )
