import json
import os
import socket
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from roll.utils.context_parallel.globals import get_ulysses_group, set_upg_manager
from roll.utils.context_parallel.monkey_patch import apply_ulysses_patch, unapply_ulysses_patch
from roll.utils.context_parallel.rmpad_ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs

_DEBUG_LOG_PATH = os.environ.get("ROLL_DEBUG_LOG_PATH", "output/debug.log")


def _dbg(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "sessionId": "debug-session",
                        "runId": "pre-fix",
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "message": message,
                        "data": data,
                        "timestamp": int(time.time() * 1000),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception:
        pass


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _worker_qwen3_hf_rmpad_equivalence(rank: int, world_size: int, port: int, model_id: str) -> None:
    pytest.importorskip("transformers")
    pytest.importorskip("flash_attn")

    if not torch.cuda.is_available():
        pytest.skip("Qwen3 HF + FlashAttention2 CP rmpad equivalence test requires CUDA")
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
        from transformers import __version__ as transformers_version

        from flash_attn import __version__ as flash_attn_version

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

        tokenizer.padding_side = "right"
        texts = [
            "Explain FlashAttention2 remove-padding (varlen) and how it interacts with rotary embeddings and position ids.",
            "Relate remove-padding to Ulysses context parallelism and all-to-all. Give a small example.",
        ]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        position_ids = (attention_mask.long().cumsum(dim=1) - 1).clamp_min(0)

        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input

        if rank == 0 and hasattr(model, "_update_causal_mask"):
            original_update_mask = model._update_causal_mask

            def _instrumented_update_mask(attention_mask, input_tensor, cache_position, **kwargs):
                result = original_update_mask(attention_mask, input_tensor, cache_position, **kwargs)
                prepare_log.append(
                    {
                        "input_attn_mask_shape": (
                            tuple(attention_mask.shape) if torch.is_tensor(attention_mask) else None
                        ),
                        "input_attn_mask_is_none": attention_mask is None,
                        "input_tensor_shape": tuple(input_tensor.shape) if torch.is_tensor(input_tensor) else None,
                        "output_mask_shape": tuple(result.shape) if torch.is_tensor(result) else None,
                    }
                )
                return result

            model._update_causal_mask = _instrumented_update_mask
        if rank == 0:
            enc1 = tokenizer(
                [texts[0]],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_len,
            )
            input_ids1 = enc1["input_ids"].to(device)
            attention_mask1 = enc1["attention_mask"].to(device)
            position_ids1 = (attention_mask1.long().cumsum(dim=1) - 1).clamp_min(0)

            with torch.no_grad():
                out_padded_1 = model(
                    input_ids=input_ids1,
                    attention_mask=attention_mask1,
                    position_ids=position_ids1,
                    use_cache=False,
                ).logits.float()
                input_ids_rmpad_1, indices_1, *_ = unpad_input(input_ids1.unsqueeze(-1), attention_mask1)
                input_ids_rmpad_1 = input_ids_rmpad_1.transpose(0, 1)
                position_ids_rmpad_1 = index_first_axis(
                    rearrange(position_ids1.unsqueeze(-1), "b s ... -> (b s) ..."),
                    indices_1,
                ).transpose(0, 1)
                out_rmpad_1 = model(
                    input_ids=input_ids_rmpad_1,
                    attention_mask=None,
                    position_ids=position_ids_rmpad_1,
                    use_cache=False,
                ).logits.float()
                out_rmpad_1 = pad_input(
                    hidden_states=out_rmpad_1.squeeze(0).unsqueeze(-1),
                    indices=indices_1,
                    batch=1,
                    seqlen=input_ids1.size(1),
                ).squeeze(-1)
                m1 = attention_mask1.to(torch.bool)
                max_abs_1 = float((out_padded_1 - out_rmpad_1).abs()[m1].max().item()) if m1.any() else 0.0
                _dbg(
                    "H9",
                    "tests/.../test_fsdp2_cp_qwen3_hf_rmpad_equivalence.py:bs1_probe",
                    "padded_vs_rmpad_bs1",
                    {"masked_max_abs_padded_vs_rmpad_bs1": max_abs_1, "mask_sum": int(attention_mask1.sum().item())},
                )
        if rank == 0:
            _dbg(
                "H1",
                "tests/.../test_fsdp2_cp_qwen3_hf_rmpad_equivalence.py:_worker",
                "env_and_batch",
                {
                    "model_id": str(model_id),
                    "transformers_version": str(transformers_version),
                    "flash_attn_version": str(flash_attn_version),
                    "world_size": int(world_size),
                    "max_len": int(max_len),
                    "mask_sum_per_sample": [int(x) for x in attention_mask.sum(dim=1).tolist()],
                    "pos0_first8": position_ids[0, :8].tolist(),
                    "pos1_first8": position_ids[1, :8].tolist(),
                },
            )
        original_fa2_forward = None
        call_log = {"padded": [], "rmpad": []}
        original_layer_forward = None
        first_layer = None
        layer_call_info = {"padded": None, "rmpad": None}
        prepare_log = []
        original_update_mask = None

        if rank == 0:

            def _instrumented_fa2(*args, **kwargs):
                import inspect

                sig = inspect.signature(original_fa2_forward)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                params = bound.arguments
                call_log["current_mode"].append(
                    {
                        "attention_mask_is_none": params.get("attention_mask") is None,
                        "has_cu_seqlens_q": "cu_seqlens_q" in params,
                        "has_cu_seqlens_k": "cu_seqlens_k" in params,
                        "query_length": int(params.get("query_length", -1)) if params.get("query_length") else -1,
                        "query_shape": (
                            tuple(params["query_states"].shape)
                            if torch.is_tensor(params.get("query_states"))
                            else None
                        ),
                    }
                )
                return original_fa2_forward(*args, **kwargs)

            try:
                from transformers.integrations import flash_attention as fa_module

                original_fa2_forward = fa_module._flash_attention_forward
                fa_module._flash_attention_forward = _instrumented_fa2
            except Exception:
                try:
                    import transformers.modeling_flash_attention_utils as mfu

                    original_fa2_forward = mfu._flash_attention_forward
                    mfu._flash_attention_forward = _instrumented_fa2
                except Exception:
                    original_fa2_forward = None

        with torch.no_grad():
            set_upg_manager(ulysses_size=1, rank=rank, world_size=world_size)

            if rank == 0 and original_fa2_forward is not None:
                call_log["current_mode"] = call_log["padded"]

            baseline_padded_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
            )
            baseline_padded = baseline_padded_out.logits

            if rank == 0:
                first_layer = model.model.layers[0].self_attn
                original_layer_forward = first_layer.forward

                def _instrumented_layer_forward(hidden_states, attention_mask=None, position_ids=None, **kwargs):
                    layer_call_info["current"]["attn_mask_shape"] = (
                        tuple(attention_mask.shape) if torch.is_tensor(attention_mask) else None
                    )
                    layer_call_info["current"]["attn_mask_is_none"] = attention_mask is None
                    layer_call_info["current"]["pos_ids_shape"] = (
                        tuple(position_ids.shape) if torch.is_tensor(position_ids) else None
                    )
                    return original_layer_forward(
                        hidden_states, attention_mask=attention_mask, position_ids=position_ids, **kwargs
                    )

                first_layer.forward = _instrumented_layer_forward

                layer_call_info["current"] = layer_call_info["padded"] = {}
                with torch.no_grad():
                    _ = model(
                        input_ids=input_ids[:1, :8],  # small probe
                        attention_mask=attention_mask[:1, :8],
                        position_ids=position_ids[:1, :8],
                        use_cache=False,
                    )
            if rank == 0 and original_fa2_forward is not None:
                call_log["current_mode"] = call_log["rmpad"]

            input_ids_rmpad_base, indices_base, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
            input_ids_rmpad_base = input_ids_rmpad_base.transpose(0, 1)  # (1, total_nnz)
            position_ids_rmpad_base = index_first_axis(
                rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                indices_base,
            ).transpose(
                0, 1
            )  # (1, total_nnz)

            if rank == 0:
                layer_call_info["current"] = layer_call_info["rmpad"] = {}
                with torch.no_grad():
                    # Create a small packed input to probe
                    probe_ids = torch.cat([input_ids[0, :4], input_ids[1, :4]], dim=0).unsqueeze(0)  # (1, 8)
                    probe_pos = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], device=device)  # position resets
                    _ = model(input_ids=probe_ids, attention_mask=None, position_ids=probe_pos, use_cache=False)

                # Restore original forward
                first_layer.forward = original_layer_forward

                _dbg(
                    "H12",
                    "tests/.../test_fsdp2_cp_qwen3_hf_rmpad_equivalence.py:layer_inputs",
                    "layer_attn_mask_comparison",
                    {
                        "padded_mask_shape": layer_call_info["padded"].get("attn_mask_shape"),
                        "padded_mask_is_none": layer_call_info["padded"].get("attn_mask_is_none"),
                        "rmpad_mask_shape": layer_call_info["rmpad"].get("attn_mask_shape"),
                        "rmpad_mask_is_none": layer_call_info["rmpad"].get("attn_mask_is_none"),
                        "padded_pos_shape": layer_call_info["padded"].get("pos_ids_shape"),
                        "rmpad_pos_shape": layer_call_info["rmpad"].get("pos_ids_shape"),
                    },
                )
            # endregion agent log

            baseline_rmpad_out = model(
                input_ids=input_ids_rmpad_base,
                attention_mask=None,
                position_ids=position_ids_rmpad_base,
                use_cache=False,
                output_hidden_states=True,
            )
            logits_rmpad_base = baseline_rmpad_out.logits  # (1, total_nnz, vocab)

            baseline_rmpad = pad_input(
                hidden_states=logits_rmpad_base.squeeze(0).unsqueeze(-1),
                indices=indices_base,
                batch=input_ids.size(0),
                seqlen=input_ids.size(1),
            ).squeeze(-1)

            # H10: locate the earliest hidden-state mismatch (after first decoder block).
            if (
                rank == 0
                and getattr(baseline_padded_out, "hidden_states", None) is not None
                and getattr(baseline_rmpad_out, "hidden_states", None) is not None
            ):
                hs_padded = baseline_padded_out.hidden_states
                hs_rmpad = baseline_rmpad_out.hidden_states
                # hidden_states[0] is embedding output; [1] is after first layer (for most HF decoder models).
                if len(hs_padded) > 1 and len(hs_rmpad) > 1:
                    hs1_padded = hs_padded[1].float()
                    hs1_rmpad = hs_rmpad[1].float()  # (1, total_nnz, hidden)
                    hs1_rmpad_padded = pad_input(
                        hidden_states=hs1_rmpad.squeeze(0).unsqueeze(-1),
                        indices=indices_base,
                        batch=input_ids.size(0),
                        seqlen=input_ids.size(1),
                    ).squeeze(-1)
                    m = attention_mask.to(torch.bool)
                    max_abs_hs1 = float((hs1_padded - hs1_rmpad_padded).abs()[m].max().item()) if m.any() else 0.0
                    tok0_abs_hs1 = float((hs1_padded[0, 0] - hs1_rmpad_padded[0, 0]).abs().max().item())
                    _dbg(
                        "H10",
                        "tests/.../test_fsdp2_cp_qwen3_hf_rmpad_equivalence.py:hidden_states",
                        "hs_layer1_diff",
                        {"masked_max_abs_hs1": max_abs_hs1, "sample0_tok0_max_abs_hs1": tok0_abs_hs1},
                    )

            set_upg_manager(ulysses_size=world_size, rank=rank, world_size=world_size)

            input_ids_rmpad, indices, cu_seqlens, max_seqlen, _ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            position_ids_rmpad = index_first_axis(
                rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                indices,
            ).transpose(
                0, 1
            )  # (1, total_nnz)

            if rank == 0:
                _dbg(
                    "H2",
                    "tests/.../test_fsdp2_cp_qwen3_hf_rmpad_equivalence.py:unpad_input",
                    "rmpad_metadata",
                    {
                        "total_nnz": int(input_ids_rmpad.size(1)),
                        "cu_seqlens_shape": tuple(cu_seqlens.shape) if torch.is_tensor(cu_seqlens) else None,
                        "cu_seqlens_head": (
                            cu_seqlens[: min(6, cu_seqlens.numel())].tolist() if torch.is_tensor(cu_seqlens) else None
                        ),
                        "max_seqlen": int(max_seqlen) if max_seqlen is not None else None,
                        "position_ids_rmpad_head": position_ids_rmpad[
                            0, : min(10, position_ids_rmpad.size(1))
                        ].tolist(),
                    },
                )
            # endregion agent log

            input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                input_ids_rmpad,
                position_ids_rmpad,
                cp_size=world_size,
                cp_rank=rank,
            )

            logits_rmpad_local = model(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                use_cache=False,
            ).logits  # (1, local_tokens, vocab)

            logits_rmpad = gather_outputs_and_unpad(
                logits_rmpad_local,
                gather_dim=1,
                unpad_dim=1,
                padding_size=pad_size,
                group=get_ulysses_group(),
            )

            logits = pad_input(
                hidden_states=logits_rmpad.squeeze(0).unsqueeze(-1),
                indices=indices,
                batch=input_ids.size(0),
                seqlen=input_ids.size(1),
            ).squeeze(-1)

        baseline_padded_full = baseline_padded.float()
        baseline_rmpad_full = baseline_rmpad.float()

        if rank == 0:
            mask = attention_mask.to(torch.bool)
            diff_cp = (logits.float() - baseline_padded_full).abs()
            diff_rmpad = (baseline_rmpad_full - baseline_padded_full).abs()
            max_abs_cp = float(diff_cp[mask].max().item()) if mask.any() else 0.0
            max_abs_rmpad = float(diff_rmpad[mask].max().item()) if mask.any() else 0.0
            tok0_abs_cp = float(diff_cp[0, 0].max().item())
            tok0_abs_rmpad = float(diff_rmpad[0, 0].max().item())
            _dbg(
                "H4",
                "tests/.../test_fsdp2_cp_qwen3_hf_rmpad_equivalence.py:compare",
                "masked_diff_stats",
                {
                    "masked_max_abs_cp_vs_padded": max_abs_cp,
                    "masked_max_abs_rmpad_vs_padded": max_abs_rmpad,
                    "sample0_tok0_max_abs_cp_vs_padded": tok0_abs_cp,
                    "sample0_tok0_max_abs_rmpad_vs_padded": tok0_abs_rmpad,
                },
            )
            if original_fa2_forward is not None:
                try:
                    from transformers.integrations import flash_attention as fa_module

                    fa_module._flash_attention_forward = original_fa2_forward
                except Exception:
                    try:
                        import transformers.modeling_flash_attention_utils as mfu

                        mfu._flash_attention_forward = original_fa2_forward
                    except Exception:
                        pass
                _dbg(
                    "H11",
                    "tests/.../test_fsdp2_cp_qwen3_hf_rmpad_equivalence.py:fa2_calls",
                    "fa2_call_comparison",
                    {
                        "padded_calls": call_log["padded"][:3],
                        "rmpad_calls": call_log["rmpad"][:3],
                        "padded_count": len(call_log["padded"]),
                        "rmpad_count": len(call_log["rmpad"]),
                    },
                )

            if original_update_mask is not None:
                model._update_causal_mask = original_update_mask
                _dbg(
                    "H13",
                    "tests/.../test_fsdp2_cp_qwen3_hf_rmpad_equivalence.py:prepare_calls",
                    "mask_generation_calls",
                    {"prepare_log": prepare_log[:6], "total_calls": len(prepare_log)},
                )

            torch.testing.assert_close(logits.float()[mask], baseline_padded_full[mask], rtol=2e-2, atol=2e-2)
    finally:
        try:
            unapply_ulysses_patch()
        except Exception:
            pass
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA + FlashAttention2")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires >= 2 CUDA devices for CP all-to-all")
def test_fsdp2_cp_qwen3_hf_rmpad_logits_equivalence():
    world_size = 2
    port = _find_free_port()
    model_id = os.environ.get(
        "ROLL_TEST_QWEN3_MODEL_ID",
        "/home/dilixiati.dlxtmhte/.cache/openlm/hub/14ffd5928d24731fd670f04c645a5928",
    )
    mp.spawn(
        _worker_qwen3_hf_rmpad_equivalence,
        args=(world_size, port, model_id),
        nprocs=world_size,
        join=True,
    )
