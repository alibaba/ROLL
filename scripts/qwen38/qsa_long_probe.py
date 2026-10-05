"""QSA sparse-window coverage and real-head-geometry forward/backward probe."""
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch


def main():
    file = Path(__file__).parents[2] / "mcore_adapter/src/mcore_adapter/models/qwen4_exp/qsa.py"
    spec = importlib.util.spec_from_file_location("qsa_long_probe", file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.cuda.set_device(0)
    torch.manual_seed(8192)
    for seq in (2047, 2048, 2049, 2052, 4096, 8192):
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        valid = torch.ones(1, seq, device="cuda", dtype=torch.bool)
        scores = torch.randn(1, seq, seq//4, device="cuda")
        selection = module.QSASelection.from_scores(scores, valid, 4, 2048)
        blocks = selection.selected_blocks[0, -1]
        selected_tokens = int(blocks.ge(0).sum()) * 4 + seq % 4
        assert selected_tokens == min(seq//4, 512)*4 + seq % 4
        q = torch.randn(1, 24, seq, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, 2, seq, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        result, lse = module.flex_qsa_attention(q, k, v, selection)
        cotangent = torch.randn_like(result)
        (result * cotangent).sum().backward()
        torch.cuda.synchronize()
        # Independent selected-key SDPA for sampled rows, preserving full geometry.
        sample_rows = [0, 3, min(2048, seq-1), seq-1]
        max_error = 0.0
        for row in sample_rows:
            selected = selection.selected_blocks[0, row]
            selected = selected[selected >= 0]
            indices = (selected[:, None]*4+torch.arange(4, device="cuda")).flatten()
            complete_end = int(selection.complete[0, row])*4
            indices = torch.cat([indices, torch.arange(complete_end, row+1, device="cuda")]).sort().values
            logits = torch.einsum("hd,hkd->hk", q.detach()[0, :, row].float(),
                                  k.detach()[0, :, indices].repeat_interleave(12, 0).float()) / 16
            expected = torch.einsum("hk,hkd->hd", logits.softmax(-1),
                                    v.detach()[0, :, indices].repeat_interleave(12, 0).float()).to(torch.bfloat16)
            torch.testing.assert_close(result[0, :, row], expected, atol=2e-2, rtol=2e-2)
            max_error = max(max_error, float((result[0, :, row]-expected).abs().max()))
        assert all(torch.isfinite(t.grad).all() for t in (q, k, v))
        print(json.dumps({"sequence": seq, "final_selected_tokens": selected_tokens,
                          "sampled_max_output_error": max_error,
                          "seconds_with_compilation": time.perf_counter()-start,
                          "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                          "finite_qkv_gradients": True}), flush=True)
        del q, k, v, result, lse, scores, selection, cotangent


if __name__ == "__main__":
    main()
