import importlib.util
import os
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).parents[1] / "src/mcore_adapter/models/qwen4_exp"
spec = importlib.util.spec_from_file_location("_qsa_training_test", ROOT / "qsa.py")
qsa = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = qsa
spec.loader.exec_module(qsa)


def reference_mask(scores, valid, block_size, budget):
    batch, seq = valid.shape
    mask = torch.zeros(batch, seq, seq, dtype=torch.bool, device=valid.device)
    for b in range(batch):
        for query in range(seq):
            if not valid[b, query]:
                continue
            visible = valid[b, :query+1].nonzero().flatten()
            n = len(visible) // block_size
            ordering = sorted(range(n), key=lambda i: (-float(scores[b, query, i]), i))[:budget//block_size]
            for block in ordering:
                mask[b, query, visible[block*block_size:(block+1)*block_size]] = True
            mask[b, query, visible[n*block_size:]] = True
    return mask


@pytest.mark.parametrize("seq", [1, 3, 4, 5, 15, 16, 17, 20, 32])
def test_selection_visible_blocks_padding_ties(seq):
    torch.manual_seed(seq)
    valid = torch.ones(2, seq, dtype=torch.bool)
    valid[0, :min(2, seq)] = False
    valid[1, max(0, seq-2):] = False
    scores = torch.randint(0, 3, (2, seq, seq//4)).float()
    selection = qsa.QSASelection.from_scores(scores, valid, block_size=4, token_budget=16)
    actual = selection.dense_mask_for_testing()
    torch.testing.assert_close(actual, reference_mask(scores, valid, 4, 16), rtol=0, atol=0)
    assert selection.selected_blocks.shape[:2] == (2, seq)


def test_single_batch_selection_shape_is_preserved():
    scores = torch.tensor([[1., 2., 3.]])
    assert qsa.select_causal_blocks(scores, 9, 4, 8).shape == (1, 9)


def test_selected_teacher_kl_detaches_teacher_and_empty_rows():
    scores = torch.tensor([[1., 3.], [4., -1.]], requires_grad=True)
    teacher = torch.tensor([[.2, .8], [.3, .7]], requires_grad=True)
    mask = torch.tensor([[True, True], [False, False]])
    loss = qsa.qsa_indexer_kl_loss(scores, teacher, mask)
    loss.backward()
    expected = (scores.detach()[0].softmax(-1) - teacher.detach()[0]) / 2
    torch.testing.assert_close(scores.grad[0], expected)
    torch.testing.assert_close(scores.grad[1], torch.zeros(2))
    assert teacher.grad is None


def test_indexer_distillation_uses_head_sum_block_max_pool_and_stop_gradient():
    torch.manual_seed(813)
    b, s, h, kv, d, ih, idim = 2, 13, 4, 2, 8, 2, 4
    valid = torch.ones(b, s, dtype=torch.bool)
    valid[0, :2] = False
    iq = torch.randn(b, s, ih, idim, requires_grad=True)
    ik = torch.randn(b, s//4, idim, requires_grad=True)
    iqref, ikref = iq.detach().clone().requires_grad_(), ik.detach().clone().requires_grad_()
    scores = torch.relu(torch.einsum("bqhd,bkd->bqhk", iq, ik)).sum(-2)
    selection = qsa.QSASelection.from_scores(scores, valid, 4, 8)
    query = torch.randn(b, h, s, d, requires_grad=True)
    key = torch.randn(b, kv, s, d, requires_grad=True)
    logits = query @ key.repeat_interleave(h//kv, 1).transpose(-1, -2) / d**0.5
    mask = reference_mask(scores.detach(), valid, 4, 8)
    logits = logits.masked_fill(~mask[:, None], -torch.inf)
    lse = logits.logsumexp(-1)
    attention = logits.softmax(-1).nan_to_num().detach()
    teacher = torch.zeros_like(scores)
    block_tokens = qsa.visible_block_tokens(valid, 4)
    for batch in range(b):
        for query_idx in range(s):
            for block in selection.selected_blocks[batch, query_idx].tolist():
                if block >= 0:
                    token_probs = attention[batch, :, query_idx, block_tokens[batch, block]].sum(0)
                    teacher[batch, query_idx, block] = token_probs.max()
    student = torch.relu(torch.einsum("bqhd,bkd->bqhk", iqref, ikref)).sum(-2)
    mask_blocks = selection.block_bitmap[:, :, :s//4]
    # Function averages over valid query positions; no-complete-block rows are 0.
    reference = qsa.qsa_indexer_kl_loss(student, teacher, mask_blocks) * (b*s) / valid.sum()
    actual = qsa.indexer_distillation_loss(iq, ik, selection, query, key, lse, tile_size=3)
    torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)
    actual.backward()
    reference.backward()
    torch.testing.assert_close(iq.grad, iqref.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(ik.grad, ikref.grad, atol=1e-5, rtol=1e-4)
    assert query.grad is None and key.grad is None


@pytest.mark.skipif(os.environ.get("QSA_CUDA_TESTS") != "1", reason="requires isolated H800 test environment")
def test_flex_sparse_core_forward_backward_and_lse():
    device = "cuda"
    torch.manual_seed(736)
    batch, seq, h, kv, d = 2, 65, 4, 2, 32
    valid = torch.ones(batch, seq, device=device, dtype=torch.bool)
    valid[0, :3], valid[1, -5:] = False, False
    scores = torch.randn(batch, seq, seq//4, device=device)
    selection = qsa.QSASelection.from_scores(scores, valid, 4, 16)
    q = torch.randn(batch, h, seq, d, device=device, requires_grad=True)
    k = torch.randn(batch, kv, seq, d, device=device, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    qr, kr, vr = [x.detach().clone().requires_grad_() for x in (q, k, v)]
    output, lse = qsa.flex_qsa_attention(q, k, v, selection)
    logits = qr @ kr.repeat_interleave(h//kv, 1).transpose(-1, -2) / d**0.5
    mask = reference_mask(scores, valid, 4, 16)[:, None]
    logits = logits.masked_fill(~mask, -torch.inf)
    probabilities = logits.softmax(-1).nan_to_num(0)
    expected = probabilities @ vr.repeat_interleave(h//kv, 1)
    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(lse[valid[:, None].expand(-1, h, -1)], logits.logsumexp(-1)[valid[:, None].expand(-1, h, -1)], atol=1e-5, rtol=1e-4)
    cotangent = torch.randn_like(output)
    (output*cotangent).sum().backward()
    (expected*cotangent).sum().backward()
    for got, want in ((q, qr), (k, kr), (v, vr)):
        torch.testing.assert_close(got.grad, want.grad, atol=1e-5, rtol=1e-4)
