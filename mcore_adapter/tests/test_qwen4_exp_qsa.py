import importlib.util
import sys
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location(
    "qsa", Path(__file__).parents[1] / "src/mcore_adapter/models/qwen4_exp/qsa.py"
)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
qsa_indexer_kl_loss = _module.qsa_indexer_kl_loss
select_causal_blocks = _module.select_causal_blocks


def test_qsa_selects_complete_blocks_and_causal_tail():
    scores = torch.tensor([[0.1, 3.0, 0.2, 2.0]])
    selected = select_causal_blocks(scores, sequence_length=9, block_size=4, token_budget=4)
    assert selected.tolist() == [[4, 5, 6, 7, 8]]


def test_qsa_kl_stops_teacher_gradient_and_is_finite_without_blocks():
    scores = torch.randn(2, 3, 4, requires_grad=True)
    teacher = torch.softmax(torch.randn(2, 3, 4), dim=-1).requires_grad_()
    loss = qsa_indexer_kl_loss(scores, teacher, torch.ones(2, 3, 4, dtype=torch.bool))
    loss.backward()
    assert teacher.grad is None
    assert torch.isfinite(loss)
    assert qsa_indexer_kl_loss(scores.detach(), teacher.detach(), torch.zeros_like(teacher, dtype=torch.bool)) == 0
