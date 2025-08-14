# tests/test_get_score_normalize_fn.py
import pytest
import torch
from dataclasses import dataclass

from roll.pipeline.agentic.utils import get_score_normalize_fn


@dataclass
class MockRNCfg:
    grouping: str = "dummy"
    method: str = "mean_std"


@pytest.mark.parametrize(
    "method,input_tensor,expected",
    [
        ("mean_std", torch.tensor([1.0, 2.0, 3.0]), torch.tensor([-1.0, 0.0, 1.0])),
        ("mean",     torch.tensor([1.0, 2.0, 3.0]), torch.tensor([-1.0, 0.0, 1.0])),
        ("asym_clip", torch.tensor([1.0, 2.0, 3.0]), torch.tensor([-1.0, 0.0, 1.0]).clamp(-1, 3)),
        ("identity", torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])),
    ],
)
def test_get_score_normalize_fn(method, input_tensor, expected):
    cfg = MockRNCfg(method=method)
    fn = get_score_normalize_fn(cfg)
    out = fn(input_tensor)
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-4)


def test_single_element_fallback():
    cfg = MockRNCfg(method="mean_std")
    fn = get_score_normalize_fn(cfg)
    out = fn(torch.tensor([5.0]))
    torch.testing.assert_close(out, torch.tensor([0.0]))