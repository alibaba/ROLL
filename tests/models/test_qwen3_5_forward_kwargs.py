"""Regression test for #457: Qwen3.5-VL's mm_token_type_ids reaching GPTModel.forward.

Builds a bare Qwen3_5Model instance without the heavy Megatron distributed init
(not needed: the bug is a pure kwargs-forwarding TypeError raised at Python's
argument-binding step, before any tensor-parallel state is touched), and drives
forward() the way a real non-first pipeline-parallel stage does.
"""

import inspect
from contextlib import contextmanager
from types import SimpleNamespace

import mcore_adapter.models.model_factory as model_factory
import torch
from mcore_adapter.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
from megatron.core.models.gpt.gpt_model import GPTModel


class _FakePackingState:
    position_ids = torch.zeros(1, 1, dtype=torch.long)
    cp_batch = {
        "input_ids": torch.zeros(1, 1, dtype=torch.long),
        "attention_mask": torch.ones(1, 1, 1, 1, dtype=torch.bool),
    }


def _bare_model():
    model = object.__new__(Qwen3_5Model)
    model.pre_process = False
    model.config = SimpleNamespace(mtp_num_layers=None)
    model.prepare_packing_state = lambda *a, **k: _FakePackingState()
    return model


@contextmanager
def _capture_super_forward():
    """Stand in for the real GPTModel.forward, reached via super() from
    Qwen3_5Model.forward, and record exactly the kwargs that arrive there."""
    captured = {}

    def fake_super_forward(self, **kw):
        # Prove the kwargs reaching this point would bind against the REAL
        # Megatron signature, without needing a fully constructed GPTModel.
        inspect.signature(GPTModel.forward).bind(self, **kw)
        captured.update(kw)
        return torch.zeros(1)

    had_own_forward = "forward" in model_factory.McaGPTModel.__dict__
    orig = model_factory.McaGPTModel.__dict__.get("forward")
    model_factory.McaGPTModel.forward = fake_super_forward
    try:
        yield captured
    finally:
        if had_own_forward:
            model_factory.McaGPTModel.forward = orig
        else:
            del model_factory.McaGPTModel.forward


def test_mm_token_type_ids_does_not_reach_gptmodel_forward():
    """The Qwen3.5-VL processor's mm_token_type_ids has no consumer and must be
    dropped before forward() reaches Megatron's GPTModel.forward, whose signature
    has no **kwargs catch-all (binding an unrecognized keyword there raises
    TypeError before the method body runs at all, matching #457's traceback)."""
    model = _bare_model()
    with _capture_super_forward() as captured:
        model.forward(
            input_ids=torch.zeros(1, 1, dtype=torch.long),
            position_ids=torch.zeros(1, 1, dtype=torch.long),
            attention_mask=torch.ones(1, 1, 1, 1, dtype=torch.bool),
            mm_token_type_ids=torch.zeros(1, 1, dtype=torch.long),
        )
    assert "mm_token_type_ids" not in captured


def test_force_vit_kwargs_still_popped():
    """Guards the two pre-existing kwargs this change sits next to, so a future
    edit can't silently stop popping them while touching this block."""
    model = _bare_model()
    with _capture_super_forward() as captured:
        model.forward(
            input_ids=torch.zeros(1, 1, dtype=torch.long),
            position_ids=torch.zeros(1, 1, dtype=torch.long),
            attention_mask=torch.ones(1, 1, 1, 1, dtype=torch.bool),
            force_vit_image=True,
            force_vit_video=True,
            mm_token_type_ids=torch.zeros(1, 1, dtype=torch.long),
        )
    assert "force_vit_image" not in captured
    assert "force_vit_video" not in captured
    assert "mm_token_type_ids" not in captured
