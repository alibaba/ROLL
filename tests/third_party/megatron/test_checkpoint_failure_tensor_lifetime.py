"""Checkpoint exception ownership, independent of optional CUDA dependencies.

The production context and its cleanup helper are AST-loaded unchanged. Small
optimizer leaves expose the restoration boundary; tensors and traceback object
ownership are real. Actual CUDA restoration is exercised in the strategy test.
"""
import ast
from contextlib import contextmanager
import gc
from pathlib import Path
import traceback
from types import SimpleNamespace
import weakref

import pytest
import torch


def _context():
    source = Path(__file__).parents[3] / "roll/third_party/megatron/offload_states_patch.py"
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {"checkpoint_grad_buffer_offload", "_clear_checkpoint_exception_frames"}]
    namespace = dict(contextmanager=contextmanager, traceback=traceback,
                     clear_memory=lambda **kwargs: gc.collect(),
                     MegatronOffloadStateType=SimpleNamespace(other_params="other"))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["checkpoint_grad_buffer_offload"]


@pytest.mark.parametrize("chained", [False, True])
def test_failed_loader_releases_tensor_before_restoring_gradients(chained):
    context = _context()
    witnesses = []
    observed = []
    failure = ValueError("invalid model shard")

    class Leaf:
        offloaded_states = set()
        model_chunks = [object()]

        def offload_states(self, **kwargs):
            pass

        def reload_states(self, **kwargs):
            observed.append(all(witness() is None for witness in witnesses))

    def read_shard():
        payload = torch.ones(1024)
        witnesses.append(weakref.ref(payload))
        raise failure

    def load_model():
        if not chained:
            return read_shard()
        try:
            return read_shard()
        except ValueError as exc:
            raise RuntimeError("checkpoint loader failed") from exc

    with pytest.raises((ValueError, RuntimeError)) as caught:
        with context(SimpleNamespace(chained_optimizers=[Leaf(), Leaf()])):
            model_state = None
            try:
                model_state = load_model()
            finally:
                del model_state

    assert observed == [True, True], "Failed loader tensors survived until gradient reconstruction"
    if chained:
        assert caught.value.__cause__ is failure
    else:
        assert caught.value is failure
    assert "read_shard" in "".join(traceback.format_exception(caught.value))


def test_cleanup_attempts_remaining_optimizer_leaves_after_one_reload_fails():
    context = _context()
    reloaded = []

    class Leaf:
        offloaded_states = set()

        def __init__(self, rank):
            self.rank = rank
            self.model_chunks = [object()]

        def offload_states(self, **kwargs):
            pass

        def reload_states(self, **kwargs):
            reloaded.append(self.rank)
            if self.rank == 0:
                raise RuntimeError("first gradient allocation failed")

    with pytest.raises(RuntimeError, match="first gradient allocation failed"):
        with context(SimpleNamespace(chained_optimizers=[Leaf(0), Leaf(1)])):
            pass
    assert reloaded == [0, 1], "One failed reload prevented cleanup of later optimizer leaves"
