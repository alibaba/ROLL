"""Execute production admission boundaries without constructing a GPU engine.

Only Ray/engine transport and model assembly are replaced. A real pending
Future and asyncio coroutine expose early completion, lost errors, and dropped
rank acknowledgements across the unchanged production methods loaded from AST.
"""
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[3]


def _method(file, owner, name, **namespace):
    source = ROOT / file
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    method = next(n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    method.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[name]


def _admission(core):
    engine = SimpleNamespace(engine_core=core)
    engine.add_lora = _method("roll/third_party/vllm/async_llm.py", "CustomAsyncLLM", "add_lora").__get__(engine)
    strategy = SimpleNamespace(model=engine, worker_config=SimpleNamespace(
        num_gpus_per_worker=8, model_args=SimpleNamespace(lora_target=r".*\.linear_qkv$")))
    strategy.add_lora = _method("roll/distributed/strategy/vllm_strategy.py", "VllmStrategy", "add_lora").__get__(strategy)
    worker = SimpleNamespace(strategy=strategy)
    worker.add_lora = _method("roll/pipeline/base_worker.py", "InferWorker", "add_lora").__get__(worker)
    return worker.add_lora


class _Core:
    def __init__(self, replies, entered=None, release=None, error=None):
        self.replies, self.entered, self.release, self.error = replies, entered, release, error

    async def collective_rpc_async(self, method, args, kwargs):
        assert method == "custom_add_lora"
        if self.entered is not None:
            self.entered.set()
            assert self.release.wait(5), "test did not release engine admission"
        await asyncio.sleep(0)
        if self.error:
            raise self.error
        return self.replies


def test_all_rank_acknowledgements_reach_worker_caller_without_mutating_config():
    replies = [True] * 8
    config = {"r": 64, "target_modules": ["old"]}
    result = asyncio.run(_admission(_Core(replies))(config))
    assert result == replies
    assert config == {"r": 64, "target_modules": ["old"]}


@pytest.mark.parametrize("replies", [[True] * 7 + [False], [True] * 7, [], None, [1] * 8])
def test_missing_or_failed_rank_acknowledgement_refuses_admission(replies):
    with pytest.raises(RuntimeError, match="LoRA.*acknowledg"):
        asyncio.run(_admission(_Core(replies))({"r": 64}))


@dataclass
class _PeftConfig:
    r: int = 64


def _updater(admission, executor):
    def remote(**kwargs):
        return executor.submit(asyncio.run, admission(**kwargs))

    worker = SimpleNamespace(add_lora=SimpleNamespace(remote=remote))
    updater = SimpleNamespace(
        _infer_parallel_cpu_group=object(), _co_infer_worker=worker,
        worker_config=SimpleNamespace(model_args=SimpleNamespace(lora_target=["linear_qkv"])),
        models_unwrapped=[SimpleNamespace(peft_config={"default": _PeftConfig()})],
        _model_update_buffer_size=1, _weights_meta={}, _broadcast_workers=[],
    )
    # Empty tensor stream isolates its final admission boundary. Tensor transfer
    # is already complete when this regression starts.
    method = _method("roll/third_party/megatron/model_update.py", "MegatronWeightUpdater", "_colocated_model_update",
                     dist=SimpleNamespace(get_world_size=lambda group: 1, get_rank=lambda group: 0),
                     ray=SimpleNamespace(get=lambda refs: [ref.result(timeout=5) for ref in refs]),
                     asdict=asdict, gather_all_hf_weights=lambda *args, **kwargs: iter(()))
    return method.__get__(updater)


def test_model_update_cannot_finish_while_engine_admission_is_pending():
    entered, release = Event(), Event()
    admission = _admission(_Core([True] * 8, entered, release))
    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(_updater(admission, executor))
        try:
            assert entered.wait(3)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.1)
        finally:
            release.set()
        assert future.result(timeout=3) == {}


def test_engine_admission_failure_reaches_model_update_caller():
    admission = _admission(_Core(None, error=RuntimeError("rank 7 native loader failed")))
    with ThreadPoolExecutor(max_workers=2) as executor:
        with pytest.raises(RuntimeError, match="rank 7 native loader failed"):
            _updater(admission, executor)()
