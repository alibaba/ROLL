import asyncio
import importlib.util
import os
import sys
from types import SimpleNamespace
import uuid

import pytest

from roll.platforms import current_platform


def _require_module(module_name: str) -> None:
    try:
        module_spec = importlib.util.find_spec(module_name)
    except ValueError:
        module_spec = None

    available = module_spec is not None or module_name in sys.modules
    if not available and not current_platform.is_npu():
        pytest.skip(f"{module_name} is not installed in this environment.")
    assert available, f"{module_name} must be installed for NPU SGLang tests."


class _CapturingScheduler:
    def __init__(self):
        self.messages = []

    def send_pyobj(self, obj):
        self.messages.append(obj)


def test_sglang_import_available():
    _require_module("sglang")
    import sglang

    assert sglang.__version__


def _run_npu_sglang_abort_smoke():
    _require_module("sglang")
    _require_module("sgl_kernel_npu")

    from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager

    request_id = uuid.uuid4().hex
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.rid_to_state = {}
    manager.send_to_scheduler = _CapturingScheduler()
    manager.enable_metrics = False

    request = SimpleNamespace(
        rid=request_id,
        stream=False,
        return_logprob=False,
        top_logprobs_num=0,
        token_ids_logprob=[],
        return_text_in_logprobs=False,
    )
    state = ReqState([], False, asyncio.Event(), request, created_time=0.0)
    state.output_ids = [101, 102, 103]
    state.text = "partial output"
    manager.rid_to_state[request_id] = state

    manager.abort_request(request_id)

    assert len(manager.send_to_scheduler.messages) == 1
    abort_req = manager.send_to_scheduler.messages[0]
    assert abort_req.rid == request_id
    assert not abort_req.abort_all

    manager._handle_abort_req(abort_req)

    assert state.finished
    assert state.event.is_set()
    assert state.out_list
    output = state.out_list[-1]
    assert output["text"] == "partial output"
    assert output["output_ids"] == [101, 102, 103]
    assert output["meta_info"]["id"] == request_id
    assert output["meta_info"]["finish_reason"]["type"] == "abort"


def test_npu_sglang_abort_smoke():
    if not current_platform.is_npu():
        pytest.skip("NPU SGLang abort smoke only applies on Ascend NPU.")
    if os.environ.get("ROLL_NPU_SGLANG_ABORT_SMOKE", "1") == "0":
        pytest.skip("ROLL_NPU_SGLANG_ABORT_SMOKE=0")

    _run_npu_sglang_abort_smoke()
