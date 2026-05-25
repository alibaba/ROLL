import asyncio
import gc
import inspect
import os
import uuid

import pytest
from transformers import AutoTokenizer

from roll.platforms import current_platform
from roll.utils.checkpoint_manager import download_model


ABORT_MAX_NEW_TOKENS = int(os.environ.get("ROLL_NPU_SGLANG_ABORT_MAX_NEW_TOKENS", "512"))
ABORT_SLEEP_SECONDS = float(os.environ.get("ROLL_NPU_SGLANG_ABORT_SLEEP_SECONDS", "0.2"))
SMOKE_MODEL = os.environ.get("ROLL_NPU_SGLANG_ABORT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def chat_format(prompt):
    system = "Please reason step by step, and put your final answer within \\boxed{}."
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


async def _generate(model, obj):
    generator = model.tokenizer_manager.generate_request(obj, None)
    chunks = None
    async for chunks in generator:
        chunks = chunks
    chunks = chunks if isinstance(chunks, list) else [chunks]
    return chunks


async def _wait_for_active_request(model, request_ids):
    request_ids = set(request_ids)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 10
    while loop.time() < deadline:
        active_request_ids = set(model.tokenizer_manager.rid_to_state)
        if request_ids <= active_request_ids:
            return
        await asyncio.sleep(0.05)


async def _check_sampling_n(model, input_ids, generate_req_cls):
    sampling_params = {
        "temperature": 0.8,
        "min_new_tokens": 32,
        "max_new_tokens": 32,
        "n": 3,
    }
    obj = generate_req_cls(
        input_ids=input_ids[0],
        sampling_params=sampling_params,
        rid=None,
        return_logprob=True,
    )
    chunks = await _generate(model, obj)
    assert all(chunk is not None for chunk in chunks)
    assert all(chunk["meta_info"]["finish_reason"]["type"] == "length" for chunk in chunks)


async def _check_abort_all(model, input_ids, generate_req_cls):
    sampling_params = {
        "temperature": 0.8,
        "min_new_tokens": ABORT_MAX_NEW_TOKENS,
        "max_new_tokens": ABORT_MAX_NEW_TOKENS,
        "n": 1,
    }
    obj1 = generate_req_cls(
        rid=str(uuid.uuid4().hex),
        input_ids=input_ids[0],
        sampling_params=sampling_params,
        return_logprob=True,
    )
    obj2 = generate_req_cls(
        rid=str(uuid.uuid4().hex),
        input_ids=input_ids[0],
        sampling_params=sampling_params,
        return_logprob=True,
    )
    tasks = [asyncio.create_task(_generate(model, obj1)), asyncio.create_task(_generate(model, obj2))]
    await _wait_for_active_request(model, [obj1.rid, obj2.rid])
    await asyncio.sleep(ABORT_SLEEP_SECONDS)
    for rid in list(model.tokenizer_manager.rid_to_state):
        model.tokenizer_manager.abort_request(rid)
    responses = await asyncio.gather(*tasks)
    assert all(isinstance(response, list) and len(response) > 0 for response in responses)
    assert all(resp["meta_info"]["finish_reason"]["type"] == "abort" for response in responses for resp in response)


async def _check_abort(model, input_ids, generate_req_cls):
    sampling_params = {
        "temperature": 0.8,
        "min_new_tokens": ABORT_MAX_NEW_TOKENS,
        "max_new_tokens": ABORT_MAX_NEW_TOKENS,
        "n": 1,
    }
    rid = uuid.uuid4().hex
    obj = generate_req_cls(
        input_ids=input_ids[0],
        sampling_params=sampling_params,
        rid=rid,
        return_logprob=True,
    )
    task = asyncio.create_task(_generate(model, obj))
    await _wait_for_active_request(model, [rid])
    await asyncio.sleep(ABORT_SLEEP_SECONDS)
    model.tokenizer_manager.abort_request(rid)
    response = await task
    assert response is not None and len(response) == 1
    assert response[0]["meta_info"]["finish_reason"]["type"] == "abort"


async def _shutdown_sglang_engine(model):
    for method_name in ("shutdown", "close"):
        method = getattr(model, method_name, None)
        if method is None:
            continue
        result = method()
        if inspect.isawaitable(result):
            await result
        return


async def _run_sglang_abort_suite():
    if not current_platform.is_npu():
        pytest.skip("SGLang abort suite only applies on Ascend NPU.")

    pytest.importorskip("sglang")
    pytest.importorskip("sgl_kernel_npu")

    from sglang.srt.managers.io_struct import GenerateReqInput

    from roll.third_party.sglang import patch as sglang_patch

    model_path = download_model(SMOKE_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompts = [
        "Write one short sentence about testing.",
        "Which number is larger, 100.25 or 100.75?",
        "Say hello in one short sentence.",
    ]
    prompts = [chat_format(prompt) for prompt in prompts]
    input_ids = tokenizer(prompts)["input_ids"]

    model = None
    try:
        model = sglang_patch.engine.engine_module.Engine(
            model_path=model_path,
            enable_memory_saver=False,  # CI skips torch-memory-saver; abort behavior does not need it.
            skip_tokenizer_init=False,  # to use min_new_tokens
            dtype="bfloat16",
            tp_size=1,
            mem_fraction_static=0.3,
            max_total_tokens=2048,
            max_running_requests=4,
            disable_custom_all_reduce=True,
        )

        await _check_sampling_n(model, input_ids, GenerateReqInput)
        await _check_abort_all(model, input_ids, GenerateReqInput)
        await _check_abort(model, input_ids, GenerateReqInput)
    finally:
        if model is not None:
            try:
                await _shutdown_sglang_engine(model)
            except Exception as e:
                print(f"Failed to shut down SGLang model cleanly: {e}")
        gc.collect()
        empty_cache = getattr(current_platform, "empty_cache", None)
        if empty_cache is not None:
            empty_cache()


def test_sglang_abort_suite():
    asyncio.run(_run_sglang_abort_suite())


if __name__ == "__main__":
    test_sglang_abort_suite()
