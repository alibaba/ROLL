import ray
import asyncio
import pytest
from packaging.version import Version

import vllm
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind
from vllm.utils import random_uuid

from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.third_party.vllm import create_async_llm
from roll.utils.checkpoint_manager import download_model
from utils import chat_prompts, print_request_output


# vLLM 0.8.4 has bug when using n_sample with output_kind other than RequestOutputKind.FINAL_ONLY
# https://github.com/vllm-project/vllm/pull/16863

async def test_vllm_sampling_n(model):
    print(">>>>>>>>>>>>>>> test_vllm_sampling_n")
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.99,
        top_k=100,
        min_tokens=8192,
        max_tokens=8192,
        n=3,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )

    async def generate(prompt):
        request_id = random_uuid()
        result_generator = model.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id)
        output = None
        async for request_output in result_generator:
            output = request_output
        assert output is not None
        return output

    output = await generate(chat_prompts[0])
    assert len(output.outputs) == 3
    # print_request_output(output)

# The semantics of AsyncLLMEngine.abort for v1 and v0 are not aligned (see 
# https://github.com/vllm-project/vllm/blob/main/tests/async_engine/test_async_llm_engine.py#L350 and
# https://github.com/vllm-project/vllm/blob/main/tests/v1/engine/test_async_llm.py#L185 for difference).
#
# What we want is the semantic of v0 that raise asyncio.CancelledError in
# AsyncLLMEngine.generate when request is aborted rather than rely on cancel of async task to abort request.
async def test_vllm_abort(model):
    print(">>>>>>>>>>>>>>> test_vllm_abort")
    sampling_params = SamplingParams(
        temperature=0.1,
        min_tokens=8192,
        max_tokens=8192,
        n=3,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )

    request_id = random_uuid()
    async def generate():
        output = None
        if Version(vllm.__version__) >= Version("0.10.2"):
            async for request_output in model.generate(chat_prompts[0], sampling_params, request_id=request_id):
                output = request_output
        else:
            with pytest.raises(asyncio.CancelledError): # we patch older version vllm
                async for request_output in model.generate(chat_prompts[0], sampling_params, request_id=request_id):
                    output = request_output
        return output

    task = asyncio.create_task(generate())
    await asyncio.sleep(1)
    await model.abort(request_id)
    output = await task
    # assume generate is longer than 1s
    if Version(vllm.__version__) >= Version("0.10.2"):
        assert output is not None and output.finished
        assert len(output.outputs) == 3
        assert all(out.finish_reason == "abort" for out in output.outputs)
    else:
        assert output is None

async def test_vllm_abort_cumulative(model):
    print(">>>>>>>>>>>>>>> test_vllm_abort_cumulative")
    sampling_params = SamplingParams(
        temperature=0.1,
        min_tokens=8192,
        max_tokens=8192,
        n=3, # the behaviour of n sample before 0.10.2 is the same as sglang, we must store output by index
        output_kind=RequestOutputKind.CUMULATIVE,
    )

    request_id = random_uuid()
    async def generate():
        output = None
        if Version(vllm.__version__) >= Version("0.10.2"):
            async for request_output in model.generate(chat_prompts[0], sampling_params, request_id=request_id):
                output = request_output
        else:
            with pytest.raises(asyncio.CancelledError): # we patch older version vllm
                async for request_output in model.generate(chat_prompts[0], sampling_params, request_id=request_id):
                    output = request_output
        return output

    task = asyncio.create_task(generate())
    await asyncio.sleep(1)
    await model.abort(request_id)
    output = await task
    # assume at least generate one iter and generate is longer than 1s
    if Version(vllm.__version__) >= Version("0.10.2"):
        assert output is not None and output.finished
        assert len(output.outputs) == 3
        assert all(out.finish_reason == "abort" for out in output.outputs)
    else:
        assert output is not None and not output.finished
        assert len(output.outputs) == 1 # does match sampling_params.n

async def main():
    model_path = "Qwen/Qwen2.5-7B-Instruct"
    model_path = download_model(model_path)

    ray.init()
    resource_manager = ResourceManager(2, 1)
    placement_groups = resource_manager.allocate_placement_group(world_size=1, device_mapping=[0, 1])
    sampling_params = SamplingParams(temperature=0.0, top_p=0.99, top_k=100, max_tokens=512)

    model = await create_async_llm(
        resource_placement_groups=placement_groups[0],
        model=model_path,
        block_size=16,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        tensor_parallel_size=2,
        distributed_executor_backend="ray",
        disable_custom_all_reduce=True,
        enable_sleep_mode=True,
        enforce_eager=False,
    )

    await test_vllm_sampling_n(model)
    await test_vllm_abort(model)
    await test_vllm_abort_cumulative(model)

if __name__ == "__main__":
    asyncio.run(main())
