import ray
import asyncio
import uuid

from sglang.srt.managers.io_struct import GenerateReqInput

from roll.third_party.sglang import patch as sglang_patch
from roll.utils.checkpoint_manager import download_model

def chat_format(prompt):
    system = "Please reason step by step, and put your final answer within \\boxed{}."
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

prompts = [
    "类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞,生成一段文案",
    "根据关键词描述生成女装/女士精品行业连衣裙品类的发在淘宝的小红书风格的推送配文，包括标题和内容。关键词：pe。要求:1. 推送标题要体现关键词和品类特点，语言通顺，有吸引力，约10个字；2. 推送内容要语言通顺，突出关键词和品类特点，对目标受众有吸引力，长度约30字。标题:",
    "100.25和90.75谁更大？",
]

chat_prompts = [chat_format(prompt) for prompt in prompts]

async def test_sampling_n(model):
    sampling_params = {
        'temperature': 0.8,
        'min_new_tokens': 8192,
        'max_new_tokens': 8192,
        'stream_interval': 50,
        'n': 3,
    }
    obj = GenerateReqInput(
        text=chat_prompts[0],
        sampling_params=sampling_params,
        rid=None,
        stream=True,
    )
    chunks: list[dict] = [None for _ in range(sampling_params['n'])]
    generator = model.tokenizer_manager.generate_request(obj, None)
    async for chunk in generator:
        index = chunk.get("index", 0)
        chunks[index] = chunk
    assert all(chunk is not None for chunk in chunks)
    assert all(chunk["meta_info"]["finish_reason"]["type"] == "length" for chunk in chunks)

async def test_abort_all(model):
    sampling_params = {
        'temperature': 0.8,
        'min_new_tokens': 8192,
        'max_new_tokens': 8192,
        'stream_interval': 50,
        'n': 3,
    }
    obj = GenerateReqInput(
        text=chat_prompts[0],
        sampling_params=sampling_params,
        stream=True,
    )
    async def _generate():
        generator = model.tokenizer_manager.generate_request(obj, None)
        chunks: list[dict] = [None for _ in range(sampling_params['n'])]
        generator = model.tokenizer_manager.generate_request(obj, None)
        async for chunk in generator:
            index = chunk.get("index", 0)
            chunks[index] = chunk
        return chunks
    task = asyncio.create_task(_generate())
    await asyncio.sleep(1)
    for rid in model.tokenizer_manager.rid_to_state:
        model.tokenizer_manager.abort_request(rid)
    chunks = await task
    assert all(chunk is not None for chunk in chunks) # assume at least generate one iter
    assert all(chunk["meta_info"]["finish_reason"]["type"] == "abort" for chunk in chunks)

async def test_abort(model):
    sampling_params = {
        'temperature': 0.8,
        'min_new_tokens': 8192,
        'max_new_tokens': 8192,
        'stream_interval': 50,
        'n': 1,
    }
    rid = uuid.uuid4().hex
    obj = GenerateReqInput(
        text=chat_prompts[0],
        sampling_params=sampling_params,
        rid=rid,
        stream=True,
    )
    async def _generate():
        generator = model.tokenizer_manager.generate_request(obj, None)
        chunk = None
        async for chunk in generator:
            chunk = chunk
        return chunk
    task = asyncio.create_task(_generate())
    await asyncio.sleep(1)
    model.tokenizer_manager.abort_request(rid)
    chunk = await task
    assert chunk is not None # assume at least generate one iter
    assert chunk["meta_info"]["finish_reason"]["type"] == "abort"

async def main():
    model_path = "Qwen/Qwen2.5-7B-Instruct"
    model_path = download_model(model_path)
    model = sglang_patch.engine.engine_module.Engine(
        enable_memory_saver= True,
        model_path=model_path,
        dtype="bfloat16",
        random_seed=1,
        tp_size=1,
        mem_fraction_static= 0.6,
        disable_custom_all_reduce=True,
    )

    await test_sampling_n(model)
    await test_abort_all(model)
    await test_abort(model)

if __name__ == "__main__":
    asyncio.run(main())
