import asyncio
import uuid

from transformers import AutoTokenizer

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang import __version__ as version

from roll.third_party.sglang import patch as sglang_patch
from roll.utils.checkpoint_manager import download_model

def chat_format(prompt):
    system = "Please reason step by step, and put your final answer within \\boxed{}."
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

async def generate(model, obj):
    generator = model.tokenizer_manager.generate_request(obj, None)
    chunks = None
    async for chunks in generator:
        chunks = chunks
    chunks = chunks if isinstance(chunks, list) else [chunks]
    return chunks

async def test_sampling_n(model, input_ids):
    sampling_params = {
        'temperature': 0.8,
        'min_new_tokens': 128,
        'max_new_tokens': 128,
        'n': 3,
    }
    obj = GenerateReqInput(
        input_ids=input_ids[0],
        sampling_params=sampling_params,
        rid=None,
        return_logprob=True,
    )
    chunks = await generate(model, obj)
    assert all(chunk is not None for chunk in chunks)
    assert all(chunk["meta_info"]["finish_reason"]["type"] == "length" for chunk in chunks)

async def test_abort_all(model, input_ids):
    sampling_params = {
        'temperature': 0.8,
        'min_new_tokens': 8192,
        'max_new_tokens': 8192,
        'n': 3,
    }
    obj1 = GenerateReqInput(
        rid=None if version < '0.5' and sampling_params["n"] > 1 else str(uuid.uuid4().hex),
        input_ids=input_ids[0],
        sampling_params=sampling_params,
        return_logprob=True,
    )
    obj2 = GenerateReqInput(
        rid=None if version < '0.5' and sampling_params["n"] > 1 else str(uuid.uuid4().hex),
        input_ids=input_ids[0],
        sampling_params=sampling_params,
        return_logprob=True,
    )
    tasks = [asyncio.create_task(generate(model, obj1)), asyncio.create_task(generate(model, obj2))]
    await asyncio.sleep(1)
    for rid in model.tokenizer_manager.rid_to_state:
        model.tokenizer_manager.abort_request(rid)
    responses = await asyncio.gather(*tasks)
    assert all(isinstance(response, list) and len(response) > 0 for response in responses) # assume at least generate one iter
    assert all(resp["meta_info"]["finish_reason"]["type"] == "abort" for response in responses for resp in response)

async def test_abort(model, input_ids):
    sampling_params = {
        'temperature': 0.8,
        'min_new_tokens': 8192,
        'max_new_tokens': 8192,
        'n': 1,
    }
    rid = uuid.uuid4().hex
    obj = GenerateReqInput(
        input_ids=input_ids[0],
        sampling_params=sampling_params,
        rid=rid,
        return_logprob=True,
    )
    task = asyncio.create_task(generate(model, obj))
    await asyncio.sleep(1)
    model.tokenizer_manager.abort_request(rid)
    response = await task
    assert response is not None and len(response) == 1 # assume at least generate one iter
    assert response[0]["meta_info"]["finish_reason"]["type"] == "abort"

async def main():
    model_path = "Qwen/Qwen2.5-7B-Instruct"
    model_path = download_model(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompts = [
        "类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞,生成一段文案",
        "根据关键词描述生成女装/女士精品行业连衣裙品类的发在淘宝的小红书风格的推送配文，包括标题和内容。关键词：pe。要求:1. 推送标题要体现关键词和品类特点，语言通顺，有吸引力，约10个字；2. 推送内容要语言通顺，突出关键词和品类特点，对目标受众有吸引力，长度约30字。标题:",
        "100.25和90.75谁更大？",
    ]
    prompts = [chat_format(prompt) for prompt in prompts]
    input_ids = tokenizer(prompts)["input_ids"]

    model = sglang_patch.engine.engine_module.Engine(
        model_path=model_path,
        enable_memory_saver= True,
        skip_tokenizer_init=False, # to use min_new_tokens
        dtype="bfloat16",
        tp_size=1,
        mem_fraction_static= 0.6,
        disable_custom_all_reduce=True,
    )

    await test_sampling_n(model, input_ids)
    await test_abort_all(model, input_ids)
    await test_abort(model, input_ids)

if __name__ == "__main__":
    asyncio.run(main())
