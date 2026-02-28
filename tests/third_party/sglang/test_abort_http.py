import asyncio
import atexit
import uuid
import httpx
import multiprocessing
from urllib.parse import quote

from transformers import AutoTokenizer

from sglang_router.launch_router import RouterArgs, launch_router
from sglang import __version__ as version

from roll.distributed.scheduler.router import wait_sglang_router_ready, wait_sglang_router_workflow, raise_for_status
from roll.distributed.strategy.sglang_strategy import SglangHttpEngine, shutdown
from roll.distributed.executor.worker import Worker
from roll.utils.checkpoint_manager import download_model
from roll.utils.logging import get_logger


logger = get_logger()

def chat_format(prompt):
    system = "Please reason step by step, and put your final answer within \\boxed{}."
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

async def generate(client, url, payload):
    response = await client.post(f"{url}/generate", json=payload)
    response.raise_for_status()
    response = response.json()
    response = response if isinstance(response, list) else [response]
    return response

async def test_sampling_n(
    url,
    client: httpx.AsyncClient,
    input_ids,
):
    payload = {
        "rid": None,
        "input_ids": input_ids[0],
        "sampling_params": {
            'temperature': 0.8,
            'min_new_tokens': 128,
            'max_new_tokens': 128,
            'n': 3,
        },
        "return_logprob": True,
    }
    response = await generate(client, url, payload)
    assert len(response) == payload["sampling_params"]["n"]
    assert all(resp["meta_info"]["finish_reason"]["type"] == "length" for resp in response)
    print(">>>>>>>>>>>>>>>>>>>>>>> TEST_sampling_n passed")

async def test_abort_all(
    url,
    client: httpx.AsyncClient,
    input_ids,
    worker_url,
):
    payload1 = {
        "rid": uuid.uuid4().hex,
        "input_ids": input_ids[0],
        "sampling_params": {
            'temperature': 0.8,
            'min_new_tokens': 8192,
            'max_new_tokens': 8192,
            'n': 1 if version < '0.5' else 3,
        },
        "return_logprob": True,
    }
    payload2 = {
        "rid": uuid.uuid4().hex,
        "input_ids": input_ids[0],
        "sampling_params": {
            'temperature': 0.8,
            'min_new_tokens': 8192,
            'max_new_tokens': 8192,
            'n': 1 if version < '0.5' else 3,
        },
        "return_logprob": True,
    }
    tasks = [asyncio.create_task(generate(client, url, payload1)), asyncio.create_task(generate(client, url, payload2))]
    await asyncio.sleep(1)

    if version < '0.5':
        response = await asyncio.gather(
            client.post(f"{worker_url}/abort_request", json={"rid": payload1["rid"]}),
            client.post(f"{worker_url}/abort_request", json={"rid": payload2["rid"]}),
        )
        for resp in response:
            resp.raise_for_status()
    else:
        response = await client.post(f"{worker_url}/abort_request", json={"abort_all": True})
        response.raise_for_status()

    responses = await asyncio.gather(*tasks)
    assert all(isinstance(response, list) and len(response) > 0 for response in responses) # assume at least generate one iter
    assert all(resp["meta_info"]["finish_reason"]["type"] == "abort" for response in responses for resp in response)
    print(">>>>>>>>>>>>>>>>>>>>>>> TEST_abort_all passed")

async def test_abort(
    url,
    client: httpx.AsyncClient,
    input_ids,
    worker_url,
):
    payload = {
        "rid": uuid.uuid4().hex,
        "input_ids": input_ids[0],
        "sampling_params": {
            'temperature': 0.8,
            'min_new_tokens': 8192,
            'max_new_tokens': 8192,
            'n': 1,
        },
        "return_logprob": True,
    }
    task = asyncio.create_task(generate(client, url, payload))
    await asyncio.sleep(1)

    response = await client.post(f"{worker_url}/abort_request", json={"rid": payload["rid"]})
    response.raise_for_status()

    response = await task
    assert response is not None and len(response) == 1 # assume at least generate one iter
    assert response[0]["meta_info"]["finish_reason"]["type"] == "abort"
    print(">>>>>>>>>>>>>>>>>>>>>>> TEST_abort passed")

def start_router():
    router_config = {
        "host": Worker.get_node_ip(),
        "port": Worker.get_free_port(),
        "prometheus_port": Worker.get_free_port(),
        # "health_check_endpoint": "/dummy_health",
    }

    router_args = RouterArgs(**router_config)
    router_process = multiprocessing.Process(
        target=launch_router,
        args=(router_args,),
        daemon=True
    )
    router_process.start()
    logger.info(f"Launch sglang-router {router_args=}")
    return router_process, router_config["host"], router_config["port"]

def start_server(model_path):
    sglang_config = {
        "model_path": model_path,
        "enable_memory_saver": True,
        "skip_tokenizer_init": False, # to use min_new_tokens
        "mem_fraction_static": 0.8,
        "trust_remote_code": True,
        "tp_size": 1,
        "log_level": "info",
        "disable_custom_all_reduce": True,
        "host": Worker.get_node_ip(),
        "port": Worker.get_free_port(),
    }
    worker_process = multiprocessing.Process(
        target=SglangHttpEngine.launch_server,
        args=(sglang_config,),
    )
    worker_process.start()
    worker_url = f"http://{sglang_config['host']}:{sglang_config['port']}"
    logger.info(f"start sglang server url={worker_url}")
    return worker_process, worker_url 

async def main():
    multiprocessing.set_start_method("spawn")
    atexit.register(shutdown)

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

    client = httpx.AsyncClient(timeout=httpx.Timeout(None))

    worker_process, worker_url = start_server(model_path)
    await SglangHttpEngine.wait_worker_healthy(worker_process=worker_process, url=worker_url, client=client)

    response = await client.post(f"{worker_url}/release_memory_occupation", json={})
    response.raise_for_status()

    enable_router = True
    if enable_router:
        router_process, router_ip, router_port = start_router()
        await wait_sglang_router_ready(router_process, f"http://{router_ip}:{router_port}")

        response = await client.post(f"http://{router_ip}:{router_port}/workers", json={"url": worker_url})
        response.raise_for_status()
        url = f"http://{router_ip}:{router_port}"
        await wait_sglang_router_workflow(f"http://{router_ip}:{router_port}", [worker_url])

        encoded_url = quote(worker_url, safe="")
        response = await client.delete(f"http://{router_ip}:{router_port}/workers/{encoded_url}")
        raise_for_status(response)
        await wait_sglang_router_workflow(f"http://{router_ip}:{router_port}", [])

        response = await client.post(f"http://{router_ip}:{router_port}/workers", json={"url": worker_url})
        response.raise_for_status()
        url = f"http://{router_ip}:{router_port}"
        await wait_sglang_router_workflow(f"http://{router_ip}:{router_port}", [worker_url])
    else:
        url = worker_url

    response = await client.post(f"{worker_url}/resume_memory_occupation", json={})
    response.raise_for_status()

    await test_sampling_n(url, client, input_ids)
    await test_abort_all(url, client, input_ids, worker_url)
    await test_abort(url, client, input_ids, worker_url)

if __name__ == "__main__":
    asyncio.run(main())
