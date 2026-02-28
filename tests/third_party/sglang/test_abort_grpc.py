import asyncio
import atexit
import grpc
import uuid
import httpx
import multiprocessing

from transformers import AutoTokenizer

from sglang_router.launch_router import RouterArgs, launch_router
from sglang.srt.grpc import sglang_scheduler_pb2, sglang_scheduler_pb2_grpc
from sglang import __version__ as version

from roll.distributed.scheduler.router import wait_sglang_router_ready, wait_sglang_router_workflow
from roll.distributed.strategy.sglang_strategy import SglangGrpcEngine, shutdown
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

async def test_sampling_n_grpc(
    url,
    client,
    stub,
    input_ids,
):
    request = sglang_scheduler_pb2.GenerateRequest(
        tokenized=sglang_scheduler_pb2.TokenizedInput(
            input_ids=input_ids[0]
        ),
        sampling_params=sglang_scheduler_pb2.SamplingParams(
            temperature=0.8,
            max_new_tokens=128,
            n=3,
        ),
        return_logprob=True,
        stream=False,
    )
    responses = []
    async for response in stub.Generate(request):
        assert not response.HasField("error")
        assert response.HasField("complete")
        responses.append(response)
    assert len(responses) == 3
    for response in responses:
        # print(f"{response.complete.finish_reason=} {list(response.complete.output_ids)=}")
        assert response.complete.finish_reason == "length"
    print(">>>>>>>>>>>>>>>>>>>>>>> TEST_sampling_n_grpc passed")

async def test_sampling_n(
    url,
    client,
    stub,
    input_ids,
):
    payload = {
        "rid": uuid.uuid4().hex,
        "input_ids": input_ids[0],
        "sampling_params": {
            'temperature': 0.8,
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
    client, 
    stub,
    input_ids,
    worker_url,
):
    payload1 = {
        "rid": uuid.uuid4().hex,
        "input_ids": input_ids[0],
        "sampling_params": {
            'temperature': 0.8,
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
            'max_new_tokens': 8192,
            'n': 1 if version < '0.5' else 3,
        },
        "return_logprob": True,
    }
    tasks = [asyncio.create_task(generate(client, url, payload1)), asyncio.create_task(generate(client, url, payload2))]
    await asyncio.sleep(1)

    # sglang grpc do not support abort all now
    responses = await asyncio.gather(
        stub.Abort(sglang_scheduler_pb2.AbortRequest(request_id=payload1["rid"])),
        stub.Abort(sglang_scheduler_pb2.AbortRequest(request_id=payload2["rid"])),
    )
    for response in responses:
        assert response.success

    responses = await asyncio.gather(*tasks)
    assert all(isinstance(response, list) and len(response) > 0 for response in responses) # assume at least generate one iter
    for response in responses:
        for resp in response:
            finish_reason = resp["meta_info"]["finish_reason"]
            if isinstance(finish_reason, dict):
                assert finish_reason["type"] in ["abort", "length", "stop"]
            else:
                assert finish_reason in ["abort", "length", "stop"]
    print(">>>>>>>>>>>>>>>>>>>>>>> TEST_abort_all passed")

async def test_abort(
    url,
    client,
    stub,
    input_ids,
    worker_url,
):
    payload = {
        "rid": uuid.uuid4().hex,
        "input_ids": input_ids[0],
        "sampling_params": {
            'temperature': 0.8,
            'max_new_tokens': 8192,
            'n': 1,
        },
        "return_logprob": True,
    }
    task = asyncio.create_task(generate(client, url, payload))
    await asyncio.sleep(1)

    response = await stub.Abort(sglang_scheduler_pb2.AbortRequest(request_id=payload["rid"]))
    assert response.success

    # will stuck if abort manually https://github.com/sgl-project/sglang/issues/14338
    response = await task
    assert response is not None and len(response) == 1 # assume at least generate one iter
    assert response[0]["meta_info"]["finish_reason"]["type"] in ["abort", "length"]
    print(">>>>>>>>>>>>>>>>>>>>>>> TEST_abort passed")

def start_router(model_path, worker_urls):
    # assert sgl-router.__version__ == '0.2.1'
    # newer version of sglang-router need to download openai harmony encoding
    # https://github.com/sgl-project/sglang/issues/14340
    router_config = {
        "host": Worker.get_node_ip(),
        "port": Worker.get_free_port(),
        "log_level": "debug",
        "enable_igw": False, # required by grpc_mode
        "model_path": model_path, # required by grpc_mode
        "worker_urls": worker_urls, # must provide at least one url at router init for grpc
        "dp_aware": False,
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
        "skip_tokenizer_init": False, # must
        "mem_fraction_static": 0.8,
        "trust_remote_code": True,
        "tp_size": 1,
        "log_level": "debug",
        "disable_custom_all_reduce": True,
        "host": Worker.get_node_ip(),
        "port": Worker.get_free_port(),
        "grpc_mode": True, # must
    }
    worker_process = multiprocessing.Process(
        target=SglangGrpcEngine.launch_server,
        args=(sglang_config,),
    )
    worker_process.start()
    worker_url = f"grpc://{sglang_config['host']}:{sglang_config['port']}"
    logger.info(f"start sglang server url={worker_url}")
    return worker_process, sglang_config["host"], sglang_config["port"] 

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

    worker_process, worker_ip, worker_port = start_server(model_path)
    worker_url = f"grpc://{worker_ip}:{worker_port}"

    channel = grpc.aio.insecure_channel(
        f"{worker_ip}:{worker_port}",
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 256),
            ("grpc.max_receive_message_length", 1024 * 1024 * 256),
        ],
    )
    stub = sglang_scheduler_pb2_grpc.SglangSchedulerStub(channel)

    await SglangGrpcEngine.wait_worker_healthy(worker_process, url=worker_url, client=stub)

    router_process, router_ip, router_port = start_router(model_path=model_path, worker_urls=[worker_url])
    await wait_sglang_router_ready(router_process, f"http://{router_ip}:{router_port}")
    # response = await client.post(f"http://{router_ip}:{router_port}/workers", json={"url": worker_url})
    # response.raise_for_status()
    url = f"http://{router_ip}:{router_port}"
    await wait_sglang_router_workflow(f"http://{router_ip}:{router_port}", [worker_url])

    await test_sampling_n_grpc(url, client, stub, input_ids)
    await test_sampling_n(url, client, stub, input_ids)
    await test_abort_all(url, client, stub, input_ids, worker_url)
    await test_abort(url, client, stub, input_ids, worker_url)

if __name__ == "__main__":
    asyncio.run(main())
