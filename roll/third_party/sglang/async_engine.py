import asyncio
import contextlib
import dataclasses
import enum
import traceback

from roll.utils.logging import get_logger


logger = get_logger()


class SglangInputType(enum.Enum):
    ADD = enum.auto()
    ABORT = enum.auto()


# 用于存放所有abort_rid_set
abort_rid_set = set()
abort_lock = asyncio.Lock()
stop_flag = False


async def producer(thread_queue, asyncio_queue):
    PRODUCER_PUT_TIMEOUT = 15 * 60
    global stop_flag
    stop_flag = False
    while True:
        if not thread_queue.empty():
            data = thread_queue.get()
            # 收到结束标记
            if data is None:
                stop_flag = True
                logger.info("[sglang async engine] receive stop signal, stoping")
                break
            command, command_data = data
            if command == SglangInputType.ABORT:
                async with abort_lock:
                    rid = command_data
                    abort_rid_set.add(rid)
            else:
                await asyncio.wait_for(asyncio_queue.put(data), timeout=PRODUCER_PUT_TIMEOUT)
        else:
            await asyncio.sleep(0.1)

async def consumer(asyncio_queue, consumer_id, llm, request_complete_callback):
    from sglang.srt.managers.io_struct import GenerateReqInput

    from roll.distributed.scheduler.protocol import DataProto

    def process_sglang_output(chunks, meta_info):
        output_data = DataProto(meta_info=meta_info)
        if chunks is None or chunks[0] is None:
            # report a abort request
            output_data.meta_info["finish_reasons"] = [None]  # not finished
            request_complete_callback(data=output_data)
            return

        output_token_ids = [chunk.get("output_ids", []) for chunk in chunks]
        output_logprobs = [chunk["meta_info"].get("output_token_logprobs", None) for chunk in chunks]
        has_logprobs = any(logprobs is not None for logprobs in output_logprobs)
        if has_logprobs:
            lens = [min(len(ids), len(logprobs)) for ids, logprobs in zip(output_token_ids, output_logprobs)]
            output_token_ids = [ids[:l] for ids, l in zip(output_token_ids, lens)]
            output_logprobs = [logprobs[:l] for logprobs, l in zip(output_logprobs, lens)]
            output_logprobs = [[prob_info[0] for prob_info in logprobs] for logprobs in output_logprobs]
            output_data.meta_info["output_logprobs"] = output_logprobs
            assert all([len(ids) == len(logprobs) for ids, logprobs in zip(output_token_ids, output_logprobs)]), (
                "output_token_ids and output_logprobs length not match"
            )
        output_data.meta_info["output_token_ids"] = output_token_ids
        output_data.meta_info["finish_reasons"] = [chunk["meta_info"].get("finish_reason") for chunk in chunks]
        request_complete_callback(data=output_data)
        logger.debug(f"worker_id:{consumer_id} request_id: {meta_info['request_id']} finish!")

    try:
        while True:
            pack_data = await asyncio_queue.get()
            asyncio_queue.task_done()
            if pack_data is None:
                break

            command, data = pack_data

            rid, input_ids, sampling_params, meta_info = data
            collect_unfinished = meta_info.get("collect_unfinished", False)
            rid_str = rid[0]

            final_chunks: list[dict] = [None for _ in range(sampling_params['n'])]
            logger.debug(f"worker_id:{consumer_id} request_id: {rid} starting!")

            if sampling_params['n'] > 1:
                rid = [rid]
                assert not collect_unfinished, "collect_unfinished is not supported in parallel sampling"

            obj_init_kw = {}  # return logprobs may be in GenerateReqInput not SamplingParams
            for field in dataclasses.fields(GenerateReqInput):
                if field.name in sampling_params:
                    obj_init_kw[field.name] = sampling_params.pop(field.name)
            from sglang import __version__ as version
            if version >= '0.4.6.post4':
                sampling_params['stream_interval'] = 50
            obj = GenerateReqInput(
                input_ids=input_ids,
                sampling_params=sampling_params,
                stream=True,
                **obj_init_kw,
            )

            need_abort = stop_flag
            async with abort_lock:
                if rid_str in abort_rid_set:
                    need_abort = True
                    logger.debug(f"request_id: {rid_str} do not running!")
            if need_abort:
                if collect_unfinished:
                    process_sglang_output(None, meta_info)
                continue

            generator = llm.tokenizer_manager.generate_request(obj, None)
            generate_success = True
            next_task = asyncio.create_task(generator.__anext__())
            while True:
                is_timeout = False
                try:
                    chunk = await asyncio.wait_for(asyncio.shield(next_task), timeout=10)
                    next_task = asyncio.create_task(generator.__anext__())
                except asyncio.TimeoutError:
                    is_timeout = True
                except StopAsyncIteration:
                    break
                if not is_timeout:
                    chunk_index = chunk.get("index", 0)
                    final_chunks[chunk_index] = chunk

                need_abort = stop_flag
                async with abort_lock:
                    if rid_str in abort_rid_set:
                        need_abort = True

                if need_abort:
                    logger.debug(f"request_id: {rid_str} aborting!")
                    if obj.is_single:
                        llm.tokenizer_manager.abort_request(obj.rid)
                    else:
                        for rid in obj.rid:
                            llm.tokenizer_manager.abort_request(rid)
                    logger.debug(f"request_id: {rid_str} abort success!")
                    generate_success = False
                    next_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await next_task
                    break

            if generate_success or collect_unfinished:
                process_sglang_output(final_chunks, meta_info)
    except Exception as e:
        logger.info(traceback.format_exc())

async def predict_in_asyncio(model, request_complete_callback, thread_queue, max_running_requests=128):
    PRODUCER_BUFFER_SIZE = 128

    logger.info("[sglang asyncio] env setup...")
    async with abort_lock:
        abort_rid_set.clear()
    asyncio_queue = asyncio.Queue(maxsize=PRODUCER_BUFFER_SIZE)
    producer_task = asyncio.create_task(producer(thread_queue, asyncio_queue))
    consumers = [
        asyncio.create_task(consumer(asyncio_queue, i, model, request_complete_callback))
        for i in range(max_running_requests)
    ]
    logger.info("[sglang asyncio] env setup (done)")

    await producer_task
    logger.info("[sglang asyncio] killing consumers ...")
    for _ in range(len(consumers)):
        await asyncio_queue.put(None)

    await asyncio_queue.join()
    logger.info("[sglang asyncio] finish signal has set")

    try:
        await asyncio.wait_for(asyncio.gather(*consumers), timeout=60)
    except asyncio.TimeoutError:
        logger.info("Timeout: Not all tasks completed within the time limit")
    # for safety, all requests should already be aborted
    for rid in model.tokenizer_manager.rid_to_state:
        model.tokenizer_manager.abort_request(rid)
    logger.info("killing workers done, AsyncSglangEngine stop success")

def start_async_sglang(loop, model, request_complete_callback, thread_queue, max_running_requests=128):
    try:
        loop.run_until_complete(
            predict_in_asyncio(
                model, request_complete_callback, thread_queue=thread_queue, max_running_requests=max_running_requests
            )
        )
    except Exception as e:
        logger.info(f"async_sglang thread raise Exception!\n{traceback.format_exc()}")

def add_request(thread_queue, data):
    thread_queue.put((SglangInputType.ADD, data))

def abort_request(thread_queue, rid):
    thread_queue.put((SglangInputType.ABORT, rid))
