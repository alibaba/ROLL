import asyncio
import traceback
import asyncio
import enum
from roll.utils.logging import get_logger

logger = get_logger()


class SglangInputType(enum.Enum):
    ADD = enum.auto()
    ABORT = enum.auto()

def list_endswith(lst, suffix):
    # Check if lst ends with suffix
    return lst[-len(suffix):] == suffix if len(suffix) <= len(lst) else False

def trim_overlap_tokens(existing_tokens, new_chunk_tokens):
    """
    copy trim_overlap in int list
    """
    max_overlap = 0
    max_possible = min(len(existing_tokens), len(new_chunk_tokens))
    for i in range(max_possible, 0, -1):
        if list_endswith(existing_tokens, new_chunk_tokens[:i]):
            max_overlap = i
            break
    return new_chunk_tokens[max_overlap:]


# Used to store all abort_rid_set
abort_rid_set = set()
abort_lock = asyncio.Lock()


async def producer(thread_queue, asyncio_queue):
    PRODUCER_PUT_TIMEOUT = 15 * 60
    while True:
        if not thread_queue.empty():
            data = thread_queue.get()
            # Received end marker
            if data is None:
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

    def process_sglang_output(token_ids, meta_info):
        # Online production use
        output_data = DataProto(meta_info=meta_info)
        output_data.meta_info["output_token_ids"] = token_ids
        request_complete_callback(data=output_data)

        # Local debugging use
        # request_complete_callback(meta_info['request_id'], token_ids)
        logger.debug(f"worker_id:{consumer_id} request_id: {meta_info['request_id']} finish!")

    try:
        while True:
            pack_data = await asyncio_queue.get()
            asyncio_queue.task_done()
            if pack_data is None:
                break

            command, data = pack_data

            rid, input_ids, sampling_params, meta_info = data
            rid_str = rid[0]
            async with abort_lock:
                if rid_str in abort_rid_set:
                    logger.debug(f"request_id: {rid_str} do not running!")
                    abort_rid_set.remove(rid_str)
                    continue

            final_tokens = [[] for _ in range(sampling_params['n'])]
            logger.debug(f"worker_id:{consumer_id} request_id: {rid} starting!")

            parallel_sample_num = 1
            if sampling_params['n'] > 1:
                rid = [rid]
                parallel_sample_num = sampling_params['n']
            
            obj = GenerateReqInput(
                # text=prompt,
                input_ids=input_ids,
                rid=rid,
                sampling_params=sampling_params,
                stream=True,
            )
            generator = llm.tokenizer_manager.generate_request(obj, None)

            # generator = await llm.async_generate(prompt, sampling_params, rid=rid, stream=True)
            generate_success = True
            async for chunk in generator:
                # chunk_text = chunk["text"]
                async with abort_lock:
                    if rid_str in abort_rid_set:
                        cur_abort_rid = chunk['meta_info']['id']
    
                        logger.debug(f"request_id: {rid_str}-{cur_abort_rid} aborting!")
                        llm.tokenizer_manager.abort_request(cur_abort_rid)
                        logger.debug(f"request_id: {rid_str}-{cur_abort_rid} abort success!")
                        parallel_sample_num -= 1

                        if parallel_sample_num == 0:
                            abort_rid_set.remove(rid_str)
                            generate_success = False
                            break

                chunk_tokens = chunk["output_ids"]
                chunk_index = chunk.get("index", 0)
                # logger.info(chunk["meta_info"])
                cleaned_chunk = trim_overlap_tokens(final_tokens[chunk_index], chunk_tokens)
                final_tokens[chunk_index] += cleaned_chunk
            # logger.info(f"consumer_id:{consumer_id} consumer finish: {final_text}")
            if generate_success:
                process_sglang_output(final_tokens, meta_info)
            # request_complete_callback(rid, final_tokens)
    except Exception as e:
        logger.info(traceback.format_exc())

async def predict_in_asyncio(model, request_complete_callback, thread_queue):
    PARALLELISM_WORKER_CNT = 128
    PRODUCER_BUFFER_SIZE = 40

    logger.info("[sglang asyncio] env setup...")
    async with abort_lock:
        abort_rid_set.clear()
    asyncio_queue = asyncio.Queue(maxsize=PRODUCER_BUFFER_SIZE)
    producer_task = asyncio.create_task(producer(thread_queue, asyncio_queue))
    consumers = [asyncio.create_task(consumer(asyncio_queue, i, model, request_complete_callback)) for i in range(PARALLELISM_WORKER_CNT)]
    logger.info("[sglang asyncio] env setup (done)")

    await producer_task
    logger.info("[sglang asyncio] killing consumers ...")
    for _ in range(len(consumers)):
        await asyncio_queue.put(None)
    # await asyncio_queue.join()
    logger.info("[sglang asyncio] finish signal has set")
    try:
        await asyncio.wait_for(asyncio.gather(*consumers), timeout=30)
    except asyncio.TimeoutError:
        logger.info("Timeout: Not all tasks completed within the time limit")
    # model.tokenizer_manager.asyncio_tasks.clear()
    # model.tokenizer_manager.no_create_loop = False
    logger.info("killing workers done, AsyncSglangEngine stop success")

def start_async_sglang(loop, model, request_complete_callback, thread_queue):
    try:
        loop.run_until_complete(predict_in_asyncio(model, request_complete_callback, thread_queue=thread_queue))    
    except Exception as e:
        logger.info(f"async_sglang thread raise Exception!\n{traceback.format_exc()}")

def add_request(thread_queue, data):
    thread_queue.put((SglangInputType.ADD, data))
    
def abort_request(thread_queue, rid):
    thread_queue.put((SglangInputType.ABORT, rid))
