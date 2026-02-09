import asyncio
import ray
from transformers import AutoTokenizer
from vllm import SamplingParams

from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.third_party.vllm import create_async_llm
from roll.utils.context_managers import cpu_memory_info
from roll.utils.logging import get_logger
from utils import generate_batch, chat_prompts

logger = get_logger()

async def main():
    model_path = "Qwen/Qwen2.5-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # os.environ["RAY_DEBUG"] = "legacy"

    ray.init()
    resource_manager = ResourceManager()
    placement_groups = resource_manager.allocate_placement_group(world_size=1, device_mapping=list(range(1)))
    sampling_params = SamplingParams(temperature=0.0, top_p=0.99, top_k=100, max_tokens=1024)

    model = await create_async_llm(
        resource_placement_groups=placement_groups[0],
        model=model_path,
        block_size=16,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        trust_remote_code=True,
        load_format="dummy",
    )


    from memory_profiler import profile
    import tracemalloc

    # tracemalloc.start()

    snapshot_1 = None
    snapshot_last = None


    # @profile
    async def generate_memory():
        global snapshot_1, snapshot_last
        for _ in range(20):
            await model.load_states()
            await generate_batch(
                model,
                sampling_params=sampling_params,
                prompts=chat_prompts,
                use_tqdm=False,
            )
            model.offload_states()
            rss = cpu_memory_info().rss / 1024**2
            logger.info(f"rss: {rss}")
            # snapshot_last = tracemalloc.take_snapshot()
            # if snapshot_1 is None:
            #     snapshot_1 = snapshot_last


    await generate_memory()

    # tracemalloc.stop()

    # snapshot.dump(f"mem_dump.pickle")
    ray.shutdown()

    # https://www.datacamp.com/tutorial/memory-profiling-python
    #
    # stats_1 = snapshot_1.compare_to(snapshot_last, 'lineno')
    #
    # with open('memory_leak_analysis.txt', 'w') as f:
    #     f.write("[ Memory usage increase from snapshot 1 to snapshot 2 ]\n")
    #     for stat in stats_1[:10]:
    #         f.write(f"{stat}\n")
    #
    #     # Detailed traceback for the top memory consumers
    #     f.write("\n[ Detailed traceback for the top memory consumers ]\n")
    #     for stat in stats_1[:-1]:
    #         f.write('\n'.join(stat.traceback.format()) + '\n\n\n')

if __name__ == "__main__":
    asyncio.run(main())
