import json
import os
import pickle

import ray
from ray.runtime_env import RuntimeEnv
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoTokenizer
from vllm import SamplingParams

from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.third_party.vllm import create_async_llm
from utils import chat_prompts, generate_batch


model_path = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# os.environ["RAY_DEBUG"] = "legacy"

# breakpoint()
runtime_env = {
    "env_vars": {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCHINDUCTOR_COMPILE_THREADS": "2",
        # "RAY_DEBUG": "legacy",
        "NCCL_CUMEM_ENABLE": "0",  # https://github.com/NVIDIA/nccl/issues/1234
        "NCCL_NVLS_ENABLE": "0",
    }
}
ray.init(log_to_driver=True, runtime_env=runtime_env)
resource_manager = ResourceManager()
placement_groups = resource_manager.allocate_placement_group(world_size=1, device_mapping=[0])


@ray.remote
class TestActor:
    async def initialize(self, placement_groups):
        self.model = await create_async_llm(
            resource_placement_groups=placement_groups[0],
            model=model_path,
            block_size=16,
            dtype="bfloat16",
            gpu_memory_utilization=0.8,
            tensor_parallel_size=1,
            trust_remote_code=True,
            distributed_executor_backend="ray",
            disable_custom_all_reduce=True,
            enable_sleep_mode=True,
        )

    async def run(self):
        sampling_params = SamplingParams(temperature=0.0, top_p=0.99, top_k=100, max_tokens=512)
        await self.model.offload_states()
        import torch

        print(f"memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3}")

        # use torch.cuda.mem_get_info()[0] in sleep mode: https://github.com/vllm-project/vllm/pull/11743
        print(f"free: {torch.cuda.mem_get_info()[0] / 1024 ** 3}")
        import pdb

        pdb.set_trace()

        await self.model.load_states()

        vllm_outputs = await generate_batch(
            self.model,
            sampling_params=sampling_params,
            prompts=chat_prompts,
        )

        print(vllm_outputs)


env_vars = {
    "WORLD_SIZE": str(1),
    "RANK": str(0),
    "LOCAL_RANK": str(0),
    "CLUSTER_NAME": "",
    "WORKER_NAME": "",
}
env_vars.update(
    {
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, list(range(0, 8)))),
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
    }
)
runtime_env = RuntimeEnv(env_vars=env_vars)

actor = TestActor.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=placement_groups[0][0]["placement_group"]),
    name="actor",
    runtime_env=runtime_env,
    num_cpus=0.01,
    num_gpus=0.01,
).remote()
ray.get(actor.initialize.remote(placement_groups=placement_groups))
ray.get(actor.run.remote())

ray.shutdown()
