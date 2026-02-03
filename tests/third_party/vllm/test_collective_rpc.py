import ray
import asyncio
import torch
import pytest
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.third_party.vllm import create_async_llm
from roll.third_party.vllm.worker_helper import WorkerHelper
from roll.utils.checkpoint_manager import download_model


def load_weight_tensor(self, name, param):
    self.load_weights([(name, param)])
WorkerHelper.load_weight_tensor = load_weight_tensor

def load_weight_numpy(self, name, param):
    param = torch.from_numpy(param)
    self.load_weights([(name, param)])
WorkerHelper.load_weight_numpy = load_weight_numpy

def load_weight_list(self, name, dtype, buffer):
    weight = torch.tensor(buffer, dtype=dtype).cuda()
    self.load_weights([(name, weight)])
WorkerHelper.load_weight_list = load_weight_list

async def test_vllm_collective_rpc():
    ray.init()
    resource_manager = ResourceManager(1, 1)
    placement_groups = resource_manager.allocate_placement_group(world_size=1, device_mapping=[0])

    model_path = "Qwen/Qwen2.5-7B-Instruct"
    model_path = download_model(model_path)
    model = await create_async_llm(
        resource_placement_groups=placement_groups[0],
        model=model_path,
        load_format="auto",
        block_size=16,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        disable_custom_all_reduce=True,
        enable_sleep_mode=True,
        enforce_eager=False,
    )

    train_model = AutoModelForCausalLM.from_pretrained(model_path)

    print(">>>>>>>>>>>>>>> test_vllm_rpc: tensor(cuda)")
    with pytest.raises(Exception):
        try:
            for name, param in tqdm(list(train_model.named_parameters()), desc="Updating parameter", unit="param"):
                await model.engine_core.collective_rpc_async(method="load_weight_tensor", args=(name, param.detach().cuda()))
        except Exception as e:
            print("<<<<<<<<<<<<<<< exception: ", e)
            raise

    print(">>>>>>>>>>>>>>> test_vllm_rpc: tensor(cpu)")
    with pytest.raises(Exception):
        try:
            for name, param in tqdm(list(train_model.named_parameters()), desc="Updating parameter", unit="param"):
                await model.engine_core.collective_rpc_async(method="load_weight_tensor", args=(name, param.detach().cpu()))
        except Exception as e:
            print("<<<<<<<<<<<<<<< exception: ", e)
            raise

    print(">>>>>>>>>>>>>>> test_vllm_rpc: numpy")
    with pytest.raises(Exception):
        try:
            for name, param in tqdm(list(train_model.named_parameters()), desc="Updating parameter", unit="param"):
                await model.engine_core.collective_rpc_async(method="load_weight_numpy", args=(name, param.detach().numpy()))
        except Exception as e:
            print("<<<<<<<<<<<<<<< exception: ", e)
            raise

    print(">>>>>>>>>>>>>>> test_vllm_rpc: list")
    for name, p in tqdm(list(train_model.named_parameters()), desc="Updating parameter", unit="param"):
        await model.engine_core.collective_rpc_async(method="load_weight_list", args=(name, p.dtype, p.tolist()))

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_vllm_collective_rpc())
