import pytest
import ray
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from vllm import SamplingParams

from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.third_party.vllm import LLM
from roll.third_party.vllm.worker_helper import WorkerHelper
from roll.utils.checkpoint_manager import download_model


def load_weight_tensor(self, name, param):
    self.load_weights([(name, param)])

WorkerHelper.load_weight_tensor = load_weight_tensor

def load_weights_tensor(self, model):
    for name, param in tqdm(list(model.named_parameters()), desc="Updating parameter", unit="param"):
        self.collective_rpc(method="load_weight_tensor", args=(name, param.detach().cuda()))

LLM.load_weights_tensor = load_weights_tensor


def load_weight_numpy(self, name, param):
    param = torch.from_numpy(param)
    self.load_weights([(name, param)])

WorkerHelper.load_weight_numpy = load_weight_numpy

def load_weights_numpy(self, model):
    for name, param in tqdm(list(model.named_parameters()), desc="Updating parameter", unit="param"):
        self.collective_rpc(method="load_weight_numpy", args=(name, param.detach().numpy()))

LLM.load_weights_numpy = load_weights_numpy


def load_weight_list(self, name, dtype, buffer):
    weight = torch.tensor(buffer, dtype=dtype).cuda()
    self.load_weights([(name, weight)])

WorkerHelper.load_weight_list = load_weight_list

def load_weights_list(self, model):
    for name, p in tqdm(list(model.named_parameters()), desc="Updating parameter", unit="param"):
        self.collective_rpc(method="load_weight_list", args=(name, p.dtype, p.tolist()))

LLM.load_weights_list = load_weights_list


def test_model_update_single_gpu():
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    model_path = download_model(model_path)

    ray.init()
    resource_manager = ResourceManager(1, 1)
    placement_groups = resource_manager.allocate_placement_group(world_size=1, device_mapping=[0])

    model = LLM(
        resource_placement_groups=placement_groups[0],
        model=model_path,
        block_size=16,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        trust_remote_code=True,
        disable_custom_all_reduce=True,
        enforce_eager=True,
        enable_sleep_mode=True,
    )

    train_model = AutoModelForCausalLM.from_pretrained(model_path)

    with pytest.raises(Exception):
        try:
            model.load_weights_tensor(train_model)
        except Exception as e:
            print("load_weights_tensor exception: ", e)
            raise

    with pytest.raises(Exception):
        try:
            model.load_weights_numpy(train_model)
        except Exception as e:
            print("load_weights_numpy exception: ", e)
            raise

    model.load_weights_list(train_model)


if __name__ == "__main__":
    test_model_update_single_gpu()
