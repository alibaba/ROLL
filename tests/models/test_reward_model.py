# Copyright (c) 2025, ALIBABA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import random

from roll.configs import DataArguments, ModelArguments, TrainingArguments

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
import torch

torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

from tqdm import tqdm

from roll.models.model_providers import default_reward_model_provider
from tests.models.load_utils import get_mock_dataloader


def reward_model_forward():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = "cuda"

    gpu_name = torch.cuda.get_device_name()

    model_name = "model.alimama_nebula2.reward_model/version=rm_qwen2_5_7B_trl/ckpt_id=checkpoint-1000"

    print(f"gpu_name: {gpu_name}")
    print(f"model_name: {model_name}")

    data_filename = "data/comparison_gpt4_data_zh.json"

    model_args: ModelArguments = ModelArguments(
        model_name_or_path=model_name, attn_implementation="disabled", dtype="bf16", model_type="trl"
    )
    data_args: DataArguments = DataArguments(
        template="qwen2_5",
        file_name=data_filename,
        prompt="instruction",
    )

    dataloader, tokenizer = get_mock_dataloader(model_args=model_args, data_args=data_args, batch_size=4)

    model = default_reward_model_provider(tokenizer, model_args, TrainingArguments(), False)

    results = []
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits
        prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        for prompt, input_id, reward in zip(prompts, input_ids, logits):
            results.append({"prompt": prompt, "input_ids": input_id.tolist(), "reward": reward.tolist()})
        with open(f"reward_forward_res_{gpu_name}_{model_args.dtype}_{model_args.attn_implementation}", "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    reward_model_forward()
