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

from roll.configs import ModelArguments, DataArguments, TrainingArguments
from roll.utils.functionals import log_probs_from_logits

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
import torch

torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

from tqdm import tqdm

from roll.models.model_providers import default_actor_model_provider
from tests.models.load_utils import get_mock_dataloader


def hf_forward_logprobs():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = "cuda"

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"model_name: {model_name}")

    data_filename = "data/comparison_gpt4_data_zh.json"

    model_args: ModelArguments = ModelArguments(model_name_or_path=model_name, flash_attn="disabled", dtype="bf16")
    data_args: DataArguments = DataArguments(
        template="qwen2_5",
        file_name=data_filename,
        prompt="instruction",
    )

    dataloader, tokenizer = get_mock_dataloader(model_args=model_args, data_args=data_args, batch_size=8)
    model = default_actor_model_provider(tokenizer, model_args, TrainingArguments(), False)

    results = []
    response_length = 32
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        output1 = model(input_ids=input_ids, attention_mask=attention_mask)
        output2 = model(input_ids=input_ids, attention_mask=attention_mask, num_logits_to_keep=response_length + 1)

        response = input_ids[:, -response_length:]
        log_probs1 = log_probs_from_logits(output1.logits[:, -response_length - 1 : -1], response)
        log_probs2 = log_probs_from_logits(output2.logits[:, -response_length - 1 : -1], response)
        prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        for prompt, input_id, logpb1, logpb2 in zip(prompts, input_ids, log_probs1, log_probs2):
            results.append(
                {
                    "prompt": prompt,
                    "input_ids": input_id.tolist(),
                    "logpb1": logpb1.tolist(),
                    "logpb2": logpb2.tolist(),
                    "equal": torch.equal(logpb1, logpb2),
                }
            )
            assert torch.equal(logpb1, logpb2)
        with open(f"hf_forward_logprobs_res.json", "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    hf_forward_logprobs()
