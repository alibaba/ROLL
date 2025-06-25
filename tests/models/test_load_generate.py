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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
from tqdm import tqdm

from roll.models.model_providers import default_actor_model_provider
from tests.models.load_utils import get_mock_dataloader


def test_load_generate():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = "cuda"

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    data_filename = "data/comparison_gpt4_data_zh.json"

    model_args: ModelArguments = ModelArguments(model_name_or_path=model_name, flash_attn="fa2", dtype="bf16")
    data_args: DataArguments = DataArguments(
        template="qwen2_5",
        file_name=data_filename,
        prompt="instruction",
    )

    dataloader, tokenizer = get_mock_dataloader(model_args=model_args, data_args=data_args, batch_size=4)

    model = default_actor_model_provider(tokenizer, model_args, TrainingArguments(), False)

    results = []
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=[tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

        output_str = tokenizer.batch_decode(output, skip_special_tokens=False)
        results.append(output_str)

        with open("generate_res.json", "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    test_load_generate()
