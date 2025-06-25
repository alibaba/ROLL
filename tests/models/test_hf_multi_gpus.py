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

from accelerate import cpu_offload_with_hook

from roll.configs import ModelArguments, DataArguments, TrainingArguments
from roll.utils.offload_states import offload_hf_model, load_hf_model
from tests.models.load_utils import get_mock_dataloader

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from tqdm import tqdm

from roll.models.model_providers import default_actor_model_provider

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
data_filename = "data/comparison_gpt4_data_zh.json"

model_args: ModelArguments = ModelArguments(model_name_or_path=model_name, flash_attn="fa2", dtype="bf16")
data_args: DataArguments = DataArguments(
    template="qwen2_5",
    file_name=data_filename,
    prompt="instruction",
)


def test_hf_multi_gpus_cpu_offload_with_hook():
    dataloader, tokenizer = get_mock_dataloader(model_args=model_args, data_args=data_args, batch_size=4)
    model = default_actor_model_provider(tokenizer, model_args, TrainingArguments(),  False)

    hook = None
    for i, batch in tqdm(enumerate(dataloader)):
        print(f"step: {i}")

        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=[tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

        print(f"before offload, hf_device_map: {model.hf_device_map}")

        if not hook:
            model, hook = cpu_offload_with_hook(model)
        print(f"after offload, hf_device_map: {model.hf_device_map}")
        print(f"after offload: {i}")
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=[tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

        output_str = tokenizer.batch_decode(output, skip_special_tokens=True)
        print(output_str)


def test_hf_multi_gpus_cpu_offload_hf_device_map():
    dataloader, tokenizer = get_mock_dataloader(model_args=model_args, data_args=data_args, batch_size=4)
    model = default_actor_model_provider(tokenizer, model_args, TrainingArguments(), False)

    hook = None
    for i, batch in tqdm(enumerate(dataloader)):
        print(f"step: {i}")

        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=[tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

        print(f"before offload, hf_device_map: {model.hf_device_map}")

        offload_hf_model(model=model)

        print(f"after offload, hf_device_map: {model.hf_device_map}")
        print(f"after offload: {i}")
        load_hf_model(model=model)

        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=[tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

        output_str = tokenizer.batch_decode(output, skip_special_tokens=True)
        print(output_str)


if __name__ == "__main__":
    # test_hf_multi_gpus_cpu_offload_with_hook()
    test_hf_multi_gpus_cpu_offload_hf_device_map()
