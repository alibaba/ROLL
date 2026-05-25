import json
import os

import pytest
from accelerate import cpu_offload_with_hook

from roll.configs import ModelArguments, DataArguments, TrainingArguments
from roll.platforms import current_platform
from roll.utils.offload_states import offload_hf_model, load_hf_model
from tests.models.load_utils import get_generation_eos_token_ids, get_mock_dataloader, get_model_input_device

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from tqdm import tqdm

from roll.models.model_providers import default_actor_model_provider

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
data_filename = "data/comparison_gpt4_data_zh.json"

attn_implementation = "sdpa" if current_platform.is_npu() else "fa2"
model_args: ModelArguments = ModelArguments(
    model_name_or_path=model_name,
    attn_implementation=attn_implementation,
    dtype="bf16",
)
data_args: DataArguments = DataArguments(
    template="qwen2_5",
    file_name=data_filename,
    prompt="instruction",
)

# This generate/offload smoke test takes too long in NPU CI.
@pytest.mark.skip_on_npu
def test_hf_multi_gpus_cpu_offload_with_hook():
    dataloader, tokenizer = get_mock_dataloader(model_args=model_args, data_args=data_args, batch_size=4)
    model = default_actor_model_provider(tokenizer, model_args, TrainingArguments(),  False)

    hook = None
    for i, batch in tqdm(enumerate(dataloader)):
        print(f"step: {i}")

        input_device = get_model_input_device(model)
        input_ids = batch["input_ids"].to(input_device)
        attention_mask = batch["attention_mask"].to(input_device)
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=get_generation_eos_token_ids(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
        )

        print(f"before offload, hf_device_map: {getattr(model, 'hf_device_map', None)}")

        if not hook:
            model, hook = cpu_offload_with_hook(model)
        print(f"after offload, hf_device_map: {getattr(model, 'hf_device_map', None)}")
        print(f"after offload: {i}")
        input_device = get_model_input_device(model)
        input_ids = input_ids.to(input_device)
        attention_mask = attention_mask.to(input_device)
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=get_generation_eos_token_ids(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
        )

        output_str = tokenizer.batch_decode(output, skip_special_tokens=True)
        print(output_str)


# This multi-GPU HF offload smoke test assumes CUDA device maps.
@pytest.mark.skip_on_npu
def test_hf_multi_gpus_cpu_offload_hf_device_map():
    dataloader, tokenizer = get_mock_dataloader(model_args=model_args, data_args=data_args, batch_size=4)
    model = default_actor_model_provider(tokenizer, model_args, TrainingArguments(), False)

    hook = None
    for i, batch in tqdm(enumerate(dataloader)):
        print(f"step: {i}")

        input_device = get_model_input_device(model)
        input_ids = batch["input_ids"].to(input_device)
        attention_mask = batch["attention_mask"].to(input_device)
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=get_generation_eos_token_ids(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
        )

        print(f"before offload, hf_device_map: {getattr(model, 'hf_device_map', None)}")

        offload_hf_model(model=model)

        print(f"after offload, hf_device_map: {getattr(model, 'hf_device_map', None)}")
        print(f"after offload: {i}")
        load_hf_model(model=model)

        input_device = get_model_input_device(model)
        input_ids = input_ids.to(input_device)
        attention_mask = attention_mask.to(input_device)
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=get_generation_eos_token_ids(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
        )

        output_str = tokenizer.batch_decode(output, skip_special_tokens=True)
        print(output_str)


if __name__ == "__main__":
    # test_hf_multi_gpus_cpu_offload_with_hook()
    test_hf_multi_gpus_cpu_offload_hf_device_map()
