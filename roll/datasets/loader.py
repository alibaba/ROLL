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
from typing import Union

from datasets import Dataset, IterableDataset, load_dataset
from transformers import PreTrainedTokenizer

from roll.configs.data_args import DataArguments
from roll.datasets.chat_template import get_chat_template
from roll.utils.logging import get_logger


logger = get_logger()


def get_dataset(
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
) -> Union["Dataset", "IterableDataset"]:
    file_name = data_args.file_name
    data_path = {"jsonl": "json"}.get(file_name.split(".")[-1], file_name.split(".")[-1])
    dataset = load_dataset(path=data_path, data_files=file_name)["train"]
    chat_template_func = get_chat_template(data_args.template, tokenizer)

    def encode_function(example):
        if data_args.messages is not None:
            messages = example[data_args.messages]
        else:
            messages = [{"role": "user", "content": example[data_args.prompt]}]
        text = chat_template_func(messages)
        encodings = tokenizer(text)
        return encodings

    dataset = dataset.map(encode_function, batched=False, desc="Encoding dataset")
    return dataset
