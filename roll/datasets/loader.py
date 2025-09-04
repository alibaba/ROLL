import json
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
            messages = example[data_args.messages] if not isinstance(example[data_args.messages], str) else json.loads(example[data_args.messages])
        else:
            messages = [{"role": "user", "content": example[data_args.prompt]}]
        text = chat_template_func(messages)
        encodings = tokenizer(text)
        return encodings

    dataset = dataset.map(encode_function, batched=False, desc="Encoding dataset")
    return dataset
