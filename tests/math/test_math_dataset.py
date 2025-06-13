from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from roll.distributed.scheduler.protocol import DataProto
from roll.datasets.collator import DataCollatorWithPaddingForPaddedKeys

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
dataset_path = "/home/weixun.wwx/Numina_hardrule_1212_lv2.json"
dataset = load_dataset("json", data_files=dataset_path)["train"]


# Add format, then convert to ids function
def encode_function(data_i):
    text_list = []
    for instruct in data_i["prompt"]:
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": instruct},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text_list.append(text)
    encodings = tokenizer(text_list)
    return encodings


# Process data
print(dataset)
dataset = dataset.map(encode_function, batched=True, desc="Encoding dataset")
print(dataset)
# Filter cutoff
dataset = dataset.filter(lambda data_i: len(data_i["input_ids"]) <= 512, desc="Filtering dataset")
print(dataset)
# ------
data_collator = DataCollatorWithPaddingForPaddedKeys(
    tokenizer=tokenizer,
    max_length=1024,
    padding="max_length",
)

dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, drop_last=True, collate_fn=data_collator)

for batch_dict in tqdm(dataloader):
    batch: DataProto = DataProto.from_single_dict(batch_dict)
    print(type(batch.non_tensor_batch))
    for key, value in batch.non_tensor_batch.items():
        print("-" * 20)
        print(key, type(value), value.shape)
        print(value[0])
        print(value[1])
        print(value[2])
        new_value = np.repeat(value, 3)
        print("*" * 20)
        print(new_value.shape)
        print(new_value[0])
        print(new_value[1])
        print(new_value[2])
        print(new_value[3])
        print(new_value[4])
        print(new_value[5])
        print(new_value[6])
        print(new_value[7])
        print(new_value[8])
    break
