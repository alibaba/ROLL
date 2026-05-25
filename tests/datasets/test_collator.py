import torch

from roll.datasets.collator import DataCollatorWithPaddingForPaddedKeys


class DummyTokenizer:
    pad_token_id = 0

    def __init__(self, padding_side="left"):
        self.padding_side = padding_side

    def encode(self, text, return_tensors=None):
        token_ids = list(range(1, len(text.split()) + 1))
        if return_tensors == "pt":
            return torch.tensor([token_ids], dtype=torch.long)
        return token_ids

    def pad(
        self,
        encoded_inputs,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors=None,
        **kwargs,
    ):
        max_input_len = max(len(feature["input_ids"]) for feature in encoded_inputs)
        target_len = max_length if padding == "max_length" and max_length is not None else max_input_len
        if pad_to_multiple_of is not None and target_len % pad_to_multiple_of:
            target_len = ((target_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in encoded_inputs:
            input_ids = feature["input_ids"].tolist()
            attention_mask = list(feature["attention_mask"])
            pad_len = target_len - len(input_ids)
            if self.padding_side == "left":
                input_ids = [self.pad_token_id] * pad_len + input_ids
                attention_mask = [0] * pad_len + attention_mask
            else:
                input_ids = input_ids + [self.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(feature["labels"])

        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.stack(batch["labels"]),
        }


def test_data_collator_with_padding_for_padded_keys():
    tokenizer = DummyTokenizer(padding_side="left")

    max_length = 32
    data_collator = DataCollatorWithPaddingForPaddedKeys(
        tokenizer=tokenizer, padding="max_length", max_length=max_length
    )

    features = [
        {
            "input_ids": tokenizer.encode("Hello, how are you?", return_tensors="pt").squeeze(0),
            "labels": torch.tensor(1),
            "auxiliary": {"type": 1},
        },
        {
            "input_ids": tokenizer.encode("I'm fine, thank you!", return_tensors="pt").squeeze(0),
            "labels": torch.tensor(0),
            "auxiliary": {"type": 2},
        },
        {
            "input_ids": tokenizer.encode("What about you?", return_tensors="pt").squeeze(0),
            "labels": torch.tensor(1),
            "auxiliary": {"type": 3},
        },
    ]
    for feature in features:
        feature["attention_mask"] = [1] * len(feature["input_ids"])

    batch = data_collator(features)

    print("Padded input_ids:")
    print(batch["input_ids"])
    print("Padded attention_mask:")
    print(batch["attention_mask"])
    print("Labels:")
    print(batch["labels"])

    assert (
        batch["input_ids"].shape[1] == max_length
    ), f"Expected max_length {max_length}, got {batch['input_ids'].shape[1]}"
    print(f"All inputs padded to length {max_length} correctly.")
