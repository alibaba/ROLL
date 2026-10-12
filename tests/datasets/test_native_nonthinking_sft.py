"""Real tokenizer checks for direct-answer SFT without reasoning traces."""
import os

import pytest


@pytest.fixture(scope="module")
def tokenizer():
    model = os.environ.get("QWEN38_TOKENIZER_PATH")
    if not model:
        pytest.skip("set QWEN38_TOKENIZER_PATH to the real Flash-Next tokenizer")
    from transformers import AutoTokenizer
    result = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    result.padding_side = "right"
    return result


def test_direct_answers_keep_exact_prompt_prefix_and_shifted_labels(tokenizer):
    import torch
    from roll.datasets.chat_template import get_chat_template
    from roll.datasets.collator import DataCollatorForSFT
    from roll.pipeline.sft.sft_pipeline import get_encode_function

    prompts = ["计算 2 + 2。", "Write a Python function that returns its argument."]
    answers = ["4", "def identity(value):\n    return value"]
    template = get_chat_template("native_nonthinking", tokenizer)
    encode = get_encode_function("native_nonthinking", tokenizer, "instruction", None, "output")
    encoded = encode({"instruction": prompts, "output": answers})
    rows = [{key: values[i] for key, values in encoded.items()} for i in range(len(prompts))]
    collator = DataCollatorForSFT(tokenizer=tokenizer, padding="max_length", max_length=128,
                                padded_keys=["input_ids", "attention_mask"], label_pad_token_id=-100)
    batch = collator(rows)
    for i, (prompt, answer) in enumerate(zip(prompts, answers)):
        user = [{"role": "user", "content": prompt}]
        prefix = template(user, add_generation_prompt=True)
        complete = template(user + [{"role": "assistant", "content": answer}],
                            add_generation_prompt=False).removesuffix("\n")
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        complete_ids = tokenizer.encode(complete, add_special_tokens=False)
        assert complete_ids[:len(prefix_ids)] == prefix_ids
        assert encoded["input_ids"][i] == complete_ids
        supervised = encoded["labels"][i][len(prefix_ids):]
        assert supervised == complete_ids[len(prefix_ids):]
        assert all(label == -100 for label in encoded["labels"][i][:len(prefix_ids)])
        assert answer in tokenizer.decode(supervised)
        assert bool((batch["labels"][i, :len(prefix_ids)-1] == -100).all())
        expected = torch.tensor(supervised, dtype=batch["labels"].dtype)
        torch.testing.assert_close(batch["labels"][i, len(prefix_ids)-1:len(complete_ids)-1], expected)
        assert bool((batch["labels"][i, len(complete_ids)-1:] == -100).all())


@pytest.fixture
def padding_tokenizer():
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]"))
    return transformers.PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                                unk_token="[UNK]", padding_side="right")


@pytest.mark.parametrize("shift", [False, True])
def test_sft_truncates_long_inputs_masks_and_labels_together(padding_tokenizer, shift):
    import copy
    import torch
    from roll.datasets.collator import DataCollatorForSFT

    rows = []
    for length in (40, 96):
        ids = list(range(100, 100 + length))
        rows.append({"input_ids": ids, "attention_mask": [1] * length,
                     "labels": [-100] * 10 + ids[10:]})
    original = copy.deepcopy(rows)
    collator = DataCollatorForSFT(tokenizer=padding_tokenizer, padding="max_length", max_length=64,
                                padded_keys=["input_ids", "attention_mask"],
                                label_pad_token_id=-100, shift_feature=shift)
    batch = collator(rows)
    assert rows == original, "collation must not truncate the shared dataset in place"
    for key in ("input_ids", "attention_mask", "position_ids", "labels"):
        assert batch[key].shape == (2, 64), key
    for i, row in enumerate(rows):
        size = min(len(row["input_ids"]), 64)
        expected_ids = row["input_ids"][:64] + [padding_tokenizer.pad_token_id] * (64 - size)
        expected_labels = row["labels"][:64] + [-100] * (64 - size)
        if shift:
            expected_labels = expected_labels[1:] + [-100]
        torch.testing.assert_close(batch["input_ids"][i], torch.tensor(expected_ids))
        torch.testing.assert_close(batch["labels"][i], torch.tensor(expected_labels))
        assert int(batch["attention_mask"][i].sum()) == size
