"""Check fixed validation data through ROLL's actual SFT encoder and collator."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from roll.datasets.collator import DataCollatorForSFT
from roll.pipeline.sft.sft_pipeline import get_encode_function


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    cli = parser.parse_args()
    manifest = json.loads((cli.data / "manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(cli.model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    encode = get_encode_function("native_nonthinking", tokenizer, "instruction", None, "output")
    report = {}
    for name, split in manifest["splits"].items():
        raw = (cli.data / split["file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == split["sha256"]
        rows = [json.loads(line) for line in raw.splitlines()]
        assert len(rows) == split["count"]
        collator = DataCollatorForSFT(tokenizer=tokenizer, padding="max_length", max_length=split["window"],
                                    padded_keys=["input_ids", "attention_mask"], label_pad_token_id=-100)
        counts = []
        domains = Counter()
        for row in rows:
            encoded = encode({"instruction": [row["instruction"]], "output": [row["output"]]})
            sample = {key: value[0] for key, value in encoded.items()}
            assert len(sample["input_ids"]) == row["total_tokens"]
            assert all(label == -100 for label in sample["labels"][:row["prompt_tokens"]])
            assert sample["labels"][row["prompt_tokens"]:] == sample["input_ids"][row["prompt_tokens"]:]
            batch = collator([sample])
            assert batch["input_ids"].shape == (1, split["window"])
            supervised = batch["labels"][0] != -100
            # ROLL truncates the full exchange then shifts labels left by one.
            expected_count = min(row["total_tokens"], split["window"]) - row["prompt_tokens"]
            assert int(supervised.sum()) == expected_count > 0
            assert bool((batch["labels"][0, :-1][supervised[:-1]] ==
                         batch["input_ids"][0, 1:][supervised[:-1]]).all())
            assert int(batch["labels"][0, -1]) == -100
            assert not bool((supervised & ~batch["attention_mask"][0].bool()).any())
            counts.append(expected_count)
            domains[row["domain"]] += 1
        report[name] = dict(count=len(rows), domains=dict(domains), supervised_tokens=sum(counts),
                            min_supervised_tokens=min(counts), max_supervised_tokens=max(counts),
                            source_sha256=split["sha256"], window=split["window"])
    result = dict(template="native_nonthinking", model=cli.model, splits=report)
    (cli.data / "roll-encoding-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
