"""Pin Chinese, English, and code forward inputs from ROLL's bundled data.

These token streams are forward-parity fixtures, not cross-sample packed
training examples. Source files and resulting tokens have content hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=8192)
    cli = parser.parse_args()
    cli.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cli.model, trust_remote_code=True)
    sources = {
        "zh": "comparison_gpt4_data_zh.json",
        "en": "general_CrossThink-QA_deal.jsonl",
        "code": "code_KodCode_data.jsonl",
    }
    manifest = {"model": cli.model, "length": cli.length, "fixtures": {}}
    for domain, filename in sources.items():
        path = cli.data_root / filename
        raw = path.read_bytes()
        rows = (json.loads(raw) if path.suffix == ".json"
                else [json.loads(line) for line in raw.decode().splitlines() if line.strip()])
        tokens, used = [], []
        for index, row in enumerate(rows):
            if domain == "zh":
                content = "\n".join(str(row.get(key, "")) for key in ("instruction", "input", "chosen"))
            else:
                content = str(row["prompt"])
                if domain == "code":
                    answers = json.loads(row["ground_truth"])
                    content += "\n" + str(answers[0])
            encoded = tokenizer.encode(content, add_special_tokens=False)
            if not encoded:
                continue
            tokens.extend(encoded + [tokenizer.eos_token_id])
            used.append(index)
            if len(tokens) >= cli.length:
                break
        if len(tokens) < cli.length:
            raise ValueError(f"{filename} has only {len(tokens)} tokens")
        ids = torch.tensor(tokens[:cli.length], dtype=torch.long).unsqueeze(0)
        destination = cli.output / f"{domain}-{cli.length}.pt"
        torch.save(ids, destination)
        manifest["fixtures"][domain] = {
            "source": filename, "source_sha256": hashlib.sha256(raw).hexdigest(),
            "record_indices": used, "tensor_file": destination.name,
            "tokens_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
            "first_tokens": tokenizer.decode(ids[0, :128]),
        }
    (cli.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({domain: value["tokens_sha256"] for domain, value in manifest["fixtures"].items()}))


if __name__ == "__main__":
    main()
