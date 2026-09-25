"""Reproduce the recorded fixed GPQA lifecycle split from ROLL's bundled data."""
import argparse
import hashlib
import json
from pathlib import Path

OLD = 'Please reason step by step and answer the following question. Put the letter of the correct option inside \\boxed{}.'
NEW = 'Answer the following multiple-choice question. Output only the letter of the correct option inside \\boxed{}, for example \\boxed{A}. Do not include any explanation.'



def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "data/gpqa_diamond_boxed.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("rl_validation_data_manifest.json").read_text())
    raw = args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["source_sha256"]:
        raise ValueError("GPQA source differs from the recorded validation source")
    rows = [json.loads(line) for line in raw.decode().split("\n") if line.strip()]
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("GPQA source IDs must be unique")
    if set(manifest["train_ids"]) & set(manifest["heldout_ids"]):
        raise ValueError("Recorded train and heldout IDs overlap")
    payloads = {}
    for split in ("train", "heldout"):
        selected = [by_id[key] for key in manifest[split + "_ids"]]
        original = ("\n".join(json.dumps(row, ensure_ascii=False) for row in selected) + "\n").encode()
        if hashlib.sha256(original).hexdigest() != manifest["original_" + split + "_sha256"]:
            raise ValueError(f"Recorded {split} split differs")
        transformed = []
        for original_row in selected:
            row = dict(original_row)
            if not row["prompt"].startswith(OLD + "\nQuestion:"):
                raise ValueError("Unexpected source prompt instruction")
            messages = json.loads(row["messages"])
            if (len(messages) != 1 or messages[0]["role"] != "user"
                    or messages[0]["content"] != row["prompt"]):
                raise ValueError("Unexpected source message format")
            row["prompt"] = NEW + row["prompt"][len(OLD):]
            messages[0]["content"] = row["prompt"]
            row["messages"] = json.dumps(messages, ensure_ascii=False)
            transformed.append(row)
        payload = ("\n".join(json.dumps(row, ensure_ascii=False) for row in transformed) + "\n").encode()
        payloads[split] = payload
    if hashlib.sha256(payloads["train"]).hexdigest() != manifest["answer_only_train_sha256"]:
        raise ValueError("Training output differs from the data used by validation")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    receipt = dict(manifest, transformed={})
    for split, payload in payloads.items():
        (args.output_dir / f"{split}.jsonl").write_bytes(payload)
        receipt["transformed"][split] = dict(sha256=hashlib.sha256(payload).hexdigest(),
                                             records=len(manifest[split + "_ids"]))
    (args.output_dir / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["transformed"], indent=2))


if __name__ == "__main__":
    main()
