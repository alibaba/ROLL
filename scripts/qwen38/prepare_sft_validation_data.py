"""Build fixed SFT engineering-validation splits from ROLL's bundled data.

The 8K split adds unmodified single-exchange records from a pinned LongWriter
source. No independent examples are concatenated. This dataset tests training
stability, not downstream quality gains.
"""
import argparse
import hashlib
import json
from pathlib import Path

def digest(value):
    return hashlib.sha256(value).hexdigest()


def load_native_long_records(path, tokenize):
    """Select original single exchanges with >=2K supervised tokens at 8K."""
    records, seen = [], set()
    for index, line in enumerate(path.read_bytes().split(b"\n")):
        if not line.strip():
            continue
        messages = json.loads(line)["messages"]
        if len(messages) != 2 or [m["role"] for m in messages] != ["user", "assistant"]:
            raise ValueError(f"Long source record {index} must contain one user/assistant exchange")
        instruction, response = (m["content"] for m in messages)
        if not instruction.strip() or not response.strip():
            continue
        identity = digest((instruction + "\0" + response).encode())
        if identity in seen:
            continue
        seen.add(identity)
        row = dict(instruction=instruction, output=response, domain="longwriter",
                   source_indices=[index], identity=identity)
        prompt_length, length = tokenize(row)
        if length >= 8192 and prompt_length <= 6144:
            row.update(prompt_tokens=prompt_length, total_tokens=length)
            records.append(row)
    return sorted(records, key=lambda row: row["identity"])


def validate_split(name, rows, heldout):
    identities = {row["identity"] for row in rows}
    if len(identities) != len(rows):
        raise ValueError(f"{name}: duplicate record identities")
    if any(len(row["source_indices"]) != 1 for row in rows):
        raise ValueError(f"{name}: each exchange must preserve exactly one source record")
    if name == "heldout":
        return
    overlap = identities & {row["identity"] for row in heldout}
    if overlap:
        raise ValueError(f"{name}: heldout identity overlap: {sorted(overlap)}")
    heldout_sources = {(row["domain"], index) for row in heldout for index in row["source_indices"]}
    train_sources = {(row["domain"], index) for row in rows for index in row["source_indices"]}
    overlap = train_sources & heldout_sources
    if overlap:
        raise ValueError(f"{name}: heldout source overlap: {sorted(overlap)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-source", type=Path, required=True)
    parser.add_argument("--long-source-manifest", type=Path, required=True)
    cli = parser.parse_args()
    cli.output.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cli.model, trust_remote_code=True)
    sources = {"zh": "comparison_gpt4_data_zh.json", "en": "general_CrossThink-QA_deal.jsonl",
               "code": "code_KodCode_data.jsonl"}
    manifest = {"model": cli.model, "enable_thinking": False, "source_sha256": {}, "splits": {}}
    provenance = json.loads(cli.long_source_manifest.read_text())
    if digest(cli.long_source.read_bytes()) != provenance["sha256"]:
        raise ValueError("Long source SHA256 differs from the pinned source manifest")
    manifest["long_source"] = provenance
    selected = {}
    seen = set()

    def tokenize(row):
        user = [{"role": "user", "content": row["instruction"]}]
        # These sources contain direct answers without reasoning traces. Use
        # the native non-thinking prefix consistently for preparation/training.
        prompt = tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
        full = tokenizer.apply_chat_template(user + [{"role": "assistant", "content": row["output"]}],
                                             tokenize=False, add_generation_prompt=False,
                                             enable_thinking=False).removesuffix("\n")
        ids = tokenizer.encode(full, add_special_tokens=False)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError("Native chat template prompt must be a token prefix of the SFT exchange")
        return len(prompt_ids), len(ids)

    for domain, name in sources.items():
        raw = (cli.data_root / name).read_bytes()
        manifest["source_sha256"][name] = digest(raw)
        source = (json.loads(raw) if name.endswith(".json")
                  else [json.loads(line) for line in raw.splitlines() if line.strip()])
        candidates = []
        for index, row in enumerate(source):
            instruction = (row["instruction"] + ("\n" + row["input"] if row["input"] else "")
                           if domain == "zh" else row["prompt"])
            response = row["chosen"] if domain == "zh" else row["ground_truth"]
            if domain == "code":
                response = json.loads(response)[0]
            if not instruction.strip() or not response.strip():
                continue
            identity = digest((instruction + "\0" + response).encode())
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append({"instruction": instruction, "output": response,
                               "domain": domain, "source_indices": [index], "identity": identity})
        # Fixed content order, independent of filesystem order and Python RNG.
        candidates.sort(key=lambda row: row["identity"])
        selected[domain] = []
        for row in candidates:
            prompt_length, length = tokenize(row)
            if prompt_length >= 1024 or length <= prompt_length:
                continue
            row.update(prompt_tokens=prompt_length, total_tokens=length)
            selected[domain].append(row)
        if len(selected[domain]) < 230:
            raise ValueError(f"Not enough usable {domain} records")

    heldout = selected["zh"][:86] + selected["en"][:85] + selected["code"][:85]
    pools = {"zh": selected["zh"][86:], "en": selected["en"][85:], "code": selected["code"][85:]}
    short = pools["zh"][:128] + pools["en"][:128] + pools["code"][:144]
    short.sort(key=lambda row: row["identity"])
    long_candidates = load_native_long_records(cli.long_source, tokenize)
    if len(long_candidates) < 96:
        raise ValueError("Need 96 unique native long records: 80 train and 16 heldout")
    heldout = heldout[:240] + long_candidates[:16]
    long = long_candidates[16:96]

    for name, rows, window in (("train-2k", short, 2048), ("train-8k", short[:320] + long, 8192),
                               ("heldout", heldout, 8192)):
        validate_split(name, rows, heldout)
        contents = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        path = cli.output / f"{name}.jsonl"
        path.write_text(contents)
        manifest["splits"][name] = {
            "file": path.name, "count": len(rows), "sha256": digest(contents.encode()), "window": window,
            "longer_than_window": sum(row["total_tokens"] >= window for row in rows),
            "minimum_supervised_tokens": min(min(window, row["total_tokens"]) - row["prompt_tokens"] for row in rows),
            "source_records": [{"domain": row["domain"], "indices": row["source_indices"]} for row in rows],
        }
    (cli.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: {k: v for k, v in entry.items() if k != "source_records"}
                      for name, entry in manifest["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
