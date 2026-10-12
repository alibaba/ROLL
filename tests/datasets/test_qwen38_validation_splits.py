"""Validation data must retain one native source exchange and split isolation."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.qwen38 import prepare_sft_validation_data as data


def test_long_records_preserve_single_source_exchange(tmp_path):
    source = [
        {"messages": [{"role": "user", "content": "Write a story."},
                      {"role": "assistant", "content": "A" * 9000}]},
        {"messages": [{"role": "user", "content": "Short answer."},
                      {"role": "assistant", "content": "ok"}]},
        {"messages": [{"role": "user", "content": "B" * 7000},
                      {"role": "assistant", "content": "C" * 3000}]},
    ]
    path = tmp_path / "long.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in source))
    def lengths(row):
        return len(row["instruction"]), len(row["instruction"]) + len(row["output"])
    rows = data.load_native_long_records(path, lengths)
    assert len(rows) == 1
    assert rows[0]["instruction"] == source[0]["messages"][0]["content"]
    assert rows[0]["output"] == source[0]["messages"][1]["content"]
    assert rows[0]["source_indices"] == [0]


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("failure", ["duplicate", "identity", "source", "composed"])
def test_split_integrity_checks_survive_optimized_python(optimized, failure):
    code = '''
from scripts.qwen38.prepare_sft_validation_data import validate_split
row = dict(identity="train", domain="code", source_indices=[1])
heldout = [dict(identity="heldout", domain="code", source_indices=[2])]
rows = [row]
failure = FAILURE
if failure == "duplicate": rows.append(dict(row))
if failure == "identity": row["identity"] = "heldout"
if failure == "source": row["source_indices"] = [2]
if failure == "composed": row["source_indices"] = [1, 3]
try:
    validate_split("train", rows, heldout)
except ValueError:
    print("REJECTED")
else:
    raise RuntimeError("invalid validation split accepted")
'''.replace("FAILURE", repr(failure))
    command = [sys.executable] + (["-O"] if optimized else []) + ["-c", code]
    result = subprocess.run(command, capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[2])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "REJECTED"
