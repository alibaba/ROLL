"""Compare sampled mmap rows to independent safetensors slices on the real model."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open


def main():
    module_file = Path(__file__).parents[2] / "mcore_adapter/src/mcore_adapter/models/qwen4_exp/ngram_embedding.py"
    spec = importlib.util.spec_from_file_location("ngram_probe_module", module_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    checkpoint = Path("/data_hdd/Qwen3.8-Flash-Next")
    start = time.monotonic()
    store = module.MMapNGramStore(checkpoint)
    row_ids, expected = [], []
    offset = 0
    generator = torch.Generator().manual_seed(773)
    for entry in store.manifest["tensors"]:
        n = entry["shape"][0]
        local_rows = [0, n - 1, int(torch.randint(n, (), generator=generator))]
        with safe_open(checkpoint / entry["file"], framework="pt", device="cpu") as source:
            tensor = source.get_slice(entry["key"])
            for row in local_rows:
                row_ids.append(offset + row)
                expected.append(tensor[row:row+1])
        offset += n
    expected = torch.cat(expected)
    ids = torch.tensor(row_ids)
    actual = store.lookup(ids)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    gpu_values = store.lookup(ids.cuda())
    torch.testing.assert_close(gpu_values.cpu(), expected, atol=0, rtol=0)
    torch.cuda.synchronize()
    summary = {
        "shards": len(store.manifest["tensors"]), "rows": store.shape[0],
        "head_dim": store.shape[1], "sampled_rows": len(row_ids), "dtype": str(store.dtype),
        "cpu_and_cuda_exact_match": True,
        "sample_sha256": hashlib.sha256(actual.view(torch.uint8).numpy().tobytes()).hexdigest(),
        "elapsed_seconds": time.monotonic()-start,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "manifest_identity_kind": store.manifest["identity_kind"],
        "source_sha256": hashlib.sha256(module_file.read_bytes()).hexdigest(),
        "rss": [x for x in Path("/proc/self/status").read_text().splitlines() if x.startswith(("VmRSS", "VmHWM"))],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
