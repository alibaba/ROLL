"""Capture reproducible environment evidence without reading credentials."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    import torch

    versions = {}
    for name in (
        "torch", "transformers", "vllm", "megatron-core", "transformer-engine",
        "flash-linear-attention", "ray", "peft", "safetensors", "triton",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    model = Path("/data_hdd/Qwen3.8-Flash-Next")
    checkout = Path("/data_nvme/workspace/roll-qwen38-validation/ROLL")
    megatron = Path("/data_nvme/workspace/Megatron-LM")
    source_files = sorted((checkout / "mcore_adapter/src/mcore_adapter/models/qwen4_exp").glob("*.py"))
    source_files += [
        megatron / "megatron/core/models/gpt/gpt_model.py",
        megatron / "megatron/core/transformer/transformer_layer.py",
        megatron / "megatron/core/transformer/transformer_block.py",
        megatron / "megatron/core/ssm/gated_delta_net.py",
        model / "config.json", model / "model.safetensors.index.json",
    ]
    info = {
        "python": platform.python_version(), "packages": versions,
        "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "files_sha256": {str(p): sha256(p) for p in source_files},
        "gpu": subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,driver_version",
            "--format=csv",
        ], text=True),
        "memory": {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                   for line in Path("/proc/meminfo").read_text().splitlines()
                   if line.startswith(("MemTotal:", "MemAvailable:"))},
    }
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
