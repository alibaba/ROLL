"""Add an overridable decoder factory to the isolated Megatron GPTModel.

This changes no default behavior. Qwen4 needs to transport original token IDs
through activation recomputation, which the stock block interface cannot do.
"""
import argparse
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("megatron_checkout", type=Path)
    args = parser.parse_args()
    file = args.megatron_checkout / "megatron/core/models/gpt/gpt_model.py"
    source = file.read_text()
    before = hashlib.sha256(source.encode()).hexdigest()
    old = "self.decoder = TransformerBlock("
    new = 'self.decoder = getattr(self, "transformer_block_class", TransformerBlock)('
    if source.count(new) == 1:
        print(f"already_applied sha256={before}")
        return
    if source.count(old) != 1:
        raise RuntimeError("Megatron GPTModel decoder factory differs; inspect before patching")
    source = source.replace(old, new)
    file.write_text(source)
    print(f"patched {file} before={before} after={hashlib.sha256(source.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
