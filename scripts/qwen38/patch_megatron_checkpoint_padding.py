"""Initialize distributed optimizer checkpoint padding deterministically.

Megatron's dp_reshardable format persists gaps between aligned parameters.
These gaps have no optimizer state, but torch.empty writes allocator contents
into the checkpoint and makes otherwise identical cold continuations differ.
Apply after patch_megatron_hybrid_checkpoint.py to the pinned dependency.
"""
import argparse
import hashlib
from pathlib import Path


SOURCE_SHA256 = "3f9c12ecc8b3f756835d09b093531b807b1518bfa766f744e09f33cfdab99bb5"

OLD = '''                            pad_tensors = {
                                k: torch.empty(
                                    next_param_start - cur_param_end, dtype=v.dtype, device=v.device
                                )
'''

NEW = '''                            # These persisted gaps are not parameter/Adam state.
                            # Never serialize uninitialized allocator contents.
                            pad_tensors = {
                                k: torch.zeros(
                                    next_param_start - cur_param_end, dtype=v.dtype, device=v.device
                                )
'''


def patch_source(source):
    original = source.replace(NEW, OLD, 1) if source.count(NEW) == 1 else source
    if hashlib.sha256(original.encode()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Unsupported Megatron checkpoint source; inspect before patching")
    if original.count(OLD) != 1:
        raise RuntimeError("Megatron checkpoint padding patch target is ambiguous")
    patched = original.replace(OLD, NEW, 1)
    compile(patched, "distrib_optimizer.py", "exec")
    return patched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("megatron_checkout", type=Path)
    args = parser.parse_args()
    target = args.megatron_checkout / "megatron/core/optimizer/distrib_optimizer.py"
    source = target.read_text()
    patched = patch_source(source)
    if patched != source:
        target.write_text(patched)
    print(f"{target}: {hashlib.sha256(patched.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
