"""Fix suffix-only adapter insertion in the pinned PEFT validation dependency.

The affected PEFT loader replaces every occurrence of the tensor suffix in a
key. For Qwen4 GR modules, inserting '.default' before the final 'weight' also
corrupts module names such as input_mix_weight_down. Preserve those names.
Pass the isolated environment's peft/utils/save_and_load.py file explicitly.
"""
import argparse
import hashlib
from pathlib import Path


SOURCE_SHA256 = "acf74f3e3c6999cd0c4c883cc0c72d100d7a46adceb4bb29e9d36c381b72d1e1"
OLD = '                k = k.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")\n'
NEW = '                k = k.removesuffix(suffix_to_replace) + f"{adapter_name}.{suffix_to_replace}"\n'


def patch_source(source):
    original = source.replace(NEW, OLD, 1) if source.count(NEW) == 1 else source
    if hashlib.sha256(original.encode()).hexdigest() != SOURCE_SHA256 or original.count(OLD) != 1:
        raise RuntimeError("Unsupported PEFT adapter loader source; inspect before patching")
    patched = original.replace(OLD, NEW, 1)
    compile(patched, "save_and_load.py", "exec")
    return patched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("save_and_load_file", type=Path)
    args = parser.parse_args()
    original = args.save_and_load_file.read_text()
    patched = patch_source(original)
    if patched != original:
        args.save_and_load_file.write_text(patched)
    print(f"{args.save_and_load_file}: {hashlib.sha256(patched.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
