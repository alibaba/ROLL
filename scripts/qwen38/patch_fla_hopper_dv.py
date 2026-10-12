"""Pad the pinned Hopper local-dv tile to avoid narrow-value backward corruption.

On the validation H800 / Triton 3.7.1 source, BV=32 misreads gradient rows at
64-token chunk tails. BV>=64 passes an independent FP64 recurrence, including
packed sequences and initial-state gradients. Only chunk_bwd_dv_local on Hopper
changes; Flash-Next's V=128 tile is unchanged. This is a dependency workaround,
not a full-model parity fix. Apply only to an isolated FLA checkout; existing
validation snapshots and imported modules must remain unchanged.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile


BEFORE_SHA256 = "73581d9de0d41fb0df7f1b5050d81736deba7cedfa3677ac69de4f4363d2a0ef"
AFTER_SHA256 = "7d4920e30c236c9cde462cefd614b2ee2185cf476f99f57e8b7f643d781f436b"
RELATIVE_PATH = Path("fla/ops/common/chunk_o.py")
OLD = b"BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)"
NEW = b"BV = min(max(triton.next_power_of_2(V), 64 if IS_NVIDIA_HOPPER else 16), CONST_TILING)"


def patch_source(source: bytes) -> bytes:
    checksum = hashlib.sha256(source).hexdigest()
    if checksum == AFTER_SHA256:
        return source
    if checksum != BEFORE_SHA256:
        raise RuntimeError(f"FLA source SHA256 {checksum} is not the pinned before/after source")
    node, = [n for n in ast.parse(source).body
             if isinstance(n, ast.FunctionDef) and n.name == "chunk_bwd_dv_local"]
    lines = source.splitlines(keepends=True)
    body = b"".join(lines[node.lineno - 1:node.end_lineno])
    if body.count(OLD) != 1:
        raise RuntimeError("Pinned local-dv tile expression must occur exactly once")
    patched = b"".join(lines[:node.lineno - 1]) + body.replace(OLD, NEW) + b"".join(lines[node.end_lineno:])
    if hashlib.sha256(patched).hexdigest() != AFTER_SHA256:
        raise RuntimeError("Patched FLA source does not match the locked after SHA256")
    return patched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fla_checkout", type=Path)
    args = parser.parse_args()
    target = args.fla_checkout / RELATIVE_PATH
    if target.is_symlink():
        raise RuntimeError("Use an isolated regular source file, not a dependency symlink")
    original = target.read_bytes()
    patched = patch_source(original)
    compile(patched, str(target), "exec")
    if patched != original:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".fla-dv-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(patched)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, target.stat().st_mode)
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(json.dumps(dict(path=str(target.resolve()), changed=patched != original,
        before_sha256=hashlib.sha256(original).hexdigest(), after_sha256=hashlib.sha256(patched).hexdigest())))


if __name__ == "__main__":
    main()
