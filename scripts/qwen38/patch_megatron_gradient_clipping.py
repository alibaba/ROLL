"""Bound gradient scaling tensor handles on the pinned Megatron source.

The global norm and clipping coefficient are computed once by native Megatron.
Only the in-place scaling calls are partitioned, bounding live TE handles even
when expert LoRA adapters produce tens of thousands of gradient tensors.

Only the known before/after SHA256 values are accepted. Reapplying is a no-op;
unrecognized source is left untouched. Existing Python processes retain their
already-imported implementation and need a fresh process to use this change.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


BEFORE_SHA256 = '4a68e9aae8cb1188f30766af93fabba00a43a8512f8fe3e9e4154d80de4c84ee'
AFTER_SHA256 = 'a7a40ac42a1658fad4f589b0da5d2963cadcb5a3c01dfffc7c4b6d8174cf4556'
RELATIVE_PATH = Path("megatron/core/optimizer/clip_grads.py")
OLD = b'        multi_tensor_applier(\n            multi_tensor_scale_impl, dummy_overflow_buf, [grads, grads], clip_coeff\n        )\n'
NEW = b'        # Bound TE tensor handles for large expert-adapter parameter lists.\n        # Every batch uses the same globally computed clipping coefficient.\n        for offset in range(0, len(grads), 2048):\n            grad_batch = grads[offset:offset + 2048]\n            multi_tensor_applier(\n                multi_tensor_scale_impl, dummy_overflow_buf, [grad_batch, grad_batch], clip_coeff\n            )\n'


def patch_source(source: bytes) -> bytes:
    checksum = hashlib.sha256(source).hexdigest()
    if checksum == AFTER_SHA256:
        return source
    if checksum != BEFORE_SHA256:
        raise RuntimeError(
            f"Megatron gradient clipping source SHA256 {checksum} is not the pinned before/after source; "
            "inspect the dependency before changing it"
        )
    if source.count(OLD) != 1:
        raise RuntimeError("Pinned gradient scaling call must occur exactly once")
    patched = source.replace(OLD, NEW)
    if hashlib.sha256(patched).hexdigest() != AFTER_SHA256:
        raise RuntimeError("Patched Megatron gradient clipping source does not match the locked after SHA256")
    return patched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("megatron_checkout", type=Path)
    args = parser.parse_args()
    target = args.megatron_checkout / RELATIVE_PATH
    original = target.read_bytes()
    patched = patch_source(original)
    compile(patched, str(target), "exec")
    if patched != original:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".gradient-clipping-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(patched)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, target.stat().st_mode)
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(json.dumps({
        "path": str(target.resolve()),
        "before_sha256": hashlib.sha256(original).hexdigest(),
        "after_sha256": hashlib.sha256(patched).hexdigest(),
        "changed": patched != original,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_mtime_ns": target.stat().st_mtime_ns,
    }))


if __name__ == "__main__":
    main()
