"""Keep GDN update gates in FP32 before the recurrence on the pinned Megatron source.

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


BEFORE_SHA256 = "205503d4f5832bbdbbb34ff295284d809bea7b91c23a75bd46cdcb1fa8abeec3"
AFTER_SHA256 = "7fa37fcbf6945353b80110b3444e92f52de3e6398c987f4c3a4243be2d4eb770"
RELATIVE_PATH = Path("megatron/core/ssm/gated_delta_net.py")
OLD = b"beta = beta.sigmoid()"
NEW = b"beta = beta.float().sigmoid()"


def patch_source(source: bytes) -> bytes:
    checksum = hashlib.sha256(source).hexdigest()
    if checksum == AFTER_SHA256:
        return source
    if checksum != BEFORE_SHA256:
        raise RuntimeError(
            f"GDN source SHA256 {checksum} is not the pinned before/after source; "
            "inspect the dependency before changing it"
        )
    if source.count(OLD) != 1:
        raise RuntimeError("Pinned GDN update gate expression must occur exactly once")
    patched = source.replace(OLD, NEW)
    if hashlib.sha256(patched).hexdigest() != AFTER_SHA256:
        raise RuntimeError("Patched GDN source does not match the locked after SHA256")
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
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".gdn-beta-", delete=False) as stream:
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
