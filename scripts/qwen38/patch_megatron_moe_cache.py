"""Detach completed MoE routing probability caches on the pinned task Megatron source.

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


BEFORE_SHA256 = "b99bf2950e7ec774a228d96fe3123e81c6dc26e2379e803d6f5e9039492ece27"
AFTER_SHA256 = "cc42e000f913c87c616a056fc7ecc6eec5d61da347f0f1f8a283526ce614846f"
RELATIVE_PATH = Path("megatron/core/transformer/moe/token_dispatcher.py")
OLD = b'            output += shared_expert_output\n        return output\n'
NEW = b'            output += shared_expert_output\n        # Keep dtype metadata without retaining the completed router graph and\n        # its checkpoint input/gradient across microbatches. Backward uses its\n        # saved tensors, not this dispatcher cache.\n        self.probs = self.probs.detach()\n        return output\n'


def patch_source(source: bytes) -> bytes:
    checksum = hashlib.sha256(source).hexdigest()
    if checksum == AFTER_SHA256:
        return source
    if checksum != BEFORE_SHA256:
        raise RuntimeError(
            f"MoE dispatcher source SHA256 {checksum} is not the pinned before/after source; "
            "inspect the dependency before changing it"
        )
    if source.count(OLD) != 1:
        raise RuntimeError("Pinned MoE dispatcher cache return site must occur exactly once")
    patched = source.replace(OLD, NEW)
    if hashlib.sha256(patched).hexdigest() != AFTER_SHA256:
        raise RuntimeError("Patched MoE dispatcher source does not match the locked after SHA256")
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
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".moe-cache-", delete=False) as stream:
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
