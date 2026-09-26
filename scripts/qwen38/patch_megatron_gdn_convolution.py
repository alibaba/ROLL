"""Expose a GDN convolution hook while preserving the default Megatron behavior.

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


# The first pair is the Megatron source after the pinned gate-precision patch.
# The second pair is the source shipped by the current .181 validation image
# (the decay patch is present, while gate precision is not).  Both variants
# contain the same direct convolution call and receive the same hook change;
# their remaining numerical patches account for the different final hashes.
BEFORE_SHA256 = "7fa37fcbf6945353b80110b3444e92f52de3e6398c987f4c3a4243be2d4eb770"
AFTER_SHA256 = "e24e34abe335df8bcc30f81829438dd71ff8f7b4bf36c16c542f6acd46bbdc42"
CURRENT_BEFORE_SHA256 = "2d573edc20bde3c83abfbcc6dacf4e813038594c90a0151a6c0bd2b4dc5e0b3a"
CURRENT_AFTER_SHA256 = "3fc8ae42e7c4bd99daec8b678265ac329ee7888f2f3fa5e26f53f1943717ef78"
RELATIVE_PATH = Path("megatron/core/ssm/gated_delta_net.py")
OLD = b'            qkv, _ = causal_conv1d(\n                x=qkv,  # FLA conv1d accepts [b, s, d] format input\n                weight=conv1d_weight.squeeze(1),  # d, 1, w -> d, w\n                bias=conv1d_bias,\n                activation=self.activation,\n                initial_state=None,\n                output_final_state=False,\n            )'
NEW = b'            qkv = self._apply_causal_conv1d(qkv, conv1d_weight, conv1d_bias)'


def patch_source(source: bytes) -> bytes:
    checksum = hashlib.sha256(source).hexdigest()
    if checksum in {AFTER_SHA256, CURRENT_AFTER_SHA256}:
        return source
    expected_after = {
        BEFORE_SHA256: AFTER_SHA256,
        CURRENT_BEFORE_SHA256: CURRENT_AFTER_SHA256,
    }.get(checksum)
    if expected_after is None:
        raise RuntimeError(
            f"GDN source SHA256 {checksum} is not a pinned before/after source; "
            "inspect the dependency before changing it"
        )
    if source.count(OLD) != 1:
        raise RuntimeError("Pinned GDN convolution call must occur exactly once")
    anchor = b'    @jit_fuser\n    def _apply_gated_norm(self, x, gate):'
    replacement = b'    def _apply_causal_conv1d(self, x, weight, bias):\n        """Convolve with the default fused activation; models may override it."""\n        output, _ = causal_conv1d(\n            x=x,\n            weight=weight.squeeze(1),\n            bias=bias,\n            activation=self.activation,\n            initial_state=None,\n            output_final_state=False,\n        )\n        return output\n\n    @jit_fuser\n    def _apply_gated_norm(self, x, gate):'
    if source.count(anchor) != 1:
        raise RuntimeError("Pinned GDN norm anchor must occur exactly once")
    patched = source.replace(OLD, NEW).replace(anchor, replacement)
    if hashlib.sha256(patched).hexdigest() != expected_after:
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
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".gdn-conv-", delete=False) as stream:
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
