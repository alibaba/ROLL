"""Create a verified local-disk copy of an immutable HF checkpoint.

Copy source files sequentially to avoid eight ranks faulting different HDD
pages concurrently. Every copied file is hashed while reading the source and
verified by reading the destination back. No source file is modified.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    cli = parser.parse_args()
    source, destination = cli.source.resolve(), cli.destination.resolve()
    if source == destination or source in destination.parents:
        raise ValueError("destination must be separate from the source checkpoint")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    files = sorted(path for path in source.iterdir() if path.is_file() and not path.name.startswith("."))
    names = {path.name for path in files}
    if not set(index["weight_map"].values()) <= names:
        raise ValueError("checkpoint index references missing or nested source shards")
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / ".staging-manifest.json"
    with (destination / ".staging.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = (json.loads(manifest_path.read_text()) if manifest_path.exists()
                    else {"source": str(source), "files": {}})
        if manifest["source"] != str(source):
            raise ValueError("existing staging manifest refers to another source")
        start = time.monotonic()
        for path in files:
            target = destination / path.name
            stat = path.stat()
            identity = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            record = manifest["files"].get(path.name)
            if record is not None:
                if any(record[key] != value for key, value in identity.items()):
                    raise ValueError(f"source changed since prior copy: {path.name}")
                if target.exists() and checksum(target) == record["sha256"]:
                    continue
            elif target.exists():
                raise FileExistsError(f"refusing to overwrite a file not owned by this copy: {target}")
            partial = destination / ("." + path.name + ".partial")
            digest = hashlib.sha256()
            with path.open("rb") as reader, partial.open("wb") as writer:
                while block := reader.read(8 * 1024 * 1024):
                    digest.update(block)
                    writer.write(block)
                writer.flush()
                os.fsync(writer.fileno())
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise ValueError(f"source changed during copy: {path.name}")
            expected = digest.hexdigest()
            if checksum(partial) != expected:
                raise ValueError(f"destination hash mismatch: {path.name}")
            partial.chmod(0o444)
            os.replace(partial, target)
            manifest["files"][path.name] = {**identity, "sha256": expected}
            manifest_tmp = manifest_path.with_suffix(".tmp")
            manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n")
            os.replace(manifest_tmp, manifest_path)
            print(json.dumps({"file": path.name, "bytes": stat.st_size, "sha256": expected,
                              "elapsed": time.monotonic() - start}), flush=True)
        print(json.dumps({"complete": True, "files": len(manifest["files"]),
                          "bytes": sum(item["size"] for item in manifest["files"].values()),
                          "elapsed": time.monotonic() - start}), flush=True)


if __name__ == "__main__":
    main()
