"""Frozen, CPU-resident Qwen4 n-gram assets and checkpoint-exact hashing.

Safetensors shards are mapped read-only. OS page cache is shared across ranks;
only requested rows are copied. The storage object is neither a module buffer
nor a parameter and therefore never follows ``Module.to`` onto an accelerator.
"""
from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn


class TensorNGramStore:
    """Small in-memory store, primarily for numerical reference fixtures."""
    def __init__(self, table):
        if table.ndim != 2 or table.device.type != "cpu":
            raise ValueError("frozen n-gram table must be a two-dimensional CPU tensor")
        self.table = table.detach()
        self.shape = tuple(table.shape)
        self.dtype = table.dtype

    def lookup(self, ids):
        flat = ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        return self.table.index_select(0, flat).reshape(*ids.shape, self.shape[1]).to(ids.device)


class MMapNGramStore:
    """A numerically ordered list of safetensors table shards.

    The manifest authenticates metadata/geometry, not every weight payload byte.
    Callers must retain immutable checkpoint assets. ``expected_manifest`` permits
    relocation while rejecting a different index, header, geometry or file size.
    """
    _DTYPES = {"BF16": (np.uint16, torch.bfloat16), "F16": (np.float16, torch.float16),
               "F32": (np.float32, torch.float32), "I64": (np.int64, torch.int64)}

    def __init__(self, checkpoint, layer_idx=1, *, expected_manifest=None, staging_rows=4096):
        if staging_rows < 1:
            raise ValueError("staging_rows must be positive")
        self.staging_rows = staging_rows
        self.checkpoint = Path(checkpoint)
        index_path = self.checkpoint / "model.safetensors.index.json"
        index_bytes = index_path.read_bytes()
        index = json.loads(index_bytes)["weight_map"]
        prefix = f"model.language_model.layers.{layer_idx}.ple.ple_embedding."
        pattern = re.compile(re.escape(prefix) + r"ngram_embedding\.shard_(\d+)\.weight$")
        shards = sorted((int(m.group(1)), key) for key in index if (m := pattern.fullmatch(key)))
        if not shards or [i for i, _ in shards] != list(range(len(shards))):
            raise ValueError("n-gram checkpoint shards must be contiguous and start at zero")
        self._headers = {}
        self._maps = []
        self._ends = []
        tensors = []
        rows, width, dtype_name = 0, None, None
        for _, key in shards:
            entry, offset, file = self._entry(index[key], key)
            if len(entry["shape"]) != 2 or min(entry["shape"]) <= 0:
                raise ValueError(f"invalid n-gram table shape: {key}")
            n, d = entry["shape"]
            width = d if width is None else width
            dtype_name = entry["dtype"] if dtype_name is None else dtype_name
            if d != width or entry["dtype"] != dtype_name or dtype_name not in ("BF16", "F16", "F32"):
                raise ValueError("n-gram shard widths and floating dtypes must match")
            storage_dtype, self.dtype = self._DTYPES[dtype_name]
            self._maps.append(np.memmap(file, mode="r", dtype=storage_dtype, offset=offset, shape=(n, d)))
            rows += n
            self._ends.append(rows)
            tensors.append({"key": key, "file": index[key], "shape": [n, d], "dtype": dtype_name,
                            "data_offsets": entry["data_offsets"]})
        self.shape = (rows, width)
        self.constants = {}
        for name in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"):
            key = prefix + name
            if key not in index:
                raise ValueError(f"missing n-gram hash constant: {key}")
            entry, offset, file = self._entry(index[key], key)
            if entry["dtype"] != "I64" or len(entry["shape"]) != 1:
                raise ValueError(f"n-gram hash constant must be int64 vector: {key}")
            values = np.memmap(file, mode="r", dtype=np.int64, offset=offset, shape=tuple(entry["shape"]))
            self.constants[name] = torch.from_numpy(np.array(values))
        self.manifest = {
            "format": 1, "identity_kind": "index_and_header_sha256", "layer_idx": layer_idx,
            "index_sha256": hashlib.sha256(index_bytes).hexdigest(), "tensors": tensors,
            "files": {name: {"header_sha256": data[2], "size": data[3]}
                      for name, data in sorted(self._headers.items())},
            "hash_constants": {k: v.tolist() for k, v in self.constants.items()},
        }
        if expected_manifest is not None and self.manifest != expected_manifest:
            raise ValueError("n-gram external asset manifest mismatch")

    def _entry(self, filename, key):
        file = (self.checkpoint / filename).resolve()
        if not file.is_relative_to(self.checkpoint.resolve()):
            raise ValueError("checkpoint index path escapes checkpoint directory")
        if filename not in self._headers:
            with file.open("rb") as source:
                raw_len = source.read(8)
                if len(raw_len) != 8:
                    raise ValueError(f"truncated safetensors header: {filename}")
                length = struct.unpack("<Q", raw_len)[0]
                if not 2 <= length <= 64 * 1024 * 1024:
                    raise ValueError(f"invalid safetensors header length: {filename}")
                raw = source.read(length)
                if len(raw) != length:
                    raise ValueError(f"truncated safetensors header: {filename}")
            self._headers[filename] = (json.loads(raw), length + 8, hashlib.sha256(raw).hexdigest(), file.stat().st_size)
        header, begin, _, size = self._headers[filename]
        entry = header[key]
        start, end = entry["data_offsets"]
        itemsize = np.dtype(self._DTYPES[entry["dtype"]][0]).itemsize
        if start < 0 or end - start != int(np.prod(entry["shape"])) * itemsize or begin + end > size:
            raise ValueError(f"invalid safetensors tensor extent: {key}")
        return entry, begin + start, file

    def lookup(self, ids):
        flat = ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        if flat.numel() and (flat.min() < 0 or flat.max() >= self.shape[0]):
            raise ValueError("n-gram row ID is outside mapped table")
        result = torch.empty((flat.numel(), self.shape[1]), dtype=self.dtype)
        ends = np.asarray(self._ends)
        starts = np.concatenate(([0], ends[:-1]))
        for begin in range(0, flat.numel(), self.staging_rows):
            requested = flat[begin:begin+self.staging_rows].numpy()
            unique, inverse = np.unique(requested, return_inverse=True)
            shard_ids = np.searchsorted(ends, unique, side="right")
            gathered = torch.empty((len(unique), self.shape[1]), dtype=self.dtype)
            for shard in np.unique(shard_ids):
                positions = np.flatnonzero(shard_ids == shard)
                values = np.array(self._maps[shard][unique[positions] - starts[shard]], copy=True)
                tensor = torch.from_numpy(values)
                if self.dtype == torch.bfloat16:
                    tensor = tensor.view(torch.bfloat16)
                gathered[torch.from_numpy(positions)] = tensor
            result[begin:begin+len(requested)] = gathered[torch.from_numpy(inverse)]
        return result.reshape(*ids.shape, self.shape[1]).to(ids.device)


class FrozenNGramEmbedding(nn.Module):
    """Hash original IDs and fetch frozen rows without registering the table."""
    def __init__(self, embedding_dim, ngram_size, heads_per_ngram, vocab_size,
                 eos_token_id, table=None, device=None, dtype=None):
        super().__init__()
        if ngram_size < 2 or heads_per_ngram < 1:
            raise ValueError("ngram_size must be >=2 and heads_per_ngram positive")
        self.embedding_dim, self.ngram_size = embedding_dim, ngram_size
        self.heads_per_ngram, self.eos_token_id = heads_per_ngram, eos_token_id
        self.ngram_heads = (ngram_size - 1) * heads_per_ngram
        if embedding_dim % self.ngram_heads:
            raise ValueError("ple_embed_dim must be divisible by n-gram heads")
        self.head_dim = embedding_dim // self.ngram_heads
        # Default constants support explicit small table fixtures only. Production
        # attaches checkpoint-provided constants before the first forward.
        self.register_buffer("layer_multipliers", torch.ones(ngram_size, dtype=torch.long, device=device))
        self.register_buffer("ngram_heads_vocab_sizes", torch.full((self.ngram_heads,), vocab_size, dtype=torch.long, device=device))
        self.register_buffer("ngram_heads_offsets", torch.zeros(self.ngram_heads, dtype=torch.long, device=device))
        self.store = TensorNGramStore(table) if table is not None else None

    def attach_checkpoint(self, checkpoint, layer_idx=1, *, expected_manifest=None, staging_rows=4096):
        store = MMapNGramStore(checkpoint, layer_idx, expected_manifest=expected_manifest, staging_rows=staging_rows)
        if store.shape[1] != self.head_dim:
            raise ValueError("checkpoint n-gram head dimension differs from model config")
        for name, value in store.constants.items():
            target = getattr(self, name)
            if value.shape != target.shape:
                raise ValueError(f"checkpoint n-gram hash shape mismatch: {name}")
        sizes, offsets = store.constants["ngram_heads_vocab_sizes"], store.constants["ngram_heads_offsets"]
        if (sizes <= 0).any() or (offsets < 0).any() or int((offsets + sizes).max()) > store.shape[0]:
            raise ValueError("checkpoint hash slots exceed n-gram table")
        for name, value in store.constants.items():
            getattr(self, name).copy_(value)
        self.store = store
        return store.manifest

    def forward(self, tokens):
        if self.store is None:
            raise RuntimeError("attach the frozen n-gram checkpoint before model forward")
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("input_ids must have shape [batch, nonempty_sequence]")
        with torch.no_grad():
            ids = self.compute_ngram_ids(tokens.long())
            if ids.numel() and (ids.min() < 0 or ids.max() >= self.store.shape[0]):
                raise ValueError("n-gram hash ID exceeds frozen table rows")
            return self.store.lookup(ids).flatten(-2)
    # -------------------------------------------------------------- id hashing
    @staticmethod
    def _segment_positions(tokens: torch.Tensor, eos_token_id: int):
        """Position within the current EOS-delimited segment.

        n-grams must not cross a document boundary, so shifts are only valid while
        they stay inside the current segment.
        """
        _, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device, dtype=torch.int64)
        eos_positions = torch.where(tokens == eos_token_id, positions, -1)
        prev_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        prev_eos = torch.cat(
            [eos_positions.new_full((tokens.shape[0], 1), -1), prev_eos_inclusive[:, :-1]],
            dim=1,
        )
        return positions, positions.unsqueeze(0) - prev_eos - 1

    @staticmethod
    def _shift(tokens, positions, position_in_segment, shift, eos_token_id):
        if shift == 0:
            return tokens
        source = positions - shift
        gather_idx = source.clamp_min(0).unsqueeze(0).expand(tokens.shape[0], -1)
        shifted = tokens.gather(1, gather_idx)
        valid = (source.unsqueeze(0) >= 0) & (position_in_segment >= shift)
        return torch.where(valid, shifted, tokens.new_full((), eos_token_id))

    def compute_ngram_ids(self, tokens: torch.Tensor) -> torch.Tensor:
        """Map ``[batch, seq]`` token ids to ``[batch, seq, ngram_heads]`` table ids.

        Integer arithmetic only -- no gradient exists here regardless of context.
        """
        positions, pos_in_seg = self._segment_positions(tokens, self.eos_token_id)
        shifted = [
            self._shift(tokens, positions, pos_in_seg, k, self.eos_token_id)
            for k in range(self.ngram_size)
        ]

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for i in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, shifted[i] * self.layer_multipliers[i])
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes) + offsets)
        return torch.cat(blocks, dim=-1)
