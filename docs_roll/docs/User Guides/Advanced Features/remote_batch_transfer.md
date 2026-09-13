# ROLL RemoteBatch and Transfer Backend

The ROLL framework supports **RemoteBatch**, a lazy data transfer mechanism that decouples data storage from data consumption. By integrating with a pluggable transfer backend (e.g., **TransferQueue**), large data batches can be transferred across Ray workers without serializing the entire payload through Ray's object store. This document provides a detailed guide on how to use this feature.

## Introduction

In RL training pipelines (especially VLM and Agentic scenarios), `DataProto` batches may contain large tensors (e.g., images, multi-modal embeddings) and non-tensor data (e.g., conversation histories). Transferring these between the RolloutScheduler and training workers via Ray's default serialization has two major problems:

1. **High memory overhead**: The full data is serialized and deserialized through the Ray object store, doubling peak memory usage.
2. **High transfer latency**: Large data batches (e.g., image data in VLM scenarios) must be fully transferred between workers, causing significant data movement overhead.

**RemoteBatch** addresses these issues by storing data in an external key-value store and only passing lightweight metadata (keys/references) through Ray. The actual data is **lazily materialized** on the consumer side only when needed, and only the requested fields are fetched.

### Key Concepts

- **RemoteBatch**: An abstract base class representing a batch of data stored remotely. It supports the same slicing, indexing, selection, concatenation, and repeat operations as `TensorDict`, but defers actual data access until `materialize()` is called.
- **RowRemoteBatch**: A concrete `RemoteBatch` where data is stored with **row IDs** as keys. Each row (sample) has a unique ID, and the transfer backend stores/retrieves data at row granularity. This is used by the **TransferQueue** backend.
- **ColumnRemoteBatch**: A concrete `RemoteBatch` where data is stored with **column IDs** as keys (one key per field/column). This is used by the **RayMemoryStore** backend.
- **BatchProxy**: A proxy object that wraps both a local `TensorDict` (or `dict`) and a `RemoteBatch`, supporting transparent fallback lookup. When a key is accessed, it first checks the local batch and then falls back to the remote batch.
- **Transfer Backend**: A pluggable storage backend responsible for `put`, `get`, and `delete` operations. Currently supported backends:
  - `None` (Dummy): No remote storage; data stays local (default).
  - `TransferQueue`: Uses the [TransferQueue](https://github.com/kvcache-ai/TransferQueue) library for high-performance distributed key-value transfer.

### How It Works

1. **Upload (`to_remote`)**: The `DataProto.to_remote()` class method converts a local `DataProto` into a remote-backed `DataProto`. It uploads all tensor and non-tensor fields to the transfer backend and returns a new `DataProto` with a `RemoteBatch` reference (no local data).
2. **Transfer**: The lightweight `DataProto` (containing only `RemoteBatch` metadata) is transferred between workers via Ray. Since the metadata is small, serialization is fast.
3. **Materialize (lazy)**: On the consumer side, when specific fields are needed, `RemoteBatch.materialize(fields)` is called to fetch only the requested columns from the backend. The fetched data is cached locally for subsequent accesses.
4. **Drop**: After the batch is consumed, `RemoteBatch.drop()` can be called to delete the data from the backend store.

## Configuration

The transfer backend is configured under the `transfer_backend` field in the top-level ROLL configuration:

```yaml
transfer_backend:
  backend_name: TransferQueue
  backend_config:
    backend:
      SimpleStorage:
        num_data_storage_units: 16
```

- `backend_name`: The name of the transfer backend to use.
  - `null` (default): Disables remote transfer; all data stays local. This is the default behavior when `transfer_backend` is not configured.
  - `TransferQueue`: Uses the TransferQueue library for high-performance data transfer.
- `backend_config`: Backend-specific configuration dictionary. For TransferQueue, this corresponds to the TransferQueue initialization config.
  - `backend.SimpleStorage.num_data_storage_units`: The number of storage units to shard data across. Can be configured based on the number of CPU cores and cluster nodes. `msgpack` serialization has a maximum 4 GB limit per object, so larger data transfers require more storage units to shard `non_tensor_batch` into smaller pieces.

### Agentic Pipeline Optimization

In the Agentic Pipeline, `to_remote` is called at the RolloutScheduler level by default. To further avoid data aggregation overhead from env workers to the RolloutScheduler, you can manually call `to_remote` in the env manager before putting data into the output queue:

```python
batch = DataProto.to_remote(batch)
output_queue.put(batch)
```

:::caution
Manually calling `to_remote` inside environment workers is incompatible with filter. When data is filtered out, the Scheduler does not call `drop()` on the filtered data, causing a leak in the remote store. Only use manual `to_remote` in env workers when filter is not required. (TODO: support automatic `drop()` on filtered RemoteBatch in the Scheduler)
:::

## Development Status

| Backend | Status | Notes |
|---------|--------|-------|
| TransferQueue | End-to-end tested | Production-ready. Tested across RLVR, VLM, and Agentic pipelines. |
| RayMemoryStore | Illustration only | Not tested. Provided as a reference implementation for the `ColumnRemoteBatch` pattern. |

### TODO

- Avoid full materialization at Trainer: Currently the Trainer calls `materialize()` on the entire RemoteBatch. This can be optimized to only materialize the fields actually needed, avoiding unnecessary data fetching.
- Selective prefetch on Driver: Implement selective prefetch in the Pipeline Driver to batch-fetch fields needed by upcoming steps, reducing the overhead of multiple small fetches.
- Automatic `drop()` on filtered RemoteBatch in the Scheduler to prevent remote storage leaks.
