# ROLL RemoteBatch and Transfer Backend

The ROLL framework supports **RemoteBatch**, a lazy data transfer mechanism that decouples data storage from data consumption. With a pluggable transfer backend, the cross-node bulk payload is stored and transferred outside Ray while Ray carries a lightweight reference. A node-scoped backend may still use the local Ray object store to hand materialized data from its node actor to a worker process. This document explains how to configure and use the available backends.

## Introduction

In RL training pipelines (especially VLM and Agentic scenarios), `DataProto` batches may contain large tensors (e.g., images, multi-modal embeddings) and non-tensor data (e.g., conversation histories). Transferring these between the RolloutScheduler and training workers via Ray's default serialization has two major problems:

1. **High memory overhead**: The full data is serialized and deserialized through the Ray object store, creating extra copies and raising peak memory usage.
2. **High transfer latency**: Large data batches (e.g., image data in VLM scenarios) must be fully transferred between workers, causing significant data movement overhead.

**RemoteBatch** addresses these issues by storing data in an external key-value store and only passing lightweight metadata (keys/references) through Ray. The actual data is **lazily materialized** on the consumer side only when needed, and only the requested fields are fetched.

### Key Concepts

- **RemoteBatch**: An abstract base class representing a batch of data stored remotely. It supports the same slicing, indexing, selection, concatenation, and repeat operations as `TensorDict`, but defers actual data access until `materialize()` is called.
- **RowRemoteBatch**: A concrete `RemoteBatch` where data is stored with **row IDs** as keys. Each row (sample) has a unique ID, and the transfer backend stores/retrieves data at row granularity. This is used by the **TransferQueue** backend.
- **ColumnRemoteBatch**: A concrete `RemoteBatch` where fields share column-oriented metadata. It is used by the **RayMemoryStore** and **Mooncake** backends.
- **BatchProxy**: A proxy object that wraps both a local `TensorDict` (or `dict`) and a `RemoteBatch`, supporting transparent fallback lookup. When a key is accessed, it first checks the local batch and then falls back to the remote batch.
- **Transfer Backend**: A pluggable storage backend responsible for `put`, `get`, and `delete` operations. Currently supported backends:
  - `None` (Dummy): No remote storage; data stays local (default).
  - `TransferQueue`: Uses the [TransferQueue](https://github.com/kvcache-ai/TransferQueue) library for high-performance distributed key-value transfer.
  - `Mooncake`: Uses Mooncake as an optional structured `DataProto` transfer backend for large tensor, non-tensor, and multimodal rollout payloads.

### How It Works

1. **Upload (`to_remote`)**: The `DataProto.to_remote()` class method converts a local `DataProto` into a remote-backed `DataProto`. It uploads all tensor and non-tensor fields and returns a new `DataProto` with a `RemoteBatch` reference. The producing process may retain a local cache; serialized consumers receive the lightweight reference and materialize data on demand.
2. **Transfer**: Ray carries the `RemoteBatch` reference together with the original lightweight `meta_info`. Bulk `batch` and `non_tensor_batch` fields remain in the transfer backend.
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

### Mooncake

#### 1. Install Mooncake

ROLL requires Mooncake's unified DataProto API, schema support, GET-result release, and the typed-ragged ndarray layout. Use a Mooncake main wheel built from commit `86b21ccf` or later. The following commands follow Mooncake's source-build flow:

```bash
git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
git checkout 86b21ccf
git submodule update --init --recursive
sudo bash dependencies.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF
cmake --build build -j
PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYTHON_VERSION="${PYTHON_VERSION}" BUILD_DIR=build bash scripts/build_wheel.sh "${PYTHON_VERSION}" dist
python -m pip install dist/mooncake_transfer_engine-*.whl
```

#### 2. Start the metadata service

Mooncake metadata infrastructure must be reachable before ROLL starts. ROLL initializes Store clients but does not start or manage `mooncake_master`, etcd, or Redis. For the standalone master configuration, start it on a reachable host:

```bash
mooncake_master --rpc_address=<master-ip> --rpc_port=<master-port>
```

Use the deployment instructions from Mooncake when etcd or Redis provides the metadata service.

#### 3. Configure the ROLL cluster

ROLL shares one `backend_config` across the cluster. Keep common settings in YAML:

```yaml
transfer_backend:
  backend_name: Mooncake
  backend_config:
    client_scope: node
    metadata_server: P2PHANDSHAKE
    global_segment_size: 8589934592
    local_buffer_size: 8589934592
    protocol: tcp
    master_server_addr: <master-host:port>
```

For RDMA, change the shared protocol:

```yaml
transfer_backend:
  backend_name: Mooncake
  backend_config:
    client_scope: node
    metadata_server: P2PHANDSHAKE
    global_segment_size: 8589934592
    local_buffer_size: 8589934592
    protocol: rdma
    master_server_addr: <master-host:port>
```

Set node-local values in each node's environment. `MOONCAKE_DEVICE` is required only for RDMA:

```bash
export MOONCAKE_LOCAL_HOSTNAME=<this-node-ip>
export MOONCAKE_DEVICE=<this-node-rdma-device>
```

`field_schemas` is optional. ROLL converts each entry to a Mooncake `FieldSchema` and passes the mapping to `put(type="dataproto")`. When it is omitted, Mooncake infers the representation from the field values. Supplying schemas for known object-array fields is recommended because it keeps the selected representation stable across batches, including batches whose sampled values are all null. For example, a workload with an `int64` ragged field can declare:

```yaml
transfer_backend:
  backend_name: Mooncake
  backend_config:
    field_schemas:
      token_rows:
        codec: typed_ragged
        nullable: true
        metadata:
          section: non_tensor_batch
          dtype: int64
```

Replace `token_rows` and its schema with fields from the workload. Multimodal processor outputs can have model-specific keys, so their schema must match the actual ROLL collator output instead of assuming a fixed `image` member.

Mooncake-specific `backend_config` options are:

| Option | Description |
| --- | --- |
| `client_scope` | `node` reuses one Mooncake client and BufferPool per Ray node. It is the only supported scope. |
| `key_prefix` | Prefix used for Mooncake Store keys. The default is `roll`. |
| `transfer_policy` | Optional arguments for Mooncake `BundleTransferPolicy`, such as `copy_mode`, `put_mode`, and `max_inflight_put`. |
| `field_schemas` | Optional per-field `codec`, `nullable`, and `metadata` settings. Unknown schema options fail during client initialization. |
| Store setup fields | `local_hostname`, `metadata_server`, segment sizes, `protocol`, RDMA device, and metadata backend address. |

Store setup can also be supplied through `setup_args` / `setup_kwargs`, or through the standard Mooncake environment variables:

```bash
export MOONCAKE_MASTER=<master-host:port>
export MOONCAKE_LOCAL_HOSTNAME=<worker-ip>
export MOONCAKE_TE_META_DATA_SERVER=P2PHANDSHAKE
export MOONCAKE_PROTOCOL=rdma  # or tcp
export MOONCAKE_DEVICE=<rdma-device>
```

When environment variables provide the node-specific Store settings, the shared ROLL configuration only needs the backend policy:

```yaml
transfer_backend:
  backend_name: Mooncake
  backend_config:
    client_scope: node
    key_prefix: roll
    transfer_policy:
      copy_mode: auto
```

Run the existing ROLL training entry point with the directory and basename of the YAML file that contains this configuration:

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python examples/start_rlvr_pipeline.py \
  --config_path <config-directory> \
  --config_name <config-filename-without-yaml>
```

The other ROLL pipeline launchers accept the same configuration fields. A successful startup logs `Initialized Mooncake transfer backend` on the driver and `Initialized transfer client` in each process that first uses RemoteBatch.

#### 4. Data and buffer lifetime

Mooncake stores each uploaded ROLL DataProto fragment as a structured object. A logical DataProto may reference several such objects after union or concatenation. Tensor fields remain in the `batch` section, NumPy fields remain in `non_tensor_batch`, and ROLL's original `meta_info` continues through the existing lightweight control path. `ColumnRemoteBatch.materialize(fields)` requests only the selected fields, while `ColumnRemoteBatch.drop()` removes the referenced remote objects.

The Mooncake GET lease belongs to the node actor. After placing a serializable view in the local Ray object store, the actor releases the BufferPool lease; a failed release is retried by a later Mooncake operation. Selections, slices, unions, and concatenations retain the original row order.

Local aliases share cleanup state, but ROLL does not provide a cross-process reference counter or invalidation broadcast. The framework must designate one cleanup path and call `drop()` only after every consumer has finished. `drop()` removes the physical remote objects; other processes holding old handles are not notified automatically and must not use them afterward.

- `backend_name`: The name of the transfer backend to use.
  - `null` (default): Disables remote transfer; all data stays local. This is the default behavior when `transfer_backend` is not configured.
  - `TransferQueue`: Uses the TransferQueue library for high-performance data transfer.
  - `Mooncake`: Uses Mooncake structured `DataProto` transfer for tensor batch fields and `non_tensor_batch`. The original `meta_info` remains on ROLL's lightweight control path.
- `backend_config`: Backend-specific configuration dictionary.
  - For TransferQueue, this corresponds to the TransferQueue initialization config.
  - For Mooncake, use the explicit configuration or standard environment variables described above.
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
| Mooncake | Experimental | Optional structured `DataProto` backend for tensor, non-tensor, and multimodal rollout payloads. |
| RayMemoryStore | Illustration only | Not tested. Provided as a reference implementation for the `ColumnRemoteBatch` pattern. |

### TODO

- Avoid full materialization at Trainer: Currently the Trainer calls `materialize()` on the entire RemoteBatch. This can be optimized to only materialize the fields actually needed, avoiding unnecessary data fetching.
- Selective prefetch on Driver: Implement selective prefetch in the Pipeline Driver to batch-fetch fields needed by upcoming steps, reducing the overhead of multiple small fetches.
- Automatic `drop()` on filtered RemoteBatch in the Scheduler to prevent remote storage leaks.
