"""Validate DCP logical chunk coverage without loading tensor payloads."""
import heapq
import math


def _validate_chunk_coverage(name, metadata):
    shape = tuple(metadata.size)
    if any(type(size) is not int or size < 0 for size in shape):
        raise ValueError(f'Invalid DCP tensor shape: {name}')
    chunks = metadata.chunks
    volume = 0
    nonempty = []
    for chunk in chunks:
        if len(chunk.offsets) != len(shape) or len(chunk.sizes) != len(shape):
            raise ValueError(f'DCP chunk dimensionality differs: {name}')
        for offset, size, bound in zip(chunk.offsets, chunk.sizes, shape):
            if (type(offset) is not int or type(size) is not int
                    or offset < 0 or size < 0 or offset + size > bound):
                raise ValueError(f'DCP chunk is out of bounds: {name}')
        count = math.prod(chunk.sizes)
        volume += count
        if count:
            nonempty.append(chunk)
    if volume != math.prod(shape):
        raise ValueError(f'DCP chunks do not cover tensor volume: {name}')
    if len(nonempty) < 2:
        return
    if not shape:
        raise ValueError(f'Overlapping scalar DCP chunks: {name}')
    # Sweep a sharded dimension; ordinary contiguous shards have no active
    # neighbors. Only intersecting slabs need comparisons in other dimensions.
    axis = max(range(len(shape)), key=lambda i: len({c.offsets[i] for c in nonempty}))
    active, ends = {}, []
    for index, chunk in enumerate(sorted(nonempty, key=lambda c: c.offsets[axis])):
        start = chunk.offsets[axis]
        while ends and ends[0][0] <= start:
            _, expired = heapq.heappop(ends)
            del active[expired]
        for other in active.values():
            if all(a < b + bs and b < a + size for a, size, b, bs in
                   zip(chunk.offsets, chunk.sizes, other.offsets, other.sizes)):
                raise ValueError(f'Overlapping DCP chunks: {name}')
        active[index] = chunk
        heapq.heappush(ends, (start + chunk.sizes[axis], index))


def validate_dcp_storage_inventory(metadata):
    """Return expected (dtype, shape) by logical storage key; bytes use None."""
    from torch.distributed.checkpoint.metadata import BytesStorageMetadata, TensorStorageMetadata

    if not metadata.state_dict_metadata or not metadata.storage_data:
        raise ValueError('Empty DCP logical or storage inventory')
    expected = {}
    for name, item in metadata.state_dict_metadata.items():
        if isinstance(item, TensorStorageMetadata):
            _validate_chunk_coverage(name, item)
            for chunk in item.chunks:
                key = (name, tuple(chunk.offsets))
                if key in expected:
                    raise ValueError(f'Duplicate declared DCP chunk: {key}')
                expected[key] = (item.properties.dtype, tuple(chunk.sizes))
        elif isinstance(item, BytesStorageMetadata):
            expected[(name, None)] = None
        else:
            raise ValueError(f'Unsupported DCP logical metadata: {name}')
    actual = {(key.fqn, tuple(key.offset) if key.offset is not None else None)
              for key in metadata.storage_data}
    if len(actual) != len(metadata.storage_data):
        raise ValueError('Duplicate logical DCP storage items')
    if actual != expected.keys():
        raise ValueError('DCP storage does not match all declared chunks/bytes items')
    return expected
