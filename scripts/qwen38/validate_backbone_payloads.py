"""Read one native checkpoint's declared DCP payloads with bounded tensor memory.

Run after the training processes exit and before cold restore. Structure and
expected-model inventory validation are separate; self-described metadata cannot
prove that every parameter belonging to the intended architecture was saved.
"""
import argparse
import hashlib
import io
import json
import math
import numbers
from pathlib import Path
import sys

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.distributed.checkpoint import FileSystemReader
from scripts.qwen38.checkpoint_integrity import validate_dcp_storage_inventory


def finite_chunks(tensor, bound=1024 * 1024):
    """Slice views along dimensions; never flatten/copy a whole strided tensor."""
    if tensor.numel() <= bound:
        yield tensor
        return
    row_elements = tensor.numel() // tensor.shape[0]
    rows = bound // row_elements
    if rows:
        for start in range(0, tensor.shape[0], rows):
            yield tensor[start:start + rows]
    else:
        for index in range(tensor.shape[0]):
            yield from finite_chunks(tensor[index], bound)


def validate_step(step, name):
    if isinstance(step, torch.Tensor):
        if step.numel() != 1:
            raise ValueError(f'Invalid Adam step: {name}')
        step = step.item()
    if (isinstance(step, bool) or not isinstance(step, numbers.Real)
            or not math.isfinite(step) or step < 0 or step != int(step)):
        raise ValueError(f'Invalid Adam step: {name}')


def validate_numeric_state(value, name):
    """Check native Adam common/object state without copying tensor payloads."""
    if isinstance(value, torch.Tensor):
        for chunk in finite_chunks(value):
            if not torch.isfinite(chunk).all():
                raise ValueError(f'Nonfinite checkpoint tensor: {name}')
    elif isinstance(value, numbers.Real):
        if not isinstance(value, numbers.Integral) and not math.isfinite(value):
            raise ValueError(f'Nonfinite checkpoint scalar: {name}')
    elif isinstance(value, dict):
        for key, item in value.items():
            if key == 'step':
                validate_step(item, name)
            validate_numeric_state(item, f'{name}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_numeric_state(item, f'{name}[{index}]')


def _iter_storage_items(directory, storage):
    """Read DCP items grouped by file and increasing offset.

    DCP commonly stores thousands of serialized tensors in one rank file. Reopening
    that file for every item turns a sequential payload scan into a metadata-heavy
    random-open workload. Keeping one bounded file handle per group preserves the
    one-item memory bound while allowing the filesystem to stream each file.
    """
    directory = Path(directory).resolve()
    grouped = {}
    for key, extent in storage.items():
        path = (directory / extent.relative_path).resolve()
        if not path.is_relative_to(directory):
            raise ValueError('DCP storage path escapes checkpoint directory')
        if (type(extent.offset) is not int or extent.offset < 0
                or type(extent.length) is not int or extent.length <= 0):
            raise ValueError('Invalid DCP storage extent')
        if getattr(extent, 'transform_descriptors', None):
            raise ValueError('Transformed DCP storage is not supported')
        grouped.setdefault(path, []).append((extent.offset, key, extent))
    for path, entries in grouped.items():
        size = path.stat().st_size
        with path.open('rb') as stream:
            for offset, key, extent in sorted(entries, key=lambda item: item[0]):
                if size < offset + extent.length:
                    raise ValueError(f'Truncated DCP payload: {path}')
                stream.seek(offset)
                data = stream.read(extent.length)
                if len(data) != extent.length:
                    raise ValueError(f'Truncated DCP payload: {path}')
                value = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
                del data
                yield key, value


def _inventory(metadata):
    """Build the logical-to-storage map without importing comparator tooling."""
    validate_dcp_storage_inventory(metadata)
    result = {(key.fqn, tuple(key.offset) if key.offset is not None else None): value
              for key, value in metadata.storage_data.items()}
    if len(result) != len(metadata.storage_data):
        raise ValueError('Duplicate logical DCP storage items')
    if {key[0] for key in result} != set(metadata.state_dict_metadata):
        raise ValueError('DCP logical and storage inventories disagree')
    return result


def validate_payloads(checkpoint, adapter_rank_payloads=None):
    checkpoint = Path(checkpoint).resolve()
    result = dict(payloads_valid=False, dcp_directories=0, storage_items=0,
                  tensor_elements=0, tensor_bytes=0, serialized_bytes=0,
                  largest_serialized_item_bytes=0, full_model_inventory_proven=False)
    roles = ('dist_model', 'dist_optimizer')
    if adapter_rank_payloads is not None:
        from scripts.qwen38.lora_checkpoint_integrity import validate_native_lora_metadata

        if not adapter_rank_payloads or len(set(adapter_rank_payloads)) != len(adapter_rank_payloads):
            raise ValueError('Expected a nonempty, unique adapter rank payload inventory')
        result.update(adapter_files=0, adapter_tensor_elements=0, adapter_tensor_bytes=0)
        model_config = json.loads((checkpoint / 'mca_config.json').read_text())
        for relative in adapter_rank_payloads:
            path = checkpoint / relative
            if Path(relative).is_absolute() or not path.resolve().is_relative_to(checkpoint):
                raise ValueError('Adapter payload escapes its checkpoint')
            # One native rank file is memory mapped at a time. Finiteness
            # workspace is bounded; no full-parameter CPU copies are created.
            state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            if not isinstance(state, dict):
                raise ValueError('Legacy adapter payload must be an object')
            model_state = state.get('model', state)
            if not isinstance(model_state, dict) or not model_state:
                raise ValueError('Legacy adapter model payload is empty')
            adapter_config = json.loads((checkpoint / Path(relative).parts[0] / 'adapter_config.json').read_text())
            validate_native_lora_metadata(model_state, adapter_config, model_config)
            for name, value in model_state.items():
                if not isinstance(name, str):
                    raise ValueError('Legacy adapter tensor names must be strings')
                is_lora = 'lora_' in name
                if not isinstance(value, torch.Tensor):
                    if is_lora:
                        raise ValueError(f'Invalid native LoRA tensor: {name}')
                    continue
                for chunk in finite_chunks(value):
                    if not torch.isfinite(chunk).all():
                        raise ValueError(f'Nonfinite native LoRA tensor: {name}')
                del chunk
                result['adapter_tensor_elements'] += value.numel()
                result['adapter_tensor_bytes'] += value.numel() * value.element_size()
            del value, model_state, state
            result['adapter_files'] += 1
        roles = ('dist_optimizer',)
    for role in roles:
        directories = sorted(path.parent for path in checkpoint.glob(f'iter_*/{role}/**/.metadata'))
        if not directories:
            raise ValueError(f'No native DCP metadata for {role}')
        for directory in directories:
            metadata = FileSystemReader(directory).read_metadata()
            storage = _inventory(metadata)
            expected = validate_dcp_storage_inventory(metadata)
            for key, value in _iter_storage_items(directory, storage):
                extent = storage[key]
                if role == 'dist_optimizer' and key[0].split('.')[-1] == 'step':
                    validate_step(value, str(key))
                if expected[key] is not None:
                    dtype, shape = expected[key]
                    if (not isinstance(value, torch.Tensor) or value.dtype != dtype
                            or tuple(value.shape) != shape):
                        raise ValueError(f'DCP payload violates declared tensor chunk: {key}')
                    for chunk in finite_chunks(value):
                        if not torch.isfinite(chunk).all():
                            raise ValueError(f'Nonfinite checkpoint tensor: {key}')
                    del chunk
                    result['tensor_elements'] += value.numel()
                    result['tensor_bytes'] += value.numel() * value.element_size()
                else:
                    validate_numeric_state(value, str(key))
                del value
                result['storage_items'] += 1
                result['serialized_bytes'] += extent.length
                result['largest_serialized_item_bytes'] = max(
                    result['largest_serialized_item_bytes'], extent.length)
            # Native DCP common state must also deserialize, without retaining it
            # alongside the next tensor storage item.
            common = torch.load(directory / 'common.pt', map_location='cpu', weights_only=False)
            validate_numeric_state(common, str(directory / 'common.pt'))
            del common
            result['dcp_directories'] += 1
    result.update(payloads_valid=True, checkpoint=str(checkpoint), torch_version=torch.__version__,
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations='Declared DCP payloads and, when configured, native LoRA rank files. One serialized DCP item or memory-mapped adapter rank file at a time with bounded finiteness workspace. Requires separate structure and architecture-inventory checks; does not prove state equality or resumed updates.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lora-config', type=Path, help='Resolved run config defining the exact LoRA rank paths')
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.checkpoint.resolve()) or args.output.exists():
        parser.error('Report must be a new file outside the checkpoint directory')
    try:
        adapters = None
        if args.lora_config:
            from scripts.qwen38.run_sft_validation import checkpoint_contract, inspect_checkpoint
            config = json.loads(args.lora_config.read_text())
            worker = dict(config['student_train'] if config.get('is_pure_opd') else config['actor_train'])
            worker['world_size'] = config['num_gpus_per_node'] * config.get('num_nodes', 1)
            contract = checkpoint_contract({'sft_train': worker})
            if contract['layout'] != 'legacy_lora':
                raise ValueError('LoRA payload validation requires the legacy adapter checkpoint layout')
            inspect_checkpoint(args.checkpoint, contract)
            adapters = contract['adapter_rank_payloads']
        result = validate_payloads(args.checkpoint, adapter_rank_payloads=adapters)
    except Exception as error:
        result = dict(payloads_valid=False, checkpoint=str(args.checkpoint.resolve()),
                      error=f'{type(error).__name__}: {error}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['payloads_valid'] else 1)


if __name__ == '__main__':
    main()
