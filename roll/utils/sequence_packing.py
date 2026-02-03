from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roll.distributed.scheduler.protocol import DataProto
    from roll.utils.functionals import get_seqlen_balanced_partitions

import torch
import math
import copy
from dataclasses import field, dataclass, asdict
from typing import Iterator, Tuple, Dict, List
import torch.distributed as dist
from roll.configs.worker_config import SequencePackingConfig

def make_micro_batch_iter_for_sequence_packing(mini_batch, tp_size, cp_size, vp_size, is_train=False, dp_group=None,
                                               micro_batch_size=None, config: SequencePackingConfig = None):
    packer = get_sequence_packing_packer(config)
    return packer.make_micro_batch_iter_for_sequence_packing(mini_batch, tp_size, cp_size, vp_size, is_train, dp_group, micro_batch_size)

def restore_results_order(
            results: Dict[str, torch.Tensor],
            partition_indices_list: List[List[int]],
            config: SequencePackingConfig = None
    ) -> Dict[str, torch.Tensor]:
    packer = get_sequence_packing_packer(config)
    return packer.restore_results_order(results, partition_indices_list)


def get_sequence_packing_packer(config: SequencePackingConfig = None):
    """Factory function to get the appropriate sequence packing algorithm."""
    if config==None:
        config = SequencePackingConfig()
    if config.algorithm == 'load_balance':
        return LoadBalancePacker(config)
    elif config.algorithm == 'none':
        return SequencePackingPacker(config)
    else:
        raise ValueError(f"Illegal sequence packing algorithm {config.algorithm},"
                         f" algorithm must be in ['none', 'load_balance']")


class SequencePackingPacker:
    """
    Sequence Packing Packer
    """

    def __init__(self, config: SequencePackingConfig = None):
        self.config = config if config is not None else SequencePackingConfig()

    def get_pad_factor(self, cp_size, tp_size):
        """Calculate padding factor based on parallelism configuration."""
        pad_factor = cp_size * 2 * tp_size if cp_size > 1 else tp_size
        pad_factor = math.lcm(16, pad_factor)
        return pad_factor

    @staticmethod
    def calculate_workload(seqlen: int) -> float:
        """
        Calculate workload (simulating Transformer FLOPs).
        FLOPs ∝ 6 * hidden_size * seqlen + seqlen^2
        Using hidden_size=4096 as reference (7B model)
        """
        return 24576 * seqlen + seqlen * seqlen

    @staticmethod
    def ceildiv(a: int, b: int) -> int:
        """Ceiling division."""
        return -(a // -b)

    def make_micro_batch_iter_for_sequence_packing(
            self,
            mini_batch: DataProto,
            tp_size, cp_size, vp_size, is_train=False,
            dp_group=None, micro_batch_size=None
    ) -> Iterator[DataProto]:
        assert micro_batch_size is not None, "SequencePackingPacker: micro_batch_size is None"
        mini_batch_size = len(mini_batch)
        mini_batch.meta_info['partition_indices_list'] = []
        num_microbatches = mini_batch_size // micro_batch_size
        mini_batch.meta_info['num_micro_batchs'] = num_microbatches
        return iter(mini_batch.chunk(chunks=num_microbatches))

    @staticmethod
    def restore_results_order(
            results: Dict[str, torch.Tensor],
            partition_indices_list: List[List[int]]
    ) -> Dict[str, torch.Tensor]:
        return results



class LoadBalancePacker(SequencePackingPacker):
    @staticmethod
    def roundup_divisible(a: int, b: int) -> int:
        """Round up a to be divisible by b."""
        return ((a + b - 1) // b) * b

    @staticmethod
    def get_device_name():
        """Get current device name."""
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"

    @staticmethod
    def calculate_workload_batch(seqlen_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculate workload for a batch of sequences.

        Args:
            seqlen_tensor: Tensor of sequence lengths

        Returns:
            Tensor of workloads
        """
        return 24576 * seqlen_tensor + seqlen_tensor * seqlen_tensor

    def make_micro_batch_iter_for_sequence_packing(
            self,
            mini_batch: DataProto,
            tp_size: int,
            cp_size: int,
            vp_size: int,
            is_train=False,
            dp_group=None,
            micro_batch_size=None
    ) -> Iterator[DataProto]:
        """
        Split mini_batch into micro batches with sequence packing strategy.

        This function:
        1. Calculates the optimal number of micro batches based on max_packed_sequence_length
        2. Ensures all DP ranks have the same number of micro batches
        3. Ensures the number of micro batches is divisible by vp_size
        4. Balances workload across micro batches using Karmarkar-Karp algorithm
        5. Optimizes scheduling by placing smaller batches at edges

        Args:
            mini_batch: Input mini batch data containing:
                - batch: TensorDict with tensors including 'input_ids' and 'attention_mask'
                - non_tensor_batch: Dict with non-tensor data
                - meta_info: Dict with metadata
            tp_size: Tensor parallel size
            cp_size: Context parallel size
            vp_size: Virtual pipeline parallel size (must divide num_micro_batches)
            max_packed_sequence_length: Maximum total sequence length per micro batch
            dp_group: Data parallel process group for synchronization

        Yields:
            DataProto: Micro batches with balanced workload

        Raises:
            AssertionError: If max_packed_sequence_length < max sequence length in batch
        """
        assert dp_group is not None, "LoadBalancePacker: dp_group is None"
        # Calculate effective sequence lengths for each sample
        # For regular tensors, use attention mask
        attention_mask = mini_batch.batch["attention_mask"]
        max_seq_len = attention_mask.shape[-1]
        seq_len_effective: torch.Tensor = attention_mask.sum(dim=1)
        pad_factor = self.get_pad_factor(cp_size, tp_size)
        seq_len_effective = ((seq_len_effective + pad_factor - 1) // pad_factor) * pad_factor

        if is_train:
            max_packed_sequence_length = self.config.max_packed_sequence_length_train
        else:
            max_packed_sequence_length = self.config.max_packed_sequence_length_forward
        assert max_packed_sequence_length is not None, "LoadBalancePacker: max_packed_sequence_length is None"
        # Validate that max_packed_sequence_length is sufficient
        assert max_packed_sequence_length >= max_seq_len, (
            f"max_packed_sequence_length ({max_packed_sequence_length}) must be >= "
            f"max sequence length in batch ({max_seq_len})"
        )

        batch_size = len(seq_len_effective)
        total_seqlen = seq_len_effective.sum().item()

        # Step 2: Calculate initial number of micro batches
        # Base calculation: how many batches do we need to fit all tokens?
        num_micro_batches = max(1, self.ceildiv(total_seqlen, max_packed_sequence_length))

        # Cannot have more micro batches than samples
        num_micro_batches = min(num_micro_batches, batch_size)

        if is_train:
            min_num_micro_batches = self.config.min_num_micro_batches_train
        else:
            min_num_micro_batches = self.config.min_num_micro_batches_forward
        num_micro_batches = max(num_micro_batches, min_num_micro_batches)

        # Step 3: Synchronize across DP ranks (all ranks must have same count)
        if dist.is_initialized() and dp_group is not None:
            num_micro_batches_tensor = torch.tensor(
                [num_micro_batches],
                device=self.get_device_name()
            )
            # Use MAX to ensure all ranks can accommodate their data
            dist.all_reduce(
                num_micro_batches_tensor,
                op=dist.ReduceOp.MAX,
                group=dp_group
            )
            num_micro_batches = num_micro_batches_tensor.cpu().item()

        # Step 4: Round up to be divisible by vp_size
        if vp_size > 1:
            num_micro_batches = self.roundup_divisible(num_micro_batches, vp_size)

        # Step 5: Calculate workload for load balancing
        # Use squared sequence length as proxy for attention computation cost
        workloads = self.calculate_workload_batch(seq_len_effective)

        from roll.utils.functionals import get_seqlen_balanced_partitions
        # Step 6: Partition samples into micro batches with balanced workload
        micro_batch_indices = get_seqlen_balanced_partitions(
            seqlen_list=workloads.tolist(),
            k_partitions=num_micro_batches,
            equal_size=False  # Allow variable sizes for better balance
        )

        # Step 7: Sort and reorder for better pipeline scheduling
        # Sort by workload (descending) to identify large and small batches
        micro_batch_indices_with_workload = [
            (
                partition,
                sum(workloads[idx].item() for idx in partition),
                partition[0] if partition else 0  # tie-breaker
            )
            for partition in micro_batch_indices
        ]

        micro_batch_indices_with_workload.sort(
            key=lambda x: (x[1], x[2]),
            reverse=True
        )

        # Reorder: place smaller batches at both ends to reduce pipeline bubbles
        # Pattern: [small, large, large, ..., large, small]
        sorted_indices = [x[0] for x in micro_batch_indices_with_workload]
        reordered_indices = sorted_indices[::2][::-1] + sorted_indices[1::2]

        mini_batch.meta_info['partition_indices_list'] = reordered_indices.copy()

        # Step 8: Generate micro batches
        generated_count = 0

        for partition in reordered_indices:
            if len(partition) == 0:
                # Skip empty partitions (shouldn't happen but be safe)
                continue

            # Use DataProto's select_idxs method to create micro batch
            micro_batch_proto = mini_batch.select_idxs(partition)

            # Add metadata about this micro batch
            micro_batch_proto.meta_info = copy.deepcopy(mini_batch.meta_info)
            micro_batch_proto.meta_info['micro_batch_idx'] = generated_count
            micro_batch_proto.meta_info['is_padding_batch'] = False
            micro_batch_proto.meta_info['partition_indices'] = partition
            micro_batch_proto.meta_info['num_micro_batchs'] = num_micro_batches
            micro_batch_proto.meta_info['mini_batch_size'] = mini_batch.batch.batch_size[0]

            yield micro_batch_proto
            generated_count += 1

        # Verify we generated the correct number of micro batches
        assert generated_count == num_micro_batches, (
            f"Generated {generated_count} micro batches but expected {num_micro_batches}"
        )

    @staticmethod
    def restore_results_order(
            results: Dict[str, torch.Tensor],
            partition_indices_list: List[List[int]]
    ) -> Dict[str, torch.Tensor]:
        """
        Restore computation results to their original order after load-balanced partitioning.

        During load balancing, samples are reordered into partitions by sequence length.
        This function reverses that reordering to match the original input order.

        Args:
            results: Dict of computation results where first dimension is in partitioned order
                     e.g., {'logits': [total_batch, ...], 'loss': [total_batch]}
            partition_indices_list: List of original indices for each partition
                                    (from mini_batch.meta_info['partition_indices_list'])

        Returns:
            Dict with same keys but tensors reordered to original sample order

        Example:
            # Create micro batches with load balancing
            micro_batches_iter = packer.make_micro_batch_iter_for_sequence_packing(
                mini_batch=mini_batch, ...
            )
            partition_indices_list = mini_batch.meta_info['partition_indices_list']

            # Compute (results are concatenated across partitions)
            results = model(micro_batches_iter)  # {'logits': [total_batch, ...]}

            # Restore original order
            restored = LoadBalancePacker.restore_results_order(
                results, partition_indices_list
            )
        """
        if not results:
            return {}

        # Flatten partition indices to get current -> original mapping
        original_indices = []
        for partition_indices in partition_indices_list:
            original_indices.extend(partition_indices)

        # Build inverse mapping: original position -> current position
        # original_indices[current_pos] = original_pos
        # reorder_indices[original_pos] = current_pos
        total_samples = len(original_indices)
        reorder_indices = [0] * total_samples
        for current_pos, original_pos in enumerate(original_indices):
            reorder_indices[original_pos] = current_pos

        reorder_indices_tensor = torch.tensor(reorder_indices, dtype=torch.long)

        # Reorder each tensor result
        restored_results = {}
        for key, tensor in results.items():
            if isinstance(tensor, torch.Tensor) and tensor.dim() > 0:
                assert tensor.shape[0] == total_samples, \
                    f"Tensor '{key}' batch size {tensor.shape[0]} != total samples {total_samples}"

                restored_results[key] = tensor[reorder_indices_tensor]
            else:
                # Scalar or non-tensor, keep as-is
                restored_results[key] = tensor

        return restored_results






