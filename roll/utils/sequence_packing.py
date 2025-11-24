import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.platforms import current_platform
from roll.utils.constants import IGNORE_INDEX

"""
Loss computation wrappers for sequence packing training.
Handles unpacking model outputs and aligning with original sequence boundaries for loss calculation.
"""


# TODO: use view of tensor in loss caculating instead of copy
class SequencePackingLossWrapper:
    """
    Base wrapper for computing loss on packed sequences.

    In sequence packing, multiple sequences are concatenated and padded to form a single packed sequence.
    This wrapper handles:
    1. Unpacking model outputs back to individual sequences
    2. Aligning original data (labels, masks) with unpacked outputs
    3. Computing loss on properly aligned data
    """

    def __init__(
        self,
        strategy,
        loss_func,
    ):
        """
        Args:
            strategy: Training strategy containing model and distributed config
            loss_func: Loss function to apply
            cu_seqlens_q: Cumulative sequence lengths of original (unpadded) sequences
            cu_seqlens_q_padded: Cumulative sequence lengths after padding for packing
            logger: Optional logger
        """
        self.strategy = strategy
        self.loss_func = loss_func
        self.cu_seqlens = None
        self.cu_seqlens_padded = None
        self.logger = None

    def set_packing_params(self, cu_seqlens, cu_seqlens_padded, logger):
        self.cu_seqlens = cu_seqlens
        self.cu_seqlens_padded = cu_seqlens_padded
        self.logger = logger

    def _unpack_output_tensor(self, output_tensor):
        """
        Unpack model output tensor from packed format back to individual sequences.

        The packed output contains multiple sequences concatenated together. This method
        splits them back using padded cumulative sequence lengths, accounting for context
        parallelism partitioning.

        Args:
            output_tensor: Packed model output with shape (batch=1, packed_seq_len, hidden_dim)

        Returns:
            List of unpacked tensors, one per original sequence, each with shape
            (batch=1, padded_seq_len, hidden_dim)
        """
        cp_size = self.strategy.worker.rank_info.cp_size

        # Calculate sequence boundaries in the packed tensor
        # Padded cumulative lengths mark where each sequence starts/ends after packing
        padded_cu_seqlens = self.cu_seqlens_padded

        # Adjust for context parallelism: each rank only holds a portion of the sequence
        seq_starts = padded_cu_seqlens[:-1] // cp_size
        seq_ends = padded_cu_seqlens[1:] // cp_size

        # Extract each sequence from the packed tensor
        unpacked_output_tensor_list = []
        for seq_idx, (seq_start, seq_end) in enumerate(zip(seq_starts, seq_ends)):
            unpacked_output_tensor_list.append(output_tensor[:, seq_start:seq_end, :])
        return unpacked_output_tensor_list

    def _pad_tensor_to_target_length(self, tensor, target_length, pad_val=0, pad_dim=0):
        """
        Pad tensor along the specified dimension to reach the target length by padding on the right.

        Args:
            tensor: Input tensor to pad
            target_length: Desired length along pad_dim
            pad_val: Value to use for padding
            pad_dim: Dimension to pad along

        Returns:
            Padded tensor with length target_length along pad_dim
        """
        seq_len = tensor.shape[pad_dim]

        if target_length > seq_len:
            pad_size = target_length - seq_len

            # Construct padding specification for torch.nn.functional.pad
            # Format: [pad_left, pad_right] for each dim from last to first
            pad_list = [0, 0] * tensor.ndim
            pad_list[2 * (tensor.ndim - 1 - pad_dim) + 1] = pad_size

            tensor = torch.nn.functional.pad(tensor, pad_list, value=pad_val)

        return tensor

    def _align_to_unpacked_output_tensor_shape(self, tensor, pad_val=0):
        """
        Align original data tensors (labels, masks) to match unpacked output shape.

        Original data comes in shape (batch, max_seq_len, ...) where batch contains multiple
        sequences with varying actual lengths. This method:
        1. Extracts each sequence's valid portion (up to its original unpadded length)
        2. Pads it to match the padded length used during packing

        This ensures original data aligns with unpacked model outputs for loss computation.

        Args:
            tensor: Original data tensor with shape (batch, seq_len, ...)
            pad_val: Value used for padding (e.g., IGNORE_INDEX for labels, 0 for masks)

        Returns:
            List of aligned tensors, each with shape (1, padded_seq_len, ...) matching
            the corresponding unpacked output tensor
        """
        # Get original unpadded sequence lengths (actual data before packing)
        unpadded_seq_lengths = self.cu_seqlens[1:] - self.cu_seqlens[:-1]

        # Get padded sequence lengths (after padding during packing)
        padded_seq_lengths = self.cu_seqlens_padded[1:] - self.cu_seqlens_padded[:-1]

        source_seq_lengths = unpadded_seq_lengths  # Valid data length
        target_seq_lengths = padded_seq_lengths  # Target length after packing

        aligned_tensor_list = []
        for seq_idx, (source_len, target_len) in enumerate(
                zip(source_seq_lengths, target_seq_lengths)
        ):
            # Extract valid portion: truncate to original unpadded length
            seq_tensor = tensor[seq_idx:seq_idx + 1, :source_len]

            # Pad to match the padded length used in packing
            seq_tensor = self._pad_tensor_to_target_length(seq_tensor, target_len, pad_val=pad_val, pad_dim=1)

            # Keep batch dimension (1) to match unpacked output format
            aligned_tensor_list.append(seq_tensor)

        return aligned_tensor_list

    def __call__(self, data: DataProto, output_tensor: torch.Tensor):
        return self.loss_func(data, output_tensor)


# SFT
class SequencePackingSFTLossWrapper(SequencePackingLossWrapper):
    """
    Wrapper for SFT loss computation with packed sequences.

    For SFT, labels are already packed in the same format as model outputs,
    so we can directly compute loss without unpacking.
    """

    def __call__(self, data: DataProto, output_tensor: torch.Tensor):
        # Use pre-packed labels that match the packed output format
        labels = data.meta_info['labels_packed']
        return self.loss_func(DataProto.from_dict(tensors={'labels': labels}), output_tensor)


# Distillation
class SequencePackingDistillForwardWrapper(SequencePackingLossWrapper):
    """
    Wrapper for teacher model forward pass in distillation with packed sequences.

    Computes teacher logits from packed outputs and prepares them for student training:
    1. Unpacks teacher outputs to individual sequences
    2. Computes full vocabulary logits or topk logits for each sequence
    3. Pads logits back to original max sequence length for easy alignment with student
    """

    def __init__(self, strategy, loss_func):
        super().__init__(strategy, loss_func)
        self.forward_func = loss_func

    def __call__(self, data: DataProto, output_tensor: torch.Tensor, non_loss_data: bool = True):
        """
        Compute teacher logits from packed outputs.

        Args:
            data: Input data protocol
            output_tensor: Packed teacher model outputs
            non_loss_data: Flag indicating this is for data generation, not loss computation

        Returns:
            Tuple of (dummy_loss, dict with teacher logits and topk indices)
        """
        # Step 1: Unpack teacher outputs to individual sequences
        unpacked_output_tensor_list = self._unpack_output_tensor(output_tensor)

        # Step 2: Compute logits for each sequence
        # Gather across tensor/context parallel ranks to get full logits
        teacher_topk_probs_list = []
        teacher_topk_log_probs_list = []
        teacher_topk_indices_list = []
        teacher_topk_inf_mask_list = []
        for idx, unpacked_output_tensor in enumerate(unpacked_output_tensor_list):
            # Compute logits with full vocabulary (or topk for efficiency)
            teacher_topk_probs, teacher_topk_log_probs, teacher_topk_indices, teacher_topk_inf_mask = self.strategy.op_compute_topk_probs_and_indices(
                unpacked_output_tensor,
                topk=self.strategy.worker.pipeline_config.logits_topk,
                target_vocab_size=self.strategy.worker.pipeline_config.target_vocab_size,
                kd_temperature=self.strategy.worker.pipeline_config.kd_temperature,
                teacher_temperature=self.strategy.worker.pipeline_config.teacher_temperature
            )

            # Step 3: Pad each sequence's logits to max sequence length
            # This makes them easy to align with original student data later
            max_length = self.strategy.worker.pipeline_config.sequence_length
            teacher_topk_probs = self._pad_tensor_to_target_length(teacher_topk_probs, max_length, pad_val=0, pad_dim=1)
            teacher_topk_log_probs = self._pad_tensor_to_target_length(teacher_topk_log_probs, max_length, pad_val=0, pad_dim=1)
            teacher_topk_indices = self._pad_tensor_to_target_length(teacher_topk_indices, max_length, pad_val=0, pad_dim=1)
            teacher_topk_inf_mask = self._pad_tensor_to_target_length(teacher_topk_inf_mask, max_length, pad_val=1, pad_dim=1)

            teacher_topk_probs_list.append(teacher_topk_probs)
            teacher_topk_log_probs_list.append(teacher_topk_log_probs)
            teacher_topk_indices_list.append(teacher_topk_indices)
            teacher_topk_inf_mask_list.append(teacher_topk_inf_mask)

        # Concatenate all sequences back into batch format
        teacher_topk_probs = torch.cat(teacher_topk_probs_list, dim=0)
        teacher_topk_log_probs = torch.cat(teacher_topk_log_probs_list, dim=0)
        teacher_topk_indices = torch.cat(teacher_topk_indices_list, dim=0)
        teacher_topk_inf_mask = torch.cat(teacher_topk_inf_mask_list, dim=0)

        # Return dummy loss (teacher forward doesn't compute loss) and teacher outputs
        return torch.tensor(0., device=output_tensor.device), {
            'topk_probs': teacher_topk_probs.detach(),
            'topk_log_probs': teacher_topk_log_probs.detach(),
            'topk_indices': teacher_topk_indices.detach(),
            'topk_inf_mask': teacher_topk_inf_mask.detach()
        }


class SequencePackingDistillLossWrapper(SequencePackingLossWrapper):
    """
    Wrapper for computing distillation loss with packed sequences.

    Combines language modeling loss and distillation loss:
    1. Unpacks student model outputs to individual sequences
    2. Aligns original labels and teacher outputs with unpacked student outputs
    3. Computes both standard LM loss and KL divergence with teacher for each sequence
    4. Combines losses with configurable weighting
    """

    def __call__(self, data: DataProto, output_tensor: torch.Tensor):
        """
        Compute combined distillation and language modeling loss.

        Args:
            data: Input data containing original labels and masks
            output_tensor: Packed student model outputs

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Step 1: Compute student logits from packed outputs
        # Keep them partitioned across tensor/context parallel for memory efficiency
        student_logits = output_tensor

        # Step 2: Unpack student logits to individual sequences (still cp-partitioned)
        student_logits_list = self._unpack_output_tensor(student_logits)

        # Step 3: Get original data from dataloader (not packed)
        labels = data.batch['labels_for_loss']
        attention_mask = data.batch['attention_mask']

        # Step 4: Align original data with unpacked outputs
        # Truncate to original length and pad to match packing padding
        aligned_labels_list = self._align_to_unpacked_output_tensor_shape(labels, pad_val=IGNORE_INDEX)
        aligned_attention_mask_list = self._align_to_unpacked_output_tensor_shape(attention_mask, pad_val=0)

        # Step 5: Get and align teacher outputs (pre-computed in teacher forward pass)
        if self.strategy.worker.teacher_probs_iterator is not None:
            teacher_probs = next(self.strategy.worker.teacher_probs_iterator)
            aligned_teacher_probs_list = self._align_to_unpacked_output_tensor_shape(teacher_probs)
        else:
            teacher_probs = None
        if self.strategy.worker.teacher_log_probs_iterator is not None:
            teacher_log_probs = next(self.strategy.worker.teacher_log_probs_iterator)
            aligned_teacher_log_probs_list = self._align_to_unpacked_output_tensor_shape(teacher_log_probs)
        else:
            teacher_log_probs = None
        if self.strategy.worker.teacher_topk_indices_iterator is not None:
            teacher_topk_indices = next(self.strategy.worker.teacher_topk_indices_iterator)
            aligned_teacher_topk_indices_list = self._align_to_unpacked_output_tensor_shape(teacher_topk_indices)
        else:
            teacher_topk_indices = None
        if self.strategy.worker.teacher_inf_mask_iterator is not None:
            teacher_inf_mask = next(self.strategy.worker.teacher_inf_mask_iterator)
            aligned_teacher_inf_mask_list = self._align_to_unpacked_output_tensor_shape(teacher_inf_mask)
        else:
            teacher_inf_mask = None


        # Step 6: Accumulate losses across all sequences in the batch
        total_gpt_loss = torch.tensor(0, device=current_platform.device_type, dtype=torch.float32)
        total_distill_loss = torch.tensor(0, device=current_platform.device_type, dtype=torch.float32)
        total_valid_tokens = 0
        total_valid_tokens_distill = 0

        batch_size = len(student_logits_list)
        for idx in range(batch_size):
            # Get aligned data for this sequence
            single_student_logits = student_logits_list[idx]
            single_label = aligned_labels_list[idx]
            single_teacher_probs = aligned_teacher_probs_list[idx] if teacher_probs is not None else None
            single_teacher_log_probs = aligned_teacher_log_probs_list[idx] if teacher_log_probs is not None else None
            single_teacher_topk_indices = aligned_teacher_topk_indices_list[idx] if teacher_topk_indices is not None else None
            single_teacher_inf_mask = aligned_teacher_inf_mask_list[idx] if teacher_inf_mask is not None else None

            # Compute standard language modeling loss (cross-entropy with labels)
            local_gpt_loss, local_valid_tokens = self.strategy.op_compute_language_loss_from_logits(
                single_student_logits, single_label,
                reduction="sum")
            total_gpt_loss += local_gpt_loss
            total_valid_tokens += local_valid_tokens

            # Compute distillation loss (KL divergence between student and teacher)
            local_distill_loss, local_valid_tokens_distill = self.strategy.op_compute_various_divergence(
                self.strategy.worker.kl_loss_func,
                single_student_logits, single_teacher_probs,
                single_teacher_log_probs, single_teacher_topk_indices,
                single_teacher_inf_mask, single_label,
                attention_mask=None, reduction="sum")

            total_distill_loss += local_distill_loss
            total_valid_tokens_distill += local_valid_tokens_distill

        # Step 7: Normalize losses by number of valid tokens
        if total_valid_tokens == 0:
            total_valid_tokens = 1
        if total_valid_tokens_distill == 0:
            total_valid_tokens_distill = 1
        gpt_loss = total_gpt_loss / total_valid_tokens
        distill_loss = total_distill_loss / total_valid_tokens_distill

        # Step 8: Combine losses with configured weighting
        # loss = (1 - α) * LM_loss + α * distill_loss
        loss = ((1 - self.strategy.worker.pipeline_config.distill_loss_weight) * gpt_loss
                + self.strategy.worker.pipeline_config.distill_loss_weight * distill_loss)

        student_metrics = {
            "train/loss": loss.detach().item(),
            "train/train_distill_loss": distill_loss.detach().item(),
            "train/train_student_loss": gpt_loss.detach().item(),
        }
        return loss, student_metrics
