from abc import ABC
from concurrent import futures
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from roll.distributed.scheduler.protocol import DataProto
from roll.platforms import current_platform
from roll.utils.checkpoint_manager import CheckpointManager
from roll.utils.constants import IGNORE_INDEX
from roll.utils.collective import collective
from roll.utils.functionals import log_probs_from_logits, get_dist_info_from_comm_plan, entropy_from_logits
from roll.utils.logging import get_logger

logger = get_logger()


class InferenceStrategy(ABC):
    strategy_name = None

    def __init__(self, worker: "Worker"):
        self.worker = worker
        self.model = None
        self.tokenizer = None

        self.worker_config = self.worker.worker_config
        self.thread_executor: futures.ThreadPoolExecutor = futures.ThreadPoolExecutor(max_workers=5)
        self.model_update_comm_plan = {}
        self.offload_nccl = self.worker_config.offload_nccl

    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    def forward_step(
        self,
        batch: DataProto,
        forward_func: Callable[[DataProto, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        """
        forward_step接口定义:
            batch: DataProto, 待forward的一批数据，batch_size = data.batch.batch_size[0]
            forward_func: 方法签名为:(data_iterator: Iterator[DataProto], model)
        """
        pass

    def get_data_input(self, batch: "DataProto") -> "DataProto":
        return batch

    def generate(self, *args, **kwargs):
        raise NotImplementedError

    def get_metrics(self, metric_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get performance metrics from the strategy.
        
        Args:
            metric_names: Optional list of specific metric names to filter
            
        Returns:
            Dictionary of metric names to aggregated values
        """
        return {}

    def start_server(self, *args, **kwargs):
        raise NotImplementedError

    def add_request(self, command, data: DataProto, *args, **kwargs):
        raise NotImplementedError()

    def unwrap_model(self, *args, **kwargs):
        raise NotImplementedError

    def save_checkpoint(self, *args, **kwargs):
        """
        save ckpt/hf model/tokenizer
        """
        raise NotImplementedError

    def load_checkpoint(self, *args, **kwargs):
        pass

    # 参数同步相关接口
    def broadcast_parameter(self, model_update_name, src_pp_rank, dtype, shape, parameter_name):
        raise NotImplementedError

    def broadcast_bucket(self, model_update_name, src_pp_rank, meta_infos, bucket_size):
        raise NotImplementedError

    def update_parameter(self, model_update_name, parameter_name, weight, ranks_in_worker):
        """
        engine模式中，p2p update要求engine能够将param 更新至指定的rank
        """
        raise NotImplementedError

    def update_parameter_in_bucket(self, model_update_name, meta_infos, buffer, ranks_in_worker):
        raise NotImplementedError

    def _setup_collective_group_impl(
            self, model_update_name, comm_plan, backend, mode
    ):
        """
        mode:
            "receiver": acts as the broadcast receiver
            "sender":   acts as the broadcast leader
        """
        if backend is None:
            backend = current_platform.communication_backend
        if mode == "receiver":
            rank, comm_plan_args = get_dist_info_from_comm_plan(
                comm_plan, rank_in_cluster=self.worker.rank, rank_in_worker=0
            )
            if rank is None:
                logger.info(f"no comm_plan found for rank {self.worker.rank}/{0}")
                return
            world_size = len(comm_plan_args["tgt_devices"]) + 1

        elif mode == "sender":
            comm_plan_args = comm_plan[self.worker.rank]
            rank = 0
            world_size = len(comm_plan_args["tgt_devices"]) + 1

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # initialize
        src_pp_rank = comm_plan_args["src_pp_rank"]
        group_name = comm_plan_args["group_name"]
        master_addr = comm_plan_args["master_addr"]
        master_port = comm_plan_args["master_port"]

        collective.init_collective_group(
            world_size, rank, backend=backend, group_name=group_name,
            master_addr=master_addr, master_port=master_port
        )
        collective.allreduce(torch.zeros(1).to(current_platform.device_type), group_name=group_name)

        if model_update_name not in self.model_update_comm_plan:
            self.model_update_comm_plan[model_update_name] = {}
        self.model_update_comm_plan[model_update_name][src_pp_rank] = dict(
            rank=rank,
            world_size=world_size,
            src_pp_rank=src_pp_rank,
            group_name=group_name,
            comm_plan=comm_plan,
            comm_plan_args=comm_plan_args,
        )
        logger.info(f"warmup setup_collective_group: {group_name} rank: {rank} world_size: {world_size}")

    def setup_collective_group(self, model_update_name, comm_plan, backend=None, mode="receiver"):
        """
        单卡infer strategy可直接复用，多卡infer strategy需要自行管理
        """
        self._setup_collective_group_impl(model_update_name, comm_plan, backend, mode=mode)

    # offload/load 相关接口
    def load_states(self):
        raise NotImplementedError

    def offload_states(self, *args, **kwargs):
        raise NotImplementedError

    # 定义一些通用的分布式op，op计算依赖分布式实现
    # 算法开发Worker时，可在worker中自行实现计算逻辑，需要分布式的可在优化时集成入op库中
    def op_compute_log_probs(self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        logits: llm logits
        input_ids [[p, p, r, r, r, 0, 0]] p: prompt, r: response, 0: pad
        attention_mask(response_mask) [[0, 0, 1, 1, 1, 0, 0]]
        """
        labels: torch.Tensor = input_ids[:, 1:].clone()
        labels[attention_mask[:, 1:] == 0] = 0  # avoid invalid token id
        log_probs = log_probs_from_logits(logits[:, :-1], labels)
        log_probs = log_probs * attention_mask[:, 1:]
        return log_probs

    def op_compute_entropy(self, logits: torch.Tensor, attention_mask: torch.Tensor):
        entropy = entropy_from_logits(logits)
        entropy = entropy[:, :-1] * attention_mask[:, 1:]
        return entropy

    def op_compute_language_loss_from_logits(self, logits: torch.Tensor, targets: torch.Tensor):
        # shift
        logits = logits[..., :-1, :].contiguous()
        targets = targets[..., 1:].contiguous()
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=IGNORE_INDEX
        )
        mask = (targets != IGNORE_INDEX)
        valid_tokens = mask.sum()
        return loss, valid_tokens

    def op_compute_topk_logits(self, logits: torch.Tensor, topk: int = 0):
        """
        Compute top-k logits from the input logits tensor.

        Args:
            logits (torch.Tensor): Input logits tensor of shape [batch_size, ..., vocab_size].
            topk (int): Number of top elements to select. If 0, returns original logits.

        Returns:
            tuple:
                - If topk == 0: (original logits, empty tensor of shape [batch_size, 1])
                - Otherwise: (top-k logits, top-k indices) from torch.topk
        """
        if topk == 0:
            batch_size = logits.shape[0]
            return logits, torch.empty([batch_size, 1], device=logits.device)
        else:
            return torch.topk(logits, k=topk, dim=-1)

    def op_compute_topk_probs_and_indices(self, logits: torch.Tensor, topk: int = 0, target_vocab_size: int = None,
                                          kd_temperature: int = 1, teacher_temperature: int = 1):
        """
        Compute top-k probabilities, log probabilities, and indices from logits with temperature scaling.

        Args:
            logits (torch.Tensor): Input logits tensor of shape [batch_size, seq_len, vocab_size].
            topk (int): Number of top elements to select. If 0, uses all logits.
            target_vocab_size (int, optional): Target vocabulary size to truncate logits. Defaults to None.
            kd_temperature (int): Knowledge distillation temperature for scaling. Defaults to 1.
            teacher_temperature (int): Teacher model temperature for scaling. Defaults to 1.

        Returns:
            tuple: (topk_probs, topk_log_probs, topk_indices, topk_inf_mask)
                - topk_probs (torch.Tensor): Softmax probabilities of top-k logits.
                - topk_log_probs (torch.Tensor): Log softmax probabilities of top-k logits.
                - topk_indices (torch.Tensor): Indices of top-k elements.
                - topk_inf_mask (torch.Tensor): Boolean mask indicating infinite values in top-k logits.
        """
        if target_vocab_size is not None and logits.shape[-1] != target_vocab_size:
            logits = logits[:, :, : min(logits.shape[-1], target_vocab_size)]
        logits = logits / kd_temperature
        logits = logits / teacher_temperature
        topk_logits, topk_indices = self.op_compute_topk_logits(logits, topk)
        topk_inf_mask = topk_logits.isinf()
        topk_probs = F.softmax(topk_logits, dim=-1, dtype=torch.float32)
        topk_log_probs = F.log_softmax(topk_logits, dim=-1)
        return topk_probs, topk_log_probs, topk_indices, topk_inf_mask

    def op_compute_various_divergence(self, loss_callable, logits, teacher_topk_probs, teacher_topk_log_probs,
                                      teacher_topk_indices,
                                      teacher_topk_inf_mask, labels, attention_mask=None, reduction="mean"):
        """
        Compute divergence loss between student and teacher distributions with support for distributed training.

        This function handles both Tensor Parallel (TP) and Context Parallel (CP) sharded logits, gathering
        full vocabulary logits for the local sequence slice before computing the divergence loss.

        Args:
            loss_callable (callable): Loss function that computes divergence between student and teacher.
            logits (torch.Tensor): Student model logits, potentially TP/CP sharded. Shape: [batch_size, seq_len, vocab_size].
            teacher_topk_probs (torch.Tensor): Teacher's top-k probabilities.
            teacher_topk_log_probs (torch.Tensor): Teacher's top-k log probabilities.
            teacher_topk_indices (torch.Tensor): Indices of teacher's top-k elements.
            teacher_topk_inf_mask (torch.Tensor): Mask for infinite values in teacher's top-k logits.
            labels (torch.Tensor, optional): Ground truth labels with padding marked as IGNORE_INDEX. Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask where 0 indicates padding. Used if labels is None.
            reduction (str): Reduction method - "mean", "sum", or "none". Defaults to "mean".

        Returns:
            tuple: (loss, token_count)
                - loss (torch.Tensor): Computed loss value based on reduction method.
                    - "mean": averaged loss over valid tokens
                    - "sum": summed loss over valid tokens
                    - "none": per-token loss tensor
                - token_count (torch.Tensor): Number of valid (non-padded) tokens.

        Raises:
            ValueError: If reduction method is not one of "mean", "sum", or "none".

        Note:
            - Input `logits` are both TP (Tensor Parallel) and CP (Context Parallel) sharded.
            - The function gathers logits across TP to obtain full-vocab logits for the local CP sequence slice.
            - `labels` and `attention_mask` are provided as full tensors with global sequence length,
              then sliced to the local CP rank's sequence shard during loss computation.
        """
        # Gather full vocabulary logits using teacher's top-k indices
        full_logits = logits
        full_logits = self.op_compute_gather_by_teacher_indices(full_logits, teacher_topk_indices)

        # Compute per-token divergence loss
        kld_per_token = loss_callable(logits=full_logits, teacher_probs=teacher_topk_probs,
                                      teacher_log_probs=teacher_topk_log_probs,
                                      teacher_inf_mask=teacher_topk_inf_mask)

        # Create padding mask from labels or attention mask
        if labels is not None:
            pad_mask = labels.eq(IGNORE_INDEX)
        else:
            pad_mask = attention_mask.eq(0)
        token_count = (~pad_mask).sum().float()

        # Early return for 'none' reduction (per-token loss)
        if reduction == 'none':
            return kld_per_token, token_count

        # Apply mask and compute aggregated loss
        kld_masked = kld_per_token.masked_fill_(pad_mask, 0.0)
        loss_sum = kld_masked.sum()

        if reduction == "sum":
            return loss_sum, token_count
        elif reduction == "mean":
            return loss_sum / token_count.clamp(min=1.0), token_count
        else:
            raise ValueError(f"Unsupported reduction: {reduction}. Use 'mean', 'sum', or 'none'.")

    # Both megatron and deepspeed can output language loss directly.
    # This op is mainly for computing context-parallel loss.
    def op_compute_language_loss(self, losses: torch.Tensor, labels: torch.Tensor):
        loss_mask = (labels != IGNORE_INDEX).float()
        loss_mask = loss_mask.view(-1).float()
        losses = torch.sum(losses.view(-1) * loss_mask)
        return losses

    def op_compute_gather_by_teacher_indices(
            self,
            student_logits: torch.Tensor,
            teacher_indices: torch.Tensor
    ):
        """
        Gather Student logits according to Teacher's selected indices.
        Assumes:
            - `student_logits` is full vocabulary logits of shape
              [batch_size, local_seq_len, vocab_size]
            - `teacher_indices` is either:
                * None: return full logits (full-vocab mode)
                * LongTensor of shape [batch_size, local_seq_len, topk] or [batch_size, local_seq_len]
                  containing teacher’s selected vocab IDs.

        Returns:
            torch.Tensor:
                - If teacher_indices is None: same as student_logits.
                - If teacher_indices is provided: logits gathered at teacher’s indices,
                  shape [batch_size, local_seq_len, topk] or [batch_size, local_seq_len] depending on input.
        """
        # Full-vocab mode: return student logits directly
        if teacher_indices is None:
            return student_logits

        # Ensure indices are long dtype for gather
        if teacher_indices.dtype != torch.long:
            teacher_indices = teacher_indices.long()

        # If top-1 indices [B, S], unsqueeze to [B, S, 1] for gather
        if teacher_indices.dim() == 2:
            teacher_indices = teacher_indices.unsqueeze(-1)

        # Gather along vocab dimension (last dim)
        gathered_logits = torch.gather(student_logits, dim=-1, index=teacher_indices)
        return gathered_logits


class TrainStrategy(InferenceStrategy):
    def __init__(self, worker: "Worker"):
        super().__init__(worker)

        self.optimizer = None
        self.scheduler = None
        self.checkpoint_manager = CheckpointManager(checkpoint_config=self.worker_config.checkpoint_config)

    def setup_collective_group(self, model_update_name, comm_plan, backend=None, mode="sender"):
        self._setup_collective_group_impl(model_update_name, comm_plan, backend, mode=mode)


    def train_step(
        self,
        batch: DataProto,
        loss_func: Callable[[DataProto, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
    ):
        """
        完成一次batch训练, 包括带ga的mini_batch, 及带vp的micro_batch
        loss func接口定义:
            data: DataProto, 由train_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """
        raise NotImplementedError

    def model_update(self, *args, **kwargs):
        raise NotImplementedError
