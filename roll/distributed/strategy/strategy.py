from abc import ABC
from concurrent import futures
from typing import Callable, Dict, Tuple

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
        return loss

    def op_compute_logits(self, logits: torch.Tensor, tp_gather: bool = False, cp_gather: bool = False, topk: int = 0):
        """
            Post-process logits.

            If topk == 0 (full-vocab mode), optionally gather across TP/CP ranks
            using tp_gather and cp_gather flags.
            If topk > 0, return top-K values and indices for each position.

            Args:
                logits: [B, local_seq_len, local_vocab_size] tensor.
                tp_gather: Gather full vocab across tensor-parallel ranks (only if topk==0).
                cp_gather: Gather full sequence across context-parallel ranks (only if topk==0).
                topk: 0 for full vocab, >0 for top-K mode.

            Returns:
                (values, indices):
                    - full-vocab: (logits, dummy indices)
                    - top-K: (topk_values, topk_indices)"""
        if topk == 0:
            batch_size = logits.shape[0]
            return logits, torch.empty([batch_size, 1], device=logits.device)
        else:
            return torch.topk(logits, k=topk, dim=-1)

    def op_compute_prepare_cp_local_iterator(self, tensor: torch.Tensor, feature_name: str, micro_batch_size: int):
        """
        Prepare a microbatch iterator for a tensor that may require Context Parallel (CP) slicing.

        Notes:
            - If the input `tensor` is None, this function returns None.
            - The input tensor is assumed to have shape [global_batch, global_seq_len, ...],
              with batch dimension first and sequence dimension second.
            - When CP size > 1, the sequence dimension (dim=1) is sliced to the local CP rank
              using `_get_feature_on_this_cp_rank`.
            - After CP slicing (if any), the tensor is split along the batch dimension (dim=0)
              into microbatches of size `micro_batch_size`.

        Args:
            tensor (torch.Tensor or None): Full tensor before CP slicing. Can be None.
            feature_name (str): Identifier passed to `_get_feature_on_this_cp_rank` for CP slicing.
                                e.g., "teacher_logits" or "teacher_topk_indices".
            micro_batch_size (int): Number of samples per microbatch; splitting is done along dim=0.

        Returns:
            iterator or None: Iterator over microbatches with shape
                              [micro_batch_size, local_seq_len, ...].
                              Returns None if input `tensor` is None.
        """
        if tensor is None:
            return None

        # Microbatch split along batch dimension
        return iter(tensor.split(micro_batch_size, dim=0))

    def op_compute_various_divergence(self, loss_callable, logits, teacher_logits, teacher_topk_indices,
                                      labels, attention_mask=None):
        """
            Note:
                `logits` here are both TP (Tensor Parallel) and CP (Context Parallel) sharded.
                `logits` here are CP (Context Parallel) sharded.
                - We gather across TP to get full-vocab logits for the local CP sequence slice.
                `labels`, and `attention_mask` are provided as full tensors
                (global sequence length). These are then sliced down to the local CP rank's
                sequence shard before loss computation.
            """

        full_logits = logits
        full_logits = self.op_compute_gather_by_teacher_indices(full_logits, teacher_topk_indices)
        if teacher_logits.shape[-1] != full_logits.shape[-1]:
            teacher_logits = teacher_logits[:, :, : min(full_logits.shape[-1], teacher_logits.shape[-1])]
        loss = loss_callable(logits=full_logits, teacher_logits=teacher_logits, labels=labels, attention_mask=attention_mask)
        return loss

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
