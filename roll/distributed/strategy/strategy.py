from abc import ABC
from concurrent import futures
from typing import Callable, Dict, Tuple

import torch

from roll.distributed.scheduler.protocol import DataProto
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

    def setup_collective_group(self, model_update_name, comm_plan, backend="nccl"):
        """
        单卡infer strategy可直接复用，多卡infer strategy需要自行管理
        """
        rank, comm_plan_args = get_dist_info_from_comm_plan(
            comm_plan, rank_in_cluster=self.worker.rank, rank_in_worker=0
        )
        if rank is None:
            logger.info(f"no comm_plan found for rank {self.worker.rank}/{0}")
            return
        group_name = comm_plan_args["group_name"]
        master_addr = comm_plan_args["master_addr"]
        master_port = comm_plan_args["master_port"]
        world_size = len(comm_plan_args["tgt_devices"]) + 1
        src_pp_rank = comm_plan_args["src_pp_rank"]
        logger.info(f"{group_name} rank: {rank} world_size: {world_size}, {comm_plan_args}")
        collective.init_collective_group(
            world_size, rank, backend=backend, group_name=group_name, master_addr=master_addr, master_port=master_port
        )
        # A small all_reduce for warmup.
        collective.allreduce(torch.zeros(1).cuda(), group_name=group_name)
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

    def op_compute_logits(self, logits: torch.Tensor):
        return logits

    # Both megatron and deepspeed can output language loss directly.
    # This op is mainly for computing context-parallel loss. 
    def op_compute_language_loss(self, losses: torch.Tensor, labels: torch.Tensor):
        loss_mask = (labels != IGNORE_INDEX).float()
        loss_mask = loss_mask.view(-1).float()
        losses = torch.sum(losses.view(-1) * loss_mask)
        return losses

class TrainStrategy(InferenceStrategy):
    def __init__(self, worker: "Worker"):
        super().__init__(worker)

        self.optimizer = None
        self.scheduler = None
        self.checkpoint_manager = CheckpointManager(checkpoint_config=self.worker_config.checkpoint_config)

    def setup_collective_group(self, model_update_name, comm_plan, backend="nccl"):
        comm_plan_args = comm_plan[self.worker.rank]
        group_name = comm_plan_args["group_name"]
        master_addr = comm_plan_args["master_addr"]
        master_port = comm_plan_args["master_port"]
        tgt_devices = comm_plan_args["tgt_devices"]
        src_pp_rank = comm_plan_args["src_pp_rank"]
        rank = 0
        world_size = len(tgt_devices) + 1
        collective.init_collective_group(
            world_size, rank, backend=backend, group_name=group_name, master_addr=master_addr, master_port=master_port
        )
        # A small all_reduce for warmup.
        collective.allreduce(torch.zeros(1).cuda(), group_name=group_name)
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
