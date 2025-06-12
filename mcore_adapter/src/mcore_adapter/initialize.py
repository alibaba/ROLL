import os
import random

import numpy as np
import torch
from megatron.core import mpu, tensor_parallel

from .training_args import TrainingArguments
from .utils import get_logger


logger = get_logger(__name__)


def is_distribute_initialized():
    return mpu.model_parallel_is_initialized()


def _set_random_seed(seed_):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        seed = seed_  # some dist dataloaders may require the same seed for all ranks
        # # Ensure that different pipeline MP stages get different seeds.
        # seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
        # # Ensure different data parallel ranks get different seeds
        # if data_parallel_random_init:
        #     seed = seed + (10 * mpu.get_data_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            tensor_parallel.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed))


def initialize_megatron(args: "TrainingArguments"):
    if not is_distribute_initialized():
        _initialize_distributed(args)
    _set_random_seed(args.seed)


def _initialize_distributed(args: "TrainingArguments"):
    """Initialize torch.distributed and core model parallel."""
    logger.info(f"Initializing mpu on device {args.device}")
    if not torch.distributed.is_initialized():
        # Manually set the device ids.
        torch.cuda.set_device(args.device)
        # Call the init process
        torch.distributed.init_process_group(
            backend=args.ddp_backend or "nccl",
            rank=int(os.getenv("RANK", "0")),
            world_size=int(os.getenv("WORLD_SIZE", "1")),
            timeout=args.ddp_timeout_delta,
        )
    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if mpu.model_parallel_is_initialized():
        logger.info("model parallel is already initialized")
    else:
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
            context_parallel_size=args.context_parallel_size if args.context_parallel_size is not None else 1,
            expert_model_parallel_size=args.expert_model_parallel_size,
            distributed_timeout_minutes=args.ddp_timeout_delta.total_seconds() // 60,
        )
        logger.info(f"initialized tensor model parallel with size {mpu.get_tensor_model_parallel_world_size()}")
        logger.info(f"initialized pipeline model parallel with size {mpu.get_pipeline_model_parallel_world_size()}")
