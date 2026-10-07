from __future__ import annotations

import os
import random
from functools import partial
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing, mpu, tensor_parallel
from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelLoadStrategyWrapper
from megatron.core.transformer.moe.moe_utils import (
    clear_aux_losses_tracker,
    get_moe_layer_wise_logging_tracker,
    reduce_aux_losses_tracker_across_ranks,
)
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

from mcore_adapter.checkpointing import get_checkpoint_dir, load_state_dict_from_checkpoint
from roll.distributed.scheduler.protocol import DataProto
from roll.platforms import current_platform
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
from roll.third_party.megatron.offload_states_patch import cleanup_ddp_buffers
from roll.utils.constants import DIST_OPTIMIZER_DIR, OPTIMIZER_NAME, RNG_STATE_DIR, SCHEDULER_NAME
from roll.utils.dynamic_batching import make_micro_batch_iter_for_dynamic_batching
from roll.utils.functionals import append_to_dict
from roll.utils.sequence_packing import make_micro_batch_iter_for_sequence_packing


def prepare_train_microbatches(strategy, batch: DataProto) -> tuple[list[DataProto], int, int]:
    worker_config = strategy.worker_config
    if worker_config.use_dynamic_batching_in_train:
        micro_batches_list = list(make_micro_batch_iter_for_dynamic_batching(batch))
        num_microbatches = batch.meta_info["num_micro_batchs"]
        mini_batch_size = 1
    elif strategy.use_sequence_packing:
        vp_size = worker_config.strategy_args.strategy_config["virtual_pipeline_model_parallel_size"] \
            if "virtual_pipeline_model_parallel_size" in worker_config.strategy_args.strategy_config else 1
        micro_batches_list = list(
            make_micro_batch_iter_for_sequence_packing(
                batch,
                tp_size=strategy.worker.rank_info.tp_size,
                cp_size=strategy.worker.rank_info.cp_size,
                vp_size=vp_size,
                is_train=True,
                dp_group=mpu.get_data_parallel_group(with_context_parallel=True),
                micro_batch_size=worker_config.training_args.per_device_train_batch_size,
                config=worker_config.sequence_packing_args,
            )
        )
        num_microbatches = micro_batches_list[0].meta_info["num_micro_batchs"]
        mini_batch_size = 1
    else:
        mini_batch_size = worker_config.training_args.per_device_train_batch_size
        num_microbatches = batch.batch.batch_size[0] // worker_config.training_args.per_device_train_batch_size
        if not batch.meta_info.get("skip_microbatch_count_check", False):
            assert (
                num_microbatches == strategy.megatron_train_args.gradient_accumulation_steps
            ), f"num_microbatches={num_microbatches} gradient_accumulation_steps={strategy.megatron_train_args.gradient_accumulation_steps}"
        micro_batches_list = batch.chunk(chunks=num_microbatches)

    for micro_batch in micro_batches_list:
        micro_batch.meta_info["loss_scale"] = num_microbatches * mpu.get_data_parallel_world_size()
        micro_batch.meta_info["micro_batch_size"] = micro_batch.batch.batch_size[0]

    return micro_batches_list, num_microbatches, mini_batch_size


def run_forward_backward_accumulate(
    strategy,
    batch: DataProto,
    loss_func: Callable[[DataProto, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
) -> List[Dict[str, torch.Tensor]]:
    strategy.model.train()

    if strategy.enable_router_replay:
        assert "routed_experts" in batch.batch
        RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

    batch.meta_info["batch_num_tokens"] = strategy._get_batch_num_tokens(
        batch, dp_group=mpu.get_data_parallel_group()
    )
    batch.meta_info["global_valid_samples"] = strategy._get_global_valid_samples(
        batch, dp_group=mpu.get_data_parallel_group()
    )

    micro_batches_list, num_microbatches, mini_batch_size = prepare_train_microbatches(strategy, batch)
    data_iterator = [iter(micro_batches_list) for _ in range(len(strategy.model))]

    metrics_tensors: List[Dict[str, torch.Tensor]] = strategy.forward_backward_func(
        forward_step_func=partial(strategy.inner_forward_step, loss_func),
        data_iterator=data_iterator,
        model=strategy.model.get_models(),
        num_microbatches=num_microbatches,
        seq_length=strategy.seq_length,
        micro_batch_size=mini_batch_size,
        forward_only=False,
    )

    if strategy.enable_router_replay:
        RouterReplay.clear_global_router_replay_action()
        RouterReplay.clear_global_indices()

    if micro_batches_list and "tinker_example_losses" in micro_batches_list[0].batch.keys():
        batch.batch["tinker_example_losses"] = torch.cat(
            [micro_batch.batch["tinker_example_losses"] for micro_batch in micro_batches_list],
            dim=0,
        )

    return metrics_tensors


def collect_train_metrics(strategy, metrics_tensors: List[Dict[str, torch.Tensor]]) -> dict:
    metrics = {}
    for mini_metrics in metrics_tensors:
        append_to_dict(metrics, mini_metrics)

    if strategy.model.config.num_moe_experts is not None and strategy.model.config.num_moe_experts > 1:
        reduce_aux_losses_tracker_across_ranks()
        tracker = get_moe_layer_wise_logging_tracker()
        loss_scale = 1 / strategy.megatron_train_args.gradient_accumulation_steps
        moe_losses = {
            strategy.worker_config.name + "/" + k: (v["values"].float() * loss_scale).mean().item()
            for k, v in tracker.items()
        }
        clear_aux_losses_tracker()
        metrics.update(moe_losses)

    if strategy.model.config.mtp_num_layers is not None and strategy.model.config.mtp_num_layers > 0:
        mtp_total_loss_dict = {}
        MTPLossLoggingHelper.reduce_loss_in_tracker()
        tracker = MTPLossLoggingHelper.tracker
        if "values" in tracker:
            loss_scale = 1 / strategy.megatron_train_args.gradient_accumulation_steps
            mtp_losses = tracker["values"] * loss_scale
            mtp_num_layers = mtp_losses.shape[0]
            for i in range(mtp_num_layers):
                name = strategy.worker_config.name + "/" + f"mtp_{i + 1} loss"
                mtp_total_loss_dict[name] = mtp_losses[i].item()
            MTPLossLoggingHelper.clean_loss_in_tracker()
            metrics.update(mtp_total_loss_dict)
    return metrics


def set_optimizer_hparams(strategy, adam_params: Dict[str, float]) -> None:
    for group in strategy.optimizer.param_groups:
        group["lr"] = adam_params["learning_rate"]
        group["betas"] = (adam_params["beta1"], adam_params["beta2"])
        group["eps"] = adam_params["eps"]
        group["weight_decay"] = adam_params["weight_decay"]


def post_optim_step_cleanup(strategy) -> None:
    # Match current ROLL optimizer cleanup so offload cannot restore stale weights.
    cleanup_ddp_buffers(strategy.optimizer, backend=strategy._get_offload_backend())
    for model in strategy.model:
        for bucket_group in model.bucket_groups + model.expert_parallel_bucket_groups:
            if hasattr(bucket_group, "per_param_grad_ready_counts") and hasattr(bucket_group, "is_first_batch"):
                if bucket_group.is_first_batch and bucket_group.per_param_grad_ready_counts:
                    for parameter in bucket_group.params:
                        bucket_group.per_param_grad_ready_counts.setdefault(parameter, 1)
        model.zero_grad_buffer()
        for bucket_group in model.bucket_groups + model.expert_parallel_bucket_groups:
            if hasattr(bucket_group, "cached_param_buffer_shard_list"):
                bucket_group.cached_param_buffer_shard_list = [None] * len(bucket_group.buckets)
            if hasattr(bucket_group, "cached_grad_buffer_shard_list"):
                bucket_group.cached_grad_buffer_shard_list = [None] * len(bucket_group.buckets)
    strategy.optimizer.zero_grad()


def apply_optim_step(strategy, adam_params: Dict[str, float]) -> dict:
    # Optional bounded probe of real model weights, not optimizer state. This
    # avoids multi-gigabyte checkpoints for the public single-question smoke run.
    probes = []
    if os.environ.get("TINKER_PARAMETER_PROBE") == "1":
        with torch.no_grad():
            for model in strategy.model:
                for parameter in model.parameters():
                    if not parameter.requires_grad or parameter.numel() == 0:
                        continue
                    flat = parameter.detach().reshape(-1)
                    stride = max(1, flat.numel() // 32)
                    indices = torch.arange(0, flat.numel(), stride, device=flat.device)[:32]
                    probes.append((parameter, indices, flat[indices].float().clone()))
                    if len(probes) >= 32:
                        break
                if len(probes) >= 32:
                    break
    set_optimizer_hparams(strategy, adam_params)
    update_successful, grad_norm, _num_zeros_in_grad = strategy.optimizer.step()
    if not update_successful:
        raise NotImplementedError("megatron optimizer step failed!")

    probe_metrics = {}
    if probes:
        with torch.no_grad():
            deltas = torch.cat([
                parameter.detach().reshape(-1)[indices].float() - before
                for parameter, indices, before in probes
            ])
            probe_metrics = {
                f"{strategy.worker_config.name}/parameter_probe_count": float(deltas.numel()),
                f"{strategy.worker_config.name}/parameter_probe_changed": float((deltas != 0).sum()),
                f"{strategy.worker_config.name}/parameter_probe_max_delta": float(deltas.abs().max()),
                f"{strategy.worker_config.name}/parameter_probe_l2_delta": float(deltas.norm()),
            }
    post_optim_step_cleanup(strategy)
    return {
        **probe_metrics,
        f"{strategy.worker_config.name}/grad_norm": grad_norm,
        f"{strategy.worker_config.name}/lr": adam_params["learning_rate"],
    }


def forward_backward_accumulate(
    strategy,
    batch: DataProto,
    loss_func: Callable[[DataProto, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
) -> dict:
    metrics_tensors = run_forward_backward_accumulate(strategy, batch, loss_func)
    return collect_train_metrics(strategy, metrics_tensors)


def load_checkpoint_for_tinker(strategy, load_dir: str, load_optimizer: bool = True) -> None:
    if load_optimizer:
        optimizer_checkpoint = get_checkpoint_dir(
            load_dir, iteration=1, return_base_dir=strategy.megatron_train_args.use_distributed_optimizer
        )
        if strategy.megatron_train_args.use_distributed_optimizer:
            optimizer_checkpoint = os.path.join(optimizer_checkpoint, DIST_OPTIMIZER_DIR)

        if strategy.megatron_train_args.use_distributed_optimizer:
            model_shared_state_dict = strategy.model.sharded_state_dict()
            sharded_state_dict = strategy.optimizer.sharded_state_dict(
                model_shared_state_dict, is_loading=True, metadata=strategy.ckpt_sharding_metadata
            )
            load_strategy = dist_checkpointing.serialization.get_default_load_sharded_strategy(optimizer_checkpoint)
            load_strategy = FullyParallelLoadStrategyWrapper(
                load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
            )
            state_dict = dist_checkpointing.load(sharded_state_dict, optimizer_checkpoint, load_strategy)
        else:
            state_dict = torch.load(
                os.path.join(optimizer_checkpoint, OPTIMIZER_NAME),
                map_location=strategy.megatron_train_args.device,
                weights_only=False,
            )
        strategy.optimizer.load_state_dict(state_dict)
        strategy.scheduler.load_state_dict(torch.load(os.path.join(load_dir, SCHEDULER_NAME)))

    state_dict = load_state_dict_from_checkpoint(load_dir)
    assert state_dict is not None, "No model state_dict found in checkpoint."
    strategy.model.models = strategy.models_unwrapped
    strategy.model.load_state_dict(state_dict)
    strategy.model.models = strategy.models_wrapped

    if load_optimizer:
        rng_file = os.path.join(load_dir, RNG_STATE_DIR, f"rng_state_{dist.get_rank()}.pth")
        if os.path.exists(rng_file):
            checkpoint_rng_state = torch.load(rng_file, weights_only=False)
            random.setstate(checkpoint_rng_state["random_rng_state"])
            np.random.set_state(checkpoint_rng_state["np_rng_state"])
            torch.set_rng_state(checkpoint_rng_state["torch_rng_state"])
            current_platform.set_rng_state(checkpoint_rng_state["cuda_rng_state"])
            if not checkpoint_rng_state["rng_tracker_states"]:
                raise KeyError
            tensor_parallel.get_cuda_rng_tracker().set_states(checkpoint_rng_state["rng_tracker_states"])
