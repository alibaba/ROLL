import inspect
import os
import threading
import time
from typing import Dict, Optional, Union, List

import ray
import torch
from codetiming import Timer
from tqdm import tqdm

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import (
    default_actor_model_provider,
    default_diffusion_module_provider,
    default_reward_model_provider,
    default_value_model_provider,
)
from roll.platforms import current_platform
from roll.utils.checkpoint_manager import download_model
from roll.utils.context_managers import state_offload_manger, log_gpu_memory_usage
from roll.utils.dynamic_batching import make_mini_batch_iter_for_dynamic_batching
from roll.utils.functionals import agg_loss, append_to_dict, compute_approx_kl, masked_mean, postprocess_generate, reduce_metrics
from roll.utils.offload_nccl import reload_process_groups
from roll.utils.offload_states import OffloadStateType


class ActorWorker(Worker):
    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy: TrainStrategy = None
        self._logprobs_cache = {}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)

        self.strategy = create_strategy(worker=self)

        if self.worker_config.model_args.model_type == "diffusion_module":
            self.strategy.initialize(model_provider=default_diffusion_module_provider)
        else:
            self.strategy.initialize(model_provider=default_actor_model_provider)

        self.tokenizer = self.strategy.tokenizer
        if self.pipeline_config.resume_from_checkpoint:
            load_dir = download_model(self.pipeline_config.resume_from_checkpoint)
            self.strategy.load_checkpoint(load_dir=load_dir, tag="checkpoint")
        self.logger.info(f"{self.worker_name} initialized")

        self.strategy.offload_states()

    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST)
    def train_step(self, data: DataProto):
        """
        return DataProto(meta_info={'metrics': metrics})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        self.logger.info(f"{self.worker_name} generate global step {global_step}")

        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/train_step",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params, OffloadStateType.other_params]},
        ):
            data = data.to(current_platform.device_type)
            data = self.strategy.get_data_input(data)
            per_device_train_batch_size = self.worker_config.training_args.per_device_train_batch_size
            backward_batch_size = (
                    per_device_train_batch_size * self.worker_config.training_args.gradient_accumulation_steps
            )
            if self.worker_config.use_dynamic_batching_in_train:
                # TODO: support `keep_mini_batch`, The number of mini_batch may be smaller than original size
                dataloader = make_mini_batch_iter_for_dynamic_batching(
                    data=data,
                    epochs=self.pipeline_config.ppo_epochs,
                    ga_steps=self.worker_config.training_args.gradient_accumulation_steps,
                )
            else:
                dataloader = data.make_iterator(
                    mini_batch_size=backward_batch_size,
                    epochs=self.pipeline_config.ppo_epochs,
                    seed=self.pipeline_config.seed,
                    dataloader_kwargs={"shuffle": True},
                )

            for batch_idx, backward_batch in tqdm(enumerate(dataloader),
                                                  desc=f"{self.worker_name} train global step {global_step}",
                                                  total=data.batch.batch_size[0] * self.pipeline_config.ppo_epochs // backward_batch_size):
                pg_metrics = self.strategy.train_step(batch=backward_batch, loss_func=self.loss_func)
                if self.worker_config.use_dynamic_batching_in_train or self.worker_config.use_sequence_packing:
                    pg_metrics = reduce_metrics(pg_metrics)
                append_to_dict(metrics, pg_metrics)

            metrics["actor/lr"] = self.strategy.scheduler.get_last_lr()[0]
            metrics["actor/backward_steps"] = data.batch.batch_size[0] * self.pipeline_config.ppo_epochs // backward_batch_size
            data.to("cpu")

        self._logprobs_cache.clear()
        output = DataProto(meta_info={"metrics": metrics})
        return output

    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST)
    def compute_log_probs(self, data: DataProto):
        """
        return DataProto.from_dict(tensors={'log_probs': output})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/compute_log_probs",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            data = self.strategy.get_data_input(data)
            data = data.to(current_platform.device_type)
            data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size

            with torch.no_grad():
                results: Dict[str, torch.Tensor] = self.strategy.forward_step(
                    batch=data, forward_func=self.forward_func_log_probs
                )
            if results is None:
                return DataProto(batch=None, meta_info={"metrics": metrics})
            output = DataProto.from_dict(tensors={"log_probs": results["log_probs"], "entropy": results["entropy"]})
            output = output.to("cpu")
            data.to("cpu")
        output.meta_info = {"metrics": metrics}
        return output

    def forward_func_log_probs(self, data: DataProto, output_tensor: torch.Tensor):
        """
        forward func 接口定义:
            data: DataProto, 由forward_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """
        log_probs = self.strategy.op_compute_log_probs(
            logits=output_tensor, input_ids=data.batch["input_ids"], attention_mask=data.batch["response_mask"]
        )
        entropy = self.strategy.op_compute_entropy(logits=output_tensor, attention_mask=data.batch["response_mask"])
        return torch.tensor(0., device=output_tensor.device), {"log_probs": log_probs.clone().detach(), "entropy": entropy.clone().detach()}

    def get_old_log_probs_with_cache(self, data: DataProto, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Get old_log_probs with intra-step caching when enable_old_logprobs_recompute == False.
        When caching is enabled, the first forward pass log_probs can be reused as old_log_probs
        since they are mathematically equivalent in on-policy settings.
        This method can be overridden by subclasses for custom caching behavior.

        Args:
            data: DataProto containing input data and sample_uuids
            log_probs: Current forward pass log_probs tensor

        Returns:
            old_log_probs tensor (detached, no gradients)
        """
        # Original computation path when caching is disabled
        if self.pipeline_config.enable_old_logprobs_recompute or "sample_uuid" not in data.non_tensor_batch:
            # When enable_old_logprobs_recompute=True, use the pre-computed old_log_probs from batch
            return data.batch["old_log_probs"]

        sample_uuids = data.non_tensor_batch["sample_uuid"]

        # Check first sample_uuid for efficiency - if it exists, all likely exist
        first_uuid = sample_uuids[0]
        if first_uuid in self._logprobs_cache:
            # All samples likely cached, retrieve all from cache
            cached_old_log_probs = []

            for sample_uuid in sample_uuids:
                cached_old_log_probs.append(self._logprobs_cache[sample_uuid])

            old_log_probs = torch.cat(cached_old_log_probs, dim=0).to(current_platform.device_type)
        else:
            # Cache miss - use current log_probs as old_log_probs (mathematically equivalent in on-policy)
            old_log_probs = log_probs.detach()
            if self.pipeline_config.ppo_epochs > 1:
                for i, sample_uuid in enumerate(sample_uuids):
                    self._logprobs_cache[sample_uuid] = old_log_probs[i : i + 1].cpu()

        return old_log_probs

    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        """
        loss func接口定义:
            data: DataProto, 由train_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """

        response_mask = data.batch["response_mask"][:, 1:].long()
        ref_log_probs = data.batch["ref_log_probs"]
        advantages = data.batch["advantages"]

        batch_num_tokens = data.meta_info['batch_num_tokens']
        global_valid_samples = data.meta_info['global_valid_samples']

        log_probs = self.strategy.op_compute_log_probs(
            logits=output_tensor, input_ids=data.batch["input_ids"], attention_mask=data.batch["response_mask"]
        )
        old_log_probs = self.get_old_log_probs_with_cache(data, log_probs)

        ratio = (log_probs - old_log_probs).exp()

        pg_clip_low = (
            self.pipeline_config.pg_clip_low
            if self.pipeline_config.use_pg_clip_range
            else self.pipeline_config.pg_clip
        )
        pg_clip_high = (
            self.pipeline_config.pg_clip_high
            if self.pipeline_config.use_pg_clip_range
            else self.pipeline_config.pg_clip
        )
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - pg_clip_low, 1 + pg_clip_high) * advantages
        pg_loss = -torch.min(surr1, surr2)
        if self.pipeline_config.dual_clip_loss:
            dual_clip_loss = -torch.max(-pg_loss, (1 + self.pipeline_config.pg_clip * 2) * advantages)
            pg_loss = torch.where(advantages < 0, dual_clip_loss, pg_loss)

        pg_loss = agg_loss(loss_mat=pg_loss, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.loss_agg_mode,
                           batch_num_tokens=batch_num_tokens['response_mask'],
                           global_valid_samples=global_valid_samples['response_mask'])

        kl_loss = compute_approx_kl(
            log_probs=log_probs, log_probs_base=ref_log_probs, action_mask=response_mask, kl_penalty="k3"
        )
        kl_loss = agg_loss(loss_mat=kl_loss, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.loss_agg_mode,
                           batch_num_tokens=batch_num_tokens['response_mask'],
                           global_valid_samples=global_valid_samples['response_mask'])

        approxkl = compute_approx_kl(
            log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="mse"
        )
        policykl = compute_approx_kl(
            log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="kl"
        )
        clipped_low = (ratio < 1 - pg_clip_low).float()
        clipped_high = (ratio > 1 + pg_clip_high).float()
        clipped = (clipped_low + clipped_high).float()

        if self.pipeline_config.use_kl_loss:
            total_loss = pg_loss + kl_loss * self.pipeline_config.kl_loss_coef
        else:
            total_loss = pg_loss
        if self.pipeline_config.entropy_loss_coef > 0:
            entropy = self.strategy.op_compute_entropy(
                logits=output_tensor, attention_mask=data.batch["response_mask"]
            )
            entropy_loss = agg_loss(
                loss_mat=entropy,
                loss_mask=response_mask,
                loss_agg_mode=self.pipeline_config.loss_agg_mode,
                batch_num_tokens=batch_num_tokens['response_mask'],
                global_valid_samples=global_valid_samples['response_mask'],
            )
            total_loss = total_loss - entropy_loss * self.pipeline_config.entropy_loss_coef

        pg_metrics = {
            "actor/ppo_ratio_high_clipfrac": clipped_high.mean().detach().item(),
            "actor/ppo_ratio_low_clipfrac": clipped_low.mean().detach().item(),
            "actor/ppo_ratio_clipfrac": clipped.mean().detach().item(),
            "actor/ratio_mean": masked_mean(ratio, response_mask, dim=-1).mean().detach().item(),
            "actor/ratio_max": torch.max(ratio * response_mask).detach().item(),
            "actor/ratio_min": torch.min(ratio * response_mask + (1 - response_mask) * 1e10).detach().item(),
            "actor/clipfrac": agg_loss(
                loss_mat=torch.lt(surr2, surr1).float(),
                loss_mask=response_mask,
                loss_agg_mode=self.pipeline_config.loss_agg_mode,
                batch_num_tokens=batch_num_tokens['response_mask'],
                global_valid_samples=global_valid_samples['response_mask'],
            )
            .detach()
            .item(),
            "actor/pg_loss": pg_loss.detach().item(),
            "actor/kl_loss": kl_loss.detach().item(),
            "actor/total_loss": total_loss.detach().item(),
            "actor/approxkl": agg_loss(
                loss_mat=approxkl, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.loss_agg_mode,
                batch_num_tokens=batch_num_tokens['response_mask'],
                global_valid_samples=global_valid_samples['response_mask'],
            )
            .detach()
            .item(),
            "actor/policykl": agg_loss(
                loss_mat=policykl, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.loss_agg_mode,
                batch_num_tokens=batch_num_tokens['response_mask'],
                global_valid_samples=global_valid_samples['response_mask'],
            )
            .detach()
            .item(),
        }

        return total_loss, pg_metrics

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def do_checkpoint(self, global_step, is_last_step=None):
        if self.worker_config.offload_nccl:
            reload_process_groups()
        with Timer("do_checkpoint") as total_timer:
            ckpt_id = f"checkpoint-{global_step}"

            # actor train是直接存在save dir目录下的，其他role是存在save_dir/cluster_name下的
            save_dir = os.path.join(self.pipeline_config.output_dir, self.worker_name, ckpt_id)
            self.logger.info(f"save checkpoint-{global_step} to {save_dir}")

            # could be passed for other strategy with kwargs
            exec_metrics: Dict = self.strategy.save_checkpoint(
                save_dir, global_step, ckpt_id, is_last_step=is_last_step
            )

        metrics = {
            f"time/{self.cluster_name}/do_checkpoint/total": total_timer.last,
        }
        metric_prefix = f"time/{self.cluster_name}/do_checkpoint"
        metrics.update({f"{metric_prefix}/{k}": v for k, v in exec_metrics.items()})
        output = DataProto(meta_info={"metrics": metrics})
        return output


class InferWorker(Worker):
    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def initialize(self, pipeline_config):
        super().initialize(pipeline_config)

        self.strategy = create_strategy(worker=self)

        await self.strategy.initialize(model_provider=default_actor_model_provider)
        self.tokenizer = self.strategy.tokenizer
        self.logger.info(f"{self.worker_name} initialized")

        await self.strategy.offload_states()

        # Platform must have been initialized when calling current_platform.reset_max_memory_allocated
        # with arguments (inside state_offload_manager). We explicitly init platform here because
        # current process is used as engine client when using vllm v1 engine, and
        # there is no chance to init platform context.
        current_platform.init()

    # TODO shigao 之前stop_server会返回一些offload_state_manager的metrics，现在删掉是否可行
    # def start_server
    # def stop_server

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def load_states(self, *args, **kwargs):
        await self.strategy.load_states(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def offload_states(self, *args, **kwargs):
        await self.strategy.offload_states(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def load_states_partial(self, target_dp_ranks: List[int]):
        """Load states for workers whose dp_rank is in target_dp_ranks."""

        # Log entry memory (only for TP rank 0 to reduce log spam)
        if self.rank_info.tp_rank == 0:
            log_gpu_memory_usage(
                head=f"Worker {self.rank} (DP {self.rank_info.dp_rank}) load_states_partial_entry",
                logger=self.logger,
                rank=None
            )

        assert getattr(self, "strategy", None) is not None, "worker has no strategy to load"
        if self.rank_info.dp_rank in target_dp_ranks:
            # AST: AST_PRECONDITION(is_model_in_gpu is False) - verify strategy offloaded before load
            is_loaded = self._get_strategy_load_state()

            assert is_loaded is False, (
                    f"Pre-condition: strategy must be offloaded before load_states_partial, "
                    f"got Worker {self.rank} (DP {self.rank_info.dp_rank}) is_model_in_gpu={is_loaded}"
                )

            await self.strategy.load_states()
            self.logger.info(f"Worker {self.rank} (DP {self.rank_info.dp_rank}) loaded states")
        else:
            self.logger.debug(f"Worker {self.rank} (DP {self.rank_info.dp_rank}) skipped load")


        # Log exit memory (only for TP rank 0 to reduce log spam)
        if self.rank_info.tp_rank == 0:
            log_gpu_memory_usage(
                head=f"Worker {self.rank} (DP {self.rank_info.dp_rank}) load_states_partial_exit",
                logger=self.logger,
                rank=None
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def offload_states_partial(self, target_dp_ranks: List[int]):
        """Offload states for workers whose dp_rank is in target_dp_ranks."""

        # Log entry memory (only for TP rank 0 to reduce log spam)
        if self.rank_info.tp_rank == 0:
            log_gpu_memory_usage(
                head=f"Worker {self.rank} (DP {self.rank_info.dp_rank}) offload_states_partial_entry",
                logger=self.logger,
                rank=None
            )

        assert getattr(self, "strategy", None) is not None, "worker has no strategy to offload"
        if self.rank_info.dp_rank in target_dp_ranks:
            # AST: AST_PRECONDITION(is_model_in_gpu is True) - verify strategy loaded before offload
            is_loaded = self._get_strategy_load_state()

            assert is_loaded is True, (
                    f"Pre-condition: strategy must be loaded before offload_states_partial, "
                    f"got Worker {self.rank} (DP {self.rank_info.dp_rank}) is_model_in_gpu={is_loaded}"
                )

            await self.strategy.offload_states()
            self.logger.info(f"Worker {self.rank} (DP {self.rank_info.dp_rank}) offloaded states")
        else:
            self.logger.debug(f"Worker {self.rank} (DP {self.rank_info.dp_rank}) skipped offload")


        # Log exit memory and verify offload success (only for TP rank 0 to reduce log spam)
        if self.rank_info.tp_rank == 0:
            log_gpu_memory_usage(
                head=f"Worker {self.rank} (DP {self.rank_info.dp_rank}) offload_states_partial_exit",
                logger=self.logger,
                rank=None
            )

            # Verify offloaded workers have near-zero GPU memory usage
            if self.rank_info.dp_rank in target_dp_ranks:
                import torch
                gpu_memory_gb = torch.cuda.memory_allocated() / 1024**3
                if gpu_memory_gb > 1.0:
                    raise RuntimeError(
                        f"GPU memory not properly offloaded for Worker {self.rank} (DP {self.rank_info.dp_rank}): "
                        f"{gpu_memory_gb:.2f} GB still allocated (expected < 1 GB after offload)"
                    )


    async def broadcast_parameter(self, *args, **kwargs):
        await self.strategy.broadcast_parameter(*args, **kwargs)

    async def setup_collective_group(self, *args, **kwargs):
        await self.strategy.setup_collective_group(*args, **kwargs)

    async def start_model_update(self, *args, **kwargs):
        raise NotImplementedError

    async def update_parameter_in_bucket(self, *args, **kwargs):
        await self.strategy.update_parameter_in_bucket(*args, **kwargs)

    async def add_lora(self, *args, **kwargs):
        await self.strategy.add_lora(*args, **kwargs)

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    async def generate(self, data: DataProto):
        """
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'old_log_probs': log_probs,
            },
            batch_size=batch_size)
        return DataProto(batch=batch)
        """
        if "generation_config" not in data.meta_info:
            generation_config = self.worker_config.generating_args.to_dict()
        else:
            generation_config = data.meta_info["generation_config"]

        generation_config["eos_token_id"] = [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
        generation_config["pad_token_id"] = self.tokenizer.pad_token_id

        global_step = data.meta_info.get("global_step", 0)
        self.logger.info(f"{self.worker_name} generate global step {global_step}")

        data = data.to("cuda")
        data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size

        is_offload_states = data.meta_info.get("is_offload_states", True)
        # state_offload_manager does not support async context
        await self.strategy.load_states()
        try:
            output = await self.strategy.generate(batch=data, generation_config=generation_config)
            output = postprocess_generate(
                prompts=data,
                output=output,
                num_return_sequences=generation_config["num_return_sequences"],
                sequence_length=self.pipeline_config.sequence_length,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            data.to("cpu")
            output = output.to("cpu")
        finally:
            if is_offload_states:
                await self.strategy.offload_states()

        return output

    async def generate_request(self, data: DataProto):
        """
        data req meta_info里需要包含: request_id: str
        generation_config, 按request设置

        on request cancellation: return DataProto with finish_reasons 'abort'
        """
        is_num_return_sequences_expand = data.meta_info.get("is_num_return_sequences_expand", False)
        if "generation_config" not in data.meta_info:
            generation_config = self.worker_config.generating_args.to_dict()
            if is_num_return_sequences_expand:
                self.worker_config.generating_args.num_return_sequences = 1
                generation_config["num_return_sequences"] = 1
                self.logger.info(f"is_num_return_sequences_expand is True, set num_return_sequences to 1.")
        else:
            generation_config = data.meta_info["generation_config"]
        generation_config["eos_token_id"] = [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
        generation_config["pad_token_id"] = self.tokenizer.pad_token_id
        data.meta_info["generation_config"] = generation_config
        data = await self.strategy.generate_request(data=data)
        data.meta_info["eos_token_id"] = self.tokenizer.eos_token_id
        data.meta_info["pad_token_id"] = self.tokenizer.pad_token_id
        return data

    async def abort_requests(self, request_ids):
        await self.strategy.abort_requests(request_ids)


class CriticWorker(Worker):

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)

        self.strategy = create_strategy(worker=self)

        self.strategy.initialize(model_provider=default_value_model_provider)
        self.tokenizer = self.strategy.tokenizer

        if self.pipeline_config.resume_from_checkpoint:
            load_dir = os.path.join(download_model(self.pipeline_config.resume_from_checkpoint), self.cluster_name)
            self.strategy.load_checkpoint(load_dir=load_dir, tag="checkpoint")

        self.logger.info(f"{self.worker_name} initialized")

        self.strategy.offload_states()

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def compute_values(self, data: DataProto):
        """
        return DataProto.from_dict(tensors={'values': values})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/compute_values",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            data = data.to(current_platform.device_type)
            data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size
            with torch.no_grad():
                results: Dict[str, torch.Tensor] = self.strategy.forward_step(
                    batch=data, forward_func=self.forward_func_values
                )

            output = DataProto.from_dict(tensors={"values": results["values"]})
            data.to("cpu")
            output = output.to("cpu")

        output.meta_info = {"metrics": metrics}
        return output

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def train_step(self, data: DataProto):
        """
        return DataProto(meta_info={'metrics': metrics})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/train_step",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params, OffloadStateType.other_params]},
        ):
            data = data.to(current_platform.device_type)
            per_device_train_batch_size = self.worker_config.training_args.per_device_train_batch_size
            backward_batch_size = (
                per_device_train_batch_size * self.worker_config.training_args.gradient_accumulation_steps
            )

            dataloader = data.make_iterator(
                mini_batch_size=backward_batch_size,
                epochs=1,
                seed=self.pipeline_config.seed,
                dataloader_kwargs={"shuffle": True},
            )

            for batch_idx, data in tqdm(
                enumerate(dataloader),
                desc=f"{self.worker_name} train global step {global_step}",
                total=data.batch.batch_size[0] * self.pipeline_config.ppo_epochs // backward_batch_size,
            ):
                vf_metrics = self.strategy.train_step(batch=data, loss_func=self.loss_func)
                append_to_dict(metrics, vf_metrics)

            data.to("cpu")
            metrics["critic/lr"] = self.strategy.scheduler.get_last_lr()[0]

        output = DataProto(meta_info={"metrics": metrics}).to("cpu")

        return output

    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        """
        loss func接口定义:
            data: DataProto, 由train_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """
        response_mask = data.batch["response_mask"][:, 1:]
        old_values = data.batch["values"]
        returns = data.batch["returns"]

        values, _ = self.forward_func_values(data=data, output_tensor=output_tensor)

        if self.pipeline_config.value_clip is not None:
            values_clipped = torch.clip(
                values,
                old_values - self.pipeline_config.value_clip,
                old_values + self.pipeline_config.value_clip,
            )
            surr1 = (values - returns) ** 2
            surr2 = (values_clipped - returns) ** 2
            vf_clipfrac = masked_mean(torch.gt(surr2, surr1).float(), response_mask, dim=-1).mean()
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2
            vf_clipfrac = masked_mean(loss, response_mask, dim=-1).mean()

        vf_loss = 0.5 * masked_mean(loss, response_mask, dim=-1).mean()

        vf_metrics = {
            "critic/loss": vf_loss.detach().item(),
            "critic/value": (masked_mean(old_values, response_mask, dim=-1)).mean().detach().item(),
            "critic/vpred": (masked_mean(values, response_mask, dim=-1)).mean().detach().item(),
            "critic/clipfrac": vf_clipfrac.detach().item(),
            "critic/error": masked_mean((values - returns) ** 2, response_mask, dim=-1).mean().detach().item(),
        }

        return vf_loss, vf_metrics

    def forward_func_values(self, data: DataProto, output_tensor: torch.Tensor):
        values = output_tensor[:, :-1]
        values = values.squeeze(dim=-1)
        return values, {"values": values.clone().detach()}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def do_checkpoint(self, global_step, is_last_step=None):
        with Timer("do_checkpoint") as total_timer:
            ckpt_id = f"checkpoint-{global_step}"
            save_dir = os.path.join(self.pipeline_config.output_dir, self.worker_name, ckpt_id, self.cluster_name)
            critic_save_dir = os.path.join(self.pipeline_config.output_dir, self.worker_name, ckpt_id)
            self.logger.info(f"save checkpoint-{global_step} to {save_dir}")
            exec_metrics: Dict = self.strategy.save_checkpoint(
                save_dir, global_step, ckpt_id, local_state_path=critic_save_dir, is_last_step=is_last_step
            )

        metrics = {
            f"time/{self.cluster_name}/do_checkpoint/total": total_timer.last,
        }
        metric_prefix = f"time/{self.cluster_name}/do_checkpoint"
        metrics.update({f"{metric_prefix}/{k}": v for k, v in exec_metrics.items()})
        output = DataProto(meta_info={"metrics": metrics})
        return output


class RewardWorker(Worker):
    """
    Reward Model 使用 AutoModelForSequenceClassification 协议
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)

        self.strategy = create_strategy(worker=self)

        self.strategy.initialize(model_provider=default_reward_model_provider)
        self.tokenizer = self.strategy.tokenizer

        self.logger.info(f"{self.worker_name} initialized")
        self.strategy.offload_states()

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        """
        return DataProto.from_dict(tensors={'rewards': rewards})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/compute_rewards",
            is_offload_states=is_offload_states,
        ):
            data = data.to(current_platform.device_type)

            # TODO: _switch_chat_template, 异构reward model

            data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size
            with torch.no_grad():
                results: Dict[str, torch.Tensor] = self.strategy.forward_step(
                    batch=data, forward_func=self.forward_func_values
                )
            token_level_rewards = results["values"]  # (bsz, input_ids.shape[1]-1)
            input_ids = data.batch["input_ids"][:, 1:]
            seq_lengths = torch.eq(input_ids, self.tokenizer.pad_token_id).int().argmax(-1) - 1
            seq_lengths = (seq_lengths % input_ids.shape[-1]).to(token_level_rewards.device)
            response_level_rewards = token_level_rewards[
                torch.arange(seq_lengths.shape[0], device=token_level_rewards.device), seq_lengths
            ]

            output = DataProto.from_dict(
                tensors={"token_level_rewards": token_level_rewards, "response_level_rewards": response_level_rewards}
            )

            data.to("cpu")
            output = output.to("cpu")

        output.meta_info = {"metrics": metrics}
        return output

    def forward_func_values(self, data: DataProto, output_tensor: torch.Tensor):
        values = output_tensor[:, 1:]
        values = values.squeeze(dim=-1)
        return values, {"values": values.clone().detach()}
