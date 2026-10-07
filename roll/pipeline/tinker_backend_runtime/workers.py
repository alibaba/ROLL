from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.base_worker import ActorWorker, InferWorker
from roll.pipeline.tinker_backend_runtime.vllm_primitives import compute_prompt_logprobs_with_vllm_strategy
from roll.platforms import current_platform
from roll.utils.context_managers import state_offload_manger
from roll.utils.functionals import reduce_metrics
from roll.utils.offload_states import OffloadStateType


class TinkerActorWorker(ActorWorker):
    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, prefetch=True)
    def forward_tinker(self, data: DataProto):
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/forward_tinker",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            # DP_MP_COMPUTE sends real input to every TP rank. Current ROLL
            # receives the dispatched DataProto directly. Older
            # ROLL strategies optionally supplied an input conversion.
            converter = getattr(self.strategy, "get_data_input", None)
            if converter is not None:
                data = converter(data)
            data = data.to(current_platform.device_type)
            data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size
            data.meta_info.setdefault("loss_mask_keys", [])

            with torch.no_grad():
                results: Dict[str, torch.Tensor] = self.strategy.forward_step(
                    batch=data, forward_func=self.forward_func_tinker_log_probs
                )
            if results is None:
                data.to("cpu")
                return DataProto(batch=None, meta_info={"metrics": metrics})
            output = DataProto.from_dict(
                tensors={"tinker_target_logprobs": results["target_logprobs"]},
                non_tensors={
                    "tinker_prompt_lengths": list(data.non_tensor_batch["tinker_prompt_lengths"]),
                    "tinker_target_lengths": list(data.non_tensor_batch["tinker_target_lengths"]),
                },
                meta_info={"metrics": metrics},
            )
            output = output.to("cpu")
            data.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, prefetch=True)
    def forward_backward_accumulate_tinker(self, data: DataProto):
        from roll.pipeline.tinker_backend_runtime import megatron_primitives

        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/forward_backward_accumulate_tinker",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params, OffloadStateType.other_params]},
        ):
            # DP_MP_COMPUTE sends real input to every TP rank. Current ROLL
            # receives the dispatched DataProto directly. Older
            # ROLL strategies optionally supplied an input conversion.
            converter = getattr(self.strategy, "get_data_input", None)
            if converter is not None:
                data = converter(data)
            data = data.to(current_platform.device_type)
            data.meta_info.setdefault("loss_mask_keys", [])
            data.meta_info["skip_microbatch_count_check"] = True
            strategy_metrics = megatron_primitives.forward_backward_accumulate(
                strategy=self.strategy,
                batch=data,
                loss_func=self.tinker_ce_loss_func,
            )
            if self.worker_config.use_dynamic_batching_in_train or self.worker_config.use_sequence_packing:
                strategy_metrics = reduce_metrics(strategy_metrics)
            metrics.update(strategy_metrics)
            if not (
                self.rank_info.tp_rank == 0
                and self.rank_info.cp_rank == 0
                and self.rank_info.is_pipeline_last_stage
            ):
                data.to("cpu")
                return DataProto(batch=None, meta_info={"metrics": metrics})
            output = DataProto.from_dict(
                tensors={"tinker_example_losses": data.batch["tinker_example_losses"].detach().cpu()},
                non_tensors={
                    "tinker_prompt_lengths": list(data.non_tensor_batch["tinker_prompt_lengths"]),
                    "tinker_target_lengths": list(data.non_tensor_batch["tinker_target_lengths"]),
                },
                meta_info={"metrics": metrics},
            )
            data.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def apply_optim_step_tinker(self, adam_params: Dict[str, float]):
        from roll.pipeline.tinker_backend_runtime import megatron_primitives

        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/apply_optim_step_tinker",
            is_offload_states=True,
            load_kwargs={
                "include": [
                    OffloadStateType.model_params,
                    OffloadStateType.other_params,
                    OffloadStateType.optimizer_states,
                ]
            },
        ):
            step_metrics = megatron_primitives.apply_optim_step(self.strategy, adam_params)
            metrics.update(step_metrics)
        return DataProto(meta_info={"metrics": metrics}).to("cpu")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint_tinker(self, checkpoint_dir: str, checkpoint_id: str):
        checkpoint_manager = getattr(self.strategy, "checkpoint_manager", None)
        original_uploader = getattr(checkpoint_manager, "uploader", None) if checkpoint_manager else None
        try:
            if checkpoint_manager is not None:
                checkpoint_manager.uploader = None
            exec_metrics: Dict = self.strategy.save_checkpoint(
                save_dir=checkpoint_dir,
                global_step=0,
                ckpt_id=checkpoint_id,
                local_state_path=checkpoint_dir,
                is_last_step=True,
            )
        finally:
            if checkpoint_manager is not None:
                checkpoint_manager.uploader = original_uploader
        return DataProto(meta_info={"metrics": exec_metrics}).to("cpu")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint_tinker(self, checkpoint_dir: str, load_optimizer: bool = True):
        from roll.pipeline.tinker_backend_runtime import megatron_primitives

        megatron_primitives.load_checkpoint_for_tinker(
            self.strategy,
            load_dir=checkpoint_dir,
            load_optimizer=load_optimizer,
        )
        return DataProto(meta_info={"metrics": {"tinker/load_optimizer": float(load_optimizer)}}).to("cpu")

    def forward_func_tinker_log_probs(self, data: DataProto, output_tensor: torch.Tensor):
        target_logprobs = self.strategy.op_compute_log_probs(
            logits=output_tensor,
            input_ids=data.batch["input_ids"],
            attention_mask=data.batch["response_mask"],
        )
        expected_shape = data.batch["tinker_weights"].shape
        if target_logprobs.shape != expected_shape:
            raise ValueError(
                f"Tinker forward log_probs shape {tuple(target_logprobs.shape)} does not match "
                f"weights shape {tuple(expected_shape)}"
            )
        return torch.tensor(0.0, device=output_tensor.device), {
            "target_logprobs": target_logprobs.clone().detach(),
        }

    def tinker_ce_loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        log_probs = self.strategy.op_compute_log_probs(
            logits=output_tensor,
            input_ids=data.batch["input_ids"],
            attention_mask=data.batch["response_mask"],
        )
        weights = data.batch["tinker_weights"]
        if weights.shape != log_probs.shape:
            raise ValueError(
                f"Tinker CE weights shape {tuple(weights.shape)} does not match "
                f"log_probs shape {tuple(log_probs.shape)}"
            )

        per_token_loss = -log_probs * weights
        per_example_loss = per_token_loss.sum(dim=-1)
        data.batch["tinker_example_losses"] = per_example_loss.detach()
        total_loss = per_example_loss.sum()
        return total_loss, {"tinker_ce_loss@sum": total_loss.detach().item()}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL_ONE)
    async def is_model_in_gpu(self) -> bool:
        return bool(getattr(self.strategy, "is_model_in_gpu", False))


class TinkerInferWorker(InferWorker):
    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    async def compute_prompt_logprobs(self, data: DataProto) -> DataProto:
        data = data.to(current_platform.device_type)
        all_prompt_logprobs = await compute_prompt_logprobs_with_vllm_strategy(self.strategy, data)
        from tensordict import TensorDict

        # ROLL requires a batch dimension even for non-tensor-only RPC output.
        out = DataProto(
            batch=TensorDict({}, batch_size=[len(all_prompt_logprobs)]),
            non_tensor_batch={"prompt_logprobs": np.array(all_prompt_logprobs, dtype=object)},
        )
        data.to("cpu")
        return out

    @register(dispatch_mode=Dispatch.ONE_TO_ALL_ONE)
    async def is_model_in_gpu(self) -> bool:
        return bool(getattr(self.strategy, "is_model_in_gpu", False))
