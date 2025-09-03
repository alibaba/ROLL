import os
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
from typing import Callable, Dict, Tuple

import deepspeed
import ray
import torch
import torch.distributed as dist
from codetiming import Timer
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.runtime.zero import GatheredParameters
from deepspeed.runtime.zero.offload_config import OffloadStateTypeEnum
from peft import get_peft_model_state_dict
from tqdm import tqdm
from transformers import get_scheduler, set_seed
from transformers.integrations import HfDeepSpeedConfig

from roll.datasets.collator import collate_fn_to_dict_list
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_processor_provider, default_tokenizer_provider
from roll.third_party.deepspeed.offload_states_patch import bind_deepspeed_offload_states_func
from roll.utils.collective import collective
from roll.utils.context_parallel import apply_ulysses_patch, get_ulysses_group, set_upg_manager
from roll.utils.deepspeed_utils import get_optimizer_grouped_parameters
from roll.utils.functionals import append_to_dict, entropy_from_logits, log_probs_from_logits
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType


logger = get_logger()


class DeepSpeedInferStrategy(InferenceStrategy):
    strategy_name = "deepspeed_infer"

    def __init__(self, worker: Worker):
        super().__init__(worker)
        self.worker_config.strategy_args.strategy_config["train_micro_batch_size_per_gpu"] = (
            self.worker_config.training_args.per_device_train_batch_size
        )

        # deepspeed的train_batch_size是全局batch_size
        self.worker_config.strategy_args.strategy_config["train_batch_size"] = (
            self.worker_config.training_args.per_device_train_batch_size
            * self.worker_config.training_args.gradient_accumulation_steps
            * self.worker.world_size
        )
        self.worker_config.strategy_args.strategy_config["gradient_clipping"] = (
            self.worker.pipeline_config.max_grad_norm
        )
        self.ds_config = HfDeepSpeedConfig(self.worker_config.strategy_args.strategy_config)

    def initialize(self, model_provider):
        set_seed(seed=self.worker.pipeline_config.seed)

        assert self.ds_config.is_zero3(), "deepspeed infer only supports zero = 3."

        deepspeed.init_distributed(timeout=timedelta(minutes=self.worker_config.backend_timeout))
        dist.all_reduce(torch.zeros(1).cuda())

        # apply Ulysses parallel
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()

        if (cp_size := self.worker_config.model_args.ulysses_size) > 1:
            apply_ulysses_patch()
            set_upg_manager(ulysses_size=cp_size, rank=global_rank, world_size=world_size)

        self.worker.rank_info.dp_rank = global_rank // cp_size
        self.worker.rank_info.dp_size = world_size // cp_size
        self.worker.rank_info.cp_rank = global_rank % cp_size
        self.worker.rank_info.cp_size = cp_size

        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.processor = default_processor_provider(model_args=self.worker_config.model_args)

        model = model_provider(tokenizer=self.tokenizer, model_args=self.worker_config.model_args, is_trainable=False)

        try:
            num_attention_heads, num_key_value_heads = model.config.num_attention_heads, model.config.num_key_value_heads
        except AttributeError:
            num_attention_heads, num_key_value_heads = (
                model.config.text_config.num_attention_heads,
                model.config.text_config.num_key_value_heads,
            )

        assert num_attention_heads % cp_size == 0, (
            f"num_attention_heads {num_attention_heads} must be divisible by ulysses_size {cp_size}"
        )
        assert num_key_value_heads % cp_size == 0 or cp_size % num_key_value_heads == 0, (
            f"num_key_value_heads {num_key_value_heads} must be divisible by ulysses_size "
            f"{cp_size}or vise versa. Upon ulysses_size % num_key_value_heads == 0,"
            f"kv heads are repeated to ensure correctness."
        )

        logger.info(f"{self.model}")

        self.model, *_ = deepspeed.initialize(
            model=model,
            config=self.worker_config.strategy_args.strategy_config,
            dist_init_required=True,
        )

        bind_deepspeed_offload_states_func(self.model)

        logger.info(f"{self.model}")
        dist.barrier()

    def get_data_input(self, batch: DataProto):
        def broadcast_obj(obj, group):
            obj_list = [obj if dist.get_rank(group) == 0 else None]
            src_rank = dist.get_process_group_ranks(group)[0]
            dist.broadcast_object_list(obj_list, src=src_rank, group=group)
            return obj_list[0]
        # to avoid making side-effect on LLM, if want to broadcast non_tensor_batch,
        # set _broadcast_non_tensor_batch into meta_info
        broadcast_non_tensor_batch = batch.meta_info.get("_broadcast_non_tensor_batch", False)
        if self.worker.rank_info.cp_size > 1:
            if broadcast_non_tensor_batch:
                tmp_batch = broadcast_obj(batch, get_ulysses_group())
                batch.batch = tmp_batch.batch
                batch.non_tensor_batch = tmp_batch.non_tensor_batch
            else:
                batch.batch = broadcast_obj(batch.batch, get_ulysses_group())
        return batch

    def forward_step(
        self,
        batch: DataProto,
        forward_func: Callable[[DataProto, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        batch_size = batch.batch.batch_size[0]
        micro_batch_size = batch.meta_info["micro_batch_size"]
        num_microbatches = max(batch_size // micro_batch_size, 1)
        micro_batches = batch.chunk(chunks=num_microbatches)
        disable_adapter = batch.meta_info.get("disable_adapter", False)
        adapter_context = self.unwrap_model().disable_adapter() if disable_adapter else nullcontext()
        losses_reduced = []
        with adapter_context:
            for data in micro_batches:
                input_ids = data.batch["input_ids"]
                attention_mask = data.batch["attention_mask"]
                position_ids = data.batch["position_ids"]
                forward_args = data.meta_info.get("forward_args", {})
                if position_ids.dim() == 3:
                    # qwen2vl mrope, maybe use a placeholder and let model generate position_ids
                    position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)
                if "multi_modal_inputs" in data.non_tensor_batch:
                    multi_modal_inputs = data.non_tensor_batch["multi_modal_inputs"]
                    multi_modal_data = defaultdict(list)
                    # mm inputs of some samples would be empty to allow text and mm
                    # mixed data
                    for sample_mm_inputs in multi_modal_inputs:
                        for key in sample_mm_inputs.keys():
                            multi_modal_data[key].append(sample_mm_inputs[key])
                    for key in multi_modal_data.keys():
                        assert key not in forward_args
                        # DataProto.to('cuda') in upper frame not work for non_tensor_batch
                        forward_args[key] = torch.concat(multi_modal_data[key], dim=0).to(input_ids.device)
                    forward_args.update({"force_vit_image": True})

                if self.worker.rank_info.cp_size > 1:
                    splited_features = self.get_feature_on_cp_rank(input_ids, attention_mask, position_ids)
                    input_ids = splited_features["input_ids"]
                    attention_mask = splited_features["attention_mask"]
                    position_ids = splited_features["position_ids"]

                # set use_cache=False manually for the same reason as HfInferStrategy
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    **forward_args,
                )
                loss, loss_reduced = forward_func(data, output.logits)
                losses_reduced.append(loss_reduced)
        results = collate_fn_to_dict_list(losses_reduced)
        return results

    def get_feature_on_cp_rank(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, position_ids: torch.Tensor = None
    ):
        seqlens_in_batch = input_ids.size(1)
        assert seqlens_in_batch % self.worker.rank_info.cp_size == 0, (
            f"input_length={seqlens_in_batch} not divisible by cp_size={self.worker.rank_info.cp_size}"
        )
        cp_middle_rank_len = seqlens_in_batch // self.worker.rank_info.cp_size
        padded_input_ids = input_ids
        result = {}
        start_index = cp_middle_rank_len * self.worker.rank_info.cp_rank
        end_index = cp_middle_rank_len * (self.worker.rank_info.cp_rank + 1)
        result["input_ids"] = padded_input_ids[:, start_index:end_index]
        if attention_mask is not None:
            result["attention_mask"] = attention_mask[:, start_index:end_index]
        if position_ids is not None:
            if position_ids.dim() == 3:
                result["position_ids"] = position_ids[:, :, start_index:end_index]
            else:
                result["position_ids"] = position_ids[:, start_index:end_index]
        return result

    def generate(self, batch: DataProto, generation_config):
        input_ids = batch.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = batch.batch["attention_mask"]  # left-padded attention_mask

        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **generation_config,
        )

        return output

    def unwrap_model(self):
        return self.model.module

    # 参数同步相关接口
    def broadcast_parameter(self, model_update_name, src_pp_rank, dtype, shape, parameter_name, is_lora=False):
        comm_plan = self.model_update_comm_plan[model_update_name][src_pp_rank]
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        collective.broadcast(tensor=weight, src_rank=0, group_name=comm_plan["group_name"])
        param = self.model.get_parameter(parameter_name)
        if not self.ds_config.is_zero3():
            param.data.copy_(weight.to("cpu"))
        else:
            with GatheredParameters([param], modifier_rank=0):
                if dist.get_rank() == 0:
                    param.data.copy_(weight)
        del weight

    def update_parameter(self, model_update_name, parameter_name, weight, ranks_in_worker):
        param = self.model.get_parameter(parameter_name)
        if not self.ds_config.is_zero3():
            param.data.copy_(weight)
        else:
            with GatheredParameters([param], modifier_rank=0):
                if dist.get_rank() == 0:
                    param.data.copy_(weight)
        del weight

    # offload/load 相关接口
    def load_states(self, include=None, non_blocking=False):
        if include is not None:
            ds_include = []
            if OffloadStateType.model_params in include:
                ds_include.append(OffloadStateTypeEnum.lp_params)
            if OffloadStateType.other_params in include:
                ds_include.append(OffloadStateTypeEnum.hp_params)
                ds_include.append(OffloadStateTypeEnum.lp_grads)
                ds_include.append(OffloadStateTypeEnum.contiguous_grad_buffer)
            if OffloadStateType.optimizer_states in include:
                ds_include.append(OffloadStateTypeEnum.optim_states)
            include = ds_include
        self.model.reload_states(include=include, non_blocking=non_blocking)

    def offload_states(self, include=None, non_blocking=False):
        if include is not None:
            ds_include = []
            if OffloadStateType.model_params in include:
                ds_include.append(OffloadStateTypeEnum.lp_params)
            if OffloadStateType.other_params in include:
                ds_include.append(OffloadStateTypeEnum.hp_params)
                ds_include.append(OffloadStateTypeEnum.lp_grads)
                ds_include.append(OffloadStateTypeEnum.contiguous_grad_buffer)
            if OffloadStateType.optimizer_states in include:
                ds_include.append(OffloadStateTypeEnum.optim_states)
            include = ds_include

        self.model.offload_states(include=include, non_blocking=non_blocking)
        torch.cuda.empty_cache()

    def op_compute_log_probs(self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        input_ids [[p, p, r, r, r, 0, 0]] p: prompt, r: response, 0: pad
        response_mask [[0, 0, 1, 1, 1, 0, 0]]
        """
        labels: torch.Tensor = input_ids[:, 1:].clone()
        labels[attention_mask[:, 1:] == 0] = 0  # avoid invalid token id
        # TODO: don't pad here but process this shift after generation
        labels = torch.cat([labels, torch.zeros_like(labels[:, :1])], dim=1)
        if self.worker.rank_info.cp_size > 1:
            labels = self.get_feature_on_cp_rank(labels)["input_ids"]
        log_probs = log_probs_from_logits(logits, labels)
        if self.worker.rank_info.cp_size > 1:
            with torch.no_grad():
                all_log_probs = [torch.empty_like(log_probs) for _ in range(self.worker.rank_info.cp_size)]
                dist.all_gather(all_log_probs, log_probs, group=get_ulysses_group())
            all_log_probs[self.worker.rank_info.cp_rank] = log_probs
            log_probs = torch.cat(all_log_probs, dim=1)
        log_probs = log_probs[:, :-1] * attention_mask[:, 1:]
        return log_probs

    def op_compute_entropy(self, logits: torch.Tensor, attention_mask: torch.Tensor):
        entropy = entropy_from_logits(logits)
        if self.worker.rank_info.cp_size > 1:
            with torch.no_grad():
                all_entropy = [torch.empty_like(entropy) for _ in range(self.worker.rank_info.cp_size)]
                dist.all_gather(all_entropy, entropy, group=get_ulysses_group())
            all_entropy[self.worker.rank_info.cp_rank] = entropy
            entropy = torch.cat(all_entropy, dim=1)
        entropy = entropy[:, :-1] * attention_mask[:, 1:]
        return entropy


class DeepSpeedTrainStrategy(DeepSpeedInferStrategy, TrainStrategy):
    strategy_name = "deepspeed_train"

    def initialize(self, model_provider):
        assert self.ds_config._stage > 0, "deepspeed train only supports zero > 0."

        set_seed(seed=self.worker.pipeline_config.seed)
        deepspeed.init_distributed(timeout=timedelta(minutes=self.worker_config.backend_timeout))
        dist.all_reduce(torch.zeros(1).cuda())

        # apply Ulysses parallel
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()

        if (cp_size := self.worker_config.model_args.ulysses_size) > 1:
            apply_ulysses_patch()
            set_upg_manager(ulysses_size=cp_size, rank=global_rank, world_size=world_size)

        self.worker.rank_info.dp_rank = global_rank // cp_size
        self.worker.rank_info.dp_size = world_size // cp_size
        self.worker.rank_info.cp_rank = global_rank % cp_size
        self.worker.rank_info.cp_size = cp_size


        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.processor = default_processor_provider(model_args=self.worker_config.model_args)

        model = model_provider(tokenizer=self.tokenizer, model_args=self.worker_config.model_args, is_trainable=True)

        try:
            num_attention_heads, num_key_value_heads = model.config.num_attention_heads, model.config.num_key_value_heads
        except AttributeError:
            num_attention_heads, num_key_value_heads = (
                model.config.text_config.num_attention_heads,
                model.config.text_config.num_key_value_heads,
            )

        assert num_attention_heads % cp_size == 0, (
            f"num_attention_heads {num_attention_heads} must be divisible by ulysses_size {cp_size}"
        )
        assert num_key_value_heads % cp_size == 0 or cp_size % num_key_value_heads == 0, (
            f"num_key_value_heads {num_key_value_heads} must be divisible by ulysses_size "
            f"{cp_size}or vise versa. Upon ulysses_size % num_key_value_heads == 0,"
            f"kv heads are repeated to ensure correctness."
        )

        adam_optimizer = DeepSpeedCPUAdam if self.ds_config.is_offload() else FusedAdam
        optim_params = get_optimizer_grouped_parameters(
            model, weight_decay=self.worker_config.training_args.weight_decay
        )
        optimizer = adam_optimizer(
            optim_params,
            lr=self.worker_config.training_args.learning_rate,
            betas=(self.worker_config.training_args.adam_beta1, self.worker_config.training_args.adam_beta2),
        )

        logger.info(f"max steps pipeline {self.worker_config.training_args.max_steps}")
        self.worker_config.training_args.max_steps = (
            self.worker_config.training_args.max_steps // self.worker.rank_info.dp_size
        )
        logger.info(f"max steps worker train {self.worker_config.training_args.max_steps}")

        scheduler = get_scheduler(
            self.worker_config.training_args.lr_scheduler_type,
            optimizer,
            num_warmup_steps=self.worker_config.training_args.get_warmup_steps(
                self.worker_config.training_args.max_steps
            ),
            num_training_steps=self.worker_config.training_args.max_steps,
        )

        self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model_parameters=model.parameters(),
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=self.worker_config.strategy_args.strategy_config,
            dist_init_required=True,
        )
        bind_deepspeed_offload_states_func(self.model)

        logger.info(f"{self.model}")
        dist.barrier()

    def train_step(
        self,
        batch: DataProto,
        loss_func: Callable[[DataProto, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
    ):
        self.model.train()
        mini_batch_size = self.worker_config.training_args.per_device_train_batch_size
        data_iter = batch.make_iterator(mini_batch_size=mini_batch_size, epochs=1)
        mini_steps = batch.batch.batch_size[0] // self.worker_config.training_args.per_device_train_batch_size
        metrics = {}

        for step in range(mini_steps):
            data: DataProto = next(data_iter)
            input_ids = data.batch["input_ids"]
            attention_mask = data.batch["attention_mask"]
            position_ids = data.batch["position_ids"]
            forward_args = data.meta_info.get("forward_args", {})
            # TODO: The offload option may be integrated into the pipeline config in the future.
            is_offload_optimizer_states_in_train_step = data.meta_info.get("is_offload_optimizer_states_in_train_step", True)
            if position_ids.dim() == 3:
                # qwen2vl mrope, maybe use a placeholder and let model generate position_ids
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)
            if "multi_modal_inputs" in data.non_tensor_batch:
                multi_modal_inputs = data.non_tensor_batch["multi_modal_inputs"]
                multi_modal_data = defaultdict(list)
                # mm inputs of some samples would be empty to allow text and mm
                # mixed data
                for sample_mm_inputs in multi_modal_inputs:
                    for key in sample_mm_inputs.keys():
                        multi_modal_data[key].append(sample_mm_inputs[key])
                for key in multi_modal_data.keys():
                    assert key not in forward_args
                    # DataProto.to('cuda') in upper frame not work for non_tensor_batch
                    forward_args[key] = torch.concat(multi_modal_data[key], dim=0).to(input_ids.device)
                forward_args.update({"force_vit_image": True})

            if self.worker.rank_info.cp_size > 1:
                splited_features = self.get_feature_on_cp_rank(input_ids, attention_mask, position_ids)
                input_ids = splited_features["input_ids"]
                attention_mask = splited_features["attention_mask"]
                position_ids = splited_features["position_ids"]

            output = self.model(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, **forward_args
            )
            loss, loss_reduced = loss_func(data, output.logits)
            append_to_dict(metrics, loss_reduced)
            loss *= self.worker.rank_info.cp_size
            self.model.backward(loss)

            is_gradient_accumulation_boundary = self.model.is_gradient_accumulation_boundary()
            if is_gradient_accumulation_boundary:
                self.load_states(include=[OffloadStateType.optimizer_states])
            self.model.step()
            if is_gradient_accumulation_boundary:
                # global_grad_norm is calculated in optimizer.step thus put it
                # into metrics after optimizer.step
                metrics.update({self.worker_config.name + "/" + "grad_norm": self.model.get_global_grad_norm().item()})
                if is_offload_optimizer_states_in_train_step:
                    self.offload_states(include=[OffloadStateType.optimizer_states], non_blocking=True)
        return metrics

    def save_checkpoint(self, save_dir, global_step, ckpt_id, tag="checkpoint", local_state_path=None, **kwargs):
        """
        save ckpt/hf model/tokenizer to local dir
        save_dir/actor_train/{hf files}
        save_dir/actor_train/checkpoint/{checkpoint files}
        """
        logger.info(f"save_dir: {save_dir}")
        if local_state_path is None:
            local_state_path = save_dir

        with Timer("load") as load_timer:
            self.load_states()

        if self.ds_config.is_zero3():
            if self.model.zero_gather_16bit_weights_on_model_save():
                state_dict = self.model._zero3_consolidated_16bit_state_dict()
            else:
                raise ValueError(
                    "Cannot get 16bit model weights because `stage3_gather_16bit_weights_on_model_save` in DeepSpeed config is False. "
                    "To save the model weights in 16bit, set `stage3_gather_16bit_weights_on_model_save` to True in DeepSpeed config file or "
                    "set `zero3_save_16bit_model` to True when using `accelerate config`. "
                    "To save the full checkpoint, run `model.save_checkpoint(save_dir)` and use `zero_to_fp32.py` to recover weights."
                )
        else:
            from deepspeed.checkpoint.utils import clone_tensors_for_torch_save

            state_dict = clone_tensors_for_torch_save(self.model.module.state_dict())

        # save huggingface pretrained model
        if dist.get_rank() == 0:
            self.model.module.save_pretrained(save_dir, state_dict=state_dict, safe_serialization=False)
            self.tokenizer.save_pretrained(save_dir)
            if getattr(self, "processor", None):
                self.processor.save_pretrained(save_dir)
            # save tokenizer
        self.model.save_checkpoint(save_dir, tag=tag, **kwargs)

        if self.worker_config.checkpoint_config.get("async_upload", True):
            self.thread_executor.submit(self.checkpoint_manager.upload, ckpt_id=ckpt_id, local_state_path=local_state_path)
        else:
            self.checkpoint_manager.upload(ckpt_id=ckpt_id, local_state_path=local_state_path)

        metrics = {
            "load": load_timer.last,
        }
        return metrics

    def load_checkpoint(self, load_dir, tag="checkpoint", **kwargs):
        logger.info(f"load checkpoint from {load_dir}")
        self.model.load_checkpoint(load_dir, tag=tag, **kwargs)

    def collect_lora_params(self):
        peft_model = self.unwrap_model()
        if not self.ds_config.is_zero3():
            lora_state_dict = get_peft_model_state_dict(peft_model)
            return lora_state_dict
        adapter_name = "default"
        state_dict = peft_model.state_dict()
        lora_state_dict = {k: state_dict[k] for k in state_dict if ("lora_" in k and adapter_name in k)}
        lora_params = []
        for name, param in lora_state_dict.items():
            lora_params.append((name.replace(f".{adapter_name}", ""), peft_model.get_parameter(name)))
        del lora_state_dict
        return lora_params

    def model_update(self, model_update_name, tgt_workers, broadcast_tgt_devices, p2p_tgt_devices):
        model = self.unwrap_model()
        if is_lora := (self.worker_config.model_args.lora_target is not None):
            all_params = self.collect_lora_params()
            peft_config = model.peft_config.get("default", None)
        else:
            all_params = list(model.named_parameters())

        comm_plan = self.model_update_comm_plan[model_update_name][self.worker.rank_info.pp_rank]
        model = self.unwrap_model()
        broadcast_time_cost = 0
        with Timer("model_update_total") as timer_total:
            for param_name, param in tqdm(all_params, desc="weight update progress", total=len(all_params)):
                shape = param.shape if not self.ds_config.is_zero3() else param.ds_shape
                if not self.ds_config.is_zero3():

                    param_weight = param.data
                    refs = []
                    for p2p_tgt_device in p2p_tgt_devices:
                        p2p_tgt_worker = tgt_workers[p2p_tgt_device["rank"]]
                        ref = p2p_tgt_worker.update_parameter.remote(
                            model_update_name=model_update_name,
                            parameter_name=param_name,
                            weight=param_weight,
                            ranks_in_worker=[p2p_tgt_device["device"]["rank"]],
                            is_lora=is_lora,
                        )
                        refs.append(ref)

                    if (
                        self.worker.rank_info.tp_rank == 0
                        and self.worker.rank_info.cp_rank == 0
                        and self.worker.rank_info.dp_rank == 0
                    ):
                        for worker in tgt_workers:
                            ref = worker.broadcast_parameter.remote(
                                model_update_name=model_update_name,
                                src_pp_rank=self.worker.rank_info.pp_rank,
                                dtype=param_weight.dtype,
                                shape=shape,
                                parameter_name=param_name,
                                is_lora=is_lora,
                            )
                            refs.append(ref)
                    if len(broadcast_tgt_devices) > 0:
                        collective.broadcast(tensor=param_weight, src_rank=0, group_name=comm_plan["group_name"])
                    ray.get(refs)

                else:
                    with GatheredParameters([param]):
                        param_weight = param.data
                        with Timer("broadcast") as timer_broadcast:
                            refs = []
                            for p2p_tgt_device in p2p_tgt_devices:
                                p2p_tgt_worker = tgt_workers[p2p_tgt_device["rank"]]
                                ref = p2p_tgt_worker.update_parameter.remote(
                                    model_update_name=model_update_name,
                                    parameter_name=param_name,
                                    weight=param_weight,
                                    ranks_in_worker=[p2p_tgt_device["device"]["rank"]],
                                    is_lora=is_lora,
                                )
                                refs.append(ref)

                            if (
                                self.worker.rank_info.tp_rank == 0
                                and self.worker.rank_info.cp_rank == 0
                                and self.worker.rank_info.dp_rank == 0
                            ):
                                for worker in tgt_workers:
                                    ref = worker.broadcast_parameter.remote(
                                        model_update_name=model_update_name,
                                        src_pp_rank=self.worker.rank_info.pp_rank,
                                        dtype=param_weight.dtype,
                                        shape=shape,
                                        parameter_name=param_name,
                                        is_lora=is_lora,
                                    )
                                    refs.append(ref)
                            if len(broadcast_tgt_devices) > 0:
                                collective.broadcast(
                                    tensor=param_weight, src_rank=0, group_name=comm_plan["group_name"]
                                )
                            ray.get(refs)
                        broadcast_time_cost += timer_broadcast.last

            if is_lora:
                with Timer("add_lora") as timer_add_lora:
                    if (
                        self.worker.rank_info.tp_rank == 0
                        and self.worker.rank_info.cp_rank == 0
                        and self.worker.rank_info.dp_rank == 0
                    ):
                        refs = []
                        for worker in tgt_workers:
                            ref = worker.add_lora.remote(peft_config=asdict(peft_config))
                            refs.append(ref)
                        ray.get(refs)

        metrics = {
            "broadcast": broadcast_time_cost,
        }
        if is_lora:
            metrics["all_gather"] = timer_total.last - broadcast_time_cost - timer_add_lora.last
            metrics["add_lora"] = timer_add_lora.last
        else:
            metrics["all_gather"] = timer_total.last - broadcast_time_cost
        return metrics
