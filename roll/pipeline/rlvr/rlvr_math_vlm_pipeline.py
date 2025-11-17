import json
import os
import uuid
from typing import Any, Dict, List, Optional

import ray
import torch
import datasets
import PIL.Image as Image
from transformers import ProcessorMixin, AutoConfig
from transformers.image_utils import load_images
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from datasets import load_dataset, load_from_disk
from codetiming import Timer
from ray.util.timer import _Timer
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from roll.datasets.collator import DataCollatorWithPaddingForMM
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.generate_scheduler import GenerateScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_processor_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.utils.checkpoint_manager import download_model
from roll.utils.constants import GENERATE_SCHEDULER_NAME, RAY_NAMESPACE
from roll.utils.functionals import (
    apply_kl_penalty,
    compute_advantage,
    reduce_metrics,
    masked_mean,
    RunningMoments,
    compute_clip_fraction,
    group_reward_norm,
    expand_to_token_level,
)
from roll.utils.kl_controller import get_kl_controller
from roll.utils.logging import get_logger

from .rlvr_vlm_pipeline import format_prompt, process_images, get_extra_data_provider

logger = get_logger()


def is_lora_training(pipeline_config: RLVRConfig) -> bool:
    if pipeline_config.actor_train.model_args.lora_target is None:
        return False
    assert pipeline_config.actor_train.strategy_args.strategy_name == "deepspeed_train", (
        "LoRA only supports deepspeed_train"
    )
    return True


def encode_function(data_i, processor, prompt_key, answer_key, image_key):
    image_flag = [True] * len(data_i[prompt_key])
    image_list = []
    for idx, image in enumerate(data_i[image_key]):
        if image is None:
            image_flag[idx] = False
        try:
            image_out = load_images(image if isinstance(image, (list, tuple)) else [image], timeout=None)
        except Exception as e:
            image_out = [Image.new("RGB", (224, 224), (255, 255, 255))]
            logger.error(f"Failed to get image due to {e}")
        # since infer-image use pil image as input while train-engine use
        # processed data, process image here to make them use same image
        image_out = process_images(image_out, processor)
        image_list.append(image_out)
    text_list = []
    for idx, instruct in enumerate(data_i[prompt_key]):
        # provide prompt_image_token if image_token in prompt
        text = format_prompt(instruct, processor, use_image=image_flag[idx], prompt_image_token=None)
        text_list.append(text)
    encodings = {
        # for area seperated validation, no need currently
        "tag": [""] * len(data_i[prompt_key]),
        "prompt": text_list,
        # no need to extract currently, answer can be by math_verify.parse
        "ground_truth": [solution for solution in data_i[answer_key]],
        "image": image_list,
        "image_flag": image_flag,
    }
    return encodings


FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}


def get_dataset(data_args, encode_function, processor, features=None, get_eval=False):
    cache_path = getattr(data_args, "cache_path", None)
    if cache_path:
        cache_path = os.path.join(cache_path, "val" if get_eval else "train")
    if cache_path and os.path.exists(cache_path):
        dataset = load_from_disk(cache_path)
        return dataset
    data_path = None
    data_name = data_args.file_name
    data_files = []
    dataset_dir = getattr(data_args, "dataset_dir", ".")
    local_path: str = os.path.join(dataset_dir, data_name)
    if os.path.isdir(local_path):
        for file_name in os.listdir(local_path):
            data_files.append(os.path.join(local_path, file_name))
            if data_path is None:
                data_path = FILEEXT2TYPE.get(file_name.split(".")[-1], None)
            elif data_path != FILEEXT2TYPE.get(file_name.split(".")[-1], None):
                raise ValueError("File types should be identical.")
    elif os.path.isfile(local_path):  # is file
        data_files.append(local_path)
        data_path = FILEEXT2TYPE.get(local_path.split(".")[-1], None)
    else:
        raise ValueError("File not found.")
    dataset = load_dataset(path=data_path, data_files=data_files)["train"]
    remove_columns = list(dataset.features.keys() - features.keys())
    # TODO: add fileds into config dataclass, actually these config attrs cannot
    # be used temporarily and equal to hard-code
    prompt_key = getattr(data_args, "prompt") if getattr(data_args, "prompt", None) else "problem"
    answer_key = getattr(data_args, "response") if getattr(data_args, "response", None) else "solution"
    image_key = getattr(data_args, "image") if getattr(data_args, "image", None) else "image"
    print(f"Begin : {dataset}")
    dataset = dataset.map(
        lambda data: encode_function(data, processor, prompt_key, answer_key, image_key),
        batched=True,
        batch_size=100,
        num_proc=32,
        features=features,
        remove_columns=remove_columns,
        desc="Encoding dataset",
    )
    print(f"Encoding: {dataset}")
    if cache_path:
        dataset.save_to_disk(cache_path)
    return dataset


def get_dataloader(dataset, batch_size, data_collator):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,  # larger shm for bigger num_workers
        collate_fn=data_collator,
    )
    return dataloader


class RLVRMathVLMPipeline(BasePipeline):
    """This pipeline is deprecated and use `RLVRPipeline` instead"""

    def __init__(self, pipeline_config: RLVRConfig):
        logger.warning(f"`{self.__class__.__name__}` is deprecated, and use `RLVRPipeline` instead")
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config
        self.is_lora = is_lora_training(self.pipeline_config)

        self.processor = default_processor_provider(self.pipeline_config.actor_train.model_args)
        # set max_pixels to avoid image token num is larger than prompt length
        self.processor.image_processor.max_pixels, self.processor.image_processor.min_pixels = (
            getattr(self.pipeline_config.actor_train.model_args, "max_pixels", 768 * 768),
            getattr(self.pipeline_config.actor_train.model_args, "min_pixels", 56 * 56),
        )
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        # regularized data filed
        features = datasets.Features(
            {
                # only support single image temporarily since sglang usage
                # "image": datasets.Image(decode=True),
                "image": datasets.Sequence(feature=datasets.Image(decode=True)),
                "prompt": datasets.Value("string"),
                "ground_truth": datasets.Value("string"),
                # for text and multi-modal mixed data usage, indicating valid image
                "image_flag": datasets.Value("bool"),
                # for area seperated validation, dummy currently
                "tag": datasets.Value("string"),
            }
        )
        dataset = get_dataset(
            self.pipeline_config.actor_train.data_args, encode_function, self.processor, features, get_eval=False
        )
        val_dataset = None
        if self.pipeline_config.validation and self.pipeline_config.validation.data_args:
            val_dataset = get_dataset(
                self.pipeline_config.validation.data_args, encode_function, self.processor, features, get_eval=True
            )

        data_collator = DataCollatorWithPaddingForMM(
            tokenizer=self.tokenizer,
            processor=self.processor,
            extra_data_provider=get_extra_data_provider(
                self.pipeline_config.actor_train.model_args.model_name_or_path, processor=self.processor
            ),
            max_length=self.pipeline_config.prompt_length,
            padding="max_length",
        )
        self.dataloader = get_dataloader(dataset, self.pipeline_config.rollout_batch_size, data_collator)
        self.val_dataloader = None
        if val_dataset:
            self.val_dataloader = get_dataloader(val_dataset, len(val_dataset), data_collator)
        max_steps = len(self.dataloader) * self.pipeline_config.actor_train.training_args.num_train_epochs
        self.pipeline_config.set_max_steps(max_steps=max_steps)

        self.actor_train: Any = Cluster(
            name=self.pipeline_config.actor_train.name,
            worker_cls=self.pipeline_config.actor_train.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_train,
        )
        self.actor_infer: Any = Cluster(
            name=self.pipeline_config.actor_infer.name,
            worker_cls=self.pipeline_config.actor_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )
        # use unwrapped model as reference for lora training
        if not self.is_lora and self.pipeline_config.enable_reference:
            self.reference: Any = Cluster(
                name=self.pipeline_config.reference.name,
                worker_cls=self.pipeline_config.reference.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.reference,
            )
        self.rewards: Dict[str, Any] = {
            key: Cluster(
                name=f"reward-{key}",
                worker_cls=worker_config.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=worker_config,
            )
            for key, worker_config in self.pipeline_config.rewards.items()
        }
        self.reward: Any = self.rewards[list(self.rewards.keys())[0]]
        if self.pipeline_config.adv_estimator == "gae":
            self.critic: Any = Cluster(
                name=self.pipeline_config.critic.name,
                worker_cls=self.pipeline_config.critic.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.critic,
            )

        self.generate_scheduler = GenerateScheduler.options(
            name=f"{GENERATE_SCHEDULER_NAME}_{self.actor_infer.cluster_name}",
            get_if_exists=True,
            namespace=RAY_NAMESPACE,
        ).remote()

        self.kl_ctrl = get_kl_controller(
            init_kl_coef=self.pipeline_config.init_kl_coef,
            target_kl=self.pipeline_config.target_kl,
            kl_horizon=self.pipeline_config.kl_horizon,
        )

        refs = []
        refs.extend(self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        refs = []
        if not self.is_lora and self.pipeline_config.enable_reference:
            refs.extend(self.reference.initialize(pipeline_config=self.pipeline_config, blocking=False))
        refs.extend(self.reward.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        refs: List[ray.ObjectRef] = []
        refs.extend(self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=False))
        if self.pipeline_config.adv_estimator == "gae":
            refs.extend(self.critic.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
            frequency=self.pipeline_config.actor_train.model_update_frequency,
        )

        if self.pipeline_config.adv_estimator == "gae":
            self.set_checkpoint_clusters(self.actor_train, self.critic)
        else:
            self.set_checkpoint_clusters(self.actor_train)

        self.running = RunningMoments()

    @torch.no_grad()
    def run(self):
        global_step = 0

        # throughput for tokens per second
        tps_timer = _Timer(window_size=5)
        actor_infer_timer = _Timer(window_size=5)
        actor_infer_response_timer = _Timer(window_size=5)
        actor_train_timer = _Timer(window_size=5)

        for epoch in range(int(self.pipeline_config.actor_train.training_args.num_train_epochs)):
            logger.info(f"epoch {epoch} start...")
            for batch_dict in tqdm(self.dataloader):
                if global_step <= self.state.step:
                    global_step += 1
                    continue

                logger.info(f"pipeline step {global_step} start...")

                metrics = {}
                with tps_timer:
                    if self.pipeline_config.adv_estimator == "gae":
                        self.critic.offload_states(blocking=True)
                    self.actor_train.offload_states(blocking=True)
                    model_update_metrics: Dict = self.model_update(global_step)
                    metrics.update(model_update_metrics)

                    if self.val_dataloader and global_step % self.pipeline_config.eval_steps == 0:
                        metrics.update(self.val())

                    batch_dict: Dict
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    batch.meta_info = {
                        "global_step": global_step,
                        # mark here to make megatron get_data_input broadcast with non_batch_tensor
                        "_broadcast_non_tensor_batch": True,
                    }

                    with actor_infer_timer, actor_infer_response_timer:
                        # donot support hf/deepspeed infer generate which use
                        # multi_modal_inputs tensors
                        gen_batch = batch.pop(
                            batch_keys=["input_ids", "attention_mask", "position_ids"],
                            non_tensor_batch_keys=(
                                ["multi_modal_data"] if "multi_modal_data" in batch.non_tensor_batch else []
                            ),
                        )
                        gen_batch.meta_info = {"global_step": global_step}
                        gen_batch.meta_info["response_callback_fn"] = self.generate_scheduler.report_response.remote
                        generate_output: DataProto = ray.get(
                            self.generate_scheduler.generate.remote(
                                data=gen_batch,
                                actor_cluster=self.actor_infer,
                                pipeline_config=self.pipeline_config,
                            ),
                            timeout=self.pipeline_config.rpc_timeout,
                        )
                        metrics.update(reduce_metrics(generate_output.meta_info.pop("metrics", {})))

                    # generate_output is repeated by num_return_sequences, thus
                    # reset batch.batch before union to make batch size same,
                    batch.batch = generate_output.batch
                    batch = batch.union(generate_output)

                    # repeat num_return_sequences for fields not in gen_batch
                    # which has been repeated in generate_scheduler
                    for key, value in batch.non_tensor_batch.items():
                        batch.non_tensor_batch[key] = np.repeat(
                            value, self.actor_infer.worker_config.generating_args.num_return_sequences
                        )
                    batch.non_tensor_batch['sample_uuid'] = np.array([str(uuid.uuid4()) for _ in range(batch.batch.shape[0])], dtype=object)

                    with Timer(name="cal_ref_log_probs_reward", logger=None) as cal_timer:
                        if self.pipeline_config.enable_reference:
                            if self.is_lora:
                                batch.meta_info["disable_adapter"] = True
                                batch.meta_info["is_offload_states"] = False
                                ref_log_probs_refs: List[ray.ObjectRef] = self.actor_train.compute_log_probs(
                                    batch, blocking=False
                                )
                            else:
                                ref_log_probs_refs: List[ray.ObjectRef] = self.reference.compute_log_probs(
                                    batch, blocking=False
                                )
                            ref_log_probs = DataProto.materialize_concat(data_refs=ref_log_probs_refs)
                            metrics.update(reduce_metrics(ref_log_probs.meta_info.pop("metrics", {})))
                            ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                            batch = batch.union(ref_log_probs)
                        rewards_refs: List[ray.ObjectRef] = self.reward.compute_rewards(batch, blocking=False)
                        rewards = DataProto.materialize_concat(data_refs=rewards_refs)
                        metrics.update(reduce_metrics(rewards.meta_info.pop("metrics", {})))
                        batch = batch.union(rewards)
                    metrics["time/ref_log_probs_values_reward"] = cal_timer.last

                    with Timer(name="cal_old_log_probs_values", logger=None) as cal_old_logpb_timer:
                        if self.is_lora:
                            batch.meta_info["disable_adapter"] = False
                        batch.meta_info["is_offload_states"] = False
                        if self.pipeline_config.adv_estimator == "gae":
                            values_refs: List[ray.ObjectRef] = self.critic.compute_values(batch, blocking=False)

                        if self.pipeline_config.enable_old_logprobs:
                            old_log_probs_refs: List[ray.ObjectRef] = self.actor_train.compute_log_probs(
                                batch, blocking=False
                            )
                            old_log_probs = DataProto.materialize_concat(data_refs=old_log_probs_refs)
                            batch.batch["old_log_probs"] = old_log_probs.batch["log_probs"]
                            metrics.update(reduce_metrics(old_log_probs.meta_info.pop("metrics", {})))
                        else:
                            # Use zeros when optimization is enabled
                            batch.batch["old_log_probs"] = torch.zeros_like(batch.batch["attention_mask"][:, 1:])

                        if self.pipeline_config.adv_estimator == "gae":
                            values = DataProto.materialize_concat(data_refs=values_refs)
                            batch = batch.union(values)
                            metrics.update(reduce_metrics(values.meta_info.pop("metrics", {})))

                        # Mock ref_log_probs using old_log_probs if reference is disabled
                        if not self.pipeline_config.enable_reference:
                            batch.batch["ref_log_probs"] = batch.batch["old_log_probs"].clone()

                    metrics["time/old_log_probs"] = cal_old_logpb_timer.last

                    with Timer(name="adv", logger=None) as timer:
                        if self.pipeline_config.use_reward_scaling:
                            self.running.update(batch.batch["response_level_rewards"])
                            reward_scaling_factor = (
                                self.running.std + torch.finfo(batch.batch["response_level_rewards"].dtype).eps
                            )
                            if self.pipeline_config.use_reward_norm:
                                batch.batch["response_level_rewards"] = (
                                    batch.batch["response_level_rewards"] - self.running.mean
                                ) / reward_scaling_factor
                            else:
                                batch.batch["response_level_rewards"] /= (
                                    reward_scaling_factor  # do not -= mean since advantage will be normalized again
                                )

                        if self.pipeline_config.reward_clip:
                            reward_clip_frac = compute_clip_fraction(
                                values=batch.batch["response_level_rewards"],
                                clip_max=self.pipeline_config.reward_clip,
                                clip_min=-self.pipeline_config.reward_clip,
                            )
                            metrics["critic/reward_clip_frac"] = reward_clip_frac
                            batch.batch["response_level_rewards"] = torch.clamp(
                                batch.batch["response_level_rewards"],
                                min=-self.pipeline_config.reward_clip,
                                max=self.pipeline_config.reward_clip,
                            )

                        if self.pipeline_config.adv_estimator == "grpo":
                            batch = group_reward_norm(
                                batch,
                                n_sample=self.pipeline_config.actor_infer.generating_args.num_return_sequences,
                                div_std=True,
                            )

                        if not self.pipeline_config.use_kl_loss:  # not grpo's kl loss
                            batch, kl_metrics = apply_kl_penalty(
                                data=batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.pipeline_config.kl_penalty
                            )
                        else:
                            token_level_rewards = expand_to_token_level(data=batch)
                            batch.batch["token_level_rewards"] = token_level_rewards
                            kl_metrics = {}

                        if self.pipeline_config.reward_clip:
                            reward_clip_frac = compute_clip_fraction(
                                values=batch.batch["token_level_rewards"],
                                clip_max=self.pipeline_config.reward_clip,
                                clip_min=-self.pipeline_config.reward_clip,
                            )
                            metrics["critic/token_reward_clip_frac"] = reward_clip_frac
                            batch.batch["token_level_rewards"] = torch.clamp(
                                batch.batch["token_level_rewards"],
                                min=-self.pipeline_config.reward_clip,
                                max=self.pipeline_config.reward_clip,
                            )

                        batch = compute_advantage(
                            data=batch,
                            gamma=self.pipeline_config.gamma,
                            lambd=self.pipeline_config.lambd,
                            adv_estimator=self.pipeline_config.adv_estimator,
                            advantage_clip=self.pipeline_config.advantage_clip,
                            whiten_advantages=self.pipeline_config.whiten_advantages,
                            whiten_rewards=self.pipeline_config.whiten_rewards,
                        )
                        metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))

                    metrics.update(kl_metrics)
                    metrics["time/adv"] = timer.last

                    if self.pipeline_config.adv_estimator == "gae":
                        critic_train_metrics_refs: List[ray.ObjectRef] = self.critic.train_step(batch, blocking=False)

                    with actor_train_timer:
                        # implement critic warmup
                        if not hasattr(self, "critic") or self.pipeline_config.critic_warmup <= global_step:
                            # update actor
                            actor_train_metrics_refs = self.actor_train.train_step(batch, blocking=False)
                            actor_train_metrics: DataProto = DataProto.materialize_concat(
                                data_refs=actor_train_metrics_refs
                            )
                            metrics.update(reduce_metrics(actor_train_metrics.meta_info.pop("metrics", {})))

                    if self.pipeline_config.adv_estimator == "gae":
                        critic_train_metrics = DataProto.materialize_concat(data_refs=critic_train_metrics_refs)
                        metrics.update(reduce_metrics(critic_train_metrics.meta_info.pop("metrics", {})))

                    tps_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())
                    actor_infer_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())
                    actor_infer_response_timer.push_units_processed(
                        n=torch.sum(batch.batch["response_mask"]).detach().item()
                    )
                    actor_train_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())

                data_metrics = compute_data_metrics(batch=batch)
                metrics.update(data_metrics)
                metrics["system/tps"] = tps_timer.mean_throughput
                metrics["system/actor_infer/tps"] = actor_infer_timer.mean_throughput
                metrics["system/actor_infer/response/tps"] = actor_infer_response_timer.mean_throughput
                metrics["system/actor_train/tps"] = actor_train_timer.mean_throughput
                metrics["system/tps_gpu"] = tps_timer.mean_throughput / self.resource_manager.num_gpus
                metrics["system/actor_infer/tps_gpu"] = actor_infer_timer.mean_throughput / self.actor_infer.world_size
                metrics["system/actor_infer//response/tps_gpu"] = (
                    actor_infer_response_timer.mean_throughput / self.actor_infer.world_size
                )
                metrics["system/actor_train/tps_gpu"] = actor_train_timer.mean_throughput / self.actor_train.world_size
                metrics["system/actor_infer/tps_dp"] = actor_infer_timer.mean_throughput / self.actor_infer.dp_size
                metrics["system/actor_infer/response/tps_dp"] = (
                    actor_infer_response_timer.mean_throughput / self.actor_infer.dp_size
                )
                metrics["system/actor_train/tps_dp"] = actor_train_timer.mean_throughput / self.actor_train.dp_size
                metrics["system/samples"] = (global_step + 1) * batch.batch.shape[0]

                # do ckpt
                self.state.step = global_step
                self.state.log_history.append(metrics)

                self.do_checkpoint(global_step=global_step)

                self.tracker.log(values=metrics, step=global_step)

                if global_step % self.pipeline_config.logging_steps == 0:
                    if int(os.environ.get("RAY_PROFILING", "0")):
                        timeline_dir = os.path.join(self.pipeline_config.profiler_output_dir, "timeline")
                        os.makedirs(timeline_dir, exist_ok=True)
                        ray.timeline(
                            filename=os.path.join(timeline_dir, f"timeline-step-{global_step}.json"),
                        )

                    prompt_ids = generate_output.batch["prompts"]
                    response_ids = generate_output.batch["responses"]

                    generate_res = []
                    # skip_special_tokens=True would output without image token, maybe do not skip
                    prompts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
                    responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                    for prompt, prompt_id, response, response_id in zip(
                        prompts,
                        prompt_ids,
                        responses,
                        response_ids,
                    ):
                        generate_res.append(
                            {
                                "prompt": prompt,
                                # "prompt_id": prompt_id.tolist(),
                                "response": response,
                                # "response_id": response_id.tolist(),
                            }
                        )
                    logger.info(json.dumps(generate_res[:10], ensure_ascii=False))
                    logger.info(json.dumps(metrics, ensure_ascii=False))

                logger.info(f"pipeline step {global_step} finished")
                global_step += 1

                if global_step >= self.pipeline_config.max_steps:
                    logger.info(f"pipeline step {global_step} finished, reached max steps: {self.pipeline_config.max_steps}")
                    return

            logger.info(f"epoch {epoch} finished")
        logger.info("pipeline complete!")

    @torch.no_grad()
    def val(self):
        # throughput for tokens per second
        tps_timer = _Timer(window_size=5)
        metrics = {}
        epoch_batch = []
        for batch_dict in tqdm(self.val_dataloader):
            with tps_timer:
                batch_dict: Dict
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["multi_modal_data"] if "multi_modal_data" in batch.non_tensor_batch else [],
                )
                gen_batch.meta_info["is_offload_states"] = False
                gen_batch.meta_info["response_callback_fn"] = self.generate_scheduler.report_response.remote
                generate_output: DataProto = ray.get(
                    self.generate_scheduler.generate.remote(
                        data=gen_batch,
                        actor_cluster=self.actor_infer,
                        pipeline_config=self.pipeline_config,
                    ),
                    timeout=self.pipeline_config.rpc_timeout,
                )
                batch.batch = generate_output.batch
                batch = batch.union(generate_output)

                for key, value in batch.non_tensor_batch.items():
                    batch.non_tensor_batch[key] = np.repeat(
                        value, self.actor_infer.worker_config.generating_args.num_return_sequences
                    )

                with Timer(name="cal_reward", logger=None) as cal_timer:
                    rewards = ray.get(self.reward.workers[0].compute_rewards.remote(batch))
                    batch = batch.union(rewards)
                logger.info(
                    json.dumps(
                        {"val_correct/mean": (batch.batch["scores"] == 1).detach().float().mean().item()},
                        ensure_ascii=False,
                    )
                )
                epoch_batch.append(batch)

        if len(epoch_batch) == 0:
            logger.info(f"len(self.val_dataloader): {len(self.val_dataloader)}, skip val...")
            return {}

        epoch_batch = DataProto.concat(epoch_batch)
        logger.info(f"total eval information: {epoch_batch}")
        logger.info(f"total eval information --- scores mean: {epoch_batch.batch['scores'].mean().item()} "
                    f"scores: {epoch_batch.batch['scores'].tolist()}")
        metrics[ f"val_correct/mean"] =  (epoch_batch.batch["scores"] == 1).detach().float().mean().item()
        return metrics


def compute_data_metrics(batch):
    sequence_score = batch.batch["scores"]
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)
    sequence_reward_mean = batch.batch["token_level_rewards"].mean(-1)

    max_response_length = batch.batch["responses"].shape[-1]
    advantages = batch.batch["advantages"]
    prompt_mask = batch.batch["prompt_mask"].bool()
    response_mask = batch.batch["response_mask"][:, 1:].bool()
    raw_advantages = batch.batch["raw_advantages"]
    prompt_length = prompt_mask.sum(-1).float()  # (batch_size,)
    response_length = response_mask.sum(-1).float()  # (batch_size,)
    returns = batch.batch["returns"]

    metrics = {
        # correct
        "critic/correct/mean": (sequence_score == 1).detach().float().mean().item(),
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        "critic/rewards_mean/mean": torch.mean(sequence_reward_mean).detach().item(),
        "critic/rewards_mean/max": torch.max(sequence_reward_mean).detach().item(),
        "critic/rewards_mean/min": torch.min(sequence_reward_mean).detach().item(),
        # adv
        "critic/advantages/mean": masked_mean(advantages, response_mask).detach().item(),
        "critic/advantages/max": torch.max(advantages[response_mask]).detach().item(),
        "critic/advantages/min": torch.min(advantages[response_mask]).detach().item(),
        # raw_adv
        "critic/raw_advantages/mean": masked_mean(raw_advantages, response_mask).detach().item(),
        "critic/raw_advantages/max": torch.max(raw_advantages[response_mask]).detach().item(),
        "critic/raw_advantages/min": torch.min(raw_advantages[response_mask]).detach().item(),
        # returns
        "critic/returns/mean": masked_mean(returns, response_mask).detach().item(),
        "critic/returns/max": torch.max(returns[response_mask]).detach().item(),
        "critic/returns/min": torch.min(returns[response_mask]).detach().item(),
        # response length
        "tokens/response_length/mean": torch.mean(response_length).detach().item(),
        "tokens/response_length/max": torch.max(response_length).detach().item(),
        "tokens/response_length/min": torch.min(response_length).detach().item(),
        # prompt length
        "tokens/prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "tokens/prompt_length/max": torch.max(prompt_length).detach().item(),
        "tokens/prompt_length/min": torch.min(prompt_length).detach().item(),
    }

    if "values" in batch.batch.keys():
        values = batch.batch["values"]
        # values
        metrics.update(
            {
                "critic/values/mean": masked_mean(values, response_mask).detach().item(),
                "critic/values/max": torch.max(values[response_mask]).detach().item(),
                "critic/values/min": torch.min(values[response_mask]).detach().item(),
            }
        )
    return metrics
