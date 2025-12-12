import copy
import json
import os
import uuid
from functools import partial
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
import PIL.Image as Image
import ray
import torch
from codetiming import Timer
from datasets import load_dataset, load_from_disk
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.timer import _Timer
from transformers import AutoConfig, ProcessorMixin
from transformers.image_utils import load_images
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from roll.datasets.collator import DataCollatorWithPaddingForMM
from roll.datasets.dataset import get_dataset
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.generate_scheduler import DynamicSamplingScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_processor_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.pipeline.rlvr.rlvr_pipeline import query_filter_fn, update_dataset_domain
from roll.utils.checkpoint_manager import download_model
from roll.utils.functionals import (
    RunningMoments,
    agg_loss,
    compute_advantage,
    compute_token_reward,
    get_sample_level_mask,
    reduce_metrics,
    reward_postprocess,
)
from roll.utils.kl_controller import get_kl_controller
from roll.utils.logging import get_logger
from roll.utils.metrics.metrics_manager import MetricsManager
from roll.utils.packages import is_transformers_version_greater_than


logger = get_logger()


def format_prompt(prompt, processor, use_image=True, prompt_image_token=None):
    question_template = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."
    if isinstance(prompt, list):
        messages = prompt
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question_template.format(Question=prompt)},
                ]
                if use_image and not prompt_image_token
                else [
                    {"type": "text", "text": question_template.format(Question=prompt)}
                ],  # image_token has been included in prompt
            }
        ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prompt_image_token:
        text = text.replace(prompt_image_token, "<|vision_start|><|image_pad|><|vision_end|>")
    return text


def process_image(image: Image.Image, processor: ProcessorMixin):
    # same as qwen2-vl image processor
    image_processor = processor.image_processor
    factor = (
        image_processor.patch_size * image_processor.merge_size
        if "Qwen" in image_processor.image_processor_type
        else 28
    )
    height, width = image.height, image.width
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=image_processor.min_pixels,
        max_pixels=image_processor.max_pixels,
    )
    resized_image = image.resize((resized_width, resized_height), resample=image_processor.resample)
    return resized_image


def process_images(
    images: Union[List, Tuple, str, Image.Image], processor: ProcessorMixin
) -> Union[Image.Image, List[Image.Image], List[List[Image.Image]]]:
    """Process images, handling different levels of nesting.

    Args:
      images: A single image, a list of images, or a list of lists of images to load.
      timeout: Timeout for loading images.

    Returns:
      A single image, a list of images, a list of lists of images.
    """
    if isinstance(images, (list, tuple)):
        if len(images) and isinstance(images[0], (list, tuple)):
            return [[process_image(image, processor=processor) for image in image_group] for image_group in images]
        else:
            return [process_image(image, processor=processor) for image in images]
    else:
        return process_image(images, processor=processor)


def encode_function(
    data, processor, prompt_getter, ground_truth_getter, image_getter, tag_getter, prompt_image_token=None
):
    image_flag = [True] * len(prompt_getter(data))
    image_list = []
    for idx, image in enumerate(image_getter(data)):
        if image is None:
            image_flag[idx] = False
        try:
            if isinstance(image, bytes): # bytes data
                # TODO: support multiple images
                image_out = Image.open(BytesIO(image))
            else:
                image_out = load_images(image if isinstance(image, (list, tuple)) else [image], timeout=None)
        except Exception as e:
            if isinstance(image, bytes):
                image_out = [Image.new("RGB", (224, 224), (255, 255, 255))]
                logger.error(f"Failed to get image with type: {type(image)}")
            else:
                image_out = [Image.new("RGB", (224, 224), (255, 255, 255))] * len(image)
                logger.error(f"Failed to get image: {image}")
        # since infer-image use pil image as input while train-engine use
        # processed data, process image here to make them use same image
        # refer to the following for Spatial Understanding with Qwen2.5-VL
        # https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/spatial_understanding.ipynb
        # NOTE: process_image from qwen2.5-vl keeps aspect ratio almostly and
        # bboxes would be normalized in detection verifier, thus nearly no need
        # to change ground-truth bboxes
        image_out = process_images(image_out, processor)
        image_list.append(image_out)
    text_list = []
    for idx, instruct in enumerate(prompt_getter(data)):
        # provide prompt_image_token if image_token in prompt
        text = format_prompt(instruct, processor, use_image=image_flag[idx], prompt_image_token=prompt_image_token)
        text_list.append(text)
    encodings = {
        "tag": tag_getter(data),
        "images": image_list,
        "prompt": text_list,
        "ground_truth": ground_truth_getter(data),
        "reward_model": data["reward_model"],
    }
    return encodings


def get_vlm_dataset(data_args, encode_function, processor, get_eval=False):
    cache_path = getattr(data_args, "cache_path", None)
    if cache_path:
        cache_path = os.path.join(cache_path, "val" if get_eval else "train")
    if cache_path and os.path.exists(cache_path):
        dataset = load_from_disk(cache_path)
        return dataset

    dataset = get_dataset(data_args=data_args)
    # regularized data filed
    features = datasets.Features(
        {
            "tag": datasets.Value(dtype="string"),  # from data_source
            "images": datasets.Sequence(feature=datasets.Image(mode=None, decode=True)),
            "prompt": datasets.Value(dtype="string"),
            "ground_truth": datasets.Value(dtype="string"),
            "reward_model": dataset.features["reward_model"],
        }
    )
    remove_columns = list(dataset.features.keys() - features.keys())
    # suit to both VLM-RL/Ocean-R1 and MiniMax-AI/One-RL-to-See-Them-All data
    prompt_getter = lambda data: data["prompt"]
    ground_truth_getter = lambda data: [x["ground_truth"] for x in data["reward_model"]]
    image_getter = lambda data: data["images"]
    tag_getter = lambda data: data["data_source"]
    print(f"Begin : {dataset}")
    dataset = dataset.map(
        lambda data: encode_function(
            data, processor, prompt_getter, ground_truth_getter, image_getter, tag_getter, prompt_image_token="<image>"
        ),
        batched=True,
        batch_size=100,
        num_proc=data_args.preprocessing_num_workers,
        features=features,
        remove_columns=remove_columns,
        desc="Encoding dataset",
    )
    print(f"Encoding: {dataset}")
    if cache_path:
        dataset.save_to_disk(cache_path)
    return dataset


def get_extra_data_provider(model_name_or_path: str, processor=None):
    model_name_or_path = download_model(model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if "qwen2" in config.model_type:
        import types

        from transformers import BatchFeature  # help define a object to accesss attr

        dummy_self = BatchFeature(
            {
                "config": BatchFeature(
                    {
                        "vision_config": BatchFeature({"spatial_merge_size": processor.image_processor.merge_size}),
                        "image_token_id": processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
                        "video_token_id": processor.tokenizer.convert_tokens_to_ids("<|video_pad|>"),
                        "vision_start_token_id": processor.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
                    }
                )
            }
        )
        if is_transformers_version_greater_than("4.52.0"):
            from transformers.models.qwen2_vl import Qwen2VLModel

            get_rope_index = types.MethodType(Qwen2VLModel.get_rope_index, dummy_self)
        else:
            from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration

            get_rope_index = types.MethodType(Qwen2VLForConditionalGeneration.get_rope_index, dummy_self)

        def extra_data_provider(
            input_ids: torch.LongTensor,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
        ):
            rope_index = get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)[0]
            # (3, bsz, seqlen) -> (bsz, 3, seqlen) to put it into DataProto,
            # transpose it batck to (3, bsz, seqlen) before forward for model
            rope_index = rope_index.transpose(0, 1)
            return {"position_ids": rope_index}

        return extra_data_provider

    def default_extra_data_provider(
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen = input_ids.shape
        position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
        if attention_mask is not None:
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        return {"position_ids": position_ids}

    return default_extra_data_provider


class RLVRVLMPipeline(BasePipeline):
    def __init__(self, pipeline_config: RLVRConfig):
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config

        self.processor = default_processor_provider(self.pipeline_config.actor_train.model_args)
        # set max_pixels to avoid image token num is larger than prompt length
        self.processor.image_processor.max_pixels, self.processor.image_processor.min_pixels = (
            getattr(self.pipeline_config.actor_train.model_args, "max_pixels", 1024 * 1024),
            getattr(self.pipeline_config.actor_train.model_args, "min_pixels", 56 * 56),
        )
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"

        dataset = get_vlm_dataset(
            self.pipeline_config.actor_train.data_args, encode_function, self.processor, get_eval=False
        )
        # update domain field, DynamicSamplingScheduler requires
        dataset = dataset.map(
            partial(update_dataset_domain, self.pipeline_config.tag_2_domain),
            num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
            desc="update_dataset_domain",
            load_from_cache_file=False,
        )

        self.domain_datasets: Dict[str, datasets.Dataset] = {}
        for domain in self.pipeline_config.actor_train.data_args.domain_interleave_probs.keys():
            self.domain_datasets[domain] = dataset.filter(
                lambda example, dom: example["domain"] == dom,
                num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
                fn_kwargs={"dom": domain},
            )
            assert len(self.domain_datasets[domain]) > 0, f"domain dataset {domain} has no data"

        self.val_dataset = None
        if self.pipeline_config.validation and self.pipeline_config.validation.data_args:
            self.val_dataset = get_vlm_dataset(
                self.pipeline_config.validation.data_args, encode_function, self.processor, get_eval=True
            )
            self.val_dataset = self.val_dataset.map(
                partial(update_dataset_domain, self.pipeline_config.tag_2_domain),
                num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
                desc="update_val_dataset_domain",
                load_from_cache_file=False,
            )

        self.kl_ctrl = get_kl_controller(
            init_kl_coef=self.pipeline_config.init_kl_coef,
            target_kl=self.pipeline_config.target_kl,
            kl_horizon=self.pipeline_config.kl_horizon,
        )

        assert self.pipeline_config.max_steps > 0, "max_steps must be greater than 0"
        self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)

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
        download_clusters = [self.actor_train, self.actor_infer]
        if self.pipeline_config.enable_reference:
            self.reference: Any = Cluster(
                name=self.pipeline_config.reference.name,
                worker_cls=self.pipeline_config.reference.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.reference,
            )
            download_clusters.append(self.reference)
        if self.pipeline_config.adv_estimator == "gae":
            self.critic: Any = Cluster(
                name=self.pipeline_config.critic.name,
                worker_cls=self.pipeline_config.critic.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.critic,
            )
            download_clusters.append(self.critic)
        # key must be same as domain, which is used in DynamicSamplingScheduler
        # to get corresponding reward
        self.rewards: Dict[str, Any] = {
            key: Cluster(
                name=f"reward-{key}",
                worker_cls=worker_config.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=worker_config,
            )
            for key, worker_config in self.pipeline_config.rewards.items()
        }
        download_clusters.extend(self.rewards.values())
        self.download_models(*download_clusters)

        domain_ratios = self.pipeline_config.actor_train.data_args.domain_interleave_probs
        self.generate_schedulers: Dict[str, DynamicSamplingScheduler] = {}
        self.domain_batch_size = {}
        domain_list = list(domain_ratios.keys())
        accumulated = 0
        for i, domain in enumerate(domain_list):
            if i == len(domain_list) - 1:
                domain_batch_size = self.pipeline_config.rollout_batch_size - accumulated
            else:
                domain_batch_size = int(domain_ratios[domain] * self.pipeline_config.rollout_batch_size)
            accumulated += domain_batch_size
            generate_scheduler = DynamicSamplingScheduler.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(), soft=False
                )
            ).remote(pipeline_config=self.pipeline_config)
            ray.get(
                generate_scheduler.set_scheduler.remote(
                    actor_cluster=self.actor_infer,
                    reward_clusters={domain: self.rewards[domain]},
                    dataset=self.domain_datasets[domain],
                    collect_fn_cls=DataCollatorWithPaddingForMM,
                    collect_fn_kwargs=dict(
                        # tokenizer passed by DynamicSamplingScheduler.set_scheduler
                        # tokenizer=self.tokenizer,
                        extra_unpadded_keys=["domain", "reward_model"],
                        extra_data_provider=get_extra_data_provider(
                            self.pipeline_config.actor_train.model_args.model_name_or_path, processor=self.processor
                        ),
                        prompt_key="prompt",
                        answer_key="ground_truth",
                        image_key="images",
                        image_flag_key=None,
                        max_length=self.pipeline_config.prompt_length,
                        padding="max_length",
                    ),
                    response_filter_fn=lambda data_item, config: True,
                    query_filter_fn=query_filter_fn,
                    response_callback_fn=generate_scheduler.report_response.remote,
                    state=self.state.kv.get(f"scheduler_state_{domain}", None),
                    is_vlm=True,
                )
            )
            self.generate_schedulers[domain] = generate_scheduler
            self.domain_batch_size[domain] = domain_batch_size

            assert domain_batch_size < len(self.domain_datasets[domain]), (
                f"domain_batch_size {domain_batch_size} must be "
                f"less than the number of domain datasets {len(self.domain_datasets[domain])}"
            )

        if self.val_dataset:
            val_pipeline_config = copy.deepcopy(self.pipeline_config)
            val_pipeline_config.is_use_additional_prompts = False
            self.val_generate_scheduler = DynamicSamplingScheduler.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(), soft=False
                )
            ).remote(pipeline_config=val_pipeline_config)
        if self.val_dataset:
            ray.get(
                self.val_generate_scheduler.set_scheduler.remote(
                    actor_cluster=self.actor_infer,
                    reward_clusters=self.rewards,
                    dataset=self.val_dataset,
                    collect_fn_cls=DataCollatorWithPaddingForMM,
                    collect_fn_kwargs=dict(
                        # tokenizer passed by DynamicSamplingScheduler.set_scheduler
                        # tokenizer=self.tokenizer,
                        # val metrics are grouped by tag rather than domain
                        extra_unpadded_keys=["domain", "reward_model", "tag"],
                        extra_data_provider=get_extra_data_provider(
                            self.pipeline_config.actor_train.model_args.model_name_or_path, processor=self.processor
                        ),
                        prompt_key="prompt",
                        answer_key="ground_truth",
                        image_key="images",
                        image_flag_key=None,
                        max_length=self.pipeline_config.prompt_length,
                        padding="max_length",
                    ),
                    response_filter_fn=lambda data_item, config: True,
                    query_filter_fn=lambda data_list, config: True,
                    response_callback_fn=self.val_generate_scheduler.report_response.remote,
                    is_vlm=True,
                )
            )

        refs = []
        refs.extend(self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        if self.pipeline_config.enable_reference:
            refs.extend(self.reference.initialize(pipeline_config=self.pipeline_config, blocking=True))
        refs = []
        for key, cluster in self.rewards.items():
            refs.extend(cluster.initialize(pipeline_config=self.pipeline_config, blocking=False))
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

        self.running = {}
        for domain in self.rewards.keys():
            self.running[domain] = RunningMoments()

    @torch.no_grad()
    def run(self):
        metrics_mgr = MetricsManager()

        tps_timer = _Timer(window_size=5)
        actor_infer_timer = _Timer(window_size=5)
        actor_infer_response_timer = _Timer(window_size=5)
        actor_train_timer = _Timer(window_size=5)

        metrics_mgr.timers["tps"] = tps_timer
        metrics_mgr.timers["actor_infer"] = actor_infer_timer
        metrics_mgr.timers["actor_infer_response"] = actor_infer_response_timer
        metrics_mgr.timers["actor_train"] = actor_train_timer

        for global_step in range(self.pipeline_config.max_steps):
            if global_step <= self.state.step:
                global_step += 1
                continue
            self.global_step = global_step
            logger.info(f"pipeline step {global_step} start...")

            metrics_mgr.clear_metrics()
            with tps_timer, Timer(name="step_total", logger=None) as step_total_timer:

                if self.pipeline_config.adv_estimator == "gae":
                    self.critic.offload_states(blocking=True)
                self.actor_train.offload_states(blocking=True)

                with Timer(name="step_model_update", logger=None) as step_model_update_timer:
                    model_update_metrics: Dict = self.model_update(global_step)
                    metrics_mgr.add_metrics(model_update_metrics)
                    metrics_mgr.add_metric("time/step_model_update", step_model_update_timer.last)

                if self.val_dataset and global_step % self.pipeline_config.eval_steps == 0:
                    with Timer(name="val_step", logger=None) as val_step_timer:
                        val_metrics = self.val()
                        metrics_mgr.add_metrics(val_metrics)
                        metrics_mgr.add_metric("time/val_step", val_step_timer.last)

                batch: DataProto = DataProto()
                batch.meta_info = {"global_step": global_step}

                # 要按domain group by生成对应的batch
                with actor_infer_timer, actor_infer_response_timer, Timer(
                    name="step_generate", logger=None
                ) as step_generate_timer:
                    domain_batches = {}
                    batch.meta_info["generation_config"] = self.actor_infer.worker_config.generating_args.to_dict()
                    self.actor_infer.start_server(data=DataProto(meta_info=batch.meta_info))
                    for reward_cluster in self.rewards.values():
                        reward_cluster.load_states()

                    batch.meta_info["is_offload_states"] = False
                    # meta mainly for dynamic reward threshold, such as global_step/max_steps
                    batch.meta_info.update(
                        {
                            "global_step": self.global_step,
                            "max_steps": self.pipeline_config.max_steps,
                            "is_training": True,
                        }
                    )
                    scheduler_refs = {}
                    for domain, scheduler in self.generate_schedulers.items():
                        scheduler_refs[domain] = scheduler.get_batch.remote(
                            data=batch, batch_size=self.domain_batch_size[domain]
                        )
                    for domain, scheduler_ref in scheduler_refs.items():
                        domain_batch: DataProto = ray.get(scheduler_ref, timeout=self.pipeline_config.rpc_timeout)
                        metrics_mgr.add_domain_metrics(
                            domain, reduce_metrics(domain_batch.meta_info.pop("metrics", {}))
                        )
                        domain_batches[domain] = domain_batch
                    generate_output = DataProto.concat([domain_batch for domain_batch in domain_batches.values()])
                    generate_output.meta_info.pop("is_offload_states", None)

                    for reward_cluster in self.rewards.values():
                        reward_cluster.offload_states()
                    gen_metrics = self.actor_infer.stop_server()
                    metrics_mgr.add_metrics(reduce_metrics(gen_metrics.meta_info.pop("metrics", {})))
                metrics_mgr.add_metric("time/step_generate", step_generate_timer.last)

                batch = generate_output
                # mark here to make megatron get_data_input broadcast with non_batch_tensor
                batch.meta_info["_broadcast_non_tensor_batch"]= True

                batch.non_tensor_batch['sample_uuid'] = np.array([str(uuid.uuid4()) for _ in range(batch.batch.shape[0])], dtype=object)
                with Timer(name="cal_ref_log_probs", logger=None) as cal_ref_log_probs_timer:
                    if self.pipeline_config.enable_reference:
                        ref_log_probs = self.reference.compute_log_probs(batch, blocking=True)
                        metrics_mgr.add_reduced_metrics(ref_log_probs.meta_info.pop("metrics", {}))
                        ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                        batch = batch.union(ref_log_probs)
                metrics_mgr.add_metric("time/ref_log_probs_values", cal_ref_log_probs_timer.last)

                with Timer(name="cal_old_log_probs_values", logger=None) as cal_old_logpb_timer:
                    batch.meta_info["is_offload_states"] = False
                    if self.pipeline_config.adv_estimator == "gae":
                        values_refs: List[ray.ObjectRef] = self.critic.compute_values(batch, blocking=False)

                    if self.pipeline_config.enable_old_logprobs_recompute:
                        old_log_probs_refs: List[ray.ObjectRef] = self.actor_train.compute_log_probs(batch, blocking=False)
                        old_log_probs = DataProto.materialize_concat(data_refs=old_log_probs_refs)
                        agg_entropy = agg_loss(
                            loss_mat=old_log_probs.batch["entropy"],
                            loss_mask=batch.batch["response_mask"][:, 1:],
                            loss_agg_mode="token-mean",
                        )
                        batch.meta_info["agg_entropy"] = agg_entropy

                        batch.batch["old_log_probs"] = old_log_probs.batch["log_probs"]
                        metrics_mgr.add_reduced_metrics(old_log_probs.meta_info.pop("metrics", {}))
                    else:
                        # Use zeros when optimization is enabled
                        batch.batch["old_log_probs"] = torch.zeros_like(batch.batch["attention_mask"][:, 1:])

                    if self.pipeline_config.adv_estimator == "gae":
                        values = DataProto.materialize_concat(data_refs=values_refs)
                        batch = batch.union(values)
                        metrics_mgr.add_reduced_metrics(values.meta_info.pop("metrics", {}))

                    # Mock ref_log_probs using old_log_probs if reference is disabled
                    if not self.pipeline_config.enable_reference:
                        batch.batch["ref_log_probs"] = batch.batch["old_log_probs"].clone()
                metrics_mgr.add_metric("time/old_log_probs", cal_old_logpb_timer.last)

                # group by domain to process reward
                batch.batch["prompt_id"] = torch.arange(batch.batch.batch_size[0], device=batch.batch.device)
                batch_grouped: Dict[str, DataProto] = batch.group_by("domain")
                batch_list = []
                for domain, domain_batch in batch_grouped.items():
                    # 1. get sample level mask
                    with Timer(name="get_sample_level_mask", logger=None) as get_sample_level_mask_timer:
                        domain_batch, mask_metrics = get_sample_level_mask(domain_batch, self.pipeline_config)
                        metrics_mgr.add_domain_metrics(domain, mask_metrics)
                    metrics_mgr.add_metric("time/get_sample_level_mask", get_sample_level_mask_timer.last)

                    # 2. process reward
                    with Timer(name="reward_postprocess", logger=None) as reward_postprocess_timer:
                        domain_batch, response_level_metrics = reward_postprocess(
                            domain_batch, self.pipeline_config, self.running
                        )
                        metrics_mgr.add_domain_metrics(domain, response_level_metrics)
                    metrics_mgr.add_domain_metrics(domain, {"time/reward_postprocess": reward_postprocess_timer.last})

                    # 3. compute token level rewards
                    with Timer(name="get_token_reward", logger=None) as get_token_reward_timer:
                        domain_batch, token_level_metrics = compute_token_reward(
                            domain_batch, self.pipeline_config, self.kl_ctrl
                        )
                        metrics_mgr.add_domain_metrics(domain, token_level_metrics)
                    metrics_mgr.add_domain_metrics(domain, {"time/get_token_reward": get_token_reward_timer.last})

                    # 4. compute advantage
                    final_response_mask = domain_batch.batch["final_response_mask"].clone()
                    with Timer(name="compute_advantage", logger=None) as compute_advantage_timer:
                        domain_batch = compute_advantage(
                            data=domain_batch,
                            gamma=self.pipeline_config.gamma,
                            lambd=self.pipeline_config.lambd,
                            adv_estimator=self.pipeline_config.adv_estimator,
                            advantage_clip=self.pipeline_config.advantage_clip,
                            whiten_advantages=self.pipeline_config.whiten_advantages,
                            whiten_rewards=self.pipeline_config.whiten_rewards,
                            response_mask=final_response_mask,
                        )
                        domain_metrics = reduce_metrics(domain_batch.meta_info.pop("metrics", {}))
                        metrics_mgr.add_domain_metrics(domain, domain_metrics)
                        batch_list.append(domain_batch)
                    metrics_mgr.add_domain_metrics(domain, {"time/compute_advantage": compute_advantage_timer.last})

                batch = DataProto.concat(batch_list)
                batch.reorder(indices=torch.argsort(batch.batch["prompt_id"]))
                batch.pop("prompt_id")

                metrics_mgr.add_all_metrics(
                    global_step,
                    batch,
                    resource_manager=self.resource_manager,
                    actor_infer=self.actor_infer,
                    actor_train=self.actor_train,
                )
                batch_grouped: Dict[str, DataProto] = batch.group_by("domain")
                metrics_mgr.add_domain_all_metrics(global_step, batch_grouped)

                with Timer(name="step_train", logger=None) as step_train_timer:
                    if self.pipeline_config.adv_estimator == "gae":
                        critic_train_metrics_refs: List[ray.ObjectRef] = self.critic.train_step(batch, blocking=False)

                    with actor_train_timer:
                        # implement critic warmup
                        if self.pipeline_config.critic_warmup <= global_step:
                            # update actor
                            actor_train_metrics_refs = self.actor_train.train_step(batch, blocking=False)
                            actor_train_metrics: DataProto = DataProto.materialize_concat(
                                data_refs=actor_train_metrics_refs
                            )
                            metrics_mgr.add_reduced_metrics(actor_train_metrics.meta_info.pop("metrics", {}))

                    if self.pipeline_config.adv_estimator == "gae":
                        critic_train_metrics = DataProto.materialize_concat(data_refs=critic_train_metrics_refs)
                        metrics_mgr.add_reduced_metrics(critic_train_metrics.meta_info.pop("metrics", {}))

                    metrics_mgr.add_metric("time/step_train", step_train_timer.last)

                tps_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())
                actor_infer_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())
                actor_infer_response_timer.push_units_processed(
                    n=torch.sum(batch.batch["response_mask"]).detach().item()
                )
                actor_train_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())

                metrics = metrics_mgr.get_metrics()
                # do ckpt
                self.state.step = global_step
                self.state.log_history.append(metrics)
                for domain, scheduler in self.generate_schedulers.items():
                    self.state.kv[f"scheduler_state_{domain}"] = ray.get(scheduler.get_scheduler_state.remote())

                self.do_checkpoint(global_step=global_step)

                self.tracker.log(values=metrics, step=global_step)

                if global_step % self.pipeline_config.logging_steps == 0:
                    if int(os.environ.get("RAY_PROFILING", "0")):
                        timeline_dir = os.path.join(self.pipeline_config.profiler_output_dir, "timeline")
                        os.makedirs(timeline_dir, exist_ok=True)
                        ray.timeline(
                            filename=os.path.join(timeline_dir, f"timeline-step-{global_step}.json"),
                        )

                    prompts = self.tokenizer.batch_decode(generate_output.batch["prompts"], skip_special_tokens=True)
                    responses = self.tokenizer.batch_decode(
                        generate_output.batch["responses"], skip_special_tokens=True
                    )
                    generate_examples = [{"prompt": p, "response": r} for p, r in zip(prompts, responses)][:10]
                    logger.info(json.dumps(generate_examples, ensure_ascii=False))
                    logger.info(json.dumps(metrics, ensure_ascii=False))

                logger.info(f"pipeline step {global_step} finished")
                global_step += 1
        logger.info("pipeline complete!")

    @torch.no_grad()
    def val(self):
        val_metrics_mgr = MetricsManager()
        batch = DataProto()

        with Timer(name="step_generate", logger=None) as step_generate_timer:
            batch.meta_info["is_offload_states"] = False
            batch.meta_info["generation_config"] = self.pipeline_config.validation.generating_args.to_dict()
            batch.meta_info.update(
                {"global_step": self.global_step, "max_steps": self.pipeline_config.max_steps, "is_training": False}
            )

            self.actor_infer.start_server(data=DataProto(meta_info=batch.meta_info))
            for reward_cluster in self.rewards.values():
                reward_cluster.load_states()
            generate_output: DataProto = ray.get(
                self.val_generate_scheduler.get_batch.remote(data=batch, batch_size=len(self.val_dataset)),
                timeout=self.pipeline_config.rpc_timeout,
            )
            self.actor_infer.stop_server()
            generate_output.meta_info.pop("is_offload_states", None)
            for reward_cluster in self.rewards.values():
                reward_cluster.offload_states()
            val_metrics_mgr.add_metric("time/step_generate", step_generate_timer.last)

        batch = generate_output
        val_score_mean = batch.batch["scores"].detach().float().mean().item()
        val_metrics_mgr.add_metric("val_score/all/mean", val_score_mean)
        logger.info(json.dumps({"val_score/all/mean": val_score_mean}, ensure_ascii=False))

        epoch_batch = batch.pop(batch_keys=["scores"], non_tensor_batch_keys=["tag"])

        grouped_batch = epoch_batch.group_by("tag")
        for group_key, group_batch in grouped_batch.items():
            score_mean = group_batch.batch["scores"].mean().item()
            print(f"{group_key}:  {score_mean}")
            val_metrics_mgr.add_domain_metrics(
                "val_score", {f"{group_key}/mean": group_batch.batch["scores"].detach().float().mean().item()}
            )

        return val_metrics_mgr.get_metrics()
