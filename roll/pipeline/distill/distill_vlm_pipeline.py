import copy
import json
import tqdm
import os
from functools import partial
from typing import Any, Dict, List, Tuple, Union, Optional

import ray
import torch
from torch.utils.data import DataLoader
import datasets
import PIL.Image as Image
from transformers.image_utils import load_images
from datasets import load_dataset, load_from_disk
from codetiming import Timer

from roll.datasets.collator import DataCollatorWithPaddingForMMWithLabels
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_processor_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.distill.distill_config import DistillConfig
from roll.utils.logging import get_logger
from roll.utils.metrics.metrics_manager import MetricsManager
from roll.pipeline.distill.logits_transfer_group import LogitsTransferGroup

from roll.pipeline.rlvr.rlvr_vlm_pipeline import process_images, get_extra_data_provider

logger = get_logger()

def format_prompt(prompt, processor, use_image=True, prompt_image_token=None):
    question_template = "{Question}  Output final answer (number) in <answer> </answer> tags."
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

def encode_function(
    data, processor, prompt_getter, ground_truth_getter, image_getter, prompt_image_token=None
):
    image_flag = [True] * len(prompt_getter(data))
    image_list = []
    for idx, image in enumerate(image_getter(data)):
        if image is None:
            image_flag[idx] = False
        try:
            image_out = load_images(image if isinstance(image, (list, tuple)) else [image], timeout=None)
        except Exception as e:
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
    for prompt, solution, has_img in zip(prompt_getter(data), ground_truth_getter(data), image_flag):
        # provide prompt_image_token if image_token in prompt
        prompt_text = format_prompt(prompt, processor, use_image=has_img, prompt_image_token=prompt_image_token)
        text = prompt_text + solution + processor.tokenizer.eos_token
        text_list.append(text)
    encodings = {
        "image": image_list,
        "text": text_list,
    }
    return encodings


def get_dataset(data_args, encode_function, processor, get_eval=False):
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
    FILEEXT2TYPE = {
        "arrow": "arrow",
        "csv": "csv",
        "json": "json",
        "jsonl": "json",
        "parquet": "parquet",
        "txt": "text",
    }
    if isinstance(data_name, list):
        local_path = ""
    else:
        local_path: str = os.path.join(dataset_dir, data_name)
    print(f"local_path: {local_path}")
    if os.path.isdir(local_path):
        for file_name in os.listdir(local_path):
            if file_name.startswith('.'):
                continue
            data_files.append(os.path.join(local_path, file_name))
            if data_path is None:
                data_path = FILEEXT2TYPE.get(file_name.split(".")[-1], None)
            elif data_path != FILEEXT2TYPE.get(file_name.split(".")[-1], None):
                raise ValueError("File types should be identical.")
    elif os.path.isfile(local_path):  # is file
        data_files.append(local_path)
        data_path = FILEEXT2TYPE.get(local_path.split(".")[-1], None)
    else:
        assert local_path == ""
        for file_name in data_name:
            data_files.append(os.path.join(dataset_dir, file_name))
            if data_path is None:
                data_path = FILEEXT2TYPE.get(file_name.split(".")[-1], None)
            elif data_path != FILEEXT2TYPE.get(file_name.split(".")[-1], None):
                raise ValueError("File types should be identical.")
    print(f"data_path: {data_path},   data_files: {data_files}")
    dataset = load_dataset(path=data_path, data_files=data_files)["train"]
    # regularized data filed
    features = datasets.Features(
        {
            "image": datasets.Sequence(feature=datasets.Image(mode=None, decode=True)),
            "text": datasets.Value(dtype="string"),
        }
    )
    remove_columns = list(dataset.features.keys() - features.keys())
    prompt_getter = lambda data: data["problem"]
    ground_truth_getter = lambda data: data['solution']
    image_getter = lambda data: data["image"]
    print(f"Begin : {dataset}")
    dataset = dataset.map(
        lambda data: encode_function(
            data, processor, prompt_getter, ground_truth_getter, image_getter,
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

def get_dataloader(dataset, batch_size, data_collator, num_proc=4):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_proc,  # larger shm for bigger num_workers
        collate_fn=data_collator,
    )
    return dataloader

class DistillVLMPipeline(BasePipeline):

    def __init__(self, pipeline_config: DistillConfig):
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config

        self.processor = default_processor_provider(self.pipeline_config.student.model_args)
        # set max_pixels to avoid image token num is larger than prompt length
        self.processor.image_processor.max_pixels, self.processor.image_processor.min_pixels = (
            getattr(self.pipeline_config.student.model_args, "max_pixels", 1024 * 1024),
            getattr(self.pipeline_config.student.model_args, "min_pixels", 56 * 56),
        )
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"

        # Load dataset
        dataset = get_dataset(
            self.pipeline_config.student.data_args, encode_function, self.processor, get_eval=False
        )

        print(f"roll student input: {dataset[0]}")

        data_collator = DataCollatorWithPaddingForMMWithLabels(
            tokenizer=self.tokenizer,
            processor=self.processor,
            extra_data_provider=get_extra_data_provider(
                self.pipeline_config.student.model_args.model_name_or_path, processor=self.processor
            ),
            prompt_key="text",
            image_key="image",
            answer_key=None,
            image_flag_key=None,
            max_length=self.pipeline_config.prompt_length,
            padding="max_length",
        )

        self.pipeline_config.set_max_steps(
            (self.pipeline_config.student.training_args.num_train_epochs * len(dataset)) // \
            (self.pipeline_config.student.training_args.per_device_train_batch_size * \
             self.pipeline_config.student.training_args.gradient_accumulation_steps))

        self.student: Any = Cluster(
            name=self.pipeline_config.student.name,
            worker_cls=self.pipeline_config.student.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.student,
        )
        self.teacher: Any = Cluster(
            name=self.pipeline_config.teacher.name,
            worker_cls=self.pipeline_config.teacher.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.teacher,
        )

        refs: List[ray.ObjectRef] = []
        refs.extend(self.student.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        refs: List[ray.ObjectRef] = []
        refs.extend(self.teacher.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        self.logits_transfer_group = LogitsTransferGroup(self.teacher, self.student,
                                                         backend=self.pipeline_config.logits_transfer_backend)

        self.dataloader = get_dataloader(dataset,
                                         self.pipeline_config.student.training_args.per_device_train_batch_size *\
                                         self.pipeline_config.student.training_args.gradient_accumulation_steps *\
                                         self.student.get_rank_info(0).dp_size,
                                         data_collator,
                                         num_proc=self.pipeline_config.student.training_args.dataloader_num_workers)

        self.set_checkpoint_clusters(self.student)

    @torch.no_grad()
    def run(self):
        metrics_mgr = MetricsManager()

        global_step = 1

        for epoch in range(self.pipeline_config.student.training_args.num_train_epochs):
            logger.info(f"epoch {epoch} start...")
            for batch_dict in self.dataloader:
                if global_step <= self.state.step:
                    global_step += 1
                    continue
                logger.info(f"pipeline step {global_step} start...")

                metrics_mgr.clear_metrics()

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info = {"global_step": global_step, "is_offload_states": False, "is_offload_optimizer_states_in_train_step": False}
                batch_offset = self.logits_transfer_group.apply_offset_by_dp(batch)
                with Timer(name="step_train", logger=None) as step_train_timer:
                    with Timer(name="teacher_forward", logger=None) as teacher_timer:
                        teacher_forward_metrics_refs = self.teacher.forward(batch_offset, blocking=False)
                        teacher_metric = DataProto.materialize_concat(
                            data_refs=teacher_forward_metrics_refs).meta_info.pop("metrics", {})
                    metrics_mgr.add_reduced_metrics(teacher_metric)

                    with Timer(name="logits_transfer", logger=None) as logits_transfer_timer:
                        logits_transfer_metrics = self.logits_transfer_group.logits_transfer()
                    metrics_mgr.add_reduced_metrics(logits_transfer_metrics)

                    with Timer(name="student_train_step", logger=None) as student_timer:
                        student_train_metrics_refs = self.student.train_step(batch, blocking=False)
                        student_train_metrics = DataProto.materialize_concat(data_refs=student_train_metrics_refs)
                        student_metric = student_train_metrics.meta_info.pop("metrics", {})
                    metrics_mgr.add_reduced_metrics(student_metric)
                metrics_mgr.add_metric("train/teacher_forward", teacher_timer.last)
                metrics_mgr.add_metric("train/student_train_step", student_timer.last)
                metrics_mgr.add_metric("train/step_train", step_train_timer.last)
                metrics = metrics_mgr.get_metrics()
                metrics = {k: float(v) for k, v in metrics.items()}

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
                    logger.info(json.dumps(metrics, ensure_ascii=False))

                logger.info(f"pipeline step {global_step} finished")
                global_step += 1
        logger.info("pipeline complete!")
