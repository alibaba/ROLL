import copy
import json
import tqdm
import os
from functools import partial
from typing import Any, Dict, List

import datasets
import ray
import torch
from torch.utils.data import DataLoader
from codetiming import Timer
from ray.util.timer import _Timer

from roll.datasets.chat_template import get_chat_template
from roll.datasets.collator import DataCollatorWithPaddingForPaddedKeys
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.distill.distill_config import DistillConfig
from roll.utils.logging import get_logger
from roll.utils.metrics.metrics_manager import MetricsManager

logger = get_logger()


def is_valid_example(example):
    for i, msg in enumerate(example["conversation"]):
        if msg.get("role") is None or msg.get("content") is None:
            return False
    if ('split' in example) and (example['split'] != 'train'):
        return False
    return True


def preprocess_dataset(dataset, template_function, encode_function, pipeline_config):
    num_proc = pipeline_config.student.data_args.preprocessing_num_workers
    dataset = dataset.map(
        sample2conversation,
        batched=True,
        num_proc=num_proc,
        desc="Sample to conversation",
        load_from_cache_file=False,
        fn_kwargs={'question_key': pipeline_config.question_key, 'answer_key': pipeline_config.answer_key}
    )
    dataset = dataset.filter(
        is_valid_example,
        num_proc=num_proc,
        desc="Filtering dataset"
    )
    dataset = dataset.map(
        template_function,
        batched=True,
        num_proc=num_proc,
        desc="Apply template",
        load_from_cache_file=False,
    )
    dataset = dataset.map(
        encode_function,
        batched=True,
        num_proc=num_proc,
        desc="Encoding dataset",
        load_from_cache_file=False,
    )

    return dataset


def sample2conversation(examples, *, question_key, answer_key):
    conversations = []

    for i in range(len(examples[question_key])):
        conversation = []
        conversation.append({"role": "user", "content": examples[question_key][i]})
        conversation.append({"role": "assistant", "content": examples[answer_key][i]})

        conversations.append(conversation)

    return {"conversation": conversations}


def get_template_function(tokenizer):
    def template_function_batch(examples):
        text = [
            tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False
            )
            for conversation in examples["conversation"]
        ]
        return {"text": text}

    return template_function_batch


def get_tokenize_function(tokenizer, pipeline_config):
    def tokenize_function_batch(examples):
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=pipeline_config.sequence_length,
            return_tensors="pt"
        )
        input_ids_list = model_inputs["input_ids"].tolist()
        labels = [
            [-100 if tid == tokenizer.pad_token_id else tid for tid in input_ids]
            for input_ids in input_ids_list
        ]
        return {
            "input_ids": input_ids_list,
            "attention_mask": model_inputs["attention_mask"].tolist(),
            "labels": labels
        }
    return tokenize_function_batch


def get_dataloader(dataset, batch_size, data_collator, num_proc):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_proc,
        collate_fn=data_collator,
    )
    return dataloader


class DistillPipeline(BasePipeline):

    def __init__(self, pipeline_config: DistillConfig):
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config

        # Load dataset
        dataset_paths = []
        if self.pipeline_config.student.data_args.file_name:
            dataset_paths.extend(self.pipeline_config.student.data_args.file_name)
        if not dataset_paths:
            raise ValueError("No dataset paths provided")
        print(f'load_dataset_paths: {chr(10)} {chr(10).join(dataset_paths)}')
        dataset = datasets.load_dataset('json', data_files=dataset_paths)['train']

        # Currently, only models where the student and teacher are of the same type are supported.
        self.tokenizer = default_tokenizer_provider(model_args=self.pipeline_config.student.model_args)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        template_function = get_template_function(self.tokenizer)
        encode_function = get_tokenize_function(self.tokenizer, self.pipeline_config)

        dataset = preprocess_dataset(
            dataset,
            template_function,
            encode_function,
            pipeline_config,
        )

        data_collator = DataCollatorWithPaddingForPaddedKeys(
            tokenizer=self.tokenizer,
            padding="longest",
        )

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
                with Timer(name="step_train", logger=None) as step_train_timer:
                    with Timer(name="teacher_forward", logger=None) as teacher_timer:
                        teacher_logits_handles = self.teacher.forward(batch, blocking=True)
                    batch.meta_info['teacher_logits_handles'] = teacher_logits_handles
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
