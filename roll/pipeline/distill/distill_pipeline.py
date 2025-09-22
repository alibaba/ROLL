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
from roll.utils.constants import IGNORE_INDEX

logger = get_logger()

def is_valid_example(example):
    """check if data are valid"""
    if "conversation" in example:
        for msg in example["conversation"]:
            if not msg.get("role") or not msg.get("content"):
                return False
    if "split" in example and example["split"] != "train":
        return False
    return True


def preprocess_dataset(dataset, tokenizer, pipeline_config):
    """
    Data preprocessing:
        - Automatically obtain template_name / keys / parameters from pipeline_config
        - Build encode_function
        - Filter out invalid data & apply map encoding
    """
    logger.info(f"Begin process dataset: {dataset}")

    template_name = (
        pipeline_config.global_template
        if getattr(pipeline_config, "global_template", None)
        else pipeline_config.student.data_args.template
    )

    num_proc = getattr(pipeline_config.student.data_args, "preprocessing_num_workers", 1)
    sequence_length = getattr(pipeline_config, "sequence_length", 2048)

    encode_func = get_encode_function(
        template_name=template_name,
        tokenizer=tokenizer,
        prompt_key=getattr(pipeline_config, "prompt_key", None),
        question_key=getattr(pipeline_config, "question_key", None),
        answer_key=getattr(pipeline_config, "answer_key", None),
        system_key=getattr(pipeline_config, "system_key", None),
        distill_on_prompt=getattr(pipeline_config, "distill_on_prompt", False),
        sequence_length=sequence_length
    )

    dataset = dataset.filter(
        is_valid_example,
        num_proc=num_proc,
        desc="Filtering dataset"
    )

    dataset = dataset.map(
        encode_func,
        batched=True,
        num_proc=num_proc,
        desc="Encoding dataset",
        load_from_cache_file=False,
    )

    logger.info(f"Encoding: {dataset}")
    return dataset


def get_encode_function(template_name, tokenizer, prompt_key, question_key, answer_key, system_key=None, distill_on_prompt=False, sequence_length=2048):
    chat_template_func = get_chat_template(template_name, tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def safe_get(batch, key, i):
        if key is None or key not in batch:
            return None
        value = batch[key]
        if isinstance(value, list) and i < len(value):
            return value[i]
        return None

    def build_conversation(system_prompt, prompt, query, response):
        conversation = []
        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": (prompt or "") + (("\n" + query) if query else "")})
        if response:
            conversation.append({"role": "assistant", "content": response})
        return conversation

    def encode_function(batch):
        tokenized_encodings = []
        responses = batch.get(answer_key, [None]*len(next(iter(batch.values()))))

        for i, response in enumerate(responses):
            system_prompt = safe_get(batch, system_key, i)
            prompt = safe_get(batch, prompt_key, i)
            query = safe_get(batch, question_key, i)

            # prompt text
            conv_prompt = build_conversation(system_prompt, prompt, query, None)
            prompt_text = chat_template_func(conv_prompt, add_generation_prompt=True)

            # full text
            conv_full = build_conversation(system_prompt, prompt, query, response)
            full_text = chat_template_func(conv_full, add_generation_prompt=False)
            if full_text.endswith("\n"):
                full_text = full_text[:-1]

            tokenized = tokenizer(full_text, truncation=True, max_length=sequence_length, padding="max_length")
            full_ids = tokenized["input_ids"]

            if distill_on_prompt:
                labels = [tid if tid != tokenizer.pad_token_id else IGNORE_INDEX for tid in full_ids]
            else:
                # match cut-off
                prompt_ids = tokenizer(prompt_text, padding=False)["input_ids"]
                cutoff = None
                for j in range(len(full_ids) - len(prompt_ids) + 1):
                    if full_ids[j:j+len(prompt_ids)] == prompt_ids:
                        cutoff = j + len(prompt_ids)
                        break
                if cutoff is None:
                    cutoff = len(prompt_ids)
                labels = [IGNORE_INDEX if idx < cutoff else (tid if tid != tokenizer.pad_token_id else IGNORE_INDEX)
                          for idx, tid in enumerate(full_ids)]

            tokenized["labels"] = labels
            tokenized_encodings.append(tokenized)

        return {k: [d[k] for d in tokenized_encodings] for k in tokenized_encodings[0]}

    return encode_function

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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dataset = preprocess_dataset(
            dataset,
            self.tokenizer,
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
