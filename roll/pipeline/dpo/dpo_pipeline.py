import copy
import json
import os
from typing import Any, Dict, List

import datasets
import ray
import torch
from codetiming import Timer
from ray.util.timer import _Timer
from torch.utils.data import DataLoader
from tqdm import tqdm

from roll.datasets.chat_template import get_chat_template
from roll.datasets.collator import DataCollatorWithPaddingForDPO
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.dpo.actor_worker import get_logps, loss_fn
from roll.pipeline.dpo.dpo_config import DPOConfig
from roll.utils.logging import get_logger
from roll.utils.metrics.metrics_manager import MetricsManager

logger = get_logger()


def preprocess_dataset(dataset, prompt_len, encode_function, num_proc):
    logger.info(f"Begin : {dataset}")
    dataset = dataset.map(
        encode_function,
        batched=True,
        num_proc=num_proc,
        desc="Encoding dataset",
        load_from_cache_file=False,
    )
    dataset = dataset.filter(
        lambda data_i: len(data_i["chosen_ids"]) <= prompt_len and len(data_i["reject_ids"]) <= prompt_len,
        num_proc=num_proc,
        desc="Filtering dataset",
    )
    logger.info(f"Filtering prompt len: {dataset}")
    logger.info(f"Encoding: {dataset}")
    return dataset


def get_encode_function(template_name, tokenizer, chosen_key, rejected_key):
    chat_template_func = get_chat_template(template_name, tokenizer)

    def build_conversation(instruction: str, response: str = None):
        conversation = [{"role": "user", "content": instruction}]
        if response is not None:
            conversation.append({"role": "assistant", "content": response})
        return conversation

    def encode_function(data_i):
        instructions = data_i["instruction"]
        chosens = data_i[chosen_key]
        rejecteds = data_i[rejected_key]
        chosen_texts = []
        rejected_texts = []
        prompt_texts = []
        for inst, chosen, rejected in zip(instructions, chosens, rejecteds):
            prompt_conversation = build_conversation(inst)  # prompt only
            chosen_conversation = build_conversation(inst, chosen)  # prompt + chosen
            rejected_conversation = build_conversation(inst, rejected)  # prompt + rejected

            prompt_text = chat_template_func(prompt_conversation)
            chosen_text = chat_template_func(chosen_conversation)
            rejected_text = chat_template_func(rejected_conversation)

            prompt_texts.append(prompt_text)
            chosen_texts.append(chosen_text)
            rejected_texts.append(rejected_text)

        prompt_encodings = tokenizer(prompt_texts)
        prompt_ids_lens = [len(ids) for ids in prompt_encodings["input_ids"]]

        chosen_encodings = tokenizer(chosen_texts)
        rejected_encodings = tokenizer(rejected_texts)

        return {
            "chosen_ids": chosen_encodings["input_ids"],
            "c_mask": chosen_encodings["attention_mask"],
            "reject_ids": rejected_encodings["input_ids"],
            "r_mask": rejected_encodings["attention_mask"],
            "prompt_ids_lens": prompt_ids_lens,
        }

    return encode_function


class DPOPipeline(BasePipeline):
    def __init__(self, pipeline_config: DPOConfig):
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config

        self.tokenizer = default_tokenizer_provider(model_args=self.pipeline_config.actor_train.model_args)

        dataset_paths = []
        dataset_paths.extend(self.pipeline_config.actor_train.data_args.file_name)
        logger.info(f"load_dataset_paths: {chr(10)} {chr(10).join(dataset_paths)}")
        self.dataset = datasets.load_dataset("json", data_files=dataset_paths)["train"]
        template_name = (
            self.pipeline_config.global_template
            if self.pipeline_config.global_template
            else self.pipeline_config.actor_train.data_args.template
        )
        encode_function = get_encode_function(template_name, self.tokenizer, self.pipeline_config.chosen_key, self.pipeline_config.rejected_key)
        self.dataset = preprocess_dataset(
            self.dataset,
            self.pipeline_config.sequence_length,
            encode_function,
            num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
        )
        data_collator = DataCollatorWithPaddingForDPO(
            tokenizer=self.tokenizer,
            max_length=self.pipeline_config.sequence_length,
        )
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.pipeline_config.train_batch_size,  # actual batch size is 2*batch_size, as there are chosen and rejected
            shuffle=True,
            drop_last=True,
            num_workers=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
            collate_fn=data_collator,
        )

        self.val_dataset = None
        if self.pipeline_config.validation.data_args:
            val_dataset_paths = self.pipeline_config.validation.data_args.file_name
            self.val_dataset = datasets.load_dataset("json", data_files=val_dataset_paths)["train"]
            self.val_dataset = preprocess_dataset(
                self.val_dataset,
                self.pipeline_config.sequence_length,
                encode_function,
                num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
            )
            self.val_dataloader = DataLoader(
                dataset=self.val_dataset,
                batch_size=self.pipeline_config.val_batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
                collate_fn=data_collator,
            )

        assert self.pipeline_config.max_steps > 0, "max_steps must be greater than 0"
        self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)

        self.actor_train: Any = Cluster(
            name=self.pipeline_config.actor_train.name,
            worker_cls=self.pipeline_config.actor_train.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_train,
        )
        self.reference: Any = Cluster(
            name=self.pipeline_config.reference.name,
            worker_cls=self.pipeline_config.reference.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.reference,
        )

        if self.val_dataset:
            val_pipeline_config = copy.deepcopy(self.pipeline_config)
            val_pipeline_config.is_use_additional_prompts = False

        refs: List[ray.ObjectRef] = []
        refs.extend(self.reference.initialize(pipeline_config=self.pipeline_config, blocking=False))

        refs: List[ray.ObjectRef] = []
        refs.extend(self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=False))

        self.set_checkpoint_clusters(self.actor_train)

    @torch.no_grad()
    def run(self):
        global_step = 0
        metrics_mgr = MetricsManager()

        for epoch in range(int(self.pipeline_config.actor_train.training_args.num_train_epochs)):
            logger.info(f"epoch {epoch} start...")
            for batch_dict in tqdm(self.dataloader):
                if global_step <= self.state.step:
                    global_step += 1
                    continue

                logger.info(f"pipeline step {global_step} start...")
                metrics_mgr.clear_metrics()

                if self.val_dataset and global_step % self.pipeline_config.eval_steps == 0:
                    with Timer(name="val_step", logger=None) as val_step_timer:
                        val_metrics = self.val()
                        metrics_mgr.add_reduced_metrics(val_metrics)
                    metrics_mgr.add_metric("time/val_step", val_step_timer.last)

                with Timer(name="step_total", logger=None) as step_total_timer:
                    batch_dict: Dict
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    batch.meta_info = {"global_step": global_step, "is_offload_states": False, "is_offload_optimizer_states_in_train_step": False}

                    with Timer(name="cal_ref_log_probs", logger=None) as cal_ref_log_probs_timer:
                        ref_log_probs = self.reference.compute_log_probs(batch, blocking=True)
                        metrics_mgr.add_reduced_metrics(ref_log_probs.meta_info.pop("metrics", {}))
                        ref_log_probs.rename(old_keys="log_probs", new_keys="reference_log_probs")
                        batch = batch.union(ref_log_probs)
                    metrics_mgr.add_metric("time/cal_ref_log_probs", cal_ref_log_probs_timer.last)

                    with Timer(name="actor_train", logger=None) as actor_train_timer:
                        actor_train_refs = self.actor_train.train_step(batch, blocking=False)
                        actor_train_refs: DataProto = DataProto.materialize_concat(data_refs=actor_train_refs)
                        metrics_mgr.add_reduced_metrics(actor_train_refs.meta_info.pop("metrics", {}))
                    metrics_mgr.add_metric("time/actor_train", actor_train_timer.last)
                metrics_mgr.add_metric("time/step_total", step_total_timer.last)

                metrics = metrics_mgr.get_metrics()
                metrics = {k: float(v) for k, v in metrics.items()}

                self.state.step = global_step
                self.state.log_history.append(metrics)
                self.tracker.log(values=metrics, step=global_step)
                self.do_checkpoint(global_step=global_step)

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
                if global_step >= self.pipeline_config.max_steps:
                    break
            
            if global_step >= self.pipeline_config.max_steps:
                break

        logger.info("pipeline complete!")

    @torch.no_grad()
    def val(self):
        metrics = {}
        for batch_dict in tqdm(self.val_dataloader):
            batch_dict: Dict
            batch: DataProto = DataProto.from_single_dict(batch_dict)

            with Timer(name="cal_ref_log_probs", logger=None) as cal_ref_log_probs_timer:
                ref_log_probs = self.reference.compute_log_probs(batch, blocking=True)
                metrics.update(ref_log_probs.meta_info.pop("metrics", {}))
                ref_log_probs.rename(old_keys="log_probs", new_keys="reference_log_probs")
                batch = batch.union(ref_log_probs)
            metrics["time/cal_ref_log_probs"] = cal_ref_log_probs_timer.last

            with Timer(name="cal_log_probs", logger=None) as cal_log_probs_timer:
                log_probs = self.actor_train.compute_log_probs(batch, blocking=True)
                metrics.update(log_probs.meta_info.pop("metrics", {}))
                batch = batch.union(log_probs)
            metrics["time/cal_log_probs"] = cal_log_probs_timer.last

            reference_chosen_logps, reference_rejected_logps = get_logps(
                batch.batch["reference_log_probs"], batch.batch["attention_mask"], batch.batch["prompt_id_lens"]
            )
            chosen_logps, rejected_logps = get_logps(
                batch.batch["log_probs"], batch.batch["attention_mask"], batch.batch["prompt_id_lens"]
            )

            ipo = batch.meta_info.get("ipo", False)
            beta = batch.meta_info.get("beta", 0.1)
            label_smoothing = batch.meta_info.get("label_smoothing", 0.0)

            loss, chosen_rewards, reject_rewards = loss_fn(
                chosen_logps,
                rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                ipo,
                beta,
                label_smoothing,
            )
            acc = (chosen_rewards > reject_rewards).float().mean().item()

            metrics = {
                "actor/loss": loss.item(),
                "actor/acc": acc,
                "actor/chosen_reward": chosen_rewards.mean().item(),
                "actor/reject_reward": reject_rewards.mean().item(),
            }

        val_metrics = {f"val/{k}": v for k, v in metrics.items()}
        return val_metrics
