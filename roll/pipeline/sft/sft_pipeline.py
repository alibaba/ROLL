from typing import Any
import os

import datasets
import numpy as np
import ray
import torch
from tqdm import tqdm
from codetiming import Timer
from torch.utils.data import DataLoader

from roll.datasets.chat_template import get_chat_template
from roll.datasets.collator import DataCollatorForSFT
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.sft.sft_config import SFTConfig
from roll.pipeline.sft.step_planning import resolve_sft_step_plan
from roll.utils.constants import IGNORE_INDEX
from roll.utils.logging import get_logger
from roll.utils.metrics.metrics_manager import MetricsManager
from roll.utils.functionals import batch_balance, reduce_metrics
from roll.utils.worker_state import WorkerState

logger = get_logger()


def preprocess_dataset(dataset, prompt_len, encode_func, num_proc):
    logger.info(f"Begin process dataset: {dataset}")
    dataset = dataset.map(
        encode_func,
        batched=True,
        num_proc=num_proc,
        desc="Encoding dataset",
        load_from_cache_file=False,
    )
    logger.info(f"Encoding: {dataset}")
    return dataset


def get_encode_function(template_name, tokenizer, prompt_key, query_key, response_key, system_key=None):
    chat_template_func = get_chat_template(template_name, tokenizer)

    def build_conversation(system_prompt, prompt, query, response):
        conversation = []
        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": prompt + ("\n" + query if query else "")})
        if response:
            conversation.append( {"role": "assistant", "content": response})
        return conversation

    def encode_function(data_i):
        system_prompts = data_i[system_key] if system_key else None
        prompts = data_i[prompt_key]
        querys = data_i[query_key] if query_key else None
        responses = data_i[response_key]

        tokenized_encodings = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            system_prompt = system_prompts[i] if isinstance(system_prompts, list) else None
            query = querys[i] if isinstance(querys, list) else None

            conversation = build_conversation(system_prompt, prompt, query, None)
            prompt_text = chat_template_func(conversation, add_generation_prompt=True)

            conversation = build_conversation(system_prompt, prompt, query, response)
            prompt_with_response_text = chat_template_func(conversation, add_generation_prompt=False) # avoid add <assistant/>
            # some template (like qwen) add `\n` in the end, remove it
            if prompt_with_response_text[-1] == "\n":
                prompt_with_response_text = prompt_with_response_text[:-1]

            tokenized_encoding = tokenizer(prompt_with_response_text)
            prompt_token_ids_len = len(tokenizer(prompt_text)["input_ids"])

            labels = [IGNORE_INDEX] * prompt_token_ids_len + tokenized_encoding["input_ids"][prompt_token_ids_len:]

            tokenized_encoding.update({"labels": labels})
            tokenized_encodings.append(tokenized_encoding)

        return {key: [tokenized_encoding[key] for tokenized_encoding in tokenized_encodings] for key in tokenized_encodings[0].keys()}

    return encode_function


class SFTPipeline(BasePipeline):
    def __init__(self, pipeline_config: SFTConfig):
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config

        self.tokenizer = default_tokenizer_provider(self.pipeline_config.sft_train.model_args)
        self.tokenizer.padding_side = "right" # padding should be on right in sft

        dataset_paths = []
        train_file_name = self.pipeline_config.sft_train.data_args.file_name
        if train_file_name:
            if isinstance(train_file_name, list):
                dataset_paths.extend(train_file_name)
            elif isinstance(train_file_name, str):
                dataset_paths.append(train_file_name)
        logger.info(f"load_dataset_paths: {chr(10)} {chr(10).join(dataset_paths)}")
        self.dataset = datasets.load_dataset("json", data_files=dataset_paths)["train"]

        self.val_dataset = None
        if self.pipeline_config.validation and self.pipeline_config.validation.data_args:
            val_dataset_paths = self.pipeline_config.validation.data_args.file_name
            self.val_dataset = datasets.load_dataset("json", data_files=val_dataset_paths)["train"]

        template_name = (
            self.pipeline_config.global_template
            if self.pipeline_config.global_template
            else self.pipeline_config.sft_train.data_args.template
        )
        encode_function = get_encode_function(template_name, self.tokenizer,
                                              self.pipeline_config.prompt_key,
                                              self.pipeline_config.query_key,
                                              self.pipeline_config.response_key,
                                              self.pipeline_config.system_key)
        self.dataset = preprocess_dataset(
            self.dataset,
            self.pipeline_config.sequence_length,
            encode_function,
            num_proc=self.pipeline_config.sft_train.data_args.preprocessing_num_workers)

        data_collator = DataCollatorForSFT(
            tokenizer=self.tokenizer,
            padding="max_length",
            max_length=self.pipeline_config.sequence_length,
            padded_keys=["input_ids", "attention_mask"],
            label_pad_token_id=IGNORE_INDEX,
        )

        training_args = self.pipeline_config.sft_train.training_args
        configured_max_steps = self.pipeline_config.max_steps
        original_num_train_epochs = training_args.num_train_epochs
        strategy_args = self.pipeline_config.sft_train.strategy_args
        strategy_config = getattr(strategy_args, "strategy_config", {}) or {}
        world_size = self.pipeline_config.sft_train.world_size
        if (configured_max_steps > 0 and world_size and
                getattr(strategy_args, "strategy_name", None) == "megatron_train"):
            model_parallel_size = (
                strategy_config.get("tensor_model_parallel_size", 1)
                * strategy_config.get("pipeline_model_parallel_size", 1)
                * strategy_config.get("context_parallel_size", 1)
            )
            if world_size % model_parallel_size:
                raise ValueError(
                    f"SFT world size {world_size} is not divisible by TP*PP*CP={model_parallel_size}"
                )
            planned_dp_size = world_size // model_parallel_size
            plan = resolve_sft_step_plan(
                configured_max_steps=configured_max_steps,
                dataset_size=len(self.dataset),
                data_parallel_size=planned_dp_size,
                per_device_train_batch_size=training_args.per_device_train_batch_size,
                gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                num_train_epochs=original_num_train_epochs,
            )
            training_args.num_train_epochs = plan.epochs
            self.pipeline_config.set_max_steps(plan.worker_max_steps)
            self._pipeline_max_steps = plan.pipeline_steps
        else:
            self.pipeline_config.set_max_steps(
                (original_num_train_epochs * len(self.dataset)) // \
                (training_args.per_device_train_batch_size * \
                 training_args.gradient_accumulation_steps))
            self._pipeline_max_steps = None

        self.sft_train: Any = Cluster(
            name=self.pipeline_config.sft_train.name,
            worker_cls=self.pipeline_config.sft_train.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.sft_train
        )
        ray.get(self.sft_train.initialize(pipeline_config=self.pipeline_config, blocking=False))

        dp_size = self.sft_train.dp_size
        ga_steps = self.pipeline_config.sft_train.training_args.gradient_accumulation_steps
        per_device_bs = self.pipeline_config.sft_train.training_args.per_device_train_batch_size
        self.global_train_batch_size = dp_size * ga_steps * per_device_bs
        logger.info(f"data parallel size = {dp_size},\n"
                    f"gradient accumulation steps = {ga_steps},\n"
                    f"per device train batch size = {per_device_bs},\n"
                    f"global train batch size = {self.global_train_batch_size}")

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.global_train_batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.pipeline_config.sft_train.training_args.dataloader_num_workers,
            collate_fn=data_collator,
        )

        if self._pipeline_max_steps is not None:
            actual_plan = resolve_sft_step_plan(
                configured_max_steps=self._pipeline_max_steps,
                dataset_size=len(self.dataset),
                data_parallel_size=dp_size,
                per_device_train_batch_size=per_device_bs,
                gradient_accumulation_steps=ga_steps,
                num_train_epochs=training_args.num_train_epochs,
            )
            if actual_plan.worker_max_steps != training_args.max_steps:
                raise RuntimeError(
                    "SFT worker data-parallel size changed during initialization: "
                    f"planned max_steps={training_args.max_steps}, "
                    f"actual={actual_plan.worker_max_steps}"
                )

        if self.val_dataset:
            self.val_dataset = preprocess_dataset(
                self.val_dataset,
                self.pipeline_config.sequence_length,
                encode_function,
                num_proc=self.pipeline_config.sft_train.data_args.preprocessing_num_workers)

            global_val_batch_size = dp_size * ga_steps * self.pipeline_config.sft_train.infer_batch_size
            self.val_dataloader = DataLoader(
                dataset=self.val_dataset,
                batch_size=global_val_batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=self.pipeline_config.sft_train.training_args.dataloader_num_workers,
                collate_fn=data_collator,
            )

        self.set_checkpoint_clusters(self.sft_train)

    @torch.no_grad()
    def run(self):
        metrics_mgr = MetricsManager()
        num_epochs = self.pipeline_config.sft_train.training_args.num_train_epochs
        steps_per_epoch = len(self.dataloader)
        if steps_per_epoch == 0:
            logger.info("pipeline complete: no full training batches")
            return
        pipeline_max_steps = getattr(self, "_pipeline_max_steps", None)
        total_steps = num_epochs * steps_per_epoch
        if pipeline_max_steps is not None:
            total_steps = min(total_steps, pipeline_max_steps)
        global_step = self.state.step + 1
        first_epoch, consumed_batches = divmod(global_step, steps_per_epoch)
        rng_directory = None
        if self.resume_from_checkpoint:
            rng_directory = os.path.join(self.resume_from_checkpoint, "pipeline")
            rng_file = os.path.join(rng_directory, "rng_state_pipeline.pth")
            if not os.path.isfile(rng_file):
                raise FileNotFoundError(f"SFT resume requires pipeline RNG state: {rng_file}")
            WorkerState.load_rng_state(rng_directory, "pipeline")

        for epoch in range(first_epoch, num_epochs):
            logger.info(f"epoch {epoch} start...")

            iterator = iter(self.dataloader)
            skipped = consumed_batches if epoch == first_epoch else 0
            for _ in range(skipped):
                next(iterator)
            if skipped and rng_directory is not None:
                # Rebuilding an interrupted iterator consumes a base seed and
                # may run random transforms for already-consumed batches. The
                # next unseen batch must start from the saved driver RNG.
                WorkerState.load_rng_state(rng_directory, "pipeline")
            pbar = tqdm(iterator, desc=f"Epoch {epoch}/{num_epochs}",
                        initial=skipped, total=steps_per_epoch)
            for batch_dict in pbar:
                # ``state.step`` and ``global_step`` are zero based.  A cap of
                # N therefore permits indices [0, N), exactly N updates.
                if pipeline_max_steps is not None and global_step >= pipeline_max_steps:
                    break
                logger.info(f"pipeline step {global_step} start...")

                metrics_mgr.clear_metrics()

                if self.val_dataset and global_step % self.pipeline_config.eval_steps == 0:
                    with Timer(name="val") as val_timer:
                        val_metrics = self.val()
                        metrics_mgr.add_reduced_metrics(val_metrics)
                    metrics_mgr.add_metric("time/val", val_timer.last)

                with Timer(name="step_train", logger=None) as step_train_timer:
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    batch.meta_info = {"global_step": global_step, "is_offload_states": False,
                                       "is_offload_optimizer_states_in_train_step": False,
                                       "loss_mask_keys": ["labels"]}
                    # Reorder data for DP rank load balancing
                    batch_balance_metrics = batch_balance(batch, dp_size=self.sft_train.dp_size,
                                                          minibatch_size=self.global_train_batch_size)
                    metrics_mgr.add_metrics(batch_balance_metrics)
                    train_metrics_refs = self.sft_train.train_step(batch, blocking=False)
                    train_metrics = DataProto.materialize_concat(data_refs=train_metrics_refs)
                    train_metrics = train_metrics.meta_info.pop("metrics", {})
                    metrics_mgr.add_reduced_metrics(train_metrics)
                metrics_mgr.add_metric("time/step_train", step_train_timer.last)

                metrics = metrics_mgr.get_metrics()
                metrics = {k: float(v) for k, v in metrics.items()}
                logger.info(f"metrics: {metrics}")

                # Update tqdm progress bar
                loss = metrics.get("sft_train/loss@sum", 0)
                pbar.set_postfix({"loss": f"{loss:.4f}", "step": f"{global_step}/{total_steps}"})

                self.state.step = global_step
                self.state.log_history.append(metrics)
                self.do_checkpoint(global_step=global_step)

                # modify custom metrics key_name
                # upload_metrics = {("train/" + k.split("/")[1]): v for k, v in metrics.items()}
                # metrics.update(upload_metrics)
                self.tracker.log(values=metrics, step=global_step)

                logger.info(f"pipeline step {global_step} finished...")

                global_step += 1

            if pipeline_max_steps is not None and global_step >= pipeline_max_steps:
                break

        logger.info("pipeline complete!")

    @torch.no_grad()
    def val(self):
        val_loss_list = []
        pbar = tqdm(self.val_dataloader, desc="Validating", leave=False)
        for batch_dict in pbar:
            batch: DataProto = DataProto.from_single_dict(batch_dict)
            batch.meta_info = {"is_offload_optimizer_states_in_train_step": False, 'loss_mask_keys': ['labels']}
            val_metrics_refs = self.sft_train.val_step(batch, blocking=False)
            val_metrics = DataProto.materialize_concat(data_refs=val_metrics_refs)
            val_metrics = reduce_metrics(val_metrics.meta_info.pop("metrics", {}))
            val_loss_list.append(val_metrics[f"sft_train/loss@sum"])
        return {"sft_train/val_loss": val_loss_list}
