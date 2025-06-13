from typing import Any, Dict

import ray
import torch
from codetiming import Timer
from ray.util.timer import _Timer
from torch.utils.data import DataLoader
from tqdm import tqdm

from roll.datasets.collator import DataCollatorWithPaddingForPaddedKeys
from roll.pipeline.base_worker import ActorWorker
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.generate_scheduler import GenerateScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.datasets.loader import get_dataset
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.utils.constants import GENERATE_SCHEDULER_NAME, RAY_NAMESPACE
from roll.utils.functionals import reduce_metrics
from roll.utils.logging import get_logger

logger = get_logger()


class GeneratePipeline(BasePipeline):

    def __init__(self, pipeline_config: RLVRConfig):
        super().__init__(pipeline_config)
        self.tokenizer = default_tokenizer_provider(
            model_args=self.pipeline_config.actor_train.model_args,
            template_name=self.pipeline_config.actor_train.data_args.template,
        )
        self.dataset = get_dataset(tokenizer=self.tokenizer, data_args=self.pipeline_config.actor_train.data_args)
        data_collator = DataCollatorWithPaddingForPaddedKeys(
            tokenizer=self.tokenizer,
            max_length=self.pipeline_config.prompt_length,
            padding="max_length",
        )
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.pipeline_config.rollout_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=data_collator,
        )
        self.actor_infer: Any = Cluster(
            name=self.pipeline_config.actor_infer.name,
            worker_cls=ActorWorker,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )
        self.generate_scheduler = GenerateScheduler.options(
            name=f"{GENERATE_SCHEDULER_NAME}_{self.actor_infer.cluster_name}",
            get_if_exists=True,
            namespace=RAY_NAMESPACE,
        ).remote()

        ray.get(self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=False))

    @torch.no_grad()
    def run(self):
        global_step = 0

        # Calculate tokens per second system throughput
        tps_timer = _Timer(window_size=5)
        metric_list = []

        for batch_dict in tqdm(self.dataloader):
            logger.info(f"pipeline step {global_step} start...")
            metrics = {}
            with tps_timer:
                batch_dict: Dict
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
                gen_batch.meta_info = {"global_step": global_step}
                with Timer() as generate_timer:
                    gen_batch.meta_info["is_offload_states"] = False
                    generate_output: DataProto = ray.get(
                        self.generate_scheduler.generate.remote(
                            data=gen_batch,
                            actor_cluster=self.actor_infer,
                            pipeline_config=self.pipeline_config,
                        )
                    )
                    batch.batch = generate_output.batch
                    batch = batch.union(generate_output)
                metrics.update(reduce_metrics(generate_output.meta_info.pop("metrics", {})))
                tps_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())

            prompt_ids = generate_output.batch["prompts"]
            response_ids = generate_output.batch["responses"]
            generate_res = []
            prompts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=False)

            for prompt, response in zip(prompts, responses):
                generate_res.append({"prompt": prompt, "response": response})

            metrics["system/tps"] = tps_timer.mean_throughput
            metrics["time/generate"] = generate_timer.last
            metrics["generate_res"] = generate_res

            metric_list.append(metrics)

            logger.info(f"global_step: {global_step} generate time: {generate_timer.last}")
            global_step += 1

        logger.info("pipeline complete!")
        return metric_list


class GenerateCmpPipeline(BasePipeline):

    def __init__(self, pipeline_config: RLVRConfig):
        super().__init__(pipeline_config)
        self.tokenizer = default_tokenizer_provider(
            model_args=self.pipeline_config.actor_train.model_args,
            template_name=self.pipeline_config.actor_train.data_args.template,
        )
        self.dataset = get_dataset(
            tokenizer=self.tokenizer,
            model_args=self.pipeline_config.actor_train.model_args,
            data_args=self.pipeline_config.actor_train.data_args,
            training_args=self.pipeline_config.actor_train.training_args,
            stage="ppo",
        )
        data_collator = DataCollatorWithPaddingForPaddedKeys(
            tokenizer=self.tokenizer,
            max_length=self.pipeline_config.prompt_length,
            padding="max_length",
        )
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.pipeline_config.rollout_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=data_collator,
        )
        self.actor_infer: Any = Cluster(
            name=self.pipeline_config.actor_infer.name,
            worker_cls=ActorWorker,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )
        self.reference: Any = Cluster(
            name=self.pipeline_config.reference.name,
            worker_cls=ActorWorker,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.reference,
        )
        self.generate_scheduler = GenerateScheduler.options(
            name=f"{GENERATE_SCHEDULER_NAME}_{self.actor_infer.cluster_name}",
            get_if_exists=True,
            namespace=RAY_NAMESPACE,
        ).remote()

        ray.get(self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(self.reference.initialize(pipeline_config=self.pipeline_config, blocking=False))

    def run(self):
        global_step = 0

        # Calculate tokens per second system throughput
        tps_timer = _Timer(window_size=5)
        metric_list = []

        for batch_dict in tqdm(self.dataloader):
            logger.info(f"pipeline step {global_step} start...")
            metrics = {}
            with tps_timer:
                batch_dict: Dict
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
                gen_batch.meta_info = {"global_step": global_step}
                with Timer() as generate_timer:
                    self.pipeline_config.generate_opt_level = 0
                    async_generate_output: DataProto = ray.get(
                        self.generate_scheduler.generate.remote(
                            data=gen_batch,
                            actor_cluster=self.actor_infer,
                            pipeline_config=self.pipeline_config,
                        )
                    )
                    self.pipeline_config.generate_opt_level = 1
                    self.pipeline_config.generate_redundancy_num = 2
                    self.pipeline_config.is_num_return_sequences_expand = True
                    batch_generate_output: DataProto = ray.get(
                        self.generate_scheduler.generate.remote(
                            data=gen_batch,
                            actor_cluster=self.actor_infer,
                            pipeline_config=self.pipeline_config,
                        )
                    )
                    hf_generate_output: DataProto = self.reference.generate(data=gen_batch)
                metrics.update(reduce_metrics(async_generate_output.meta_info.pop("metrics", {})))
                tps_timer.push_units_processed(
                    n=torch.sum(async_generate_output.batch["attention_mask"]).detach().item()
                )

            prompt_ids_async = async_generate_output.batch["prompts"]
            response_ids_async = async_generate_output.batch["responses"]
            prompt_ids_batch = batch_generate_output.batch["prompts"]
            response_ids_batch = batch_generate_output.batch["responses"]
            prompt_ids_hf = hf_generate_output.batch["prompts"]
            response_ids_hf = hf_generate_output.batch["responses"]
            generate_res = []
            prompts_async = self.tokenizer.batch_decode(prompt_ids_async, skip_special_tokens=True)
            responses_async = self.tokenizer.batch_decode(response_ids_async, skip_special_tokens=True)
            prompts_batch = self.tokenizer.batch_decode(prompt_ids_batch, skip_special_tokens=True)
            responses_batch = self.tokenizer.batch_decode(response_ids_batch, skip_special_tokens=True)
            prompts_hf = self.tokenizer.batch_decode(prompt_ids_hf, skip_special_tokens=True)
            responses_hf = self.tokenizer.batch_decode(response_ids_hf, skip_special_tokens=True)

            for prompt_async, response_async, prompt_batch, response_batch, prompt_hf, response_hf in zip(
                prompts_async, responses_async, prompts_batch, responses_batch, prompts_hf, responses_hf
            ):
                generate_res.append(
                    {
                        "prompt_async": prompt_async,
                        "response_async": response_async,
                        "prompt_batch": prompt_batch,
                        "response_batch": response_batch,
                        "prompt_hf": prompt_hf,
                        "response_hf": response_hf,
                    }
                )

            metrics["system/tps"] = tps_timer.mean_throughput
            metrics["time/generate"] = generate_timer.last
            metrics["generate_res"] = generate_res

            metric_list.append(metrics)

            logger.info(f"global_step: {global_step} generate time: {generate_timer.last}")
            global_step += 1

        logger.info("pipeline complete!")
        return metric_list
