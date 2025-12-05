import copy
import json
from functools import partial
from typing import Any, Dict, List, Optional

import datasets
import ray
import torch
from codetiming import Timer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from roll.datasets.collator import DataCollatorWithPaddingForPaddedKeys
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.generate_scheduler import DynamicSamplingScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.pipeline.rlvr.rlvr_pipeline import RLVRPipeline, get_encode_function, preprocess_dataset, \
    update_dataset_domain
from roll.utils.logging import get_logger
from roll.utils.metrics.metrics_manager import MetricsManager


logger = get_logger()


class RLVRRolloutPipeline(RLVRPipeline):

    def __init__(self, pipeline_config: RLVRConfig):
        BasePipeline.__init__(self, pipeline_config)
        self.pipeline_config = pipeline_config

        if self.pipeline_config.actor_infer.strategy_args.strategy_name in ["vllm", "sglang"]:
            assert self.pipeline_config.actor_infer.strategy_args.strategy_config.get("load_format", "dummy") != "dummy", (
                "rollout pipeline should strategy load model, set load_formant: auto."
            )
        if self.pipeline_config.actor_infer.strategy_args.strategy_name == "vllm":
            assert self.pipeline_config.actor_infer.strategy_args.strategy_config.get("sleep_level", 1) == 1, (
                "rollout pipeline should strategy sleep_level 1, set sleep_level: 1."
            )

        scheduler_cls = DynamicSamplingScheduler
        self.tokenizer = default_tokenizer_provider(model_args=self.pipeline_config.actor_infer.model_args)

        self.val_dataset = None
        assert self.pipeline_config.validation and self.pipeline_config.validation.data_args, "validation should set in RLVRRolloutPipeline"
        val_dataset_paths = self.pipeline_config.validation.data_args.file_name
        self.val_dataset = datasets.load_dataset("json", data_files=val_dataset_paths)["train"]

        # 加上format，然后转ids的func
        template_name = self.pipeline_config.global_template
        encode_function = get_encode_function(template_name, self.tokenizer, self.pipeline_config.actor_train.data_args)

        self.val_dataset = preprocess_dataset(
            self.val_dataset,
            self.pipeline_config.prompt_length,
            encode_function,
            data_args=self.pipeline_config.actor_train.data_args,
        )
        self.val_dataset = self.val_dataset.map(
            partial(update_dataset_domain, self.pipeline_config.tag_2_domain),
            num_proc=1,
            desc="update_val_dataset_domain",
            load_from_cache_file=False,
        )
        assert "domain" in self.val_dataset.column_names, "domain field should set in val dataset"

        print(self.val_dataset)

        self.actor_infer: Any = Cluster(
            name=self.pipeline_config.actor_infer.name,
            worker_cls=self.pipeline_config.actor_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )
        download_clusters = [self.actor_infer]
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

        val_pipeline_config = copy.deepcopy(self.pipeline_config)
        val_pipeline_config.is_use_additional_prompts = False
        self.val_generate_scheduler = scheduler_cls.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            )
        ).remote(pipeline_config=val_pipeline_config)
        ray.get(
            self.val_generate_scheduler.set_scheduler.remote(
                actor_cluster=self.actor_infer,
                reward_clusters=self.rewards,
                dataset=self.val_dataset,
                collect_fn_cls=DataCollatorWithPaddingForPaddedKeys,
                collect_fn_kwargs=dict(max_length=self.pipeline_config.prompt_length, padding="max_length"),
                response_filter_fn=lambda data_item, config: True,
                query_filter_fn=lambda data_list, config: True,
                response_callback_fn=self.val_generate_scheduler.report_response.remote,
                is_val=True,
            )
        )

        refs = []
        refs.extend(self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        refs = []
        for key, cluster in self.rewards.items():
            refs.extend(cluster.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

    @torch.no_grad()
    def run(self):
        global_step = 0
        logger.info(f"pipeline step {global_step} start...")
        val_metrics_mgr = MetricsManager()
        batch = DataProto()

        with Timer(name="step_generate", logger=None) as step_generate_timer:
            batch.meta_info["is_offload_states"] = False
            batch.meta_info["generation_config"] = self.pipeline_config.validation.generating_args.to_dict()
            self.actor_infer.start_server(data=DataProto(meta_info=batch.meta_info))
            for reward_cluster in self.rewards.values():
                reward_cluster.load_states()
            generate_output: DataProto = ray.get(
                self.val_generate_scheduler.get_batch.remote(data=batch, batch_size=len(self.val_dataset)),
                timeout=self.pipeline_config.rpc_timeout,
            )
            self.actor_infer.stop_server()
            for reward_cluster in self.rewards.values():
                reward_cluster.offload_states()
            generate_output.meta_info.pop("is_offload_states", None)
        val_metrics_mgr.add_metric("time/step_generate", step_generate_timer.last)

        batch = generate_output
        val_correct_mean = (batch.batch["scores"] == 1).detach().float().mean().item()
        val_metrics_mgr.add_metric("val_correct/all/mean", val_correct_mean)
        logger.info(json.dumps({"val_correct/all/mean": val_correct_mean}, ensure_ascii=False))

        epoch_batch = batch.pop(batch_keys=["scores"], non_tensor_batch_keys=["tag"])

        grouped_batch = epoch_batch.group_by("tag")
        for group_key, group_batch in grouped_batch.items():
            score_mean = group_batch.batch["scores"].mean().item()
            print(f"{group_key}:  {score_mean}")
            val_metrics_mgr.add_domain_metrics(
                "val_correct", {f"{group_key}/mean": (group_batch.batch["scores"] == 1).detach().float().mean().item()}
            )

        prompts = self.tokenizer.batch_decode(generate_output.batch["prompts"], skip_special_tokens=True)
        responses = self.tokenizer.batch_decode(
            generate_output.batch["responses"], skip_special_tokens=True
        )
        generate_examples = [{"prompt": p, "response": r} for p, r in zip(prompts, responses)][:10]
        logger.info(json.dumps(generate_examples, ensure_ascii=False))
        logger.info(json.dumps(val_metrics_mgr.get_metrics(), ensure_ascii=False))

        logger.info(f"pipeline step {global_step} finished")

        logger.info("pipeline complete!")
