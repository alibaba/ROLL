import copy
import json
import math
import os
import time
import uuid
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Optional

import datasets
import numpy as np
import ray
import torch
from codetiming import Timer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.timer import _Timer

from roll.configs import GeneratingArguments
from roll.datasets.chat_template import get_chat_template
from roll.datasets.collator import DataCollatorWithPaddingForPaddedKeys
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.async_generate_scheduler import AsyncDynamicSamplingScheduler
from roll.distributed.scheduler.generate_scheduler import DynamicSamplingScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.pipeline.rlvr.utils import dump_rollout_to_specific_path
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
from roll.utils.dynamic_batching import dynamic_batching_shard


logger = get_logger()


def is_lora_training(pipeline_config: RLVRConfig) -> bool:
    if pipeline_config.actor_train.model_args.lora_target is None:
        return False
    assert pipeline_config.actor_train.strategy_args.strategy_name == "deepspeed_train", (
        "LoRA only supports deepspeed_train"
    )
    return True


def preprocess_dataset(dataset, prompt_len, encode_function, data_args):
    logger.info(f"Begin : {dataset}")
    dataset = dataset.map(
        encode_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        desc="Encoding dataset",
        load_from_cache_file=False,
    )
    # 过滤cutoff
    dataset = dataset.filter(
        lambda data_i: 5 < len(data_i["input_ids"]) <= prompt_len,
        num_proc=data_args.preprocessing_num_workers,
        desc="Filtering dataset",
    )
    logger.info(f"Filtering prompt len: {dataset}")
    logger.info(f"Encoding: {dataset}")
    return dataset


def get_encode_function(template_name, data_args, tokenizer):
    chat_template_func = get_chat_template(template_name, tokenizer)

    def encode_function(data_i):
        text_list = []
        if (message_key := getattr(data_args, "messages", "messages")) in data_i:
            for messages in data_i[message_key]:
                if isinstance(messages, str):
                    messages = json.loads(messages)
                text_list.append(chat_template_func(messages))
        elif (prompt_key := getattr(data_args, "prompt", "prompt")) in data_i:
            for prompt in data_i[prompt_key]:
                text_list.append(prompt)
        encodings = tokenizer(text_list)
        return encodings

    return encode_function

def update_dataset_domain(tag_2_domain: Dict[str, set[str]], row):
    if "domain" in row and row["domain"] is not None:
        return row
    row["domain"] = tag_2_domain.get(row["tag"], "math_rule")
    return row

def query_filter_fn(data_list: List[DataProto], config: RLVRConfig) -> bool:
    """
    各domain的过滤规则可以自定义
    """
    response_level_rewards = [data.batch["response_level_rewards"] for data in data_list]
    if len(response_level_rewards) == 1:
        return True
    rewards = torch.cat(response_level_rewards, dim=0)

    domain = data_list[0].non_tensor_batch["domain"][0]
    query_filter_config = config.rewards[domain].query_filter_config

    if query_filter_config.type == "no_filter":
        return True
    elif query_filter_config.type == "mean_filter":
        threshold_up = query_filter_config.filter_args.get("threshold_up", math.inf)
        threshold_down = query_filter_config.filter_args.get("threshold_down", -1)
        if torch.mean(rewards) <= threshold_down or torch.mean(rewards) >= threshold_up:
            return False
    elif query_filter_config.type == "std_filter":
        std_threshold = query_filter_config.filter_args.get("std_threshold", -1)
        if torch.std(rewards) <= std_threshold:
            return False
    return True


class RLVRPipeline(BasePipeline):

    def __init__(self, pipeline_config: RLVRConfig):
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config
        self.is_lora = is_lora_training(self.pipeline_config)
        scheduler_cls = (
            AsyncDynamicSamplingScheduler if self.pipeline_config.async_pipeline else DynamicSamplingScheduler
        )
        self.tokenizer = default_tokenizer_provider(model_args=self.pipeline_config.actor_train.model_args)

        dataset_paths = []
        if self.pipeline_config.actor_train.data_args.file_name:
            dataset_paths.extend(self.pipeline_config.actor_train.data_args.file_name)

        print(f"load_dataset_paths: {chr(10)} {chr(10).join(dataset_paths)}")
        dataset = datasets.load_dataset("json", data_files=dataset_paths)["train"]

        self.val_dataset = None
        if self.pipeline_config.validation and self.pipeline_config.validation.data_args:
            val_dataset_paths = self.pipeline_config.validation.data_args.file_name
            self.val_dataset = datasets.load_dataset("json", data_files=val_dataset_paths)["train"]

        # 加上format，然后转ids的func
        template_name = (
            self.pipeline_config.global_template
            if self.pipeline_config.global_template
            else self.pipeline_config.actor_train.data_args.template
        )
        encode_function = get_encode_function(template_name, self.pipeline_config.actor_train.data_args, self.tokenizer)

        dataset = preprocess_dataset(
            dataset,
            self.pipeline_config.prompt_length,
            encode_function,
            data_args=self.pipeline_config.actor_train.data_args,
        )
        # update domain field
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

        if self.val_dataset:
            self.val_dataset = preprocess_dataset(
                self.val_dataset,
                self.pipeline_config.prompt_length,
                encode_function,
                data_args=self.pipeline_config.validation.data_args,
            )
            self.val_dataset = self.val_dataset.map(
                partial(update_dataset_domain, self.pipeline_config.tag_2_domain),
                num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
                desc="update_val_dataset_domain",
                load_from_cache_file=False,
            )
            assert "domain" in self.val_dataset.column_names, "domain field should set in val dataset"

        assert "domain" in dataset.column_names, "domain field should set in dataset"
        print(dataset)

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
        # use unwrapped model as reference for lora training
        if not self.is_lora and self.pipeline_config.enable_reference:
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
            generate_scheduler = scheduler_cls.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                )
            ).remote(pipeline_config=self.pipeline_config)
            ray.get(
                generate_scheduler.set_scheduler.remote(
                    actor_cluster=self.actor_infer,
                    reward_clusters={domain: self.rewards[domain]},
                    dataset=self.domain_datasets[domain],
                    collect_fn_cls=DataCollatorWithPaddingForPaddedKeys,
                    collect_fn_kwargs=dict(max_length=self.pipeline_config.prompt_length, padding="max_length"),
                    response_filter_fn=lambda data_item, config: True,
                    query_filter_fn=query_filter_fn,
                    response_callback_fn=generate_scheduler.report_response.remote,
                    state=self.state.kv.get(f"scheduler_state_{domain}", None),
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
            self.val_generate_scheduler = scheduler_cls.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                )
            ).remote(pipeline_config=val_pipeline_config)
        if self.val_dataset:
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

        if not self.is_lora and self.pipeline_config.enable_reference:
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
    def save_metrics(self, batch):
        def remove_leading_zeros(A, r_mask):
            B = []
            for i in range(len(A)):
                row = A[i]
                mask = (r_mask[i] != 0).to(torch.int32)
                if not mask.any():  # 如果该行全为零
                    B.append([])  # 添加空列表
                else:
                    first_non_zero = mask.argmax().item()  # 找到第一个非零元素的索引
                    B.append(row[first_non_zero:].tolist())
            return B

        res_dict = {}
        batch_size = batch.batch["old_log_probs"].shape[0]
        res_dict["log_probs2"] = remove_leading_zeros(
            batch.batch["old_log_probs2"], batch.batch["response_mask"][:, 1:]
        )
        res_dict["logprobs"] = remove_leading_zeros(
            batch.batch["old_log_probs"], batch.batch["response_mask"][:, 1:]
        )  # 这里注意batch.batch["old_log_probs"]是right pad 格式
        res_dict["new_response_mask"] = remove_leading_zeros(
            batch.batch["response_mask"][:, 1:], batch.batch["response_mask"][:, 1:]
        )
        res_dict["prompt"] = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=False)
        res_dict["response"] = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=False)
        res_dict["old_log_probs2_entropy"] = remove_leading_zeros(batch.batch["old_log_probs2_entropy"], batch.batch["response_mask"][:, 1:])
        res_dict["ref_logprobs"] = remove_leading_zeros(
            batch.batch["ref_log_probs"], batch.batch["response_mask"][:, 1:]
        )
        res_dict["values"] = batch.batch["token_level_rewards"].numpy().tolist()
        res_dict["token_rewards"] = batch.batch["token_level_rewards"].numpy().tolist()
        res_dict["reward"] = batch.batch["response_level_rewards"].numpy().tolist()
        step = batch.meta_info["global_step"]

        assert (
            len(res_dict["logprobs"]) == batch_size
        ), f"len of logprobs is : {len(res_dict['logprobs'])}, len of batch size is :{batch_size}"
        assert (
            len(res_dict["ref_logprobs"]) == batch_size
        ), f"len of ref_logprobs is : {len(res_dict['ref_logprobs'])}, len of batch size is :{batch_size}"
        assert (
            len(res_dict["log_probs2"]) == batch_size
        ), f"len of log_probs2 is : {len(res_dict['log_probs2'])}, len of batch size is :{batch_size}"

        res = []
        for i in range(batch_size):
            temp_dict = {}
            for k, _ in res_dict.items():
                temp_dict[k] = res_dict[k][i]
            temp_dict["step"] = step
            temp_dict["response_tokens"] = self.tokenizer.convert_ids_to_tokens(
                batch.batch["responses"][i], skip_special_tokens=False
            )
            res.append(temp_dict)

        file_dir = self.pipeline_config.save_logging_board_dir
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        # 用时间命名domain batch 文件名
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        milliseconds = str(int(time.time() * 1000) % 1000).zfill(3)
        file_name = f"data_{current_time}_{milliseconds}.jsonl"

        try:
            with open(file_dir + file_name, "a", encoding="utf-8") as f:
                for r in res:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.info(f"Writing files catch error :{e}")

    def get_generation_config(self, generating_args: Optional[GeneratingArguments] = None):
        generating_args = (
            generating_args if generating_args is not None else self.actor_infer.worker_config.generating_args
        )
        generation_config = generating_args.to_dict()
        if self.pipeline_config.async_pipeline:
            generation_config["logprobs"] = 1
        return generation_config

    @torch.no_grad()
    def run(self):
        # 计算tokens per second 系统吞吐

        # 创建一个专门管理监控指标的类
        metrics_mgr = MetricsManager()

        tps_timer = _Timer(window_size=5)
        actor_infer_timer = _Timer(window_size=5)
        actor_infer_response_timer = _Timer(window_size=5)
        actor_train_timer = _Timer(window_size=5)

        pre_step_total_time = 0
        if self.pipeline_config.async_pipeline and self.pipeline_config.generate_opt_level == 1:
            for reward_cluster in self.rewards.values():
                reward_cluster.load_states()

        first_step = True
        for global_step in range(self.pipeline_config.max_steps):
            if global_step <= self.state.step:
                global_step += 1
                continue
            logger.info(f"pipeline step {global_step} start...")
            should_eval = self.val_dataset and global_step % self.pipeline_config.eval_steps == 0

            metrics_mgr.clear_metrics()
            with tps_timer, Timer(name="step_total", logger=None) as step_total_timer:
                # if global_step > self.state.step + 1:
                logger.info(f"pre_step_total_time: {pre_step_total_time}")
                metrics_mgr.add_metric("time/step_total", pre_step_total_time)
                batch: DataProto = DataProto(
                    meta_info={"global_step": global_step, "collect_unfinished": self.pipeline_config.async_pipeline}
                )

                # 先model update，resume时不需要保存infer cluster的状态
                if self.pipeline_config.adv_estimator == "gae":
                    self.critic.offload_states(blocking=True)
                self.actor_train.offload_states(blocking=True)

                with Timer(name="step_stop_server", logger=None) as step_stop_server_timer:
                    if self.pipeline_config.async_pipeline and not first_step and self.pipeline_config.generate_opt_level == 1:
                        scheduler_refs = []
                        for scheduler in self.generate_schedulers.values():
                            scheduler_refs.append(scheduler.pause_sampling.remote(data=batch))
                        ray.get(scheduler_refs, timeout=self.pipeline_config.rpc_timeout)
                        self.actor_infer.stop_server()
                metrics_mgr.add_metric("time/step_stop_server", step_stop_server_timer.last)

                with Timer(name="step_model_update", logger=None) as step_model_update_timer:
                    model_update_metrics: Dict = self.model_update(global_step)
                    metrics_mgr.add_metrics(model_update_metrics)
                    batch.meta_info["generation_config"] = self.get_generation_config()
                metrics_mgr.add_metric("time/step_model_update", step_model_update_timer.last)

                if should_eval and not self.pipeline_config.async_pipeline:
                    with Timer(name="val_step", logger=None) as val_step_timer:
                        val_metrics = self.val()
                    metrics_mgr.add_metrics(val_metrics)
                    metrics_mgr.add_metric("time/val_step", val_step_timer.last)

                # 要按domain group by生成对应的batch
                with (
                    actor_infer_timer,
                    actor_infer_response_timer,
                    Timer(name="step_generate", logger=None) as step_generate_timer,
                ):
                    domain_batches = {}
                    if self.pipeline_config.generate_opt_level == 1:
                        self.actor_infer.start_server(data=DataProto(meta_info=batch.meta_info))
                        batch.meta_info["is_offload_states"] = False
                    if self.pipeline_config.async_pipeline:
                        if should_eval:
                            # 为Validation创建独立的DataProto
                            val_batch = DataProto()
                            val_batch.meta_info = {
                                "global_step": global_step,
                                "generation_config": self.pipeline_config.validation.generating_args.to_dict()
                            }
                            self.val_generate_scheduler.start_sampling.remote(data=val_batch, batch_size=len(self.val_dataset))

                        scheduler_refs = []
                        for domain, scheduler in self.generate_schedulers.items():
                            scheduler_refs.append(
                                scheduler.start_sampling.remote(data=batch, batch_size=self.domain_batch_size[domain])
                            )
                        ray.get(scheduler_refs, timeout=self.pipeline_config.rpc_timeout)
                    else:
                        for reward_cluster in self.rewards.values():
                            reward_cluster.load_states()

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
                    dump_rollout_to_specific_path(self.pipeline_config.rollout_dump_dir, global_step, generate_output, self.tokenizer)
                    generate_output.meta_info.pop("is_offload_states", None)

                    if not self.pipeline_config.async_pipeline and self.pipeline_config.generate_opt_level == 1:
                        for reward_cluster in self.rewards.values():
                            reward_cluster.offload_states()
                        gen_metrics = self.actor_infer.stop_server()
                        metrics_mgr.add_domain_metrics(
                            domain, reduce_metrics(gen_metrics.meta_info.pop("metrics", {}))
                        )
                metrics_mgr.add_metric("time/step_generate", step_generate_timer.last)

                batch = generate_output
                batch.meta_info["global_step"] = global_step
                batch.non_tensor_batch['sample_uuid'] = np.array([str(uuid.uuid4()) for _ in range(batch.batch.shape[0])], dtype=object)


                with Timer(name="cal_ref_log_probs", logger=None) as cal_ref_log_probs_timer:
                    if self.pipeline_config.enable_reference:
                        if self.is_lora:
                            batch.meta_info["disable_adapter"] = True
                            batch.meta_info["is_offload_states"] = False
                            ref_log_probs = self.actor_train.compute_log_probs(batch, blocking=True)
                        else:
                            if self.pipeline_config.reference.use_dynamic_batching_in_infer:
                                batch, dynamic_batching_metrics = dynamic_batching_shard(
                                    batch,
                                    self.reference.dp_size,
                                    self.pipeline_config.reference.max_tokens_per_microbatch_in_infer,
                                    self.pipeline_config.reference.sequence_length_round_in_infer,
                                    "reference/compute_log_probs",
                                )
                                metrics_mgr.add_metrics(dynamic_batching_metrics)
                            ref_log_probs = self.reference.compute_log_probs(batch, blocking=True)
                        metrics_mgr.add_reduced_metrics(ref_log_probs.meta_info.pop("metrics", {}))
                        ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                        batch = batch.union(ref_log_probs)
                metrics_mgr.add_metric("time/ref_log_probs_values", cal_ref_log_probs_timer.last)

                with Timer(name="cal_old_log_probs_values", logger=None) as cal_old_logpb_timer:
                    if self.is_lora:
                        batch.meta_info["disable_adapter"] = False
                    batch.meta_info["is_offload_states"] = False
                    if self.pipeline_config.adv_estimator == "gae":
                        values_refs: List[ray.ObjectRef] = self.critic.compute_values(batch, blocking=False)

                    if self.pipeline_config.enable_old_logprobs:
                        if self.pipeline_config.actor_train.use_dynamic_batching_in_infer:
                            batch, dynamic_batching_metrics = dynamic_batching_shard(
                                batch,
                                self.actor_train.dp_size,
                                self.pipeline_config.actor_train.max_tokens_per_microbatch_in_infer,
                                self.pipeline_config.actor_train.sequence_length_round_in_infer,
                                "actor_train/compute_log_probs",
                            )
                            metrics_mgr.add_metrics(dynamic_batching_metrics)
                        old_log_probs_refs: List[ray.ObjectRef] = self.actor_train.compute_log_probs(batch, blocking=False)
                        old_log_probs = DataProto.materialize_concat(data_refs=old_log_probs_refs)

                        # Customize_logging metrics, Double check call twice
                        if self.pipeline_config.save_logging_board_dir:
                            old_log_probs_refs2: List[ray.ObjectRef] = self.actor_train.compute_log_probs(
                                batch, blocking=False
                            )
                            old_log_probs2 = DataProto.materialize_concat(data_refs=old_log_probs_refs2)
                            batch.batch["old_log_probs2"] = old_log_probs2.batch["log_probs"]
                            batch.batch["old_log_probs2_entropy"] = old_log_probs2.batch["entropy"]

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

                # 要按domain group by处理reward
                batch.batch["prompt_id"] = torch.arange(batch.batch.batch_size[0], device=batch.batch.device)
                batch_grouped: Dict[str, DataProto] = batch.group_by("domain")
                batch_list = []
                for domain, domain_batch in batch_grouped.items():
                    # 1. 处理mask相关策略， 获取sample level mask
                    with Timer(name="get_sample_level_mask", logger=None) as get_sample_level_mask_timer:
                        domain_batch, mask_metrics = get_sample_level_mask(domain_batch, self.pipeline_config)
                        metrics_mgr.add_domain_metrics(domain, mask_metrics)
                    metrics_mgr.add_domain_metrics(domain, {"time/get_sample_level_mask": get_sample_level_mask_timer.last})

                    # 2. 处理reward相关策略
                    with Timer(name="reward_postprocess", logger=None) as reward_postprocess_timer:
                        domain_batch, response_level_metrics = reward_postprocess(
                            domain_batch, self.pipeline_config, self.running
                        )
                        metrics_mgr.add_domain_metrics(domain, response_level_metrics)
                    metrics_mgr.add_domain_metrics(domain, {"time/reward_postprocess": reward_postprocess_timer.last})

                    # 3. 计算token level rewards
                    with Timer(name="get_token_reward", logger=None) as get_token_reward_timer:
                        domain_batch, token_level_metrics = compute_token_reward(
                            domain_batch, self.pipeline_config, self.kl_ctrl
                        )
                        metrics_mgr.add_domain_metrics(domain, token_level_metrics)
                    metrics_mgr.add_domain_metrics(domain, {"time/get_token_reward": get_token_reward_timer.last})

                    # 4. 计算advantage
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
                    if self.pipeline_config.save_logging_board_dir:
                        self.save_metrics(domain_batch)

                batch = DataProto.concat(batch_list)

                if batch.batch["final_response_mask"].sum() == 0:
                    logger.info("Warning: final_response_mask.sum() == 0! Current step will be skipped.")
                    metrics_mgr.add_metric("mask/final_mask_sum_eq_0", 1)
                    metrics = metrics_mgr.get_metrics()
                    # do ckpt
                    self.state.step = global_step
                    self.state.log_history.append(metrics)
                    for domain, scheduler in self.generate_schedulers.items():
                        self.state.kv[f"scheduler_state_{domain}"] = ray.get(scheduler.get_scheduler_state.remote())
                    self.do_checkpoint(global_step=global_step)
                    self.tracker.log(values=metrics, step=global_step)
                    continue
                else:
                    metrics_mgr.add_metric("mask/final_mask_sum_eq_0", 0)

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
                            if self.pipeline_config.actor_train.use_dynamic_batching_in_train:
                                batch, dynamic_batching_metrics = dynamic_batching_shard(
                                    batch,
                                    self.actor_train.dp_size,
                                    self.pipeline_config.actor_train.max_tokens_per_microbatch_in_train,
                                    self.pipeline_config.actor_train.sequence_length_round_in_train,
                                    "actor_train/train_step",
                                )
                                metrics_mgr.add_metrics(dynamic_batching_metrics)
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

                for domain, scheduler in self.generate_schedulers.items():
                    self.state.kv[f"scheduler_state_{domain}"] = ray.get(scheduler.get_scheduler_state.remote())

                if (
                    self.pipeline_config.async_pipeline
                    and self.val_dataset
                    and global_step % self.pipeline_config.eval_steps == 0
                ):
                    with Timer(name="val_step", logger=None) as val_step_timer:
                        val_metrics = self.val()
                        metrics_mgr.add_metrics(val_metrics)
                    metrics_mgr.add_metric("time/val_step", val_step_timer.last)

                metrics = metrics_mgr.get_metrics()
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

                    prompts = self.tokenizer.batch_decode(generate_output.batch["prompts"], skip_special_tokens=True)
                    responses = self.tokenizer.batch_decode(
                        generate_output.batch["responses"], skip_special_tokens=True
                    )
                    generate_examples = [{"prompt": p, "response": r} for p, r in zip(prompts, responses)][:10]
                    logger.info(json.dumps(generate_examples, ensure_ascii=False))
                    logger.info(json.dumps(metrics, ensure_ascii=False))

                logger.info(f"pipeline step {global_step} finished")
                global_step += 1
                first_step = False
            pre_step_total_time = step_total_timer.last
        

        logger.info("pipeline complete!")

    @torch.no_grad()
    def val(self):
        val_metrics_mgr = MetricsManager()
        batch = DataProto()

        with Timer(name="step_generate", logger=None) as step_generate_timer:
            batch.meta_info["is_offload_states"] = False
            batch.meta_info["generation_config"] = self.pipeline_config.validation.generating_args.to_dict()
            if not self.pipeline_config.async_pipeline:
                self.actor_infer.start_server(data=DataProto(meta_info=batch.meta_info))
                for reward_cluster in self.rewards.values():
                    reward_cluster.load_states()
            generate_output: DataProto = ray.get(
                self.val_generate_scheduler.get_batch.remote(data=batch, batch_size=len(self.val_dataset)),
                timeout=self.pipeline_config.rpc_timeout,
            )
            if not self.pipeline_config.async_pipeline and self.pipeline_config.generate_opt_level == 1:
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
            logger.info(f"val_correct/{group_key}:  {score_mean}")
            val_metrics_mgr.add_domain_metrics(
                "val_correct", {f"{group_key}/mean": (group_batch.batch["scores"] == 1).detach().float().mean().item()}
            )

        return val_metrics_mgr.get_metrics()
