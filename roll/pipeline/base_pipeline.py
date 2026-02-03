import os
import re
import shutil
from collections import defaultdict
from concurrent import futures
from typing import Any, Dict, List

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import set_seed

from roll.distributed.executor.cluster import Cluster
from roll.distributed.executor.model_update_group import ModelUpdateGroup
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.utils.checkpoint_manager import CheckpointManager, download_model
from roll.utils.functionals import reduce_metrics
from roll.utils.logging import get_logger
from roll.utils.tracking import create_tracker
from roll.utils.worker_state import WorkerState

logger = get_logger()


class BasePipeline:
    model_update_groups: List[ModelUpdateGroup] = []
    checkpoint_clusters: List = []

    def __init__(self, pipeline_config):
        set_seed(seed=pipeline_config.seed)
        self.pipeline_config = pipeline_config
        self.resource_manager = ResourceManager(
            num_nodes=self.pipeline_config.num_nodes, num_gpus_per_node=self.pipeline_config.num_gpus_per_node
        )
        self.state = WorkerState()
        self.checkpoint_manager = CheckpointManager(checkpoint_config=self.pipeline_config.checkpoint_config)
        self.tracker = create_tracker(
            tracker_name=self.pipeline_config.track_with,
            config=self.pipeline_config.to_dict(),
            **self.pipeline_config.tracker_kwargs,
        )
        self.resume_from_checkpoint = False
        self.executor: futures.ThreadPoolExecutor = futures.ThreadPoolExecutor(max_workers=5)
        self.resume_futures = []

        if self.pipeline_config.resume_from_checkpoint:
            self.resume_from_checkpoint = download_model(self.pipeline_config.resume_from_checkpoint)

            logger.info(f"resume_from_checkpoint: {self.resume_from_checkpoint}")
            load_dir = os.path.join(self.resume_from_checkpoint, "pipeline")
            self.state = WorkerState.load_from_json(load_dir=load_dir, tag="pipeline")

            def resume_metrics():
                for metrics in self.state.log_history:
                    self.tracker.log(values=metrics, step=metrics["system/step"])

            self.resume_futures.append(self.executor.submit(resume_metrics))

    def run(self):
        pass

    def set_model_update_pair(self, src_cluster, tgt_cluster, frequency=1):
        self.model_update_groups.append(
            ModelUpdateGroup(src_cluster=src_cluster, tgt_cluster=tgt_cluster, frequency=frequency, pipeline_config=self.pipeline_config)
        )

    def set_checkpoint_clusters(self, *clusters):
        self.checkpoint_clusters.extend(clusters)

    def model_update(self, global_step):
        metrics = {}
        for model_update_group in self.model_update_groups:
            metrics.update(model_update_group.model_update(global_step))
            model_update_group.tgt_cluster.process_weights_after_loading()
        return metrics

    def do_checkpoint(self, global_step, is_last_step=None):
        if is_last_step is None:
            is_last_step = global_step == self.pipeline_config.max_steps - 1

        metrics = self.state.log_history[-1]
        metrics["system/step"] = global_step
        if global_step > 0 and (
            global_step % self.pipeline_config.save_steps == 0 or global_step == self.pipeline_config.max_steps - 1
        ):
            ckpt_metrics_refss = []
            for cluster in self.checkpoint_clusters:
                ckpt_metrics_refss.append(
                    cluster.do_checkpoint(global_step=global_step, is_last_step=is_last_step, blocking=False)
                )

            for ckpt_metrics_refs in ckpt_metrics_refss:
                ckpt_metrics = DataProto.materialize_concat(data_refs=ckpt_metrics_refs)
                metrics.update(reduce_metrics(ckpt_metrics.meta_info.pop("metrics", {})))

            ckpt_id = f"checkpoint-{global_step}"
            pipeline_save_dir = os.path.join(self.pipeline_config.output_dir, "pipeline", ckpt_id)
            save_dir = os.path.join(self.pipeline_config.output_dir, "pipeline", ckpt_id, "pipeline")
            self.state.save_to_json(save_dir=save_dir, tag="pipeline")
            self.state.save_rng_state(save_dir=save_dir, tag="pipeline")
            self.checkpoint_manager.upload(ckpt_id=ckpt_id, local_state_path=pipeline_save_dir)

            # Clean up old checkpoints if max_ckpt_to_keep is set
            self._cleanup_old_checkpoints()

        futures.wait(self.resume_futures)
        self.resume_futures.clear()

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints if max_ckpt_to_keep is set."""
        max_ckpt = getattr(self.pipeline_config, 'max_ckpt_to_keep', 0)
        if max_ckpt <= 0:
            return

        output_dir = self.pipeline_config.output_dir
        if not os.path.exists(output_dir):
            return

        # Pattern to match checkpoint directories: checkpoint-{step}
        ckpt_pattern = re.compile(r'^checkpoint-(\d+)$')

        # Collect all checkpoint steps across all subdirectories
        all_ckpt_steps = set()
        for subdir in os.listdir(output_dir):
            subdir_path = os.path.join(output_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for item in os.listdir(subdir_path):
                match = ckpt_pattern.match(item)
                if match:
                    all_ckpt_steps.add(int(match.group(1)))

        # Sort steps and determine which to delete
        sorted_steps = sorted(all_ckpt_steps, reverse=True)
        steps_to_delete = sorted_steps[max_ckpt:]

        if not steps_to_delete:
            return

        logger.info(f"Cleaning up old checkpoints. Keeping {max_ckpt}, deleting steps: {steps_to_delete}")

        # Delete old checkpoints from all subdirectories
        for subdir in os.listdir(output_dir):
            subdir_path = os.path.join(output_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for step in steps_to_delete:
                ckpt_dir = os.path.join(subdir_path, f"checkpoint-{step}")
                if os.path.exists(ckpt_dir):
                    try:
                        shutil.rmtree(ckpt_dir)
                        logger.info(f"Deleted old checkpoint: {ckpt_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to delete checkpoint {ckpt_dir}: {e}")

    def download_models(self, *clusters: Cluster):
        node2pg: Dict[str, PlacementGroup] = {}
        node2model_names: Dict[str, set[str]] = defaultdict(set)
        for cluster in clusters:
            assert cluster.placement_groups is not None
            for pg_list in cluster.placement_groups:
                assert len(pg_list) > 0
                worker_nodes = set()
                for pg in pg_list:
                    node_rank = pg["node_rank"]
                    if node_rank not in worker_nodes:
                        worker_nodes.add(node_rank)
                        node2pg[node_rank] = pg["placement_group"]
                        if cluster.worker_config.model_args.model_name_or_path:
                            node2model_names[node_rank].add(cluster.worker_config.model_args.model_name_or_path)
                        if self.pipeline_config.resume_from_checkpoint:
                            node2model_names[node_rank].add(self.pipeline_config.resume_from_checkpoint)
        ray.get(
            [
                download_models.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=node2pg[node_rank])
                ).remote(model_name_or_paths=model_names)
                for node_rank, model_names in node2model_names.items()
            ]
        )

@ray.remote
def download_models(model_name_or_paths: set[str]):
    with futures.ThreadPoolExecutor(max_workers=5) as thread_executor:
        futures.wait([thread_executor.submit(download_model, model_name_or_path)
                      for model_name_or_path in model_name_or_paths])
