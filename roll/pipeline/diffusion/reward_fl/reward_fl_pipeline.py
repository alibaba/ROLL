from typing import Any, Dict, List
import os
import ray
import torch
import torchvision
from codetiming import Timer
from tqdm import tqdm
import numpy as np

from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.diffusion.reward_fl.reward_fl_config import RewardFLConfig
from roll.utils.logging import get_logger
from roll.utils.metrics.metrics_manager import MetricsManager

from diffsynth.trainers.unified_dataset import UnifiedDataset

logger = get_logger()


def collate_fn(examples):
    video = torch.stack([
        torch.stack([
            torchvision.transforms.functional.to_tensor(frame) 
            for frame in example['video']],
            dim=0
        )
        for example in examples
    ], dim=0)
    prompt = np.array([example['prompt'] for example in examples], dtype=object)
    return {'video': video, 'prompt': prompt}


class RewardFLPipeline(BasePipeline):
    def __init__(self, pipeline_config: RewardFLConfig):
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config

        assert self.pipeline_config.max_steps > 0, "max_steps must be greater than 0"
        self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)

        self.actor_train: Any = Cluster(
            name=self.pipeline_config.actor_train.name,
            worker_cls=self.pipeline_config.actor_train.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_train,
        )
        metadata_path = self.pipeline_config.actor_train.data_args.file_name
        base_path = os.path.dirname(metadata_path)
        dataset = UnifiedDataset(
            base_path=base_path,
            metadata_path=metadata_path,
            data_file_keys=("video", "image"),
            repeat=100,
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=base_path,
                max_pixels=480*480, height=None, width=None,
                height_division_factor=16, width_division_factor=16,
            ),
        )
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=pipeline_config.train_batch_size, collate_fn=collate_fn)
        refs: List[ray.ObjectRef] = []
        refs.extend(self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

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

                with Timer(name="step_total", logger=None) as step_total_timer:
                    batch_dict: Dict
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    batch.meta_info = {"global_step": global_step, "is_offload_states": False, "is_offload_optimizer_states_in_train_step": False}

                    with Timer(name="actor_train", logger=None) as actor_train_timer:
                        actor_train_refs = self.actor_train.train_step(batch, blocking=False)
                        actor_train_refs: DataProto = DataProto.materialize_concat(data_refs=actor_train_refs)
                        metrics_mgr.add_reduced_metrics(actor_train_refs.meta_info.pop("metrics", {}))

                        metrics_mgr.add_metric("time/actor_train", actor_train_timer.last)

                    metrics_mgr.add_metric("time/step_total", step_total_timer.last)

                metrics = metrics_mgr.get_metrics()
                
                self.state.step = global_step
                self.state.log_history.append(metrics)
                self.tracker.log(values=metrics, step=global_step)
                self.do_checkpoint(global_step=global_step)
                
                timeline_dir = os.path.join(self.pipeline_config.profiler_output_dir, "timeline")
                os.makedirs(timeline_dir, exist_ok=True)
                ray.timeline(
                    filename=os.path.join(timeline_dir, f"timeline-step-{global_step}.json"),
                )

                logger.info(f"pipeline step {global_step} finished")
                global_step += 1
                if global_step >= self.pipeline_config.max_steps:
                    break
                
                if global_step >= self.pipeline_config.max_steps:
                    break

        logger.info("pipeline complete!")
