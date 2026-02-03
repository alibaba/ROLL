from typing import Any

import torch

from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.initialize import init
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.base_worker import ActorWorker
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.utils.logging import get_logger
from tests.distributed.strategy.make_baseline_config import make_baseline_config

logger = get_logger()


class TestModelCheckpointPipeline(BasePipeline):

    def __init__(self, pipeline_config: RLVRConfig):
        super().__init__(pipeline_config)

        self.tokenizer = default_tokenizer_provider(
            model_args=self.pipeline_config.actor_train.model_args,
        )
        max_steps = 10240 * self.pipeline_config.actor_train.training_args.num_train_epochs
        self.pipeline_config.set_max_steps(max_steps=max_steps)

        self.actor_train: Any = Cluster(
            name=self.pipeline_config.actor_train.name,
            worker_cls=ActorWorker,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_train,
        )
        self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=True)
        self.set_checkpoint_clusters(self.actor_train)

    @torch.no_grad()
    def run(self):
        # self.actor_train.strategy.save_checkpoint(self.pipeline_config.output_dir, global_step)
        self.state.log_history.append({"global_step": 1})
        self.do_checkpoint(global_step=1)
        self.do_checkpoint(global_step=2, is_last_step=True)


if __name__ == "__main__":
    init()

    ppo_config = make_baseline_config(config_path="./checkpoint", config_name="fsdp_config")
    # ppo_config = make_baseline_config(
    #     config_path="./checkpoint", config_name="fsdp_lora_config"
    # )

    pipeline = TestModelCheckpointPipeline(ppo_config)
    metric_list = pipeline.run()
