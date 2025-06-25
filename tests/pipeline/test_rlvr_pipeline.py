# Copyright (c) 2025, ALIBABA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse

from dacite import from_dict
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.rlvr.rlvr_config import RLVRConfig

parser = argparse.ArgumentParser(description="PPO Configuration")

parser.add_argument(
    "--config_name", type=str, default="rlvr_megatron_config", help="Name of the PPO configuration."
)
args = parser.parse_args()


def make_ppo_config():

    config_path = "."
    config_name = args.config_name

    initialize(config_path=config_path)
    cfg = compose(config_name=config_name)
    ppo_config = from_dict(data_class=RLVRConfig, data=OmegaConf.to_container(cfg, resolve=True))

    return ppo_config


def test_make_ppo_config():
    ppo_config = make_ppo_config()
    print(ppo_config)


def test_ppo_pipeline():

    ppo_config = make_ppo_config()

    init()

    from roll.pipeline.rlvr.rlvr_pipeline import RLVRPipeline
    pipeline = RLVRPipeline(pipeline_config=ppo_config)

    pipeline.run()


if __name__ == "__main__":
    test_ppo_pipeline()
