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
from dacite import from_dict
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from roll.pipeline.rlvr.rlvr_config import RLVRConfig


def make_baseline_config(config_path, config_name):

    initialize(config_path=config_path)
    cfg = compose(config_name=config_name)
    ppo_config = from_dict(data_class=RLVRConfig, data=OmegaConf.to_container(cfg, resolve=True))

    return ppo_config
