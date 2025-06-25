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
import dataclasses
import json
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Union

import numpy as np
import torch

from roll.utils.logging import logger


WORKER_STATE_NAME = "worker_state_{tag}.json"


@dataclass
class WorkerState:
    step: int = -1
    log_history: List[Dict[str, float]] = None
    kv: Dict[str, Union[float, Dict]] = None

    def __post_init__(self):
        if self.log_history is None:
            self.log_history = []
        if self.kv is None:
            self.kv = {}

    def save_to_json(self, save_dir: str, tag):
        """Save the content of this instance in JSON format inside `json_path`."""
        json_path = os.path.join(save_dir, WORKER_STATE_NAME.format(tag=tag))
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, load_dir: str, tag):
        """Create an instance from the content of `json_path`."""
        json_path = os.path.join(load_dir, WORKER_STATE_NAME.format(tag=tag))
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))

    @staticmethod
    def save_rng_state(save_dir, tag):
        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
            "cuda": torch.cuda.random.get_rng_state_all(),
        }
        os.makedirs(save_dir, exist_ok=True)
        torch.save(rng_states, os.path.join(save_dir, f"rng_state_{tag}.pth"))

    @staticmethod
    def load_rng_state(load_dir, tag):
        # Load RNG states from `checkpoint`
        if load_dir is None:
            return
        rng_file = os.path.join(load_dir, f"rng_state_{tag}.pth")
        if not os.path.isfile(rng_file):
            logger.info(
                f"Didn't find an RNG file for process {tag}, if you are resuming a training that "
                "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
            )
            return

        checkpoint_rng_state = torch.load(rng_file)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
