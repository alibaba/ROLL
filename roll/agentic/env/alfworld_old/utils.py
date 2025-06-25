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
import os
import yaml
import re
from typing import List, Any


def load_config(config_file: str, params: List[str] = []):
    assert os.path.exists(config_file), f"Invalid config file: {config_file}"
    with open(config_file) as reader:
        config = yaml.safe_load(reader)
    # Parse overriden params.
    for param in params:
        fqn_key, value = param.split("=")
        entry_to_change = config
        keys = fqn_key.split(".")
        for k in keys[:-1]:
            entry_to_change = entry_to_change[k]
        entry_to_change[keys[-1]] = value
    return config


def check_format(action: str, templates: Any) -> bool:
    """
    Validate that the action matches one of our action templates.
    Returns True if valid, False otherwise.
    """
    if "None" in action:
        return False

    # Skip validation for basic actions that don't have placeholders
    basic_actions = ["look", "inventory"]
    if action in basic_actions:
        return True

    # Check if the action follows any of our templates
    for template in templates:
        # Skip "None" and basic actions we already checked
        if template == "None" or template in basic_actions:
            continue

        # Convert template to regex pattern
        # Replace <something> with regex that matches any word(s)
        pattern = (
            template.replace("<receptacle>", "([\\w\\s]+)")
            .replace("<object>", "([\\w\\s]+)")
            .replace("<something>", "([\\w\\s]+)")
        )
        pattern = f"^{pattern}$"  # Match the entire string

        if re.match(pattern, action):
            return True

    return False


def check_correctness(action: str, target: str) -> bool: ...
