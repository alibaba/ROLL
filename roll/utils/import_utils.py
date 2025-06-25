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
from importlib.util import find_spec

import importlib
from typing import Any, Optional


def is_vllm_available() -> bool:
    return find_spec("vllm") is not None


def can_import_class(class_path: str) -> bool:
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        getattr(module, class_name)
        return True
    except (ModuleNotFoundError, AttributeError) as e:
        print(e)
        return False


def safe_import_class(class_path: str) -> Optional[Any]:
    if can_import_class(class_path):
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls
    else:
        return None
