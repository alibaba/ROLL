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
import inspect
from functools import partial
from typing import Optional, Iterator

import torch
from torch import Tensor

from roll.distributed.scheduler.protocol import DataProto
from roll.utils.functionals import masked_mean, log_probs_from_logits


def log_probs_post_func(data: DataProto, output_tensor: torch.Tensor):
    """
    input_ids [[p, p, r, r, r, 0, 0]] p: prompt, r: response, 0: pad
    response_mask [[0, 0, 1, 1, 1, 0, 0]]
    """
    labels = data.batch["input_ids"][:, 1:].clone()
    labels[data.batch["response_mask"][:, 1:] == 0] = 0  # avoid invalid token id
    log_probs = log_probs_from_logits(output_tensor[:, :-1], labels)
    log_probs = log_probs * data.batch["response_mask"][:, 1:]
    return log_probs, {"log_probs": log_probs.clone().detach()}


def log_probs_forward_step_func(data_iterator: Iterator, model):
    data: DataProto = next(data_iterator)
    input_ids = data.batch["input_ids"]
    attention_mask = data.batch["attention_mask"]
    position_ids = data.batch["position_ids"]
    forward_args = filter_forward_args(model, data.meta_info.get("forward_args", {}))
    output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, **forward_args)
    if isinstance(output, Tensor):
        logits = output
    else:
        logits = output.logits
    return logits, partial(log_probs_post_func, data)


def filter_forward_args(model, forward_args):
    forward_func = model.forward
    signature = inspect.signature(forward_func)
    forward_params = signature.parameters.keys()
    valid_args = {k: v for k, v in forward_args.items() if k in forward_params}
    return valid_args
