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
from torch.utils.data import DistributedSampler, DataLoader
from transformers import DataCollatorWithPadding

from roll.configs import ModelArguments, DataArguments
from roll.configs.training_args import TrainingArguments
from roll.datasets.loader import get_dataset
from roll.models.model_providers import default_tokenizer_provider


def get_mock_dataloader(model_args: ModelArguments, data_args: DataArguments, batch_size: int = 4):

    tokenizer = default_tokenizer_provider(model_args=model_args)

    dataset = get_dataset(
        tokenizer=tokenizer,
        data_args=data_args,
    )
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
    sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=42,
        drop_last=True,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return dataloader, tokenizer
