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
from dataclasses import dataclass

@dataclass
class SetupCollectiveGroupReqInput:
    comm_plan: dict
    backend: int
    rank_in_cluster: int


@dataclass
class SetupCollectiveGroupReqOutput:
    success: bool
    message: str

@dataclass
class BroadcastBucketReqInput:
    src_pp_rank: str
    meta_infos: dict
    bucket_size: int


@dataclass
class BroadcastBucketReqOutput:
    success: bool
    message: str

@dataclass
class BroadcastParameterReqInput:
    src_pp_rank: str
    dtype: int
    shape: dict
    parameter_name: str


@dataclass
class BroadcastParameterReqOutput:
    success: bool
    message: str

@dataclass
class UpdateParameterReqInput:
    parameter_name: str
    weight: int
    ranks_in_worker: dict


@dataclass
class UpdateParameterReqOutput:
    success: bool
    message: str

@dataclass
class UpdateParameterInBucketReqInput:
    meta_infos: str
    buffer: int
    ranks_in_worker: dict


@dataclass
class UpdateParameterInBucketReqOutput:
    success: bool
    message: str