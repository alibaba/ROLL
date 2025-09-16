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