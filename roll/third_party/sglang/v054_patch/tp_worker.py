from sglang.srt.managers.tp_worker import TpModelWorker


from roll.third_party.sglang.io_struct import (
    SetupCollectiveGroupReqInput,
    BroadcastBucketReqInput,
    BroadcastParameterReqInput,
    UpdateParameterInBucketReqInput,
    UpdateParameterReqInput,
)

class TpModelWorkerSA(TpModelWorker):
    def __init__(self, *args, **kwargs):
        import sys
        from roll.third_party.sglang.v054_patch.model_runner import ModelRunnerSA
        sys.modules['sglang.srt.managers.tp_worker'].__dict__['ModelRunner'] = ModelRunnerSA
        super().__init__(*args, **kwargs)

    def setup_collective_group(self, recv_req: SetupCollectiveGroupReqInput):
        success, message = self.model_runner.setup_collective_group(
            recv_req.comm_plan,
            recv_req.backend,
            recv_req.rank_in_cluster,
        )
        return success, message

    def broadcast_bucket(self, recv_req: BroadcastBucketReqInput):
        success, message = self.model_runner.broadcast_bucket(
            recv_req.src_pp_rank,
            recv_req.meta_infos,
            recv_req.bucket_size,
        )
        return success, message

    def broadcast_parameter(self, recv_req: BroadcastParameterReqInput):
        success, message = self.model_runner.broadcast_parameter(
            recv_req.src_pp_rank,
            recv_req.dtype,
            recv_req.shape,
            recv_req.parameter_name,
        )
        return success, message

    def update_parameter(self, recv_req: UpdateParameterReqInput):
        success, message = self.model_runner.update_parameter(
            recv_req.parameter_name,
            recv_req.weight,
            recv_req.ranks_in_worker,
        )
        return success, message

    def update_parameter_in_bucket(self, recv_req: UpdateParameterInBucketReqInput):
        success, message = self.model_runner.update_parameter_in_bucket(
            recv_req.meta_infos,
            recv_req.buffer,
            recv_req.ranks_in_worker,
        )
        return success, message