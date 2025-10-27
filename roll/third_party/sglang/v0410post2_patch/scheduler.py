import torch
from roll.platforms import current_platform
from sglang.srt.managers.io_struct import (
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
)
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_update_weights_mixin import _import_static_state, _export_static_state


from roll.third_party.sglang.io_struct import (
    SetupCollectiveGroupReqInput,
    BroadcastBucketReqInput,
    BroadcastParameterReqInput,
    UpdateParameterInBucketReqInput,
    UpdateParameterReqInput,
    SetupCollectiveGroupReqOutput,
    BroadcastBucketReqOutput,
    BroadcastParameterReqOutput,
    UpdateParameterInBucketReqOutput,
    UpdateParameterReqOutput,
)

class SchedulerSA(Scheduler):
    def __init__(self, *args, **kwargs):
        import sys
        from roll.third_party.sglang.v0410post2_patch.tp_worker import TpModelWorkerClientSA, TpModelWorkerSA
        sys.modules['sglang.srt.managers.scheduler'].__dict__['TpModelWorkerClient'] = TpModelWorkerClientSA
        sys.modules['sglang.srt.managers.scheduler'].__dict__['TpModelWorker'] = TpModelWorkerSA
        super().__init__(*args, **kwargs)
        func_map_patch = [(SetupCollectiveGroupReqInput, self.setup_collective_group),
                          (BroadcastBucketReqInput, self.broadcast_bucket),
                          (BroadcastParameterReqInput, self.broadcast_parameter),
                          (UpdateParameterInBucketReqInput, self.update_parameter_in_bucket),
                          (UpdateParameterReqInput, self.update_parameter)]
        self._request_dispatcher._mapping += func_map_patch

    def setup_collective_group(self, recv_req: SetupCollectiveGroupReqInput):
        success, message = self.tp_worker.setup_collective_group(recv_req)
        return SetupCollectiveGroupReqOutput(success, message)

    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        self.stashed_model_static_state = _export_static_state(
            self.tp_worker.worker.model_runner.model
        )
        self.tp_worker.worker.model_runner.model.to('cpu')
        self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
        self.flush_cache()
        return ReleaseMemoryOccupationReqOutput()
    
    def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):
        self.tp_worker.worker.model_runner.model.to(current_platform.current_device())
        self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)

        # gc.collect()
        # current_platform.empty_cache()
        # self.tp_worker.worker.model_runner.model.to(current_platform.current_device())
        _import_static_state(
            self.tp_worker.worker.model_runner.model, self.stashed_model_static_state
        )
        del self.stashed_model_static_state

        self.tp_worker.worker.model_runner.init_cublas()
        self.tp_worker.worker.model_runner.init_attention_backend()
        from sglang.srt.model_executor.cuda_graph_runner import set_global_graph_memory_pool
        set_global_graph_memory_pool(None)
        self.tp_worker.worker.model_runner.init_cuda_graphs()

        return ResumeMemoryOccupationReqOutput()

    def broadcast_bucket(self, recv_req: BroadcastBucketReqInput):
        success, message = self.tp_worker.broadcast_bucket(recv_req)
        return BroadcastBucketReqOutput(success, message)

    def broadcast_parameter(self, recv_req: BroadcastParameterReqInput):
        success, message = self.tp_worker.broadcast_parameter(recv_req)
        return BroadcastParameterReqOutput(success, message)

    def update_parameter(self, recv_req: UpdateParameterReqInput):
        success, message = self.tp_worker.update_parameter(recv_req)
        return UpdateParameterReqOutput(success, message)

    def update_parameter_in_bucket(self, recv_req: UpdateParameterInBucketReqInput):
        success, message = self.tp_worker.update_parameter_in_bucket(recv_req)
        return UpdateParameterInBucketReqOutput(success, message)


def run_scheduler_process(*args, **kwargs):
    import sys
    sys.modules['sglang.srt.managers.scheduler'].__dict__['Scheduler'] = SchedulerSA
    from sglang.srt.managers.scheduler import run_scheduler_process
    return run_scheduler_process(*args, **kwargs)