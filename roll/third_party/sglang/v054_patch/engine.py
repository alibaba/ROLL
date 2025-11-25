import asyncio
from sglang.srt.entrypoints.engine import Engine

from roll.third_party.sglang.io_struct import (
    SetupCollectiveGroupReqInput,
    BroadcastBucketReqInput,
    BroadcastParameterReqInput,
    UpdateParameterInBucketReqInput,
    UpdateParameterReqInput,
)
import sglang.srt.entrypoints.engine as engine_module


class EngineSA(Engine):

    def setup_collective_group(
        self,
        comm_plan: str,
        backend: str,
        rank_in_cluster: int,
    ):
        obj = SetupCollectiveGroupReqInput(
            comm_plan=comm_plan,
            backend=backend,
            rank_in_cluster=rank_in_cluster,
        )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.setup_collective_group(obj, None)
        )
    
    def broadcast_bucket(
        self,
        src_pp_rank: int, 
        meta_infos: dict, 
        bucket_size: int,
    ):
        obj = BroadcastBucketReqInput(
            src_pp_rank=src_pp_rank,
            meta_infos=meta_infos,
            bucket_size=bucket_size,
        )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.broadcast_bucket(obj, None)
        )
    
    def broadcast_parameter(
        self,
        src_pp_rank, 
        dtype, 
        shape, 
        parameter_name
    ):
        obj = BroadcastParameterReqInput(
            src_pp_rank=src_pp_rank,
            dtype=dtype,
            shape=shape,
            parameter_name=parameter_name,
        )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.broadcast_parameter(obj, None)
        )
    
    def update_parameter(
        self,
        parameter_name, 
        weight, 
        ranks_in_worker
    ):
        obj = UpdateParameterReqInput(
            parameter_name=parameter_name,
            weight=weight,
            ranks_in_worker=ranks_in_worker,
        )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.update_parameter(obj, None)
        )
    
    def update_parameter_in_bucket(
        self,
        meta_infos, 
        buffer, 
        ranks_in_worker
    ):
        """Initialize parameter update group."""
        obj = UpdateParameterInBucketReqInput(
            meta_infos=meta_infos,
            buffer=buffer,
            ranks_in_worker=ranks_in_worker,
        )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.update_parameter_in_bucket(obj, None)
        )
    
class _roll_launch_subprocesses(object):
    def __init__(self, _launch_subprocesses):
        self._launch_subprocesses = _launch_subprocesses
    
    def __call__(self, *args, **kwargs):
        import sys
        from roll.third_party.sglang.v054_patch.tokenizer_manager import TokenizerManagerSA
        from roll.third_party.sglang.v054_patch.scheduler import run_scheduler_process, run_data_parallel_controller_process
        
        sys.modules['sglang.srt.entrypoints.engine'].__dict__['TokenizerManager'] = TokenizerManagerSA
        sys.modules['sglang.srt.entrypoints.engine'].__dict__['run_scheduler_process'] = run_scheduler_process
        sys.modules['sglang.srt.entrypoints.engine'].__dict__['run_data_parallel_controller_process'] = run_data_parallel_controller_process
        return self._launch_subprocesses(*args, **kwargs)



engine_module._launch_subprocesses = _roll_launch_subprocesses(engine_module._launch_subprocesses)