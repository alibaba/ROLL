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
from queue import Queue
import psutil
import threading
from typing import Optional
import torch

from sglang.srt.server_args import ServerArgs
from sglang.srt.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.tp_worker_overlap_thread import TpModelWorkerClient
from sglang.srt.distributed import get_pp_group, get_world_group


from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.hf_transformers_utils import get_processor, get_tokenizer
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, TokenToKVPoolAllocator
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import broadcast_pyobj, set_random_seed


from roll.third_party.sglang.v046post4_patch.io_struct import (
    SetupCollectiveGroupReqInput,
    BroadcastBucketReqInput,
    BroadcastParameterReqInput,
    UpdateParameterInBucketReqInput,
    UpdateParameterReqInput,
)
from roll.third_party.sglang.v046post4_patch.model_runner import ModelRunnerSA

class TpModelWorkerSA(TpModelWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[TokenToKVPoolAllocator] = None,
    ):
        # Parse args
        self.tp_size = server_args.tp_size
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank

        # Init model and tokenizer
        self.model_config = ModelConfig.from_server_args(
            server_args,
            model_path=(
                server_args.model_path
                if not is_draft_worker
                else server_args.speculative_draft_model_path
            ),
            is_draft_model=is_draft_worker,
        )

        self.model_runner = ModelRunnerSA(
            model_config=self.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            pp_rank=pp_rank,
            pp_size=server_args.pp_size,
            nccl_port=nccl_port,
            server_args=server_args,
            is_draft_worker=is_draft_worker,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
        self.device = self.model_runner.device

        # Init nccl groups
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        # Profile number of tokens
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.max_running_requests = min(
            (
                self.max_total_num_tokens // 2
                if server_args.max_running_requests is None
                else server_args.max_running_requests
                // (server_args.dp_size if server_args.enable_dp_attention else 1)
            ),
            self.model_runner.req_to_token_pool.size,
        )
        assert self.max_running_requests > 0, "max_running_request is zero"
        self.max_req_len = min(
            self.model_config.context_len - 1,
            self.max_total_num_tokens - 1,
        )
        self.max_req_input_len = self.max_req_len - 5
        assert (
            self.max_req_len > 0 and self.max_req_input_len > 0
        ), "Memory pool size is too small"

        # Sync random seed across TP workers
        self.random_seed = broadcast_pyobj(
            [server_args.random_seed],
            self.tp_size * self.pp_rank + tp_rank,
            self.world_group.cpu_group,
            src=self.world_group.ranks[0],
        )[0]
        set_random_seed(self.random_seed)

        # A reference make this class has the same member as TpModelWorkerClient
        self.worker = self

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


class TpModelWorkerClientSA(TpModelWorkerClient):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
    ):
        # Load the model
        self.worker = TpModelWorkerSA(
            server_args, gpu_id, tp_rank, pp_rank, dp_rank, nccl_port
        )
        self.max_running_requests = self.worker.max_running_requests
        self.device = self.worker.device
        self.gpu_id = gpu_id

        # Init future mappings
        self.future_token_ids_ct = 0
        self.future_token_ids_limit = self.max_running_requests * 3
        self.future_token_ids_map = torch.empty(
            (self.max_running_requests * 5,), dtype=torch.int64, device=self.device
        )

        # Launch threads
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.forward_stream = torch.get_device_module(self.device).Stream()
        self.forward_thread = threading.Thread(
            target=self.forward_thread_func,
        )
        self.forward_thread.start()
        self.parent_process = psutil.Process().parent()
        self.scheduler_stream = torch.get_device_module(self.device).current_stream()
        if self.device == "cpu":
            self.scheduler_stream.synchronize = lambda: None  # No-op for CPU

    def setup_collective_group(self, recv_req: SetupCollectiveGroupReqInput):
        success, message = self.worker.setup_collective_group(recv_req)
        return success, message

    def broadcast_bucket(self, recv_req: BroadcastBucketReqInput):
        success, message = self.worker.broadcast_bucket(recv_req)
        return success, message

    def broadcast_parameter(self, recv_req: BroadcastParameterReqInput):
        success, message = self.worker.broadcast_parameter(recv_req)
        return success, message

    def update_parameter(self, recv_req: UpdateParameterReqInput):
        success, message = self.worker.update_parameter(recv_req)
        return success, message

    def update_parameter_in_bucket(self, recv_req: UpdateParameterInBucketReqInput):
        success, message = self.worker.update_parameter_in_bucket(recv_req)
        return success, message