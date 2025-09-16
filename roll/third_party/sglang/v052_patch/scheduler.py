import faulthandler
import logging
import os
import signal
import threading
import time
from types import SimpleNamespace
from typing import Dict, List, Optional

import psutil
import setproctitle
import torch
import zmq

from sglang.global_config import global_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constrained.base_grammar_backend import (
    create_grammar_backend,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.io_struct import (
    AbortReq,
    CloseSessionReqInput,
    ExpertDistributionReq,
    FlushCacheReqInput,
    GetInternalStateReq,
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterReqInput,
    OpenSessionReqInput,
    ProfileReq,
    BatchTokenizedGenerateReqInput,
    BatchTokenizedEmbeddingReqInput,
    ClearHiCacheReqInput,
    FreezeGCReq,
    MultiTokenizerRegisterReq,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    RpcReqInput,
    SetInternalStateReq,
    SlowDownReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UnloadLoRAAdapterReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import (
    Req,
    ScheduleBatch,
    global_server_args_dict,
)
from sglang.srt.managers.schedule_policy import (
    SchedulePolicy,
)
from sglang.srt.managers.scheduler_input_blocker import SchedulerInputBlocker
from sglang.srt.managers.session_controller import Session
from sglang.srt.managers.utils import DPBalanceMeta
from sglang.srt.managers.scheduler import Scheduler, IdleSleeper
from sglang.srt.managers.scheduler_recv_skipper import SchedulerRecvSkipper

from sglang.srt.managers.scheduler_update_weights_mixin import _import_static_state, _export_static_state

from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import (
    configure_gc_logger,
    configure_logger,
    get_available_gpu_memory,
    get_bool_env_var,
    get_zmq_socket,
    kill_itself_when_parent_died,
    numa_bind_to_node,
    set_gpu_proc_affinity,
    set_random_seed,
    suppress_other_loggers,
)
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

from roll.third_party.sglang.v052_patch.tp_worker import TpModelWorkerClientSA, TpModelWorkerSA
from roll.third_party.sglang.v052_patch.io_struct import (
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

logger = logging.getLogger(__name__)

# Test retract decode for debugging purposes
TEST_RETRACT = get_bool_env_var("SGLANG_TEST_RETRACT")
GRAMMAR_TIMEOUT = float(os.environ.get("SGLANG_GRAMMAR_TIMEOUT", 300))



class SchedulerSA(Scheduler):
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        dp_balance_meta: Optional[DPBalanceMeta] = None,
    ):
        # Parse args
        self.server_args = server_args
        self.tp_rank = tp_rank
        self.moe_ep_rank = moe_ep_rank
        self.pp_rank = pp_rank
        self.dp_rank = dp_rank
        self.tp_size = server_args.tp_size
        self.moe_ep_size = server_args.ep_size
        self.pp_size = server_args.pp_size
        self.dp_size = server_args.dp_size
        self.schedule_policy = server_args.schedule_policy
        self.enable_lora = server_args.enable_lora
        self.max_loras_per_batch = server_args.max_loras_per_batch
        self.enable_overlap = not server_args.disable_overlap_schedule
        self.skip_tokenizer_init = server_args.skip_tokenizer_init
        self.enable_metrics = server_args.enable_metrics
        self.enable_metrics_for_all_schedulers = (
            server_args.enable_metrics_for_all_schedulers
        )
        self.enable_kv_cache_events = server_args.kv_events_config is not None
        self.stream_interval = server_args.stream_interval
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.gpu_id = gpu_id
        self.enable_hierarchical_cache = server_args.enable_hierarchical_cache
        self.enable_hicache_storage = server_args.hicache_storage_backend is not None
        self.page_size = server_args.page_size

        self.attn_tp_rank, self.attn_tp_size, self.attn_dp_rank = (
            compute_dp_attention_world_info(
                server_args.enable_dp_attention,
                self.tp_rank,
                self.tp_size,
                self.dp_size,
            )
        )

        # Init model config
        self.model_config = ModelConfig.from_server_args(server_args)

        # Init inter-process communication
        context = zmq.Context(2)
        self.idle_sleeper = None
        if self.pp_rank == 0 and self.attn_tp_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )
            self.recv_from_rpc = get_zmq_socket(
                context, zmq.DEALER, port_args.rpc_ipc_name, False
            )

            self.send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )
            if server_args.skip_tokenizer_init:
                # Directly send to the TokenizerManager
                self.send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.tokenizer_ipc_name, False
                )
            else:
                # Send to the DetokenizerManager
                self.send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.detokenizer_ipc_name, False
                )

            if self.server_args.sleep_on_idle:
                self.idle_sleeper = IdleSleeper(
                    [
                        self.recv_from_tokenizer,
                        self.recv_from_rpc,
                    ]
                )
        else:
            self.recv_from_tokenizer = None
            self.recv_from_rpc = None
            self.send_to_tokenizer = SimpleNamespace(send_pyobj=lambda x: None)
            self.send_to_detokenizer = SimpleNamespace(send_pyobj=lambda x: None)

        if self.current_scheduler_metrics_enabled():
            self.send_metrics_from_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.metrics_ipc_name, False
            )

        # Init tokenizer
        self.init_tokenizer()

        # Init moe config
        self.init_moe_config()

        # Set reasoning_parser and think_end_id if --reasoning_parser is enabled
        if self.server_args.reasoning_parser and self.tokenizer:
            reasoning_parser = ReasoningParser(
                model_type=self.server_args.reasoning_parser, stream_reasoning=False
            )
            self.tokenizer.think_end_id = self.tokenizer.encode(
                reasoning_parser.detector.think_end_token, add_special_tokens=False
            )[0]

        # Check whether overlap can be enabled
        if not self.is_generation:
            self.enable_overlap = False
            logger.info("Overlap scheduler is disabled for embedding models.")

        # Launch a tensor parallel worker
        if self.enable_overlap:
            TpWorkerClass = TpModelWorkerClientSA
        else:
            TpWorkerClass = TpModelWorkerSA

        self.tp_worker = TpWorkerClass(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            nccl_port=port_args.nccl_port,
        )

        # Launch a draft worker for speculative decoding
        if self.spec_algorithm.is_eagle():
            from sglang.srt.speculative.eagle_worker import EAGLEWorker

            self.draft_worker = EAGLEWorker(
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                moe_ep_rank=moe_ep_rank,
                server_args=server_args,
                nccl_port=port_args.nccl_port,
                target_worker=self.tp_worker,
                dp_rank=dp_rank,
            )
        elif self.spec_algorithm.is_standalone():
            from sglang.srt.speculative.standalone_worker import StandaloneWorker

            self.draft_worker = StandaloneWorker(
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                moe_ep_rank=moe_ep_rank,
                server_args=server_args,
                nccl_port=port_args.nccl_port,
                target_worker=self.tp_worker,
                dp_rank=dp_rank,
            )
        else:
            self.draft_worker = None

        # Get token and memory info from the model worker
        (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_queued_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            worker_global_server_args_dict,
            _,
            _,
            _,
        ) = self.tp_worker.get_worker_info()
        if global_server_args_dict["max_micro_batch_size"] is None:
            global_server_args_dict["max_micro_batch_size"] = max(
                self.max_running_requests // server_args.pp_size, 1
            )

        self.tp_group = self.tp_worker.get_tp_group()
        self.tp_cpu_group = self.tp_group.cpu_group
        self.attn_tp_group = self.tp_worker.get_attention_tp_group()
        self.attn_tp_cpu_group = self.tp_worker.get_attention_tp_cpu_group()
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()
        global_server_args_dict.update(worker_global_server_args_dict)
        set_random_seed(self.random_seed)

        # Hybrid memory pool
        self.is_hybrid = self.tp_worker.is_hybrid
        if self.is_hybrid:
            self.sliding_window_size = self.tp_worker.sliding_window_size
            self.full_tokens_per_layer, self.swa_tokens_per_layer = (
                self.tp_worker.get_tokens_per_layer_info()
            )

        # Print debug info
        if tp_rank == 0:
            avail_mem = get_available_gpu_memory(
                self.device, self.gpu_id, empty_cache=False
            )
            logger.info(
                f"max_total_num_tokens={self.max_total_num_tokens}, "
                f"chunked_prefill_size={server_args.chunked_prefill_size}, "
                f"max_prefill_tokens={self.max_prefill_tokens}, "
                f"max_running_requests={self.max_running_requests}, "
                f"context_len={self.model_config.context_len}, "
                f"{'available_cpu_mem' if self.device == 'cpu' else 'available_gpu_mem'}={avail_mem:.2f} GB"
            )

        # Init memory pool and cache
        self.init_memory_pool_and_cache()

        # Init running status
        self.waiting_queue: List[Req] = []
        # The running decoding batch for continuous batching
        self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)
        # The current forward batch
        self.cur_batch: Optional[ScheduleBatch] = None
        # The last forward batch
        self.last_batch: Optional[ScheduleBatch] = None
        self.forward_ct = 0
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.last_prefill_tokens = 0
        self.last_decode_stats_tic = time.perf_counter()
        self.last_prefill_stats_tic = time.perf_counter()
        self.return_health_check_ct = 0
        self.num_retracted_reqs: int = 0
        self.num_paused_reqs: int = 0
        self.kv_transfer_speed_gb_s: float = 0.0
        self.kv_transfer_latency_ms: float = 0.0
        self.sessions: Dict[str, Session] = {}
        self.current_stream = torch.get_device_module(self.device).current_stream()
        if self.device == "cpu":
            self.current_stream.synchronize = lambda: None  # No-op for CPU
        self.forward_sleep_time = None

        # Init chunked prefill
        self.chunked_prefill_size = server_args.chunked_prefill_size
        if self.chunked_prefill_size <= 0:  # -1 means disable
            self.chunked_prefill_size = None
        self.chunked_req = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None and server_args.enable_mixed_chunk
        )

        # Init the grammar backend for constrained generation
        self.grammar_queue: List[Req] = []
        if not server_args.skip_tokenizer_init:
            self.grammar_backend = create_grammar_backend(
                server_args,
                self.tokenizer,
                self.model_config.vocab_size,
                self.model_config.hf_eos_token_id,
            )
        else:
            self.grammar_backend = None

        # Init schedule policy and new token estimation
        self.policy = SchedulePolicy(
            self.schedule_policy,
            self.tree_cache,
            self.enable_hierarchical_cache,
        )
        assert (
            server_args.schedule_conservativeness >= 0
        ), "Invalid schedule_conservativeness"
        self.init_new_token_ratio = min(
            global_config.default_init_new_token_ratio
            * server_args.schedule_conservativeness,
            1.0,
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio
            * global_config.default_min_new_token_ratio_factor,
            1.0,
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / global_config.default_new_token_ratio_decay_steps
        self.new_token_ratio = self.init_new_token_ratio

        # Init watchdog thread
        self.watchdog_timeout = server_args.watchdog_timeout
        t = threading.Thread(target=self.watchdog_thread, daemon=True)
        t.start()
        self.parent_process = psutil.Process().parent()

        # Init memory saver, profiler and metric stats
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )
        self.offload_tags = set()
        self.init_profiler()

        self.recv_skipper = SchedulerRecvSkipper.maybe_create(server_args)
        self.input_blocker = (
            SchedulerInputBlocker(noop=self.attn_tp_rank != 0)
            if get_bool_env_var("SGLANG_ENABLE_COLOCATED_BATCH_GEN")
            else None
        )

        # Init metrics stats
        self.init_metrics(tp_rank, pp_rank, dp_rank)
        self.init_kv_events(server_args.kv_events_config)
        self.init_dp_balance(dp_balance_meta)

        # Init disaggregation
        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        self.init_disaggregation()

        if get_bool_env_var("SGLANG_GC_LOG"):
            configure_gc_logger()

        # Init request dispatcher
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                (TokenizedEmbeddingReqInput, self.handle_embedding_request),
                (BatchTokenizedGenerateReqInput, self.handle_batch_generate_request),
                (BatchTokenizedEmbeddingReqInput, self.handle_batch_embedding_request),
                (FlushCacheReqInput, self.flush_cache_wrapped),
                (ClearHiCacheReqInput, self.clear_hicache_storage_wrapped),
                (AbortReq, self.abort_request),
                (OpenSessionReqInput, self.open_session),
                (CloseSessionReqInput, self.close_session),
                (UpdateWeightFromDiskReqInput, self.update_weights_from_disk),
                (InitWeightsUpdateGroupReqInput, self.init_weights_update_group),
                (
                    UpdateWeightsFromDistributedReqInput,
                    self.update_weights_from_distributed,
                ),
                (SetupCollectiveGroupReqInput, self.setup_collective_group),
                (BroadcastBucketReqInput, self.broadcast_bucket),
                (BroadcastParameterReqInput, self.broadcast_parameter),
                (UpdateParameterInBucketReqInput, self.update_parameter_in_bucket),
                (UpdateParameterReqInput, self.update_parameter),
                (UpdateWeightsFromTensorReqInput, self.update_weights_from_tensor),
                (GetWeightsByNameReqInput, self.get_weights_by_name),
                (ReleaseMemoryOccupationReqInput, self.release_memory_occupation),
                (ResumeMemoryOccupationReqInput, self.resume_memory_occupation),
                (SlowDownReqInput, self.slow_down),
                (ProfileReq, self.profile),
                (FreezeGCReq, self.handle_freeze_gc),
                (GetInternalStateReq, self.get_internal_state),
                (SetInternalStateReq, self.set_internal_state),
                (RpcReqInput, self.handle_rpc_request),
                (ExpertDistributionReq, self.expert_distribution_handle),
                (LoadLoRAAdapterReqInput, self.load_lora_adapter),
                (UnloadLoRAAdapterReqInput, self.unload_lora_adapter),
                (MultiTokenizerRegisterReq, self.register_multi_tokenizer),
            ]
        )

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
        self.tp_worker.worker.model_runner.model.to(torch.cuda.current_device())
        self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)

        # gc.collect()
        # torch.cuda.empty_cache()
        # self.tp_worker.worker.model_runner.model.to(torch.cuda.current_device())
        _import_static_state(
            self.tp_worker.worker.model_runner.model, self.stashed_model_static_state
        )
        del self.stashed_model_static_state

        self.tp_worker.worker.model_runner.init_cublas()
        self.tp_worker.worker.model_runner.init_attention_backend()
        from sglang.srt.model_executor.cuda_graph_runner import set_global_graph_memory_pool
        set_global_graph_memory_pool(None)
        self.tp_worker.worker.model_runner.init_device_graphs()

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


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
    balance_meta: Optional[DPBalanceMeta] = None,
):
    if (numa_node := server_args.numa_node) is not None:
        numa_bind_to_node(numa_node[gpu_id])

    # Generate the prefix
    prefix = ""
    if dp_rank is not None:
        prefix += f" DP{dp_rank}"
    if server_args.tp_size > 1:
        prefix += f" TP{tp_rank}"
    if server_args.ep_size > 1:
        prefix += f" EP{moe_ep_rank}"
    if server_args.pp_size > 1:
        prefix += f" PP{pp_rank}"

    # Config the process
    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    # Configure the logger
    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, gpu_id)

    # Create a scheduler and run the event loop
    try:
        scheduler = SchedulerSA(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            dp_rank,
            dp_balance_meta=balance_meta,
        )
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": scheduler.max_total_num_tokens,
                "max_req_input_len": scheduler.max_req_input_len,
            }
        )

        disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode
        if disaggregation_mode == DisaggregationMode.NULL:
            if server_args.pp_size > 1:
                scheduler.event_loop_pp()
            elif scheduler.enable_overlap:
                scheduler.event_loop_overlap()
            else:
                scheduler.event_loop_normal()
        elif disaggregation_mode == DisaggregationMode.PREFILL:
            if scheduler.enable_overlap:
                scheduler.event_loop_overlap_disagg_prefill()
            else:
                if server_args.pp_size > 1:
                    scheduler.event_loop_pp_disagg_prefill()
                else:
                    scheduler.event_loop_normal_disagg_prefill()

        elif disaggregation_mode == DisaggregationMode.DECODE:
            if scheduler.enable_overlap:
                scheduler.event_loop_overlap_disagg_decode()
            else:
                scheduler.event_loop_normal_disagg_decode()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)

