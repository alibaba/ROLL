import os
import time
from collections.abc import Mapping, Sequence
from copy import copy
from typing import Any, Optional, Union

from vllm import envs
from vllm.config import VllmConfig
from vllm.usage.usage_lib import UsageContext
from vllm.v1.metrics.loggers import StatLoggerFactory
from vllm.v1.engine.processor import Processor
from vllm.config import VllmConfig
from vllm.inputs import ProcessorInputs, PromptType, SingletonInputs
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.inputs.parse import split_enc_dec_inputs
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.lora.request import LoRARequest
from vllm.multimodal.utils import argsort_mm_positions
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core_client import SyncMPClient
from vllm.v1.executor.abstract import Executor
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.metrics.stats import IterationStats
from roll.utils.logging import get_logger

logger = get_logger()

def custom_process_inputs(
    self,
    request_id: str,
    prompt: ProcessorInputs,
    params: Union[SamplingParams, PoolingParams],
    arrival_time: Optional[float] = None,
    lora_request: Optional[LoRARequest] = None,
    trace_headers: Optional[Mapping[str, str]] = None,
    priority: int = 0,
    data_parallel_rank: Optional[int] = None,
) -> tuple[Optional[str], EngineCoreRequest]:

    # TODO(woosuk): Support pooling models.
    self._validate_lora(lora_request)
    self._validate_params(params)

    data_parallel_size = self.vllm_config.parallel_config.data_parallel_size
    if data_parallel_rank is not None and not (0 <= data_parallel_rank <
                                               data_parallel_size):
        raise ValueError(f"data_parallel_rank {data_parallel_rank} "
                         f"is out of range [0, {data_parallel_size}).")

    assert arrival_time is not None
    processed_inputs: ProcessorInputs = prompt

    eos_token_id = self.input_preprocessor.get_eos_token_id()

    encoder_inputs, decoder_inputs = split_enc_dec_inputs(processed_inputs)
    self._validate_model_inputs(encoder_inputs, decoder_inputs)

    sampling_params = None
    pooling_params = None
    if isinstance(params, SamplingParams):
        # TODO: can we avoid cloning here in multiproc case?
        sampling_params = params.clone()
        # If unset max tokens, then generate up to the max_model_len.
        if sampling_params.max_tokens is None:
            sampling_params.max_tokens = (
                self.model_config.max_model_len -
                len(decoder_inputs["prompt_token_ids"]))
        sampling_params.update_from_generation_config(
            self.generation_config_fields, eos_token_id)
        if self.tokenizer is not None:
            sampling_params.update_from_tokenizer(self.tokenizer)
    else:
        pooling_params = params.clone()

    # Multimodal related.
    mm_features: Optional[list[MultiModalFeatureSpec]] = None

    if decoder_inputs["type"] == "multimodal":
        decoder_mm_inputs = decoder_inputs["mm_kwargs"]
        decoder_mm_positions = decoder_inputs["mm_placeholders"]
        decoder_mm_hashes = decoder_inputs["mm_hashes"]

        # Merge and flatten multimodal placeholders, hashes and inputs
        # from dictionaries to lists, and sort them by each item's position
        # in the input sequence.
        sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)

        mm_features = []
        for modality, idx in sorted_mm_idxs:
            mm_features.append(
                MultiModalFeatureSpec(
                    data=decoder_mm_inputs[modality][idx],
                    modality=modality,
                    identifier=decoder_mm_hashes[modality][idx],
                    mm_position=decoder_mm_positions[modality][idx]))

    return decoder_inputs.get("prompt"), EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=decoder_inputs["prompt_token_ids"],
        mm_features=mm_features,
        sampling_params=sampling_params,
        pooling_params=pooling_params,
        eos_token_id=eos_token_id,
        arrival_time=arrival_time,
        lora_request=lora_request,
        cache_salt=decoder_inputs.get("cache_salt"),
        priority=priority,
        data_parallel_rank=data_parallel_rank,
        trace_headers=trace_headers,
    )

Processor.custom_process_inputs = custom_process_inputs

def get_output_nowait(self) -> EngineCoreOutputs:
    """
    Only get an item if one is immediately available. Otherwise
    raise the queue.Empty exception.
    """
    # If an exception arises in process_outputs_socket task,
    # it is forwarded to the outputs_queue so we can raise it
    # from this (run_output_handler) task to shut down the server.
    outputs = self.outputs_queue.get_nowait()
    if isinstance(outputs, Exception):
        raise self._format_exception(outputs) from None
    if outputs.wave_complete is not None:
        self.engines_running = False
    return outputs

# Function 'step' of vllm v1 and v0 engine has different semantic.
# Function vllm.v1.engine.LLMEngine.step is blocking but that of v0 is not.
# This will cause deadlock when calling roll.third_party.vllm.vllm_0_8_4.Llm084.fetch_output
# inside VllmStrategy if set generate_opt_level to 1.
SyncMPClient.get_output_nowait = get_output_nowait

class LLMEngine0110(LLMEngine):

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[list[StatLoggerFactory]] = None,
        disable_log_stats: bool = False,
    ) -> "LLMEngine":
        parallel_config = vllm_config.parallel_config

        executor_class = Executor.get_class(vllm_config)
        if parallel_config.distributed_executor_backend == "ray":
            from roll.third_party.vllm.vllm_0_11_0.v1.ray_distributed_executor import (
                CustomRayDistributedExecutor as V1CustomeRayDistributedExecutor)
            executor_class = V1CustomeRayDistributedExecutor

        # Default fork method is not compatible with ScaleAligner.
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

        logger.info(f"Using executor_class: {executor_class}")
        logger.info(f"Using worker cls: {parallel_config.worker_cls}")
        return cls(vllm_config=vllm_config,
                   executor_class=executor_class,
                   log_stats=(not disable_log_stats),
                   usage_context=usage_context,
                   stat_loggers=stat_loggers,
                   multiprocess_mode=envs.VLLM_ENABLE_V1_MULTIPROCESSING)

    def _add_processed_request(
        self,
        request_id: str,
        processed_inputs: ProcessorInputs,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: float,
        lora_request: Optional[LoRARequest],
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> None:
        prompt_str, request = self.processor.custom_process_inputs(request_id, processed_inputs, params,
                                                arrival_time, lora_request,
                                                trace_headers,
                                                priority)

        n = params.n if isinstance(params, SamplingParams) else 1

        if n == 1:
            # Make a new RequestState and queue.
            self.output_processor.add_request(request, prompt_str, None, 0)
            # Add the request to EngineCore.
            self.engine_core.add_request(request)
            return

        # Fan out child requests (for n>1).
        parent_req = ParentRequest(request_id, params)
        for idx in range(n):
            request_id, params = parent_req.get_child_info(idx)
            child_request = request if idx == n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = params

            # Make a new RequestState and queue.
            self.output_processor.add_request(child_request,prompt_str, parent_req, idx)
            # Add the request to EngineCore.
            self.engine_core.add_request(child_request)

    def step_nowait(self) -> Union[list[RequestOutput], list[PoolingRequestOutput]]:

        if self.should_execute_dummy_batch:
            self.should_execute_dummy_batch = False
            self.engine_core.execute_dummy_batch()
            return []

        # 1) Get EngineCoreOutput from the EngineCore.
        outputs = self.engine_core.get_output_nowait()

        # 2) Process EngineCoreOutputs.
        iteration_stats = IterationStats() if self.log_stats else None
        processed_outputs = self.output_processor.process_outputs(
            outputs.outputs,
            engine_core_timestamp=outputs.timestamp,
            iteration_stats=iteration_stats)

        # 3) Abort any reqs that finished due to stop strings.
        self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

        # 4) Record stats
        if self.logger_manager is not None:
            assert outputs.scheduler_stats is not None
            self.logger_manager.record(scheduler_stats=outputs.scheduler_stats,
                                    iteration_stats=iteration_stats)
            self.do_log_stats_with_interval()

        return processed_outputs.request_outputs
