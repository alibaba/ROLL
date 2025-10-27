import os
from collections.abc import Mapping, Sequence
from copy import copy
from typing import Optional, Union

from vllm import envs
from vllm.config import VllmConfig
from vllm.usage.usage_lib import UsageContext
from vllm.engine.metrics_types import StatLoggerBase
from vllm.v1.engine.processor import Processor
from vllm.config import VllmConfig
from vllm.inputs import ProcessorInputs
from vllm.inputs.parse import split_enc_dec_inputs
from vllm.outputs import RequestOutput
from vllm.lora.request import LoRARequest
from vllm.multimodal import MultiModalKwargs
from vllm.multimodal.inputs import PlaceholderRange
from vllm.multimodal.utils import merge_and_sort_multimodal_metadata
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core_client import SyncMPClient
from vllm.v1.executor.abstract import Executor
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.engine.parallel_sampling import ParentRequest
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
) -> EngineCoreRequest:

    self._validate_lora(lora_request)
    self._validate_params(params,lora_request)
    if priority != 0:
        raise ValueError("V1 does not support priority yet.")
    if trace_headers is not None:
        raise ValueError("V1 does not support tracing yet.")

    assert arrival_time is not None

    processed_inputs: ProcessorInputs = prompt
    eos_token_id = self.input_preprocessor.get_eos_token_id(lora_request)

    self._validate_model_inputs(processed_inputs, lora_request)

    encoder_inputs, decoder_inputs = split_enc_dec_inputs(processed_inputs)

    if encoder_inputs is not None:
        raise NotImplementedError

    assert isinstance(params, SamplingParams)
    sampling_params = params.clone()
    # If unset max tokens, then generate up to the max_model_len.
    if sampling_params.max_tokens is None:
        sampling_params.max_tokens = (
            self.model_config.max_model_len -
            len(decoder_inputs["prompt_token_ids"]))
    sampling_params.update_from_generation_config(
        self.generation_config_fields, eos_token_id)
    sampling_params.update_from_tokenizer(
        self.tokenizer.get_lora_tokenizer(lora_request))
    pooling_params = None

    # Multimodal related.
    sorted_mm_inputs: Optional[Sequence[Optional[MultiModalKwargs]]] = None
    sorted_mm_positions: Optional[list[PlaceholderRange]] = None
    sorted_mm_hashes: Optional[list[str]] = None
    if decoder_inputs["type"] == "multimodal":
        decoder_mm_inputs = decoder_inputs["mm_kwargs"]

        # Merge and flatten multimodal placeholders, hashes and inputs
        # from dictionaries to lists, and sort them by each item's position
        # in the input sequence.
        (
            sorted_item_modalities,
            sorted_mm_positions,
            sorted_mm_hashes,
        ) = merge_and_sort_multimodal_metadata(
            decoder_inputs["mm_placeholders"],
            decoder_inputs["mm_hashes"] if self.use_hash else None,
        )

        # The output of merged multi-modal processor (`decoder_mm_inputs`)
        # is a single MultiModalKwargs for all items from all modalities.
        # This code flattens kwargs for individual items in a list and
        # sorts them by each item's position in the input sequence if there
        # are multiple modalities.
        unique_modalities = set(sorted_item_modalities)
        if len(unique_modalities) > 1:
            orig_sorted_mm_inputs = []
            used_indices = {modality: 0 for modality in unique_modalities}

            for modality in sorted_item_modalities:
                items = decoder_mm_inputs.get_items(modality)
                item = items[used_indices[modality]]

                orig_sorted_mm_inputs.append(
                    MultiModalKwargs.from_items([item]))
                used_indices[modality] += 1
        else:
            orig_sorted_mm_inputs = [
                MultiModalKwargs.from_items([item]) for item in
                decoder_mm_inputs.get_items(sorted_item_modalities[0])
            ]

        if sorted_mm_hashes is not None:
            sorted_mm_inputs = self.mm_input_cache_client.get_and_update_p0(
                orig_sorted_mm_inputs, sorted_mm_hashes)
        else:
            sorted_mm_inputs = orig_sorted_mm_inputs

    return decoder_inputs.get("prompt"),EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=decoder_inputs["prompt_token_ids"],
        mm_inputs=sorted_mm_inputs,
        mm_hashes=sorted_mm_hashes,
        mm_placeholders=sorted_mm_positions,
        sampling_params=sampling_params,
        pooling_params=pooling_params,
        eos_token_id=eos_token_id,
        arrival_time=arrival_time,
        lora_request=lora_request,
        cache_salt=None,
        data_parallel_rank=None,
    )

Processor.custom_process_inputs = custom_process_inputs

def get_output_nowait(self) -> EngineCoreOutputs:
    """
    Only get an item if one is immediately available. Otherwise
    raise the queue.Empty exception.
    """
    return self.outputs_queue.get_nowait()

# Function 'step' of vllm v1 and v0 engine has different semantic.
# Function vllm.v1.engine.LLMEngine.step is blocking but that of v0 is not.
# This will cause deadlock when calling roll.third_party.vllm.vllm_0_8_4.Llm084.fetch_output
# inside VllmStrategy if set generate_opt_level to 1.
SyncMPClient.get_output_nowait = get_output_nowait

class LLMEngine0100(LLMEngine):

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[dict[str, StatLoggerBase]] = None,
        disable_log_stats: bool = False,
    ) -> "LLMEngine":
        if stat_loggers is not None:
            raise NotImplementedError(
                "Passing StatLoggers to V1 is not yet supported. "
                "Set VLLM_USE_V1=0 and file and issue on Github.")

        parallel_config = vllm_config.parallel_config

        executor_class = Executor.get_class(vllm_config)
        if parallel_config.distributed_executor_backend == "ray":
            from roll.third_party.vllm.vllm_0_10_0.v1.ray_distributed_executor import (
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

    def step_nowait(self) -> list[RequestOutput]:

        if self.should_execute_dummy_batch:
            self.should_execute_dummy_batch = False
            self.engine_core.execute_dummy_batch()
            return []

        # 1) Get EngineCoreOutput from the EngineCore.
        outputs = self.engine_core.get_output_nowait()

        # 2) Process EngineCoreOutputs.
        processed_outputs = self.output_processor.process_outputs(
            outputs.outputs)

        # 3) Abort any reqs that finished due to stop strings.
        self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

        return processed_outputs.request_outputs
