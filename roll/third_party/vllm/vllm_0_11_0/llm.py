import os
import queue
import time
from typing import Any, Dict, Iterable, List, Optional, Union

import cloudpickle
import torch
from pydantic import ValidationError
from vllm import LLM, EngineArgs, SamplingParams, envs
from vllm.config import (
    CompilationConfig,
    StructuredOutputsConfig,
    is_init_field,
)

try:
    # 0.11.1rc2.dev0+gc3a722fcb.d20251021 has import diff
    from vllm.config.model import ModelDType, TokenizerMode
except ImportError:
    from vllm.config import ModelDType, TokenizerMode

from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.engine.arg_utils import (ConvertOption, EngineArgs, HfOverrides,
                                   PoolerConfig, RunnerOption)
from vllm.v1.sample.logits_processor import LogitsProcessor
from vllm.entrypoints.utils import (log_non_default_args)
from vllm.lora.request import LoRARequest
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter
from vllm.plugins.io_processors import get_io_processor
from vllm.envs import get_default_cache_root

from roll.third_party.vllm.vllm_0_11_0.llm_engine import LLMEngine0110
from roll.utils.send_recv_utils import SendBucketManager
from roll.utils.logging import get_logger
from roll.platforms import current_platform

logger = get_logger()

class Llm0110(LLM):

    def __init__(
        self,
        resource_placement_groups: List[Dict],
        model: str,
        *,
        runner: RunnerOption = "auto",
        convert: ConvertOption = "auto",
        tokenizer: Optional[str] = None,
        tokenizer_mode: TokenizerMode = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        allowed_local_media_path: str = "",
        allowed_media_domains: Optional[list[str]] = None,
        tensor_parallel_size: int = 1,
        dtype: ModelDType = "auto",
        quantization: Optional[QuantizationMethods] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: bool = False,
        disable_custom_all_reduce: bool = False,
        hf_token: Optional[Union[bool, str]] = None,
        hf_overrides: Optional[HfOverrides] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        pooler_config: Optional[PoolerConfig] = None,
        override_pooler_config: Optional[PoolerConfig] = None,
        structured_outputs_config: Optional[Union[dict[
            str, Any], StructuredOutputsConfig]] = None,
        kv_cache_memory_bytes: Optional[int] = None,
        compilation_config: Optional[Union[int, dict[str, Any],
                                           CompilationConfig]] = None,
        logits_processors: Optional[list[Union[str,
                                               type[LogitsProcessor]]]] = None,
        **kwargs: Any,
    ) -> None:
        """LLM constructor."""
        # setup envs for vllm
        # https://github.com/vllm-project/vllm/pull/14189/files
        # TODO do not override other options in PYTORCH_CUDA_ALLOC_CONF
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
        # torch.cuda may already init, explicitly disable expandable_segments
        # here (only matters when VLLM_USE_RAY_SPMD_WORKER=0)
        current_platform.set_allocator_settings("expandable_segments:False")

        os.environ["VLLM_CACHE_ROOT"] = os.path.join(
            get_default_cache_root(), "vllm", os.environ.get("WORKER_NAME", ""))

        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True

        if "worker_cls" in kwargs:
            worker_cls = kwargs["worker_cls"]
            # if the worker_cls is not qualified string name,
            # we serialize it using cloudpickle to avoid pickling issues
            if isinstance(worker_cls, type):
                kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)

        if "kv_transfer_config" in kwargs and isinstance(
                kwargs["kv_transfer_config"], dict):
            from vllm.config.kv_transfer import KVTransferConfig
            raw_config_dict = kwargs["kv_transfer_config"]
            try:
                kwargs["kv_transfer_config"] = KVTransferConfig(
                    **raw_config_dict)
            except ValidationError as e:
                logger.error(
                    "Failed to convert 'kv_transfer_config' dict to "
                    "KVTransferConfig object. Dict: %s. Error: %s",
                    raw_config_dict, e)
                # Consider re-raising a more specific vLLM error or ValueError
                # to provide better context to the user.
                raise ValueError(
                    f"Invalid 'kv_transfer_config' provided: {e}") from e

        if hf_overrides is None:
            hf_overrides = {}

        if compilation_config is not None:
            if isinstance(compilation_config, int):
                compilation_config_instance = CompilationConfig(
                    level=compilation_config)
            elif isinstance(compilation_config, dict):
                predicate = lambda x: is_init_field(CompilationConfig, x[0])
                compilation_config_instance = CompilationConfig(
                    **dict(filter(predicate, compilation_config.items())))
            else:
                compilation_config_instance = compilation_config
        else:
            compilation_config_instance = CompilationConfig()

        if structured_outputs_config is not None:
            if isinstance(structured_outputs_config, dict):
                structured_outputs_instance = StructuredOutputsConfig(
                    **{
                        k: v
                        for k, v in structured_outputs_config.items()
                        if is_init_field(StructuredOutputsConfig, k)
                    })
            else:
                structured_outputs_instance = structured_outputs_config
        else:
            structured_outputs_instance = StructuredOutputsConfig()

        kwargs["enable_sleep_mode"] = True
        engine_args = EngineArgs(
            model=model,
            runner=runner,
            convert=convert,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            allowed_local_media_path=allowed_local_media_path,
            allowed_media_domains=allowed_media_domains,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            disable_custom_all_reduce=disable_custom_all_reduce,
            hf_token=hf_token,
            hf_overrides=hf_overrides,
            mm_processor_kwargs=mm_processor_kwargs,
            pooler_config=pooler_config,
            override_pooler_config=override_pooler_config,
            structured_outputs_config=structured_outputs_instance,
            compilation_config=compilation_config_instance,
            logits_processors=logits_processors,
            **kwargs,
        )
        engine_args.resource_placement_groups = resource_placement_groups

        log_non_default_args(engine_args)

        # Create the Engine (autoselects V0 vs V1)
        self.llm_engine = LLMEngine0110.from_engine_args(
            engine_args=engine_args, usage_context=UsageContext.LLM_CLASS)
        self.engine_class = type(self.llm_engine)

        self.request_counter = Counter()
        self.default_sampling_params: Union[dict[str, Any], None] = None

        supported_tasks = self.llm_engine.get_supported_tasks()  # type: ignore

        logger.info("Supported_tasks: %s", supported_tasks)

        self.supported_tasks = supported_tasks

        # Load the Input/Output processor plugin if any
        self.model_config = self.llm_engine.model_config
        io_processor_plugin = self.llm_engine.model_config.io_processor_plugin
        self.io_processor = get_io_processor(self.llm_engine.vllm_config,
                                             io_processor_plugin)

    def load_states(self):
        self.collective_rpc(method="load_states")

    def offload_states(self, level=1):
        self.reset_prefix_cache()
        self.collective_rpc(method="offload_states", args=(level,))

    def fetch_output(self):
        # simulating non blocking semantic when using v1 engine
        if envs.VLLM_USE_V1:
            try:
                request_outputs = self.llm_engine.step_nowait()
            except queue.Empty:
                request_outputs = []
        else:
            request_outputs = self.llm_engine.step()

        return request_outputs

    def get_num_waiting(self):
        stats = self.llm_engine._get_stats(scheduler_outputs=None)
        return stats.num_waiting_sys

    def add_requests(
        self,
        prompt_token_ids: List[List[int]],
        request_ids: List[int] | None,
        sampling_params: SamplingParams,
        multi_modal_data: List[int] | None,
        lora_requests: List[LoRARequest] | None,
    ):
        assert len(prompt_token_ids) == len(request_ids)
        if multi_modal_data:
            assert len(multi_modal_data) == len(request_ids)
        for i, (token_ids, request_id)in enumerate(zip(prompt_token_ids, request_ids)):
            if request_id is None:
                request_id = next(self.request_counter)
            lora_request = lora_requests[i] if lora_requests is not None else None
            if multi_modal_data:
                # in v1, input_preprocessor is in engine.processor
                processor = getattr(self.llm_engine, "processor", None)
                input_preprocessor = processor.input_preprocessor if processor else self.llm_engine.input_preprocessor
                preprocessed_inputs = input_preprocessor.preprocess(
                    prompt={"prompt_token_ids": token_ids, "multi_modal_data": multi_modal_data[i]},
                    lora_request=lora_request,
                )
                # in v1, engine does not use a input_processor
                processed_inputs = (
                    self.llm_engine.input_processor(preprocessed_inputs)
                    if hasattr(self.llm_engine, "input_processor")
                    else preprocessed_inputs
                )
            else:
                processed_inputs = {
                    "type": "token",
                    "prompt_token_ids": token_ids
                }
            self.llm_engine._add_processed_request(
                request_id=request_id,
                processed_inputs=processed_inputs,
                params=sampling_params,
                arrival_time=time.time(),
                lora_request=lora_request,
            )

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        self.llm_engine.abort_request(request_id)

    def clear_unfinished_requests(self):
        self._run_engine(use_tqdm=True)

    # 参数同步接口
    def setup_collective_group(self, *args, **kwargs):
        self.collective_rpc(method="setup_collective_group", args=args, kwargs=kwargs)

    def broadcast_bucket(self, src_pp_rank, meta_infos, bucket_size):
        if envs.VLLM_USE_V1:
            SendBucketManager.meta_to_dict(meta_infos)
        self.collective_rpc(method="broadcast_bucket", args=(src_pp_rank, meta_infos, bucket_size))

    def broadcast_parameter(self, *args, **kwargs):
        self.collective_rpc(method="broadcast_parameter", args=args, kwargs=kwargs)

    def update_parameter(self, parameter_name, weight, ranks_in_worker, is_lora):
        if envs.VLLM_USE_V1:
            weight_dict = {
                "dtype": weight.dtype,
                "weight": weight.cpu().tolist()
            }
        self.collective_rpc(method="update_parameter", args=(parameter_name, weight_dict, ranks_in_worker, is_lora))

    def update_parameter_in_bucket(self, meta_infos, buffer, ranks_in_worker):
        if envs.VLLM_USE_V1:
            SendBucketManager.meta_to_dict(meta_infos)
            # vllm 084 does not support serialization of torch.Tensor(GPU), must use custom
            # numpy array encoder or use pickle.
            # Can not convert to numpy array here, because of bug in encoder/decoder of vllm 084.
            # Newer version of vllm support efficient serilization of torch.Tensor.
            buffer = buffer.cpu().tolist()
        self.collective_rpc(method="update_parameter_in_bucket", args=(meta_infos, buffer, ranks_in_worker))

    def add_lora(self, *args, **kwargs):
        self.collective_rpc(method="add_lora", args=args, kwargs=kwargs)
