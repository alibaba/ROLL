import os
from typing import Iterable, Tuple, List, Dict, Type, Optional, Union, Any
import time 
import torch
import cloudpickle
from vllm import LLM, SamplingParams, EngineArgs, LLMEngine, envs
from vllm.config import CompilationConfig, PoolerConfig, TaskOption, HfOverrides
from vllm.core.scheduler import Scheduler
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter

from roll.third_party.vllm.vllm_0_7_3.llm_engine import LLMEngine073


class Llm073(LLM):

    def __init__(self, resource_placement_groups: List[Dict],
                 model: str,
                 tokenizer: Optional[str] = None,
                 tokenizer_mode: str = "auto",
                 skip_tokenizer_init: bool = False,
                 trust_remote_code: bool = False,
                 allowed_local_media_path: str = "",
                 tensor_parallel_size: int = 1,
                 dtype: str = "auto",
                 quantization: Optional[str] = None,
                 revision: Optional[str] = None,
                 tokenizer_revision: Optional[str] = None,
                 seed: int = 0,
                 gpu_memory_utilization: float = 0.9,
                 swap_space: float = 4,
                 cpu_offload_gb: float = 0,
                 enforce_eager: Optional[bool] = None,
                 max_seq_len_to_capture: int = 8192,
                 disable_custom_all_reduce: bool = False,
                 disable_async_output_proc: bool = False,
                 hf_overrides: Optional[HfOverrides] = None,
                 mm_processor_kwargs: Optional[Dict[str, Any]] = None,
                 # After positional args are removed, move this right below `model`
                 task: TaskOption = "auto",
                 override_pooler_config: Optional[PoolerConfig] = None,
                 compilation_config: Optional[Union[int, Dict[str, Any]]] = None, **kwargs,):

        # setup envs for vllm
        # https://github.com/vllm-project/vllm/pull/14189/files
        # TODO do not override other options in PYTORCH_CUDA_ALLOC_CONF
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
        # torch.cuda may already init, explicitly disable expandable_segments
        # here (only matters when VLLM_USE_RAY_SPMD_WORKER=0)
        torch.cuda.memory._set_allocator_settings("expandable_segments:False")

        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True

        if "worker_cls" in kwargs:
            worker_cls = kwargs["worker_cls"]
            # if the worker_cls is not qualified string name,
            # we serialize it using cloudpickle to avoid pickling issues
            if isinstance(worker_cls, type):
                kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)

        if compilation_config is not None:
            if isinstance(compilation_config, (int, dict)):
                compilation_config_instance = CompilationConfig.from_cli(
                    str(compilation_config))
            else:
                compilation_config_instance = compilation_config
        else:
            compilation_config_instance = None

        kwargs["enable_sleep_mode"] = True
        engine_args = EngineArgs(
            model=model,
            task=task,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            allowed_local_media_path=allowed_local_media_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            disable_async_output_proc=disable_async_output_proc,
            hf_overrides=hf_overrides,
            mm_processor_kwargs=mm_processor_kwargs,
            override_pooler_config=override_pooler_config,
            compilation_config=compilation_config_instance,
            **kwargs,
        )
        engine_args.resource_placement_groups = resource_placement_groups
        # Logic to switch between engines is done at runtime instead of import
        # to avoid import order issues
        self.engine_class = self.get_engine_class()
        self.llm_engine = self.engine_class.from_engine_args(
            engine_args, usage_context=UsageContext.LLM_CLASS)

        self.request_counter = Counter()

    @staticmethod
    def get_engine_class() -> Type[LLMEngine]:
        if envs.VLLM_USE_V1:
            # Lazy import: the v1 package isn't distributed
            from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine
            return V1LLMEngine  # type: ignore
        return LLMEngine073

    def load_states(self):
        self.collective_rpc(method="load_states")

    def offload_states(self, level=2):
        self.reset_prefix_cache()
        self.collective_rpc(method="offload_states")

    def fetch_output(self):
        output_list = []
        request_outputs = self.llm_engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                output_list.append(request_output)
        return output_list

    def add_requests(self, prompt_token_ids: List[List[int]], request_ids: List[int] | None, sampling_params: SamplingParams):
        assert len(prompt_token_ids) == len(request_ids)
        for token_ids, request_id in zip(prompt_token_ids, request_ids):
            if request_id is None:
                request_id = next(self.request_counter)
            self.llm_engine._add_processed_request(request_id=request_id,
                                                   processed_inputs={"type": "token", "prompt_token_ids": token_ids},
                                                   params=sampling_params,
                                                   arrival_time=time.time(),
                                                   lora_request=None,
                                                   prompt_adapter_request=None
                                                )

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        self.llm_engine.abort_request(request_id)

    def clear_unfinished_requests(self):
        self._run_engine(use_tqdm=True)

    # Parameter synchronization interface
    def setup_collective_group(self, *args, **kwargs):
        self.collective_rpc(method="setup_collective_group", args=args, kwargs=kwargs)

    def broadcast_bucket(self, *args, **kwargs):
        self.collective_rpc(method="broadcast_bucket", args=args, kwargs=kwargs)

    def broadcast_parameter(self, *args, **kwargs):
        self.collective_rpc(method="broadcast_parameter", args=args, kwargs=kwargs)

    def update_parameter(self, *args, **kwargs):
        self.collective_rpc(method="update_parameter", args=args, kwargs=kwargs)

    def update_parameter_in_bucket(self, *args, **kwargs):
        self.collective_rpc(method="update_parameter_in_bucket", args=args, kwargs=kwargs)
