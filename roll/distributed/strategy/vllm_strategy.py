import asyncio
import copy
import gc
import os
import queue
import threading
import time
from collections import defaultdict, deque
from concurrent import futures
from typing import Dict, List, Optional, Union
from packaging.version import Version

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from transformers import set_seed
import vllm
from vllm import RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import RequestOutputKind, BeamSearchParams
from vllm.utils import random_uuid

from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto, list_of_dict_to_dict_of_list
from roll.distributed.strategy.strategy import InferenceStrategy
from roll.third_party.vllm import LLM, AsyncLLM
from roll.utils.functionals import GenerateRequestType, concatenate_input_and_output, reduce_metrics
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType
from roll.platforms import current_platform


logger = get_logger()


class VllmStrategy(InferenceStrategy):
    strategy_name = "vllm"

    def __init__(self, worker: Worker):
        super().__init__(worker)
        self.model: Union[LLM, AsyncLLM]
        self.executor: futures.ThreadPoolExecutor = futures.ThreadPoolExecutor(max_workers=1)
        self.pending_size = 1
        self.command_queue: Optional[queue.Queue] = None

        self.request_metas = {}
        self.running = False
        
        # Metrics snapshot infrastructure
        self._metrics_snapshots = deque(maxlen=3600)
        self._metrics_snapshot_interval = 1.0  # Snapshot every 1 second
        self._metrics_thread = None

    def initialize(self, model_provider):
        set_seed(seed=self.worker.pipeline_config.seed)
        vllm_config = copy.deepcopy(self.worker_config.strategy_args.strategy_config)
        engine_mode = vllm_config.pop("engine_mode", "sync")  # sync/async
        self.pending_size = vllm_config.pop("pending_size", 1)
        self.sleep_level = vllm_config.pop("sleep_level", 1)
        self.command_queue = queue.Queue()

        if self.worker_config.model_args.dtype == "fp32":
            dtype = "float32"
        elif self.worker_config.model_args.dtype == "fp16":
            dtype = "float16"
        elif self.worker_config.model_args.dtype == "bf16":
            dtype = "bfloat16"
        else:
            dtype = "auto"
        vllm_config.update(
            {
                "model": self.worker_config.model_args.model_name_or_path,
                "dtype": dtype,
                "enforce_eager": vllm_config.get("enforce_eager", False),
                "trust_remote_code": True,
                "seed": self.worker.pipeline_config.seed,
                "disable_custom_all_reduce": vllm_config.get(
                    "disable_custom_all_reduce", True
                ),  # potentially hangs in tp>1
                "enable_prefix_caching": vllm_config.get("enable_prefix_caching", True),
                "load_format": vllm_config.get("load_format", "dummy"),  # use model update passed value
            }
        )

        self.is_lora = self.worker_config.model_args.lora_target is not None
        if self.is_lora:
            lora_kwargs = {
                "enable_lora": True,
                "max_loras": 1,
                "max_lora_rank": self.worker_config.model_args.lora_rank,
            }
            vllm_config.update(lora_kwargs)
            vllm_config["load_format"] = "auto"  # enables vLLM to load the base model for add_lora

        logger.info(f"vllm_config: {vllm_config}")
        assert not dist.is_initialized()

        # set VLLM_PORT to avoid port conflict applied by vllm
        vllm_port = self.worker.get_free_port()
        os.environ["VLLM_PORT"] = str(vllm_port)

        if engine_mode == "sync":
            self.model = LLM(resource_placement_groups=self.worker_config.resource_placement_groups, **vllm_config)
            self.tokenizer = self.model.get_tokenizer()
        else:
            self.model = AsyncLLM(
                resource_placement_groups=self.worker_config.resource_placement_groups, **vllm_config
            )
            loop = asyncio.get_event_loop()
            self.tokenizer = loop.run_until_complete(self.model.get_tokenizer())
        additional_special_tokens = self.tokenizer.additional_special_tokens
        special_tokens = [
            add_token
            for add_token in self.tokenizer.added_tokens_decoder.values()
            if add_token.special and add_token.content not in additional_special_tokens
        ]
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}, replace_additional_special_tokens=False
        )
        logger.info(f"add {special_tokens} to additional_special_tokens: {self.tokenizer.additional_special_tokens}")

        self.worker.rank_info.dp_rank = self.worker.rank
        self.worker.rank_info.dp_size = self.worker.world_size

        self.is_model_in_gpu = True

        self._metrics_thread = threading.Thread(
            target=self._collect_metrics_snapshot,
            name="metrics-collection"
        )
        self._metrics_thread.start()

    def op_compute_log_probs(self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        vllm实现compute log probs在这里实现即可
        """
        pass

    def generate(self, batch: DataProto, generation_config) -> torch.Tensor:
        # Check if beam search is requested
        if self._should_use_beam_search(generation_config):
            return self._generate_with_beam_search(batch, generation_config)
        else:
            return self._generate_standard(batch, generation_config)

    def _should_use_beam_search(self, generation_config) -> bool:
        """Check if beam search should be used based on generation_config."""
        return generation_config.get("num_beams", 1) > 1 or generation_config.get("use_beam_search", False)

    def _generate_standard(self, batch: DataProto, generation_config) -> torch.Tensor:
        """Standard generate method for non-beam search cases."""
        sampling_params = create_sampling_params_for_vllm(gen_kwargs=generation_config)

        input_ids = batch.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = batch.batch["attention_mask"]  # left-padded attention_mask

        vllm_input_args = {}
        if "multi_modal_data" in batch.non_tensor_batch:
            vllm_input_args["prompts"] = batch.non_tensor_batch["multi_modal_data"]
        else:
            if Version(vllm.__version__) >= Version("0.11.0"):
                from vllm.inputs import TokensPrompt
                prompt_token_ids_list=gather_unpadded_input_ids(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                vllm_input_args["prompts"] = [TokensPrompt(prompt_token_ids=prompt_token_ids)for prompt_token_ids in prompt_token_ids_list]
            else:
                vllm_input_args["prompt_token_ids"] = gather_unpadded_input_ids(
                    input_ids=input_ids, attention_mask=attention_mask
                )

        lora_requests = None
        if self.is_lora:
            batch_size = len(input_ids)
            lora_int_ids = list(self.model.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(
                        lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="dummy_lora_path"
                    )
                ] * batch_size

        vllm_outputs = self.model.generate(
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=lora_requests,
            **vllm_input_args,
        )

        # (bs * num_return_sequences, max_response_len)
        output_ids = gather_outputs_to_pad_tensor(
            request_outputs=vllm_outputs,
            pad_token_id=self.tokenizer.pad_token_id,
            device=input_ids.device,
        )

        # (bs * num_return_sequences, input_len + max_response_len)
        output = concatenate_input_and_output(
            input_ids=input_ids, output_ids=output_ids, num_return_sequences=sampling_params.n
        )

        return output

    def _generate_with_beam_search(self, batch: DataProto, generation_config) -> torch.Tensor:
        """Generate using beam search method."""
        # Create beam search parameters
        beam_params = BeamSearchParams(
            beam_width=generation_config.get("num_beams", 1),
            max_tokens=generation_config.get("max_new_tokens", 50),
            temperature=generation_config.get("temperature", 0.0),
            ignore_eos=generation_config.get("ignore_eos", False),
            length_penalty=generation_config.get("length_penalty", 1.0),
            include_stop_str_in_output=generation_config.get("include_stop_str_in_output", False),
        )

        input_ids = batch.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = batch.batch["attention_mask"]  # left-padded attention_mask

        # Prepare prompts for beam_search
        if "multi_modal_data" in batch.non_tensor_batch:
            # For multimodal data, we need to handle it differently
            # This is a simplified approach - may need refinement based on actual multimodal format
            prompts = batch.non_tensor_batch["multi_modal_data"]
        else:
            # Convert to token lists format expected by beam_search
            token_lists = gather_unpadded_input_ids(
                input_ids=input_ids, attention_mask=attention_mask
            )
            # Convert to TokensPrompt format expected by vLLM beam_search
            prompts = [{"prompt_token_ids": token_ids} for token_ids in token_lists]

        # Call beam_search method
        beam_search_outputs = self.model.beam_search(
            prompts=prompts,
            params=beam_params,
        )

        generated_token_ids = []
        token_ids = [prompt['prompt_token_ids'] for prompt in prompts]
        for batch_idx, output in enumerate(beam_search_outputs):
            # Each output contains beam_width sequences
            for beam_idx, sequence in enumerate(output.sequences):
                # Get prompt length for this input
                prompt_length = len(token_ids[batch_idx])
                # Extract only the generated tokens (exclude prompt)
                generated_tokens = sequence.tokens[prompt_length:]
                generated_token_ids.append(torch.tensor(generated_tokens, device=input_ids.device))

        # Pad the sequences
        output_ids = pad_sequence(generated_token_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        # Concatenate input and output
        output = concatenate_input_and_output(
            input_ids=input_ids,
            output_ids=output_ids,
            num_return_sequences=beam_params.beam_width
        )

        return output

    def process_vllm_output(self, vllm_outputs: List[RequestOutput], request_complete_callback, collect_unfinished=False):
        # 转成response id, request_complete_callback
        report_request_ids = []
        for request_output in vllm_outputs:
            if not (request_output.finished or collect_unfinished):
                continue
            request_id = request_output.request_id
            meta_info = self.request_metas.pop(request_id, None)
            if meta_info is None:
                continue
            output_token_ids, finish_reasons, logprobs = [], [], []
            for completion_output in request_output.outputs:
                output_token_ids.append(completion_output.token_ids)
                finish_reasons.append(completion_output.finish_reason)
                if completion_output.logprobs is not None:
                    logprobs.append(
                        [
                            float(lps[token_id].logprob)
                            for token_id, lps in zip(completion_output.token_ids, completion_output.logprobs)
                        ]
                    )
            output_data = DataProto(meta_info=meta_info)
            output_data.meta_info["output_token_ids"] = output_token_ids
            output_data.meta_info["finish_reasons"] = finish_reasons
            output_data.meta_info["output_logprobs"] = logprobs
            request_complete_callback(data=output_data)
            report_request_ids.append(request_id)
        return report_request_ids

    def start_server(self, data: DataProto, request_complete_callback):
        self.command_queue = queue.Queue()
        self.running = True
        collect_unfinished = data.meta_info.get("collect_unfinished", False)

        while True:
            while not self.command_queue.empty():
                command, batch = self.command_queue.get_nowait()
                if command == GenerateRequestType.ADD:
                    input_ids = batch.batch["input_ids"]
                    attention_mask = batch.batch["attention_mask"]
                    request_id = batch.meta_info["request_id"]
                    self.request_metas[request_id] = batch.meta_info
                    generation_config = batch.meta_info.get("generation_config")
                    max_new_tokens = batch.meta_info.get("max_new_tokens", generation_config["max_new_tokens"])
                    max_new_tokens = min(max_new_tokens, generation_config["max_new_tokens"])
                    output_kind = RequestOutputKind.CUMULATIVE if collect_unfinished else RequestOutputKind.FINAL_ONLY
                    sampling_params = create_sampling_params_for_vllm(
                        gen_kwargs={**generation_config, "max_new_tokens": max_new_tokens, "output_kind": output_kind}
                    )
                    if "multi_modal_data" in batch.non_tensor_batch:
                        prompt_token_ids = [
                            batch.non_tensor_batch["multi_modal_data"][0]
                            ["prompt_token_ids"]
                        ]
                        multi_modal_data = (
                            [batch.non_tensor_batch["multi_modal_data"][0]["multi_modal_data"]]
                            if "multi_modal_data" in batch.non_tensor_batch["multi_modal_data"][0]
                            else None
                        )
                    else:
                        prompt_token_ids = gather_unpadded_input_ids(
                            input_ids=input_ids, attention_mask=attention_mask
                        )
                        multi_modal_data = None
                    lora_requests = None
                    if self.is_lora:
                        batch_size = len(prompt_token_ids)
                        lora_int_ids = list(self.model.llm_engine.list_loras())
                        if len(lora_int_ids) > 0:
                            lora_int_id = lora_int_ids[0]
                            lora_requests = [
                                LoRARequest(
                                    lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="dummy_lora_path"
                                )
                            ] * batch_size
                    self.model.add_requests(
                        request_ids=[request_id],
                        prompt_token_ids=prompt_token_ids,
                        sampling_params=sampling_params,
                        multi_modal_data=multi_modal_data,
                        lora_requests=lora_requests,
                    )
                elif command == GenerateRequestType.ABORT:
                    request_id = batch.meta_info["request_id"]
                    self.model.abort_request(request_id=request_id)
                elif command == GenerateRequestType.STOP:
                    stop_time = time.time()
                    wait_seconds = 120
                    while collect_unfinished and len(self.request_metas) > 0:  # for partial rollout
                        vllm_outputs: List[RequestOutput] = self.model.fetch_output()
                        processed_request_ids = self.process_vllm_output(
                            vllm_outputs=vllm_outputs,
                            request_complete_callback=request_complete_callback,
                            collect_unfinished=collect_unfinished,
                        )
                        if time.time() - stop_time > wait_seconds:
                            logger.warning(f"Timeout after {wait_seconds}s waiting for running requests to complete. "
                                           f"Remaining running requests: {len(self.request_metas)}")
                            break
                        self.model.abort_request(request_id=processed_request_ids)
                    self.model.abort_request(request_id=list(self.request_metas.keys()))
                    self.request_metas.clear()
                    while not self.command_queue.empty():
                        self.command_queue.get_nowait()
                    # Run llm_engine again to consume all out standing requests and
                    # stop model execute loop, otherwise collective_rpc will stuck by
                    # model execute loop or there will be garbage output at next step.
                    self.model.clear_unfinished_requests()
                    self.running = False
                    return

            vllm_outputs: List[RequestOutput] = self.model.fetch_output()
            self.process_vllm_output(vllm_outputs=vllm_outputs, request_complete_callback=request_complete_callback)

    def add_request(self, command, data: DataProto):
        self.command_queue.put((command, data))

    async def async_generate(self, batch: DataProto, generation_config: Dict) -> torch.Tensor:
        # TODO: refactor async_generate interface. not supported now!
        raise NotImplementedError()
        from vllm.inputs.data import TokensPrompt

        sampling_params = create_sampling_params_for_vllm(gen_kwargs=generation_config)

        input_ids = batch.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = batch.batch["attention_mask"]  # left-padded attention_mask
        assert input_ids.size(0) == 1, f"async_generate: batch['input_ids'] must have exactly one batch dimension"

        prompt_token_ids = gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)

        # TODO meaningful request id?
        #   async_generate如何实现abort_request
        request_id = random_uuid()
        result_generator = self.model.generate(
            prompt=TokensPrompt(prompt_token_ids=prompt_token_ids[0]),
            sampling_params=sampling_params,
            request_id=request_id,
        )
        vllm_output: Optional[RequestOutput] = None
        async for request_output in result_generator:
            vllm_output = request_output
        assert vllm_output is not None

        # (bs * num_return_sequences, max_response_len)
        output_ids = gather_outputs_to_pad_tensor(
            request_outputs=[vllm_output], pad_token_id=self.tokenizer.pad_token_id, device=input_ids.device
        )
        # (bs * num_return_sequences, input_len + max_response_len)
        output = concatenate_input_and_output(
            input_ids=input_ids, output_ids=output_ids, num_return_sequences=sampling_params.n
        )
        return output

    # offload/reload 接口
    def load_states(self, *args, **kwargs):
        self.model.reset_prefix_cache()
        if not self.is_model_in_gpu:
            self.model.load_states()
            self.is_model_in_gpu = True

    def offload_states(self, include=None, non_blocking=False):
        if include is None or OffloadStateType.model_params in include:
            if self.is_model_in_gpu and self.worker.pipeline_config.is_train_infer_colocated:
                self.model.offload_states(self.sleep_level)
                self.is_model_in_gpu = False
        gc.collect()
        current_platform.empty_cache()

    # 参数同步相关接口
    def setup_collective_group(self, model_update_name, comm_plan, backend=None):
        if backend is None:
            backend = current_platform.communication_backend
        self.model.setup_collective_group(comm_plan=comm_plan, backend=backend, rank_in_cluster=self.worker.rank)

    def broadcast_parameter(self, model_update_name, src_pp_rank, dtype, shape, parameter_name, is_lora=False):
        self.model.broadcast_parameter(src_pp_rank, dtype, shape, parameter_name, is_lora)

    def broadcast_bucket(self, model_update_name, src_pp_rank, meta_infos, bucket_size):
        self.model.broadcast_bucket(src_pp_rank, meta_infos, bucket_size)

    def update_parameter(self, model_update_name, parameter_name, weight, ranks_in_worker, is_lora=False):
        self.model.update_parameter(parameter_name, weight, ranks_in_worker, is_lora)

    def update_parameter_in_bucket(self, model_update_name, meta_infos, buffer, ranks_in_worker):
        self.model.update_parameter_in_bucket(meta_infos, buffer, ranks_in_worker)

    def add_lora(self, peft_config):
        self.model.add_lora(peft_config)

    def _collect_metrics_snapshot(self):
        """Collect metrics snapshots periodically in a background thread."""
        try:
            while True:
                raw_metrics = self.model.get_metrics()
                snapshot = {
                    'vllm/kv_cache_usage_perc_max': [],
                    'vllm/num_requests_waiting_max': [],
                    'vllm/num_preemptions_max': []
                }
                for metric in raw_metrics:
                    if metric.name == "vllm:kv_cache_usage_perc":
                        snapshot['vllm/kv_cache_usage_perc_max'].append(metric.value)
                    elif metric.name == "vllm:num_requests_waiting":
                        snapshot['vllm/num_requests_waiting_max'].append(metric.value)
                    elif metric.name == "vllm:num_preemptions":
                        snapshot['vllm/num_preemptions_max'].append(metric.value)
                self._metrics_snapshots.append(snapshot)

                time.sleep(self._metrics_snapshot_interval)
        except Exception as e:
            logger.warning(f"Failed to get metrics: {e}")

    def get_metrics(self, metric_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get aggregated metrics for the time interval since last call.

        Args:
            metric_names: Optional list of specific metric names to filter

        Returns:
            Dictionary of metric names to aggregated values
        """
        if not self._metrics_snapshots:
            return {}
        metrics_snapshots = list_of_dict_to_dict_of_list(self._metrics_snapshots)
        self._metrics_snapshots.clear()
        return reduce_metrics(metrics_snapshots)

def gather_unpadded_input_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    gathered_input_ids = [ids[mask.bool()].tolist() for ids, mask in zip(input_ids, attention_mask)]
    return gathered_input_ids


def gather_outputs_to_pad_tensor(request_outputs: List["RequestOutput"], pad_token_id, device=None) -> torch.Tensor:
    if device is None:
        device = current_platform.device_type
    token_ids_list_of_lists = [
        torch.tensor(completion_output.token_ids, device=device)
        for request_output in request_outputs
        for completion_output in request_output.outputs
    ]
    output_tensor = pad_sequence(token_ids_list_of_lists, batch_first=True, padding_value=pad_token_id)
    return output_tensor


def create_sampling_params_for_vllm(gen_kwargs):
    output_kind = gen_kwargs.get("output_kind", RequestOutputKind.FINAL_ONLY)
    if output_kind != RequestOutputKind.FINAL_ONLY:
        assert gen_kwargs["num_return_sequences"] == 1, (
            "fetch_output only supports num_return_sequences=1 or output_kind=FINAL"
        )
    return SamplingParams(
        max_tokens=gen_kwargs["max_new_tokens"],
        temperature=gen_kwargs["temperature"],
        top_p=gen_kwargs["top_p"],
        top_k=gen_kwargs["top_k"],
        stop_token_ids=gen_kwargs["eos_token_id"],
        repetition_penalty=gen_kwargs["repetition_penalty"],
        n=gen_kwargs["num_return_sequences"],
        stop=gen_kwargs["stop_strings"],
        logprobs=gen_kwargs.get("logprobs", 0),
        output_kind=output_kind,
        include_stop_str_in_output=gen_kwargs.get("include_stop_str_in_output", True),
    )
