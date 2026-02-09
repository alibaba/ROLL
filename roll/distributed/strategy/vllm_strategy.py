import asyncio
import copy
import gc
import os
from collections import deque
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from transformers import set_seed
from vllm import RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import RequestOutputKind, BeamSearchParams
from vllm.inputs.data import TokensPrompt
from vllm.utils import random_uuid

from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto, list_of_dict_to_dict_of_list
from roll.distributed.strategy.strategy import InferenceStrategy
from roll.third_party.vllm import create_async_llm
from roll.utils.functionals import concatenate_input_and_output, reduce_metrics
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType
from roll.platforms import current_platform


logger = get_logger()


class VllmStrategy(InferenceStrategy):
    strategy_name = "vllm"

    def __init__(self, worker: Worker):
        super().__init__(worker)

        # Metrics snapshot infrastructure
        self._metrics_snapshots = deque(maxlen=3600)
        self._metrics_snapshot_interval = 1.0  # Snapshot every 1 second
        self._metrics_task = None

    async def initialize(self, model_provider):
        set_seed(seed=self.worker.pipeline_config.seed)
        vllm_config = copy.deepcopy(self.worker_config.strategy_args.strategy_config)
        # Must explicitly set VLLM_USE_V1 to pass this check: https://github.com/vllm-project/vllm/pull/14972
        os.environ["VLLM_USE_V1"] = str(vllm_config.pop("VLLM_USE_V1", 1))
        self.sleep_level = vllm_config.pop("sleep_level", 1)

        data_parallel_size = vllm_config.get("data_parallel_size", 1)
        if data_parallel_size > 1:
            logger.info(
                f"VllmStrategy {self.worker.cluster_name} enable data parallel {data_parallel_size=} data_parallel_rank={self.worker.rank}"
                f" data_parallel_address={os.environ['MASTER_ADDR']} data_parallel_rpc_port={os.environ['MASTER_PORT']}"
            )
            assert data_parallel_size == self.worker.world_size, f"{data_parallel_size=} != {self.worker.world_size=}"
            vllm_config.update(
                {
                    "data_parallel_rank": self.worker.rank, # set data_parallel_rank to use external load balancing
                    "data_parallel_address": os.environ["MASTER_ADDR"],
                    "data_parallel_rpc_port": os.environ["MASTER_PORT"],
                }
            )

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
                "max_num_batched_tokens": vllm_config.get("max_num_batched_tokens", 8192), # use default value of LLM class usage context
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

        # Can not set VLLM_PORT explicitly in DP. Each call of get_engine_client_zmq_addr in
        # DPCoordinator will return the same port, which will cause port conflict.
        # https://github.com/vllm-project/vllm/blob/releases/v0.10.0/vllm/v1/engine/coordinator.py#L72
        if not data_parallel_size > 1:
            # set VLLM_PORT to avoid port conflict applied by vllm
            vllm_port = self.worker.get_free_port()
            os.environ["VLLM_PORT"] = str(vllm_port)

        self.model = await create_async_llm(resource_placement_groups=self.worker_config.resource_placement_groups, **vllm_config)

        self.tokenizer = await self.model.get_tokenizer()
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

        try:
            from vllm.v1.metrics.reader import get_metrics_snapshot
            self._metrics_task = asyncio.create_task(self._collect_metrics_snapshot())
        except Exception as e:
            logger.warning(f"Failed to create metrics collector task: {e}")

    def op_compute_log_probs(self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        vllm实现compute log probs在这里实现即可
        """
        pass

    async def generate(self, batch: DataProto, generation_config) -> torch.Tensor:
        # Check if beam search is requested
        if self._should_use_beam_search(generation_config):
            return await self._generate_with_beam_search(batch, generation_config)
        else:
            return await self._generate_standard(batch, generation_config)

    def _should_use_beam_search(self, generation_config) -> bool:
        """Check if beam search should be used based on generation_config."""
        return generation_config.get("num_beams", 1) > 1 or generation_config.get("use_beam_search", False)

    async def _generate_standard(self, batch: DataProto, generation_config: Dict) -> torch.Tensor:
        """Standard generate method for non-beam search cases."""
        sampling_params = create_sampling_params_for_vllm(gen_kwargs=generation_config)

        input_ids = batch.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = batch.batch["attention_mask"]  # left-padded attention_mask

        if "multi_modal_data" in batch.non_tensor_batch:
            prompts = [TokensPrompt(data) for data in batch.non_tensor_batch["multi_modal_data"]]
        else:
            prompts = [TokensPrompt(prompt_token_ids=prompt)
                for prompt in gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)
            ]

        lora_request = None
        if self.is_lora:
            lora_int_ids = list(await self.model.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_request = LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="dummy_lora_path")

        async def _generate(prompt):
            request_id = random_uuid()
            result_generator = self.model.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
            )
            output: Optional[RequestOutput] = None
            async for result in result_generator:
                output = result
            return output

        vllm_outputs = await asyncio.gather(*[_generate(prompt) for prompt in prompts])

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

    async def _generate_with_beam_search(self, batch: DataProto, generation_config: Dict) -> torch.Tensor:
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
        async def _beam_search(prompt):
            request_id = random_uuid()
            result_generator = self.model.beam_search(
                prompt=prompt,
                request_id=request_id,
                params=beam_params,
            )
            output: Optional[RequestOutput] = None
            async for result in result_generator:
                output = result
            return output

        beam_search_outputs = await asyncio.gather(*[_beam_search(prompt) for prompt in prompts])

        generated_token_ids = []
        for request_output in beam_search_outputs:
            for completion_output in request_output.outputs:
                generated_tokens = completion_output.token_ids
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

    async def generate_request(self, data: DataProto):
        collect_unfinished = data.meta_info.get("collect_unfinished", False)
        input_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        request_id = data.meta_info["request_id"]
        generation_config = data.meta_info.get("generation_config")
        max_new_tokens = data.meta_info.get("max_new_tokens", generation_config["max_new_tokens"])
        max_new_tokens = min(max_new_tokens, generation_config["max_new_tokens"])
        output_kind = RequestOutputKind.CUMULATIVE if collect_unfinished else RequestOutputKind.FINAL_ONLY
        sampling_params = create_sampling_params_for_vllm(
            gen_kwargs={**generation_config, "max_new_tokens": max_new_tokens, "output_kind": output_kind}
        )
        assert sampling_params.n == 1 or not collect_unfinished, "collect_unfinished is not supported in parallel sampling"
        if "multi_modal_data" in data.non_tensor_batch:
            assert len(data.non_tensor_batch["multi_modal_data"]) == 1
            prompt_token_ids = data.non_tensor_batch["multi_modal_data"][0]["prompt_token_ids"]
            multi_modal_data = (data.non_tensor_batch["multi_modal_data"][0]["multi_modal_data"]
                                if "multi_modal_data" in data.non_tensor_batch["multi_modal_data"][0] else None)
            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids, multi_modal_data=multi_modal_data)
        else:
            assert input_ids.size(0) == 1, f"data['input_ids'] must have exactly one batch dimension"
            prompt_token_ids = gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)
            assert len(prompt_token_ids) == 1
            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids[0])
        lora_request = None
        if self.is_lora:
            lora_int_ids = list(await self.model.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_request = LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="dummy_lora_path")

        result_generator = self.model.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
        )
        output: Optional[RequestOutput] = None
        # vLLM support partial rollout in v1 from 0.10.1, and will return finished output
        # with finish_reason setted no matter what RequestOutputKind is.
        # For compatibility, the following except block are only for v0 and older version of v1.
        try:
            async for result in result_generator:
                output = result
        except asyncio.CancelledError:
            if output is None:
                output_data = DataProto(meta_info=data.meta_info)
                output_data.meta_info["finish_reasons"] = ["abort"]
                return output_data

        output_token_ids, finish_reasons, logprobs = [], [], []
        for completion_output in output.outputs:
            output_token_ids.append(completion_output.token_ids)
            # For compatibility, older version may return unfinished result, set finish_reason of those to 'abort'.
            finish_reason = "abort" if completion_output.finish_reason is None else completion_output.finish_reason
            finish_reasons.append(finish_reason)
            if completion_output.logprobs is not None:
                logprobs.append(
                    [
                        float(lps[token_id].logprob)
                        for token_id, lps in zip(completion_output.token_ids, completion_output.logprobs)
                    ]
                )
        output_data = DataProto(meta_info=data.meta_info)
        output_data.meta_info["output_token_ids"] = output_token_ids
        output_data.meta_info["finish_reasons"] = finish_reasons
        output_data.meta_info["output_logprobs"] = logprobs
        return output_data

    async def abort_requests(self, request_ids):
        for id in request_ids:
            await self.model.abort(request_id=id)

    # offload/reload 接口
    async def load_states(self, *args, **kwargs):
        await self.model.reset_prefix_cache()
        if not self.is_model_in_gpu:
            await self.model.load_states()
            self.is_model_in_gpu = True

    async def offload_states(self, include=None, non_blocking=False):
        await self.model.reset_prefix_cache()
        if include is None or OffloadStateType.model_params in include:
            if self.is_model_in_gpu and self.worker.pipeline_config.is_actor_infer_colocated:
                await self.model.offload_states(self.sleep_level)
                self.is_model_in_gpu = False
        gc.collect()
        current_platform.empty_cache()
    
    def process_weights_after_loading(self,*args, **kwargs):
        self.model.process_weights_after_loading()

    # 参数同步相关接口
    async def setup_collective_group(self, master_address, master_port, rank_offset, world_size, group_name, backend=None):
        logger.info(f"setup_collective_group {group_name=}")
        backend = backend if backend is not None else current_platform.communication_backend
        await self.model.setup_collective_group(master_address, master_port, rank_offset, world_size, group_name, backend)

    async def broadcast_parameter(self, names, dtypes, shapes, group_name, is_lora=False):
        await self.model.broadcast_parameter(names, dtypes, shapes, group_name, is_lora)

    async def update_parameter_in_bucket(self, serialized_named_tensors, is_lora=False):
        await self.model.update_parameter_in_bucket(serialized_named_tensors, is_lora)

    async def add_lora(self, peft_config):
        peft_config["target_modules"] = set(self.worker_config.model_args.lora_target)
        await self.model.add_lora(peft_config)

    async def _collect_metrics_snapshot(self):
        """Collect metrics snapshots periodically in a background thread."""
        from vllm.v1.metrics.reader import get_metrics_snapshot
        while True:
            raw_metrics = get_metrics_snapshot()
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

            await asyncio.sleep(self._metrics_snapshot_interval)

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
    # TODO vllm 0.10.2 support partial rollout, and do not need to set RequestOutputKind to CUMULATIVE
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
