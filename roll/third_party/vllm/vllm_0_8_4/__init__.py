# Patch CustomAsyncLLM.generate and OutputProcessor.abort_requests
# (more on tests.third_party.vllm.test_vllm_local.test_vllm_abort)
from typing import Optional
from collections.abc import AsyncGenerator, Mapping, Iterable
import asyncio

from vllm.inputs import PromptType
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.output_processor import OutputProcessor

from roll.third_party.vllm.async_llm import CustomAsyncLLM

async def generate(
    self,
    prompt: PromptType,
    sampling_params: SamplingParams,
    request_id: str,
    lora_request: Optional[LoRARequest] = None,
    trace_headers: Optional[Mapping[str, str]] = None,
    prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    priority: int = 0,
) -> AsyncGenerator[RequestOutput, None]:
    try:
        if self.output_handler is None:
            self.output_handler = asyncio.create_task(
                self._run_output_handler())

        q = await self.add_request(
            request_id,
            prompt,
            sampling_params,
            lora_request=lora_request,
            trace_headers=trace_headers,
            prompt_adapter_request=prompt_adapter_request,
            priority=priority,
        )

        finished = False
        while not finished:
            out = q.get_nowait() or await q.get()

            if isinstance(out, BaseException) or (isinstance(out, type) and issubclass(out, BaseException)):
                # raise asyncio.CancelledError, will not cause dead recursive
                raise out

            finished = out.finished
            yield out

    except asyncio.CancelledError:
        await self.abort(request_id)
        raise
CustomAsyncLLM.generate = generate

def abort_requests(
    self,
    request_ids: Iterable[str],
) -> list[str]:
    request_ids_to_abort = []
    for request_id in request_ids:
        req_state = self.request_states.pop(request_id, None)
        if req_state is not None:
            self.lora_states.abort_request(req_state)
            request_ids_to_abort.append(request_id)
            req_state.queue.put(asyncio.CancelledError) # wakeup generate coroutine with asyncio.CancelledError
        else:
            parent = self.parent_requests.pop(request_id, None)
            if parent and parent.child_requests:
                self.abort_requests(parent.child_requests)
                request_ids_to_abort.extend(parent.child_requests)
    return request_ids_to_abort
OutputProcessor.abort_requests = abort_requests


# patch qwen3 fp8
# https://github.com/vllm-project/vllm/issues/17327
# https://github.com/vllm-project/vllm/pull/17318
from vllm.model_executor.layers.linear import QKVParallelLinear
from typing import Optional
import torch
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           BlockQuantScaleParameter,
                                           PerTensorScaleParameter,
                                           RowvLLMParameter)
def weight_loader_v2(self,
                     param: BasevLLMParameter,
                     loaded_weight: torch.Tensor,
                     loaded_shard_id: Optional[str] = None):
    if loaded_shard_id is None:  # special case for certain models
        if isinstance(param, PerTensorScaleParameter):
            param.load_qkv_weight(loaded_weight=loaded_weight, shard_id=0)
            return
        elif type(param) in (RowvLLMParameter, BasevLLMParameter):
            param.load_qkv_weight(loaded_weight=loaded_weight)
            return
        # TODO: @dsikka - move to parameter.py
        self._load_fused_module_from_checkpoint(param, loaded_weight)
        return

    assert loaded_shard_id in ["q", "k", "v"]

    shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
    shard_size = self._get_shard_size_mapping(loaded_shard_id)

    # Note(simon): This is needed for Qwen3's fp8 quantization.
    if isinstance(param, BlockQuantScaleParameter):
        assert self.quant_method is not None
        assert hasattr(self.quant_method, "quant_config")
        weight_block_size = self.quant_method.quant_config.weight_block_size
        block_n, _ = weight_block_size[0], weight_block_size[1]
        shard_offset = (shard_offset + block_n - 1) // block_n
        shard_size = (shard_size + block_n - 1) // block_n

    param.load_qkv_weight(loaded_weight=loaded_weight,
                          num_heads=self.num_kv_head_replicas,
                          shard_id=loaded_shard_id,
                          shard_offset=shard_offset,
                          shard_size=shard_size)
QKVParallelLinear.weight_loader_v2 = weight_loader_v2


__all__ = []
