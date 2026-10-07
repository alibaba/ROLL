from __future__ import annotations

import asyncio
from typing import List, Optional

from vllm import RequestOutput, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.utils import random_uuid

from roll.distributed.scheduler.protocol import DataProto
from roll.utils.functionals import gather_unpadded_input_ids


async def compute_prompt_logprobs_with_vllm_strategy(
    strategy,
    batch: DataProto,
    max_tokens: int = 1,
) -> List[List[Optional[float]]]:
    sampling_params = SamplingParams(
        max_tokens=max(1, int(max_tokens)),
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        stop_token_ids=[strategy.tokenizer.eos_token_id],
        repetition_penalty=1.0,
        n=1,
        stop=None,
        logprobs=0,
        include_stop_str_in_output=False,
        prompt_logprobs=0,
    )

    input_ids = batch.batch["input_ids"]
    attention_mask = batch.batch["attention_mask"]
    unpadded_prompts = gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in unpadded_prompts]

    lora_request = None
    if strategy.is_lora:
        lora_int_ids = list(await strategy.model.list_loras())
        if len(lora_int_ids) > 0:
            lora_int_id = lora_int_ids[0]
            lora_request = LoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="dummy_lora_path",
            )

    async def _score(prompt, prompt_token_ids):
        request_id = random_uuid()
        result_generator = strategy.model.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
        )
        output: Optional[RequestOutput] = None
        async for result in result_generator:
            output = result
        return output, prompt_token_ids

    results = await asyncio.gather(*[_score(p, ids) for p, ids in zip(prompts, unpadded_prompts)])

    all_prompt_logprobs: List[List[Optional[float]]] = []
    for output, prompt_token_ids in results:
        raw = getattr(output, "prompt_logprobs", None) if output is not None else None
        if raw is None:
            all_prompt_logprobs.append([None] * len(prompt_token_ids))
            continue
        row: List[Optional[float]] = []
        for pos, entry in enumerate(raw):
            if entry is None:
                row.append(None)
                continue
            token_id = prompt_token_ids[pos]
            value = entry.get(token_id)
            row.append(getattr(value, "logprob", None) if value is not None else None)
        all_prompt_logprobs.append(row)
    return all_prompt_logprobs
