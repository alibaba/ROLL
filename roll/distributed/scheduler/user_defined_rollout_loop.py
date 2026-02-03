import asyncio
import copy
import math
from typing import List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from roll.distributed.scheduler.generate_scheduler import (
    RolloutContext,
    expand_requests,
    is_report_data_finished,
)
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.distributed.scheduler.protocol import DataProto
from roll.utils.functionals import (
    postprocess_generate,
    concatenate_input_and_output,
)
from roll.utils.logging import get_logger


logger = get_logger()

# ================= helper functions =================

def query_filter(data_list: List[DataProto], config: RLVRConfig) -> bool:
    """
    各domain的过滤规则可以自定义
    """
    response_level_rewards = [data.batch["response_level_rewards"] for data in data_list]
    if len(response_level_rewards) == 1:
        return True
    rewards = torch.cat(response_level_rewards, dim=0)

    domain = data_list[0].non_tensor_batch["domain"][0]
    query_filter_config = config.rewards[domain].query_filter_config

    if query_filter_config.type == "no_filter":
        return True
    elif query_filter_config.type == "mean_filter":
        threshold_up = query_filter_config.filter_args.get("threshold_up", math.inf)
        threshold_down = query_filter_config.filter_args.get("threshold_down", -1)
        if torch.mean(rewards) <= threshold_down or torch.mean(rewards) >= threshold_up:
            return False
    elif query_filter_config.type == "std_filter":
        std_threshold = query_filter_config.filter_args.get("std_threshold", -1)
        if torch.std(rewards) <= std_threshold:
            return False
    return True

def response_filter(data_item, config):
    return True

def postprocess_paused_data(pre_data, data: DataProto, sequence_length, prompt_length) -> DataProto:
    if "output_token_ids" not in data.meta_info:  # abort without inferred a token
        # too many this log means need more infer workers
        logger.info(f"received data without output_token_ids, request_id: {data.meta_info['request_id']}")
        return pre_data
    logger.debug(f"received paused data, request_id: {data.meta_info['request_id']}")

    assert len(data.meta_info["output_token_ids"]) == 1, (
        "async pipeline only support num_return_sequences=1 or is_num_return_sequences_expand=True"
    )

    # value: list[list[int|float]]
    for key in ["output_token_ids", "output_logprobs"]:
        cur_value = data.meta_info.pop(key)
        pre_value = pre_data.meta_info.get(f"pre_{key}", [[]] * len(cur_value))
        assert len(pre_value) == len(cur_value)
        pre_value = [pre_value[i] + cur_value[i] for i in range(len(pre_value))]
        data.meta_info[f"pre_{key}"] = pre_value
    new_batch = {**pre_data.batch}

    init_attention_mask = pre_data.batch.get("init_attention_mask", pre_data.batch["attention_mask"])
    new_batch["init_attention_mask"] = init_attention_mask
    new_batch["init_input_ids"] = pre_data.batch.get("init_input_ids", pre_data.batch["input_ids"])

    # concat pre output_ids and input_ids
    new_input_ids = concatenate_input_and_output(
        input_ids=new_batch["init_input_ids"],
        output_ids=torch.LongTensor(data.meta_info["pre_output_token_ids"]),
        num_return_sequences=len(data.meta_info["pre_output_token_ids"]),
    )
    new_batch["input_ids"] = new_input_ids

    new_attention_mask = torch.ones_like(new_input_ids, dtype=init_attention_mask.dtype)
    new_attention_mask[:, :init_attention_mask.shape[1]] = init_attention_mask
    new_batch["attention_mask"] = new_attention_mask

    max_new_tokens = sequence_length - new_input_ids.shape[1]
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens {max_new_tokens} <= 0, init_input_ids {new_batch['init_input_ids'].shape}, "
        f"pre_output_token_ids {len(data.meta_info['pre_output_token_ids'][0])}")
    data.meta_info["max_new_tokens"] = max_new_tokens
    new_non_tensor_batch = dict(
        [(k, v.repeat(len(data.meta_info["pre_output_token_ids"]))) for k, v in pre_data.non_tensor_batch.items()]
    )  # repeat num_return_sequences=1
    if "multi_modal_data" in pre_data.non_tensor_batch:
        for i, (mm_data, prompt_token_ids) in enumerate(
            zip(new_non_tensor_batch["multi_modal_data"], data.meta_info["pre_output_token_ids"])
        ):
            # use new dict to replace repeated reference
            mm_data = new_non_tensor_batch["multi_modal_data"][i] = dict(mm_data)
            # VLM uses prompt_ids (without replaced image tokens) in multi_modal_data
            prompt_token_ids = mm_data["prompt_token_ids"] + prompt_token_ids
            mm_data.update({"prompt_token_ids": prompt_token_ids})
    data = DataProto.from_dict(
        new_batch, non_tensors=new_non_tensor_batch, meta_info={**pre_data.meta_info, **data.meta_info}
    )
    assert data.batch["init_attention_mask"].shape[1] == prompt_length
    assert data.batch["init_input_ids"].shape[1] == prompt_length
    return data

def postprocess_output_data(request, data: DataProto, sequence_length) -> DataProto:
    # postprocess_generate, input_ids, attention_mask, left pad
    eos_token_id = data.meta_info["eos_token_id"]
    pad_token_id = data.meta_info["pad_token_id"]
    input_ids = request.batch.pop("init_input_ids", request.batch["input_ids"])
    request.batch["input_ids"] = input_ids
    request.batch["attention_mask"] = request.batch.pop("init_attention_mask", request.batch["attention_mask"])
    output_token_ids = data.meta_info["output_token_ids"]
    pre_output_token_ids = request.meta_info.pop("pre_output_token_ids", [[]] * len(output_token_ids))
    output_token_ids = [pre_output_token_ids[i] + output_token_ids[i] for i in range(len(pre_output_token_ids))]

    output_logprobs = data.meta_info.get("output_logprobs", None)
    if output_logprobs is not None:
        pre_output_logprobs = request.meta_info.get("pre_output_logprobs", [[]] * len(output_token_ids))
        output_logprobs = [pre_output_logprobs[i] + output_logprobs[i] for i in range(len(pre_output_logprobs))]

    output_tokens = [torch.tensor(token_ids) for token_ids in output_token_ids]
    output_tensor = pad_sequence(output_tokens, batch_first=True, padding_value=pad_token_id)
    output_tensor = concatenate_input_and_output(
        input_ids=input_ids, output_ids=output_tensor, num_return_sequences=len(output_tokens)
    )
    output: DataProto = postprocess_generate(
        prompts=request,
        output=output_tensor,
        num_return_sequences=len(output_tokens),
        sequence_length=sequence_length,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        output_logprobs=output_logprobs,
    )
    request_repeat = request.repeat(repeat_times=len(output_tokens))
    output.non_tensor_batch = request_repeat.non_tensor_batch
    output.meta_info = request_repeat.meta_info
    return output

# ================= example of user defined rollout loop =================

class UserDefinedRolloutLoop:
    """
    Default user defined rollout loop.

    User should write there own udrl class with an async function name process_new_prompt
    with signature (self, context: RolloutContext) -> Optional[DataProto|List[DataProto]].

    RolloutContext hide almost all the implementation details of DynamicSamplingScheduler, LoadBalancer,
    ReplayBuffer, and sync/async training.

    A typical process_new_prompt has few steps:
        1. get and filter dataset
        2: spawn tasks to process requests, including generate, reward, and response level filter
        3. prompt level filter
        4. return responses to commit to ReplayBuffer

    To abort this prompt(or dataset), just return None at any where.

    Exception safe:
        The framework will only raise asyncio.CancelledError exception. (process_new_prompt will be called by scheduler
        as an asyncio.Task and scheduler may cancel this task if needed. User should not suppress asyncio.CancelledError
        exception and should handle clean up by themself.)
    
        User should catch all other exceptions, any other exceptions will be treat as sys.exit by framework.
    """
    def __init__(self):
        pass

    async def process_new_prompt(self, context: RolloutContext) -> Optional[DataProto|List[DataProto]]:
        num_return_sequences = context.meta_info["generation_config"]["num_return_sequences"]
        # TODO user can control whether to expand requests at prompt level
        is_num_return_sequences_expand = context.is_num_return_sequences_expand

        ################# STEP 1: get and filter dataset
        # TODO shigao dataset这一层应该暴露哪些部分(是collect前还是后面的数据呢)，需要用户自定义collect_fn吗
        request_data, domain = context.get_request_data(meta_info=context.meta_info)
        request_data_list = expand_requests(data=request_data, num_return_sequences=num_return_sequences,
                            is_num_return_sequences_expand=is_num_return_sequences_expand)
        # TODO data filter

        ################# STEP 2: spawn tasks to process requests, including generate, reward, and filter at response level
        # Must run inside RolloutContext.do_generate_and_reward context.
        # RolloutContext.do_generate_and_reward will wait until can send new request (controlled by LoadBalancer).
        # And at exit, RolloutContext will enforce there is no running requests.
        async with context.do_generate_and_reward(max_concurrency=num_return_sequences):
            responses_list: List[List[DataProto]] = await asyncio.gather(
                *[self._generate_and_reward(context=context, req=req, domain=domain) for req in request_data_list]
            )
            responses: List[DataProto] = [item for sublist in responses_list for item in sublist]
            # User can call RolloutContext.abort_running_requests to abort any running generate requests (generate will return a response
            # with finish_reason=="abort", user should distinguish this from partial rollout to avoid dead loop).
        # assert there is no running requests outside do_generate_and_reward context.

        ################# STEP 3: prompt level filter
        if not context.is_val and not query_filter(responses, context.pipeline_config):
            # TODO add metrics (query_filter_count)
            logger.debug(f"prompt_id {context.prompt_id} is filtered")
            return

        ################# STEP 4: return responses to commit to ReplayBuffer
        return responses

    async def _generate_and_reward(
        self,
        context: RolloutContext,
        req: DataProto,
        domain: str,
    ):
        responses: List[DataProto] = []

        for _ in range(5): # limit max retry times, otherwise may cause dead loop
            original_req = copy.deepcopy(req)

            # TODO deprecate collect_unfinished after sglang support partial rollout
            collect_unfinished = req.meta_info.get("collect_unfinished", False)

            # TODO: multi-turn rollout
            while True:
                # TODO: user defined request preprocessor

                data = await context.generate(req=req, domain=domain)

                # TODO: user defined response postprocessor

                # Scheduler may abort request in async training. Should resend partial output
                # to support partial rollout.
                if is_report_data_finished(data):
                    req = postprocess_output_data(req, data, context.sequence_length)
                    break
                else:
                    if not collect_unfinished:
                        # return None to abort this prompt
                        return
                    else:
                        req = postprocess_paused_data(req, data, context.sequence_length, context.prompt_length)

            rewards = await context.compute_rewards(req=req, domain=domain)
            req.union(rewards)

            output_count = req.batch.batch_size[0]
            assert output_count == req.meta_info["generation_config"]["num_return_sequences"]
            batch_expanded = [req[[idx]] for idx in range(output_count)]

            response_filter_count = 0
            for batch_item in batch_expanded:
                if context.is_val or response_filter(batch_item, context.pipeline_config):
                    responses.append(batch_item)
                else:
                    # TODO add metrics (response_filter_count)
                    response_filter_count += 1

            if response_filter_count == 0:
                break
            else:
                # retry if filter out some responses
                original_req.meta_info["generation_config"]["num_return_sequences"] = response_filter_count
                req = original_req

        return responses
