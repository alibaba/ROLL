from typing import List, Dict, Any, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from roll.pipeline.agentic.llm_proxy import BaseLLMProxy, register_llm_proxy
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.router import RouterManager, RouterClient, is_report_data_finished
from roll.utils.functionals import (
    postprocess_generate,
    concatenate_input_and_output,
)


@register_llm_proxy("policy")
class PolicyProxy(BaseLLMProxy):
    """
    A proxy for policy model that invokes the policy model's engine (e.g. vllm/sglang) to perform generation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router_client: RouterClient = RouterManager.create_client_sync(self.generate_scheduler)

    def generate(self,
                 messages: List[Dict[str, str]],
                 lm_input: DataProto,
                 generation_config: Dict[str, Any]) -> DataProto:

        lm_input.meta_info["generation_config"] = generation_config
        lm_input.meta_info["pad_to_seq_len"] = False
        src_rank = lm_input.meta_info.pop("src_rank")
        response_data: Optional[DataProto] = self.router_client.generate_request_sync(req=lm_input, request_id=None, uid=src_rank)

        if response_data is None or not is_report_data_finished(response_data):
            return None

        # postprocess_generate, input_ids, attention_mask, left pad
        eos_token_id = response_data.meta_info["eos_token_id"]
        pad_token_id = response_data.meta_info["pad_token_id"]
        output_token_ids = response_data.meta_info["output_token_ids"]
        output_tokens = [torch.tensor(token_ids) for token_ids in output_token_ids]

        output_logprobs = response_data.meta_info.get("output_logprobs", None)

        output_tensor = pad_sequence(output_tokens, batch_first=True, padding_value=pad_token_id)
        output_tensor = concatenate_input_and_output(
            input_ids=lm_input.batch["input_ids"], output_ids=output_tensor, num_return_sequences=len(output_tokens)
        )
        lm_output: DataProto = postprocess_generate(
            prompts=lm_input,
            output=output_tensor,
            num_return_sequences=len(output_tokens),
            sequence_length=output_tensor.shape[-1],
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            pad_to_seq_len=lm_input.meta_info.get("pad_to_seq_len", True),
            output_logprobs=output_logprobs,
        )
        request_repeat = lm_input.repeat(repeat_times=len(output_tokens))
        lm_output.non_tensor_batch = request_repeat.non_tensor_batch
        lm_output.meta_info = request_repeat.meta_info
        lm_output.meta_info.pop("generation_config", None)
        return lm_output
