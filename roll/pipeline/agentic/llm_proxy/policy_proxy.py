from typing import List, Dict, Any

import ray

from roll.pipeline.agentic.llm_proxy import BaseLLMProxy, register_llm_proxy
from roll.distributed.scheduler.protocol import DataProto


@register_llm_proxy("policy")
class PolicyProxy(BaseLLMProxy):
    """
    A proxy for policy model that invokes the policy model's engine (e.g. vllm/sglang) to perform generation.
    """

    def generate(self,
                 messages: List[Dict[str, str]],
                 lm_input: DataProto,
                 generation_config: Dict[str, Any]) -> DataProto:

        lm_input.meta_info["generation_config"] = generation_config
        lm_input.meta_info['response_callback_fn'] = self.generate_scheduler.report_response.remote
        lm_input.meta_info["pad_to_seq_len"] = False
        lm_output: DataProto = ray.get(self.generate_scheduler.generate_one_request.remote(data=lm_input))

        if lm_output is not None:
            lm_output.meta_info.pop("generation_config", None)

        return lm_output
