from typing import List, Dict, Any

import numpy as np

from roll.pipeline.agentic.llm_proxy import BaseLLMProxy, register_llm_proxy
from roll.distributed.scheduler.protocol import DataProto


@register_llm_proxy("random")
class RandomProxy(BaseLLMProxy):
    def generate(self,
                 messages: List[Dict[str, str]],
                 lm_input: DataProto,
                 generation_config: Dict[str, Any]) -> Any:

        response_text = f"{self.env.sample_random_action()}"
        responses = self.tokenizer([response_text], return_tensors="pt")
        lm_input.batch["responses"] = responses["input_ids"]
        lm_input.non_tensor_batch["response_text"] = np.array([response_text], dtype=object)

        return lm_input
