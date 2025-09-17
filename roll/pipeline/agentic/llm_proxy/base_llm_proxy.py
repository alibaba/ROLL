from abc import abstractmethod, ABC
from typing import List, Dict, Optional, Any

from transformers import PreTrainedTokenizer

import gem
from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import LLMProxyConfig


class BaseLLMProxy(ABC):
    """
    LLMProxy defines a unified interface for generating responses based on messages or lm_input DataProto.
    Subclasses will implement specific inference apis.
    """
    def __init__(self, generate_scheduler: RequestScheduler, llm_proxy_config: LLMProxyConfig, tokenizer: PreTrainedTokenizer, env: gem.Env):
        """

        """
        self.generate_scheduler = generate_scheduler
        self.llm_proxy_config = llm_proxy_config
        self.tokenizer = tokenizer
        self.env = env

    @abstractmethod
    def generate(self,
                 messages: List[Dict[str, str]],
                 lm_input: DataProto,
                 generation_config: Dict[str, Any]) -> DataProto:
        """
        Generates a response based on the provided conversation messages and model input.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries representing the conversation history,
                                             e.g., `[{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi!"}]`.
            lm_input (DataProto): Input data structure containing tokenized prompts and other tensor inputs for the model.
            generation_config (Dict[str, Any]): configuration to override default generation parameters.

        Returns:
            DataProto: The output data protocol containing generated sequences and associated metadata.
                        The batch contains 'responses' key with returned token_ids.
        """
        pass


LLM_PROXY_REGISTRY = {}


def register_llm_proxy(name):
    def register_class(cls):
        LLM_PROXY_REGISTRY[name] = cls
        return cls
    return register_class