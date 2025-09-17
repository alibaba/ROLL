import gem
import time
from typing import List, Dict, Any, Optional

import numpy as np
from openai import OpenAI, OpenAIError
from transformers import PreTrainedTokenizer

from roll.pipeline.agentic.llm_proxy import BaseLLMProxy, register_llm_proxy
from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import LLMProxyConfig
from roll.utils.logging import get_logger

logger = get_logger()


@register_llm_proxy("openai")
class OpenAIProxy(BaseLLMProxy):
    """
    A proxy class that uses the OpenAI API to perform text generation.
    It encapsulates the OpenAI client and handles API calls, retries,
    and mapping to the BaseLLMProxy interface.
    """

    def __init__(self,
                 generate_scheduler: RequestScheduler,
                 llm_proxy_config: LLMProxyConfig,
                 tokenizer: PreTrainedTokenizer,
                 env: gem.Env):
        """
        Initializes the OpenAIProxy with the given configuration.

        Args:
            generate_scheduler (RequestScheduler): Scheduler for managing requests.
            llm_proxy_config (LLMProxyConfig): Configuration specific to the LLM proxy (e.g., API key, base URL).
            tokenizer (PreTrainedTokenizer): Tokenizer for the model.
            env (gem.Env): sample_random_action (if applicable).
        """
        super().__init__(generate_scheduler, llm_proxy_config, tokenizer, env)

        self.base_url = llm_proxy_config.proxy_config["base_url"]
        self.api_key = llm_proxy_config.proxy_config["api_key"]
        self.model_name = llm_proxy_config.proxy_config["model_name"]
        self.timeout = llm_proxy_config.proxy_config.get("timeout", 60)
        self.max_retries = llm_proxy_config.proxy_config.get("max_retries", 3)
        self.retry_delay = llm_proxy_config.proxy_config.get("retry_delay", 2)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        logger.info(f"OpenAIProxy initialized: base_url={self.base_url}, model_name={self.model_name}")

    def generate(self,
                 messages: List[Dict[str, str]],
                 lm_input: DataProto,
                 generation_config: Dict[str, Any]) -> Optional[DataProto]:
        """
        Generates a response using the OpenAI API.
        Args:
            messages (List[Dict[str, str]]): Conversation history.
            lm_input (DataProto): Input data protocol (not directly used by this proxy).
            generation_config (Dict[str, Any]): Dictionary of generation parameters.
                                                Supports: model, temperature, max_tokens, top_p, stream,
                                                presence_penalty, top_k, enable_thinking.

        Returns:
            DataProto: The generated response and metadata.
        """
        model_name = generation_config.get("model_name", self.model_name)

        # GeneratingArguments to OpenAI args
        temperature = generation_config.get("temperature", 0.7)
        max_tokens = generation_config.get("max_new_tokens", 8192)
        top_p = generation_config.get("top_p", 0.8)
        top_k = generation_config.get("top_k", None) # Default to None, only add if specified
        # presence_penalty = generation_config.get("repetition_penalty", 0.0) # OpenAI default is 0.0

        enable_thinking = generation_config.get("enable_thinking", False)

        extra_body = {}
        if top_k is not None:
            extra_body["top_k"] = top_k
        if enable_thinking:
            # According to the example, enable_thinking goes under extend_fields
            extra_body.setdefault("extend_fields", {})["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

        attempt = 0
        while attempt < self.max_retries:
            try:
                logger.debug(f"Attempt {attempt + 1}/{self.max_retries}: Calling OpenAI API for model '{model_name}'...")
                logger.debug(f"Messages: {messages[0] if messages else 'No messages'}, Config: {generation_config}")

                completion = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    # presence_penalty=presence_penalty,
                    # Pass extra_body only if it's not empty
                    extra_body=extra_body if extra_body else None
                )

                response_text = completion.choices[0].message.content
                responses = self.tokenizer([response_text], return_tensors="pt")
                lm_input.batch["responses"] = responses["input_ids"]
                lm_input.non_tensor_batch["response_text"] = np.array([response_text], dtype=object)
                return lm_input

            except OpenAIError as e:
                # Catch specific OpenAI API errors
                attempt += 1
                error_msg = f"OpenAI API error (Attempt {attempt}/{self.max_retries}): {e}"
                logger.error(error_msg)
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay + attempt * 0.5) # Simple exponential backoff
                else:
                    return None
            except Exception as e:
                attempt += 1
                error_msg = f"Unexpected error during OpenAI API call (Attempt {attempt}/{self.max_retries}): {e}"
                logger.error(error_msg)
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay + attempt * 0.5)
                else:
                    return None

        # Fallback if somehow loop exits without returning (shouldn't happen with proper retry logic)
        final_error_msg = f"Failed to generate response after {self.max_retries} attempts."
        logger.critical(final_error_msg)
        return None

