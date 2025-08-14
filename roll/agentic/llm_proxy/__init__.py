from typing import Dict, List

from transformers import PreTrainedTokenizer

import gem
from roll.agentic.llm_proxy.base_llm_proxy import BaseLLMProxy, LLM_PROXY_REGISTRY, register_llm_proxy
from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from roll.pipeline.agentic.agentic_config import LLMProxyConfig
from roll.agentic.llm_proxy.random_proxy import RandomProxy
from roll.agentic.llm_proxy.openai_proxy import OpenAIProxy
from roll.agentic.llm_proxy.policy_proxy import PolicyProxy

def create_llm_proxy(
        generate_scheduler: RequestScheduler,
        llm_proxy_config: LLMProxyConfig,
        tokenizer: PreTrainedTokenizer,
        env: gem.Env) -> BaseLLMProxy:
    proxy_type = llm_proxy_config.proxy_type
    if proxy_type in LLM_PROXY_REGISTRY:
        cls = LLM_PROXY_REGISTRY[proxy_type]
        return cls(generate_scheduler, llm_proxy_config, tokenizer, env)
    else:
        raise ValueError(f"Unknown proxy type: {proxy_type}")

