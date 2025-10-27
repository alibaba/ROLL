import vllm
from packaging.version import Version

from roll.utils.logging import get_logger

logger = get_logger()

LLM = None
AsyncLLM = None

if Version("0.8.4") == Version(vllm.__version__):
    from roll.third_party.vllm.vllm_0_8_4.llm import Llm084
    from roll.third_party.vllm.vllm_0_8_4.v1.async_llm import AsyncLLM084
    LLM = Llm084
    AsyncLLM = AsyncLLM084
elif Version("0.10.0") <= Version(vllm.__version__) < Version("0.10.2"):
    from roll.third_party.vllm.vllm_0_10_0.llm import Llm0100
    from roll.third_party.vllm.vllm_0_10_0.v1.async_llm import AsyncLLM0100
    LLM = Llm0100
    AsyncLLM = AsyncLLM0100
elif Version("0.10.2") == Version(vllm.__version__):
    from roll.third_party.vllm.vllm_0_10_2.llm import Llm0102
    LLM = Llm0102
elif Version("0.11.0") == Version(vllm.__version__):
    from roll.third_party.vllm.vllm_0_11_0.llm import Llm0110
    LLM = Llm0110
else:
    raise NotImplementedError(f"roll vllm version {vllm.__version__} is not supported.")

__all__ = ["LLM", "AsyncLLM"]
