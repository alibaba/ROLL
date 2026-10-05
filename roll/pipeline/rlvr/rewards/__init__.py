from typing import TYPE_CHECKING, Any

from roll.pipeline.rlvr.rewards.code_sandbox_reward_worker import CodeSandboxRewardWorker
from roll.pipeline.rlvr.rewards.crossthinkqa_rule_reward_worker import CrossThinkQARuleRewardWorker
from roll.pipeline.rlvr.rewards.general_val_rule_reward_worker import GeneralValRuleRewardWorker
from roll.pipeline.rlvr.rewards.gsm8k_math_reward_worker import GSM8KMathRewardWorker
from roll.pipeline.rlvr.rewards.ifeval_rule_reward_worker import GeneralRuleRewardWorker
from roll.pipeline.rlvr.rewards.llm_judge_reward_worker import LLMJudgeRewardWorker
from roll.pipeline.rlvr.rewards.math_rule_reward_worker import MathRuleRewardWorker
from roll.pipeline.rlvr.rewards.remote_reward_system_worker import RemoteRewardSystemWorker

if TYPE_CHECKING:
    from roll.pipeline.rlvr.rewards.geo3k_reward_worker import Geo3kRewardWorker

# Workers whose third-party scoring dependencies are not part of the base install.
# Importing them eagerly here would make the whole rewards package unimportable,
# which breaks every RLVR pipeline at worker pre-import even when no job asks for
# the worker. Resolve them on first attribute access instead so the dependency is
# only required by the runs that actually use the worker.
_LAZY_WORKERS = {"Geo3kRewardWorker": "roll.pipeline.rlvr.rewards.geo3k_reward_worker"}

__all__ = [
    "CodeSandboxRewardWorker",
    "CrossThinkQARuleRewardWorker",
    "GSM8KMathRewardWorker",
    "GeneralRuleRewardWorker",
    "GeneralValRuleRewardWorker",
    "Geo3kRewardWorker",
    "LLMJudgeRewardWorker",
    "MathRuleRewardWorker",
    "RemoteRewardSystemWorker",
]


def __getattr__(name: str) -> Any:
    """Import a lazily exported reward worker on first access (PEP 562)."""
    module_path = _LAZY_WORKERS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list:
    return sorted(__all__)
