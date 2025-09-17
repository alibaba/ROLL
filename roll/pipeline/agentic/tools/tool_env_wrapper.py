from typing import Dict, List, Optional, Tuple, Any, SupportsFloat

from gem import Env
from gem.tools.tool_env_wrapper import ToolEnvWrapper as GEMToolEnvWrapper

from roll.pipeline.agentic.tools.registration import make_tool

class ToolEnvWrapper(GEMToolEnvWrapper):
    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        observation, info = super().reset(seed=seed)
        metrics = {
            "tool_use_counter": info.pop("tool_use_counter"),
            "tool_success_counter": info.pop("tool_success_counter"),
        }
        metrics_agg_mode = {
            "tool_use_counter": "last",
            "tool_success_counter": "last",
        }
        metrics.update(info.pop("metrics", {}))
        metrics_agg_mode.update(info.pop("metrics_agg_mode", {}))
        info["metrics"] = metrics
        info["metrics_agg_mode"] = metrics_agg_mode
        return observation, info

    def step(
            self,
            action: str,
            verbose: bool = False,
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action, verbose)
        metrics = {
            "tool_use_counter": info.pop("tool_use_counter"),
            "tool_success_counter": info.pop("tool_success_counter"),
        }
        metrics_agg_mode = {
            "tool_use_counter": "last",
            "tool_success_counter": "last",
        }
        metrics.update(info.pop("metrics", {}))
        metrics_agg_mode.update(info.pop("metrics_agg_mode", {}))
        info["metrics"] = metrics
        info["metrics_agg_mode"] = metrics_agg_mode
        return observation, reward, terminated, truncated, info


def tool_wrapper(env: Env, wrapper_args: Dict, tool_configs: List[Dict]):
    tools = []

    for tool_config in tool_configs:
        tool_id = tool_config["tool_id"]
        tool_args = tool_config["tool_args"]
        tools.append(make_tool(tool_id=tool_id, **tool_args))

    tool_env_wrapper = ToolEnvWrapper(env, tools, **wrapper_args)
    return tool_env_wrapper

