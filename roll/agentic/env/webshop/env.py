import random
import string
from typing import Optional, Union

from roll.agentic.env.parse_action_utils import default_parser_action_func
from webshop_minimal import WebAgentTextEnv

from roll.agentic.env.base import BaseEnv
from roll.agentic.env.webshop.config import WebShopEnvConfig
from roll.agentic.utils import all_seed


class WebShopEnv(BaseEnv, WebAgentTextEnv):
    def __init__(self, config: Optional[WebShopEnvConfig] = None, **kwargs: any) -> None:
        """
        Adapter for WebAgentTextEnv to conform to the BaseLanguageBasedEnv interface.
        """

        # 未修改API
        raise NotImplementedError

        BaseEnv.__init__(self, config=config)
        self.config = config or WebShopEnvConfig()
        self.observation_mode = self.config.observation_mode
        self.file_path = self.config.file_path
        self.server = self.config.server
        self.filter_goals = self.config.filter_goals
        self.limit_goals = self.config.limit_goals
        self.num_products = self.config.num_products
        self.human_goals = self.config.human_goals
        self.show_attrs = self.config.show_attrs
        self.render_cache = None

        WebAgentTextEnv.__init__(
            self,
            observation_mode=self.observation_mode,
            file_path=self.file_path,
            server=self.server,
            filter_goals=self.filter_goals,
            limit_goals=self.limit_goals,
            num_products=self.num_products,
            human_goals=self.human_goals,
            show_attrs=self.show_attrs,
            **kwargs,
        )
        self.step_count = 0

    def reset(
        self, seed=None, session: Optional[Union[str, int]] = None, instruction_text: Optional[str] = None
    ) -> any:
        self.step_count = 0
        if session is None:
            with all_seed(seed):
                session = "".join(random.choices(string.ascii_lowercase, k=10))
        obs, _ = WebAgentTextEnv.reset(self, session=session, instruction_text=instruction_text)
        self.prepare_render_cache(WebAgentTextEnv.get_instruction_text(self))
        self.prepare_render_cache(obs)
        return self.render(), {}

    def step(self, action):
        action_info = self.parse_action(action)
        if action_info["action"] is None:
            metrics = {
                "action_is_effective": False,
                "action_is_valid": False,
                "success": False,
            }
            info = {
                "metrics": metrics,
            }
            info.update(action_info)
            return self.render(), 0, False, False, info

        state, reward, done, info = WebAgentTextEnv.step(self, action_info["action"])
        self.prepare_render_cache(self.observation)
        metrics = {
            "action_is_effective": tuple(self.get_available_actions())
            == ("click[back to search]", "click[< prev]", "click[next >]"),
            "action_is_valid": True,
            "success": done,
        }
        info = {
            "metrics": metrics,
        }
        info.update(action_info)
        obs_with_actions = self._attach_actions(state)
        self.step_count += 1
        terminated, truncated = done, False
        if terminated:
            if not metrics["success"] and self.step_count >= self.config.max_steps:
                truncated = True
        return obs_with_actions, reward, terminated, truncated, info

    def _attach_actions(self, observation: str) -> str:
        actions = ", ".join(self.get_available_actions())
        return observation + "\n" + "Available actions: " + actions

    def parse_action(self, text):
        return default_parser_action_func(text, self.config.action_pattern, None, None)

    def render(self, mode=None):
        """
        Render the environment.
        """
        return self.render_cache

    def close(self):
        """
        Close the environment.
        """
        WebAgentTextEnv.close(self)

    def prepare_render_cache(self, observation: str):
        """
        Prepare the render cache for the environment.
        """
        available_actions = self.get_available_actions()
        self.render_cache = observation + "\n" + "Available actions: " + ", ".join(available_actions)

    def get_available_actions(self):
        """
        Parse the available actions in the environment to a list of strings.
        """
        orig_available_actions = WebAgentTextEnv.get_available_actions(self)
        available_actions = []

        if orig_available_actions["has_search_bar"]:
            available_actions.append("search[<content>]")

        for clickable in orig_available_actions["clickables"]:
            if clickable != "search":
                available_actions.append(f"click[{clickable}]")
        # TODO: we may need to purge the case when available_actions == ['click[back to search]', 'click[< prev]', 'click[next >]']
        return available_actions


if __name__ == "__main__":
    env = WebShopEnv()
    print(env.reset())
    while True:
        print(env.observation)
        print(f"Available actions: {env.get_available_actions()}")
        action = input("Enter action: ")
        if action == "q":
            break
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
    env.close()
