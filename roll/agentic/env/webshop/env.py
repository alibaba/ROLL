# Copyright (c) 2025, ALIBABA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from roll.agentic.env.base import BaseLanguageBasedEnv
from roll.agentic.env.webshop.config import WebShopEnvConfig
from webshop_minimal import WebAgentTextEnv
from typing import Optional, Union
from roll.agentic.utils import all_seed
import random
import string


class WebShopEnv(BaseLanguageBasedEnv, WebAgentTextEnv):
    def __init__(self, config: Optional[WebShopEnvConfig] = None, **kwargs: any) -> None:
        """
        Adapter for WebAgentTextEnv to conform to the BaseLanguageBasedEnv interface.
        """
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

        BaseLanguageBasedEnv.__init__(self)
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

    def reset(
        self, seed=None, session: Optional[Union[str, int]] = None, instruction_text: Optional[str] = None
    ) -> any:
        """
        Reset the environment and return the initial observation.

        Args:
            session (str|int|None): The new session ID.
            instruction_text (str|None): Optional new instruction text.

        Returns:
            The initial observation.
        """
        if session is None:
            with all_seed(seed):
                session = "".join(random.choices(string.ascii_lowercase, k=10))
        obs, _ = WebAgentTextEnv.reset(self, session=session, instruction_text=instruction_text)
        self.prepare_render_cache(WebAgentTextEnv.get_instruction_text(self))
        return obs

    def step(self, action):
        """
        Take an action in the environment and return the next observation, reward, done, and info.
        """
        state, reward, done, info = WebAgentTextEnv.step(self, action)
        self.prepare_render_cache(self.observation)
        info = {
            "action_is_effective": tuple(self.get_available_actions())
            == ("click[back to search]", "click[< prev]", "click[next >]"),
            "action_is_valid": True,
            "success": done,
        }
        return self.observation, reward, done, info

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
