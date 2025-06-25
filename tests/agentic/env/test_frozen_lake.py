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
from roll.agentic.env import FrozenLakeEnvConfig, FrozenLakeEnv
from roll.agentic.utils import dump_frames_as_gif


def test_frozen_lake():
    import matplotlib.pyplot as plt

    config = FrozenLakeEnvConfig(size=4, p=0.8, is_slippery=False, map_seed=42)
    env = FrozenLakeEnv(config)
    frames = []
    print(env.reset(seed=42))
    while True:
        keyboard = input("Enter action: ")
        if keyboard.lower() == "q":
            break
        try:
            action = int(keyboard)
        except Exception as e:
            print("Invalid action, please enter a number")
            continue
        if action not in env.ACTION_LOOKUP:
            print(f"Invalid action {action}, please enter a number between 1 and 4")
            continue
        obs, reward, done, info = env.step(action)
        print()
        print(obs, reward, done, info)
        if action in env.ACTION_LOOKUP:
            frames.append(env.render(mode="rgb_array"))
        if done:
            break

    # save the image
    dump_frames_as_gif(filename="./frozen_lake_result.gif", frames=frames)


if __name__ == "__main__":
    test_frozen_lake()
