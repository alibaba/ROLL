"""
Adapted from the nicely written code from gymnasium.envs.toy_text.frozen_lake.generate_random_map
Modify it so that the start and end points are random

## Description
The game starts with the player at random location of the frozen lake grid world with the
goal located at another random location for the 4x4 environment.

## Action Space
The action shape is `(1,)` in the range `{0, 3}` indicating
which direction to move the player.
NOTE the action space is different from gymnasium.envs.toy_text.frozen_lake.FrozenLakeEnv, start from 1
- 0: Still
- 1: Left
- 2: Down
- 3: Right
- 4: Up

## Starting State
The episode starts with the player at random location

## Rewards
NOTE added -0.1 as penalty for invalid action
Reward schedule:
- Reach goal: +1
- Reach hole: 0
- Reach frozen: 0

## Arguments
`is_slippery`: if action is left and is_slippery is True, then:
- P(move left)=1/3
- P(move up)=1/3
- P(move down)=1/3

## Example
P   _   _   _
_   _   _   O
O   _   O   _
O   _   _   G
"""

from .env import FrozenLakeEnv

__all__ = ["FrozenLakeEnv"]
