from numbers import Real

from roll.pipeline.agentic.env.frozen_lake import FrozenLakeEnv


def test_frozen_lake():
    env = FrozenLakeEnv(size=4, p=0.8, is_slippery=False, map_seed=42)

    obs, info = env.reset(seed=42)
    assert isinstance(obs, str)
    assert "env_instruction" in info

    action_text = f"<answer>{env.ACTION_LOOKUP[0]}</answer>"
    obs, reward, terminated, truncated, info = env.step(action_text)
    assert isinstance(obs, str)
    assert isinstance(reward, Real)
    assert not isinstance(reward, bool)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["metrics"]["action_is_valid"]


if __name__ == "__main__":
    test_frozen_lake()
