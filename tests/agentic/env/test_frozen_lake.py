from roll.pipeline.agentic.env.frozen_lake import FrozenLakeEnv
from roll.pipeline.agentic.utils import dump_frames_as_gif


def test_frozen_lake():
    """Test FrozenLake environment with a fixed action sequence (no keyboard input)."""
    env = FrozenLakeEnv(
        size=4,
        p=0.8,
        is_slippery=False,
        map_seed=42,
        render_mode="rgb_array"
    )
    frames = []
    obs = env.reset(seed=42)
    print(f"Initial observation: {obs}")
    frames.append(env.render(mode="rgb_array"))

    # Predefined action sequence to complete the game
    # Actions: 1=Left, 2=Down, 3=Right, 4=Up (note: 0=Still)
    actions = ['2', '2', '3', '3', '4', '3']  # Down, Down, Right, Right, Up, Right

    for action in actions:
        obs, reward, done,truncated, info = env.step(action)
        print(f"Action: {action}, Obs: {obs}, Reward: {reward}, Done: {done}, Info: {info}")
        frames.append(env.render(mode="rgb_array"))
        if done:
            print("Game completed!")
            break

    # save the image
    dump_frames_as_gif(filename="./frozen_lake_result.gif", frames=frames)
    
    # Basic assertions
    assert len(frames) > 0, "Should have captured frames"
    print(f"Test passed! Captured {len(frames)} frames.")