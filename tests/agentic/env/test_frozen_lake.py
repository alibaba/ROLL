from roll.pipeline.agentic.env import FrozenLakeEnvConfig, FrozenLakeEnv
from roll.pipeline.agentic.utils import dump_frames_as_gif


def test_frozen_lake():
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
