from roll.pipeline.agentic.env.sandbox import SokobanSandboxEnv
import traceback

def test_sandbox():
    """
    Main function to run an interactive test session with the SokobanSandboxEnv.
    """
    try: 
        env = SokobanSandboxEnv()
        print("--- Initialization Successful! ---")
        
        # Initial reset to start the first game
        obs, info = env.reset(seed=1)
        print_game_state(obs, info)
        
        while True:
            keyboard = input("Enter action(up, down, left, right), render, reset, or enter exit to quit): ").strip().lower()
            
            if not keyboard:
                continue
            
            if keyboard == "exit":
                break
            
            if keyboard == "render":
                print(env.render())
                continue
            
            if keyboard.startswith("reset"):
                parts = keyboard.split()
                seed = None
                if len(parts) > 1:
                    try:
                        seed = int(parts[1])
                        print(f"--- Resetting with seed: {seed} ---")
                    except (ValueError, IndexError):
                        print("Invalid seed provided. Resetting with a random seed.")
                
                obs, info = env.reset(seed=seed)
                print_game_state(obs, info)
                continue
            
            # Wrap the action in the format expected by the LLM parser
            action = f"<answer>{keyboard}</answer>"
            obs, reward, terminated, truncated, info = env.step(action)
            print_game_state(obs, info)
            print(f"Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")
            
            if terminated or truncated:
                print("\n!!! GAME OVER !!!Starting a new game...")
                obs, info = env.reset()
                print_game_state(obs, info)
    
    except Exception as e:
        print("\n!!! An error occurred during SokobanSandboxEnv initialization !!!")
        # traceback.format_exc() is more informative than just printing the exception 'e'
        print("--- Full Traceback ---")
        print(traceback.format_exc())
        print("--- End of Traceback ---")
    
    finally:
        if env:
            print("\n--- Closing environment ---")
            env.close()    

def print_game_state(obs, info):
    """
    A helper function to neatly print the current game state.
    
    Args:
        obs (str): The observation string, which contains rules or turn feedback.
        info (dict): The info dictionary, which should contain the game map.
    """
    print("\n" + "="*20 + " CURRENT STATE " + "="*20)
    
    # Print the observation (game rules or turn feedback)
    print("\n[Observation]")
    print(obs)
    
    # Extract and print the game map from the info dictionary
    game_map = info.get('suffix', 'No map data found in info.')
    print("\n[Map]")
    print(game_map.strip())  # .strip() removes potential leading/trailing whitespace
    
    print("="*55 + "\n")

if __name__ == "__main__":
    test_sandbox()