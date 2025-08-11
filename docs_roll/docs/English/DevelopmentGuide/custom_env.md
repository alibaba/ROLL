# Custom Environment

## Reinforcement Learning Environment
In reinforcement learning, the environment is the world where the agent interacts. It defines the states (State) that the agent can perceive, the actions (Action) that the agent can perform, and the rewards (Reward) that the agent receives after each interaction. The environment is responsible for simulating real-world dynamics, updating states based on agent actions, and providing feedback.

To help you quickly get started and understand the adaptability and performance of our ROLL framework's Agentic Pipeline in different task scenarios, we have specially provided two core example environments:
- Traditional RL environments based on discrete actions (inheriting from BaseDiscreteActionEnv): Such as Sokoban (push box) and FrozenLake (ice lake). They represent classic RL challenges such as discrete action control and uncertain state transitions.
- Complex environments based on natural language interaction (inheriting from BaseLanguageBasedEnv): Such as WebShop (simulated online shopping) and Countdown (number game). They represent advanced LLM Agent challenges such as complex natural language understanding and generation, multi-step planning, and reasoning.

## Core Functions
A standard Env typically needs to implement the following functions:
- Observation Space
  - Defines the format, range, and type of information that the agent can obtain from the environment.
  - Example: Box(low=0, high=255, shape=(84, 84, 3)) for image input, or Text(max_length=8192) for long text input.
- Action Space
  - Defines the type and range of actions that the agent can perform.
  - Example: Discrete(n=4) for discrete actions (such as up, down, left, right), or Text(max_length=256) for text generation actions.
- reset() method
  - Called at the beginning of each training episode.
  - Resets the environment to its initial state and returns the initial observation.
  - Standard return: initial_observation, info (where info is an optional auxiliary information dictionary).
- step(action) method
  - Called after the agent performs an action.
  - Updates the environment state based on the agent's action, calculates the reward, and determines whether the episode is over.
  - Standard return:
    - next_observation: New observation after performing the action.
    - reward: Reward (floating point number) that the agent receives for performing the action.
    - terminated: Boolean value indicating whether the episode ends due to reaching a termination condition (such as game failure, reaching the goal).
    - truncated: Boolean value indicating whether the episode ends due to non-natural termination conditions such as time limits.
    - info: Dictionary containing diagnostic information (such as debugging data, which should not be used as input to the agent).
- render() method (optional)
    - Used to visualize the environment state, such as displaying a graphical interface on the screen.
    - For headless training scenarios, this method usually does not need to be implemented.
- close() method (optional)
    - Used to clean up environment resources, such as closing rendering windows or releasing file handles.

## Code Examples

### Sokoban Environment: Classic Puzzle Task with Discrete Actions
1. Environment Configuration SokobanEnvConfig
```python
class SokobanEnvConfig:
    # Room dimensions (rows, columns)
    dim_room: Tuple[int, int] = (6, 6) 
    # Maximum steps per episode
    max_steps: int = 100 
    # Number of boxes in the room
    num_boxes: int = 3 
    # Search depth used when generating solvable rooms
    search_depth: int = 300 
    # Mapping from integer IDs of grid elements to character representations for text rendering
    grid_lookup: Optional[Dict[int, str]] = field(
        default_factory=lambda: {0: "#", 1: "_", 2: "O", 3: "√", 4: "X", 5: "P", 6: "S"}
    )
    # Mapping from grid element characters to readable names
    grid_vocab: Optional[Dict[str, str]] = field(
        default_factory=lambda: {
            "#": "wall",
            "_": "empty",
            "O": "target",
            "√": "box on target",
            "X": "box",
            "P": "player",
            "S": "player on target",
        }
    )
    # Mapping from action IDs to action names (1:Up, 2:Down, 3:Left, 4:Right)
    action_lookup: Optional[Dict[int, str]] = field(
        default_factory=lambda: {1: "Up", 2: "Down", 3: "Left", 4: "Right"}
    )
    # Compatibility fields for setting dim_room via dim_x, dim_y
    dim_x: Optional[int] = None
    dim_y: Optional[int] = None
    render_mode: str = "text"
```

2. Environment Implementation SokobanEnv
This is a standard reinforcement learning environment implementation that inherits from BaseDiscreteActionEnv (a generic interface for discrete action environments) and GymSokobanEnv (the core logic of the Sokoban game) in the framework.
- Define action space: 4 discrete actions with IDs starting from 1 (1, 2, 3, 4)
```python
self.ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1)
```

- reset method: Generate a new Sokoban room layout and reset the game's internal state
```python
def reset(self, seed=None):
    try:
        # Use all_seed to ensure reproducibility of room generation
        with all_seed(seed):
            # Call generate_room to generate a new room layout
            self.room_fixed, self.room_state, self.box_mapping, action_sequence = generate_room(
                dim=self.dim_room,
                num_steps=self.num_gen_steps, # Steps required for room generation
                num_boxes=self.num_boxes,
                search_depth=self.search_depth,
            )
        # Reset episode-related counters and states
        self.num_env_steps, self.reward_last, self.boxes_on_target = 0, 0, 0
        self.player_position = np.argwhere(self.room_state == 5)[0] # Find player position
        
        # Return initial observation (obtained via render method)
        return self.render()
    except (RuntimeError, RuntimeWarning) as e:
        # If room generation fails, try again with a new seed
        next_seed = abs(hash(str(seed))) % (2**32) if seed is not None else None
        return self.reset(next_seed)
``` 

- step method: Update the environment state based on the action performed by the agent.
```python
def step(self, action: int):
    # Record the player's old position to determine if the action was effective
    previous_pos = self.player_position
    
    # Call the parent class GymSokobanEnv's step method to execute the action
    _, reward, done, _ = GymSokobanEnv.step(self, action)
    
    # Get the new observation after executing the action
    next_obs = self.render()
    
    # Determine if the action actually changed the player's position
    action_effective = not np.array_equal(previous_pos, self.player_position)
    
    # Construct and return the additional information dictionary
    info = {
        "action_is_effective": action_effective, # Whether the action actually moved the player or boxes
        "action_is_valid": True, # Whether the passed action ID is valid (even if it hits a wall)
        "success": self.boxes_on_target == self.num_boxes, # Whether all boxes are on targets (game won)
    }

    # Return the standard reinforcement learning environment step result (next_observation, reward, terminated, info)
    return next_obs, reward, done, info
```

- render method: Render the current environment state as text or image.
```python
def render(self, mode=None):
    # Use the specified mode or default mode
    render_mode = mode if mode is not None else self.render_mode 
    
    if render_mode == "text":
        # Text rendering: Convert the internal numeric representation of room state to ASCII character grid
        room = np.where((self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state)
        return "\n".join("".join(self.GRID_LOOKUP.get(cell, "?") for cell in row) for row in room.tolist())
    elif render_mode == "rgb_array":
        # Image rendering: Delegate to the parent class GymSokobanEnv's get_image method
        return self.get_image(mode="rgb_array", scale=1)
    else:
        raise ValueError(f"Invalid mode: {render_mode}")
```

3. Module Testing
```python
import matplotlib.pyplot as plt
# Create a Sokoban environment configuration
config = SokobanEnvConfig(dim_room=(6, 6), num_boxes=1, max_steps=100, search_depth=10)
# Create a Sokoban environment instance using this configuration
env = SokobanEnv(config)
# Loop 10 times, each time resetting the environment with a different seed and printing the initial state to observe different room layouts.
for i in range(10):
    # Reset the environment and pass in the seed
    print(env.reset(seed=1010 + i))
    print()
# Enter an interactive loop that allows the user to control the agent through keyboard input.  
while True:
    keyboard = input("Enter action: ")
    if keyboard == "q":
        break
    # Convert input to integer action ID  
    action = int(keyboard)
    assert action in env.ACTION_LOOKUP, f"Invalid action: {action}"
    # Execute the action and get the new observation, reward, termination state, and information
    obs, reward, done, info = env.step(action)
    print(obs, reward, done, info)
# If the environment supports RGB array rendering, get the final game screen image  
np_img = env.get_image("rgb_array")
# Save the image
plt.imsave("sokoban1.png", np_img)
```

### WebShop Environment: Complex Interaction Task Driven by Natural Language

WebShop is a simulated online shopping task environment that requires the agent to complete operations such as searching, selecting products, viewing details, and placing orders according to natural language instructions. Each trajectory can contain up to 50 steps, which places high demands on the model's context understanding ability and task execution efficiency.

The following focuses on the differences from Sokoban:

1. WebShop parses available actions in the environment and converts them into a list of text strings that the agent can generate.
```python
def get_available_actions(self):
    # Get raw available action information from the underlying WebShop simulator
    # Unlike Sokoban's fixed action set, WebShop's action space is dynamic.
    orig_available_actions = WebAgentTextEnv.get_available_actions(self) 
    available_actions = []
    # Define the text format for search actions
    if orig_available_actions["has_search_bar"]:
        available_actions.append("search[<content>]") 
    # Define the text format for click actions
    for clickable in orig_available_actions["clickables"]:
        if clickable != "search":
            available_actions.append(f"click[{clickable}]") 
    # Return a string list to guide the Agent on which string to generate      
    return available_actions
```

2. WebShop's reset can specify a session ID and initial instruction text.
```python
def reset(
    self, seed=None, session: Optional[Union[str, int]] = None, instruction_text: Optional[str] = None
) -> any:
  
    # Session ID management: If not provided, generate a random one
    if session is None:
        with all_seed(seed):
            session = "".join(random.choices(string.ascii_lowercase, k=10))
    
    # Call the parent class WebAgentTextEnv's reset, which returns text observation
    obs, _ = WebAgentTextEnv.reset(self, session=session, instruction_text=instruction_text)
    
    # Prepare render cache: Add the initial instruction to the cache for the render method
    self.prepare_render_cache(WebAgentTextEnv.get_instruction_text(self))
    return obs
```

3. WebShop's action is a natural language text string.
```python
def step(self, action):
    # Call the parent class WebAgentTextEnv's step, which parses and executes the text action
    state, reward, done, info = WebAgentTextEnv.step(self, action)
    
    # Prepare render cache: Update the cached observation
    self.prepare_render_cache(self.observation)
    
    # Construct additional information dictionary
    info = {
        "action_is_effective": tuple(self.get_available_actions()) 
        == ("click[back to search]", "click[< prev]", "click[next >]"), 
        "action_is_valid": True,
        "success": done, 
    }
    return self.observation, reward, done, info
```

## Creating Custom Environments

### Step Overview
1. Choose a base class: Select to inherit from BaseDiscreteActionEnv or BaseLanguageBasedEnv based on your task type (discrete actions or language interaction)

2. Define init: Initialize environment parameters, define observation_space and action_space

3. Implement reset(): Define the initial state of the environment

4. Implement step(action): Define how the environment updates state, calculates rewards, and determines episode termination based on actions

5. Implement render(): Define the environment's rendering logic

6. Implement close(): Define resource cleanup logic

### Design Recommendations
1. State Representation
   - Discrete action environments: Structured grid states, position information, etc.
   - Language environments: Text observations should contain all relevant context (such as complete web content, instructions) and consider context window limitations. Too much redundant information will reduce LLM efficiency or make it unable to process.
2. Action Space Design
   - Discrete action environments: Actions are predefined integers or enumeration values.
   - Language environments: Actions are natural language text. This requires the agent to have natural language generation capabilities, and the environment needs to be able to parse and validate these text actions.
3. Reward Function Design
   - Clear objectives: Rewards should clearly guide the agent to achieve the behavior you expect.
   - Sparse rewards vs. dense rewards:
     - Discrete action environments: Rewards are usually given when completing sub-goals or final goals.
     - Language environments:
        - WebShop may have sparse rewards but can also design intermediate rewards.
        - Countdown uses hierarchical rewards (0, format score, full score) to guide learning.
    - Avoid reward hacking: Ensure the agent cannot obtain high rewards through unintended means.
    - Format penalty terms: In language environments, it is crucial to impose penalties on text actions that do not conform to expected formats, as it can effectively guide LLM to generate structured and parseable outputs.
4. Episode Termination Conditions
   - Clearly define conditions for success, failure, or timeout to end a training episode. Use terminated and truncated to represent natural termination and non-natural termination respectively.
   - WebShop also has a maximum step limit
5. Uncertainty/Randomness: If the environment contains uncertainty (such as FrozenLake), ensure its behavior is a predictable probability distribution, and control randomness through seed in reset.
6. Reproducibility: Use the seed parameter to initialize the random number generator to ensure that the environment's behavior is reproducible each time it runs.