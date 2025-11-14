# Customer Env

## Reinforcement Learning Environment

In Reinforcement Learning (RL), the **Environment** is the world where the **Agent** interacts with. It defines the **States** that the agent can perceive, the **Actions** it can execute, and the **Reward** the agent receives after each interaction. The environment is responsible for simulating the dynamics of the real world, updating its state based on the agent's actions, and providing feedback.

To help you quickly get started and understand the adaptability and performance of our ROLL framework's Agentic Pipeline across various task scenarios, we specifically provide two main types of example environments:

- Traditional RL Environments based on **Discrete Actions** (inheriting from *BaseDiscreteActionEnv*): Such as Sokoban (Push Box Puzzle) and FrozenLake (Sliding Maze). These represent classic RL challenges like discrete action control and uncertain state transitions.
- Complex Environments based on **Natural Language Interaction** (inheriting from *BaseLanguageBasedEnv): Such as WebShop (Online Shopping Simulation) and Countdown (Number Game). These represent advanced LLM Agent challenges like complex natural language understanding and generation, multi-step planning, and reasoning.

## Core Functional Requirements

A standard environment (Env) typically implements the following functionalities:

- Observation Space
  - Definition: Specifies the format, range, and type of information the agent can obtain from the environment.
  - Examples: Box(low=0, high=255, shape=(84, 84, 3)) for image inputs, or Text(max_length=8192) for long text inputs.
- Action Space
  - Definition: Specifies the types and ranges of actions the agent can execute.
  - Examples: Discrete(n=4) for discrete actions (e.g., up, down, left, right), or Text(max_length=256) for text generation actions (e.g., WebShop search operations).
- reset() 
  - Triggered: At the start of each training episode.
  - Function: Resets the environment to an initial state and returns the initial observation.
  - Standard Output: initial_observation, info (an optional auxiliary dictionary).
- step(action)
  - Triggered: After the agent executes an action.
  - Function: Updates the environment state, calculates rewards, and determines if the episode has ended.
  - Standard Output:
    - next_observation: The new observation after the action.
    - reward: The reward received by the agent (float).
    - terminated: Boolean indicating if the episode ended naturally (e.g., game failure, goal achieved).
    - truncated: Boolean indicating if the episode ended due to time limits or other non-natural conditions.
    - info:  A dictionary containing diagnostic information (e.g., debug data, not for agent input).
- render() (Optional)
    - Function: Visualizes the environment state (e.g., graphical interface).
    - Headless Mode: Not required in headless training scenarios.
- close() 方法 (Optional)
    - Function: Cleans up environment resources (e.g., closes rendering windows or releases file handles).

## Code Examples

### Sokoban Environment: Discrete Action Classic Puzzle Task
1. Environment Configuration
```python
class SokobanEnvConfig:
    # Room dimensions (rows, columns)
    dim_room: Tuple[int, int] = (6, 6) 
    # Maximum steps per episode
    max_steps: int = 100 
    # Number of boxes in the room
    num_boxes: int = 3 
    # Search depth for generating solvable rooms
    search_depth: int = 300 
    # Mapping from grid element integer IDs to character representations (for text rendering)
    grid_lookup: Optional[Dict[int, str]] = field(
        default_factory=lambda: {0: "#", 1: "_", 2: "O", 3: "√", 4: "X", 5: "P", 6: "S"}
    )
    # Mapping from grid elements to readable names
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
    # Mapping from action IDs to action names (1: Up, 2: Down, 3: Left, 4: Right)
    action_lookup: Optional[Dict[int, str]] = field(
        default_factory=lambda: {1: "Up", 2: "Down", 3: "Left", 4: "Right"}
    )
    # Compatibility fields for setting dim_room via dim_x/dim_y
    dim_x: Optional[int] = None
    dim_y: Optional[int] = None
    render_mode: str = "text"
```

2. Environment Implementation **SokobanEnv**
This is a standard RL environment implementation, inheriting from BaseDiscreteActionEnv (generic interface for discrete-action environments) and GymSokobanEnv (core logic for the Sokoban game).

- Action Space Definition: 4 discrete actions starting from 1 (up, down, left, right):
```python
self.ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1)
```

- reset()：Generates a new Sokoban room layout and resets internal state:
```python
def reset(self, seed=None):
    try:
        # Ensures reproducibility of room generation
        with all_seed(seed):
            # Call generate_room to create a new room
            self.room_fixed, self.room_state, self.box_mapping, action_sequence = generate_room(
                dim=self.dim_room,
                num_steps=self.num_gen_steps,  # The number of steps required to generate a room
                num_boxes=self.num_boxes,
                search_depth=self.search_depth,
            )
        # Reset counters and state
        self.num_env_steps, self.reward_last, self.boxes_on_target = 0, 0, 0
        self.player_position = np.argwhere(self.room_state == 5)[0]  # Find player's initial position
        
        # Return initial observation
        return self.render()
    except (RuntimeError, RuntimeWarning) as e:
        # Retry with a new seed if room generation fails
        next_seed = abs(hash(str(seed))) % (2**32) if seed is not None else None
        return self.reset(next_seed)
``` 

- step(action): Executes an action and updates the state:
```python
def step(self, action: int):
    # Record player's old position to determine if action was effective
    previous_pos = self.player_position
    
    # Call parent class GymSokobanEnv's step method to execute the action
    _, reward, done, _ = GymSokobanEnv.step(self, action)
    
    # Get the new observation after executing the action
    next_obs = self.render()
    
    # Determine if the action actually changed the player's position
    action_effective = not np.array_equal(previous_pos, self.player_position)
    
    # Construct and return auxiliary information dictionary
    info = {
        "action_is_effective": action_effective,  # Whether the action actually moved the player or a box
        "action_is_valid": True, # Whether the input action ID is valid (even if hitting a wall)
        "success": self.boxes_on_target == self.num_boxes, # Whether all boxes are on target (game won)
    }

    # Return standard reinforcement learning environment step results (next_observation, reward, terminated, info)
    return next_obs, reward, done, info
```

- render()：Renders the current environment state as text or an image.
```python
def render(self, mode=None):
    # Use specified mode or default mode
    render_mode = mode if mode is not None else self.render_mode 
    
    if render_mode == "text":
        # Text rendering: Convert internal numeric representation of room state to ASCII character grid
        room = np.where((self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state)
        return "\n".join("".join(self.GRID_LOOKUP.get(cell, "?") for cell in row) for row in room.tolist())
    elif render_mode == "rgb_array":
        # Image rendering: Delegate to parent class GymSokobanEnv's get_image method
        return self.get_image(mode="rgb_array", scale=1)
    else:
        raise ValueError(f"Invalid mode: {render_mode}")
```

3. Module Test
```python
import matplotlib.pyplot as plt
# Create a Sokoban environment configuration
config = SokobanEnvConfig(dim_room=(6, 6), num_boxes=1, max_steps=100, search_depth=10)
# Create a Sokoban environment instance using this configuration
env = SokobanEnv(config)
# Loop 10 times, resetting the environment with a different seed each time, and print the initial state to observe different room layouts.
for i in range(10):
    # Reset environment with a seed
    print(env.reset(seed=1010 + i))
    print()
# Enter an interactive loop, allowing the user to control the agent via keyboard input.
while True:
    keyboard = input("Enter action: ")
    if keyboard == "q":
        break
    # Convert input to integer action ID
    action = int(keyboard)
    assert action in env.ACTION_LOOKUP, f"Invalid action: {action}"
    # Execute the action, get new observation, reward, done state, and info
    obs, reward, done, info = env.step(action)
    print(obs, reward, done, info)
# If the environment supports RGB array rendering, get the final game screen image.
np_img = env.get_image("rgb_array")
# Save the image
plt.imsave("sokoban1.png", np_img)
```

### WebShop Environment: Complex Natural Language-Driven Interaction Task

WebShop is a simulated online shopping environment that requires agents to complete tasks like searching, selecting products, viewing details, and placing orders based on natural language instructions. Each trajectory includes up to 50 steps, demanding strong contextual understanding and task execution efficiency.

The following section focuses on the differences from the Sokoban environment:

1. WebShop parses available actions in the environment and converts them into a list of text strings that the agent can generate.
```python
def get_available_actions(self):
    # Get raw available action information from the underlying WebShop simulator
    # Unlike Sokoban's fixed action set, WebShop's action space is dynamic.
    orig_available_actions = WebAgentTextEnv.get_available_actions(self) 
    available_actions = []
    # Define text format for search actions
    if orig_available_actions["has_search_bar"]:
        available_actions.append("search[<content>]") 
    # Define text format for click actions
    for clickable in orig_available_actions["clickables"]:
        if clickable != "search":
            available_actions.append(f"click[{clickable}]") 
    # Return a list of strings, instructing the Agent which string to generate 
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
    
    # Call parent class WebAgentTextEnv's reset, which returns text observation
    obs, _ = WebAgentTextEnv.reset(self, session=session, instruction_text=instruction_text)
    
    # Prepare render cache: Add initial instruction to cache for render method
  self.prepare_render_cache(WebAgentTextEnv.get_instruction_text(self))
    return obs
```

3. WebShop's action is a natural language text string.
```python
def step(self, action):
    # Call parent class WebAgentTextEnv's step, which parses and executes text actions
    state, reward, done, info = WebAgentTextEnv.step(self, action)
    
    # Prepare render cache: Update cached observation
    self.prepare_render_cache(self.observation)
    
    # Construct auxiliary information dictionary
    info = {
        "action_is_effective": tuple(self.get_available_actions()) 
        == ("click[back to search]", "click[< prev]", "click[next >]"), 
        "action_is_valid": True,
        "success": done, 
    }
    return self.observation, reward, done, info
```

## Creating a Custom Env

### Step Overview
1. Choose a Base Class: Select to inherit from BaseDiscreteActionEnv or BaseLanguageBasedEnv based on your task type (discrete actions or language interaction).

2. Define init: Initialize environment parameters, define observation_space and action_space.

3. Implement reset(): Define the initial state of the environment.

4. Implement step(action): Define how the environment updates its state, calculates rewards, and determines episode termination based on an action.

5. Implement render(): Define the environment's rendering logic.

6. Implement close(): Define resource cleanup logic.

### Design Suggestions
1. State Representation
   - Discrete Action Environments: Structured grid states, position information, etc.
   - Language Environments: Text observations should contain all relevant context (e.g., full web page content, instructions) and consider context window limits. Too much redundant information can lead to LLM inefficiency or inability to process. 
2. Action Space Design
   - Discrete Action Environments: Actions are predefined integer or enumerated values.
   - Language Environments: Actions are natural language text. This requires the agent to have natural language generation capabilities, and the environment needs to be able to parse and validate these text actions.
3. Reward Function Design
   - Clear Goals: Rewards should clearly guide the agent towards the desired behavior. 
   - Sparse vs. Dense Rewards
     - Discrete Action Environments: Rewards are usually given upon completing subgoals or final goals.
     - Language Environments:
        - WebShop may have sparse rewards, but intermediate rewards can also be designed.
        - Countdown uses hierarchical rewards (0, format score, full score) to guide learning.
    - Avoid Reward Hacking: Ensure the agent cannot achieve high rewards through unintended means.
    - Format Penalty: In language environments, imposing penalties for text actions that do not conform to the expected format is crucial; it effectively guides the LLM to generate structured and parsable output.
4. Episode Termination Conditions
   - Clearly define conditions for success, failure, or timeout to end a training episode. Use terminated and truncated to denote natural and non-natural termination, respectively. 
   - WebShop also has a maximum step limit.
5. Uncertainty/Randomness: If the environment includes uncertainty (like FrozenLake), ensure its behavior follows a predictable probability distribution and that randomness can be controlled via a seed in reset.
6. Reproducibility: Use the seed parameter to initialize random number generators to ensure that the environment's behavior is reproducible across runs. 
