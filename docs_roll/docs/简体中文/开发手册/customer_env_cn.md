# 自定义Env

## 强化学习Env
在强化学习中，环境（Environment）是智能体（Agent）进行交互的世界。它定义了智能体可以感知的状态（State）、可以执行的动作（Action）、以及智能体每次交互后获得的奖励（Reward）。环境负责模拟真实世界的动态，并根据智能体的动作更新状态并给出反馈。

为了帮助您快速入门并了解我们ROLL框架的 Agentic Pipeline 在不同任务场景下的适应性与性能表现，我们特地提供了两类核心示例环境：
- 基于离散动作的传统RL环境 (继承自 BaseDiscreteActionEnv)：如 Sokoban（推箱子） 和 FrozenLake（冰湖）。它们代表了离散动作控制、不确定状态转移等经典RL挑战。
- 基于自然语言交互的复杂环境 (继承自 BaseLanguageBasedEnv)：如 WebShop（模拟在线购物） 和 Countdown（数字游戏）。它们代表了复杂自然语言理解与生成、多步规划和推理等先进的LLM Agent挑战。

## 核心功能
一个标准的Env通常需要实现以下功能：
- 观察空间（Observation Space）
  - 定义智能体可以从环境中获取的信息的格式、范围和类型。
  - 示例：Box(low=0, high=255, shape=(84, 84, 3)) 用于图像输入，或 Text(max_length=8192) 用于长文本输入。
- 动作空间（Action Space）
  - 定义智能体可以执行的动作的类型和范围。
  - 示例：Discrete(n=4) 用于离散动作（如上下左右），或 Text(max_length=256) 用于文本生成动作。
- reset() 方法
  - 在每个训练回合（Episode）开始时调用。
  - 将环境重置到初始状态，并返回初始观测。
  - 标准返回：initial_observation, info （其中 info 是可选的辅助信息字典）。
- step(action) 方法
  - 在智能体执行一个动作后调用。
  - 根据智能体的动作更新环境状态，计算奖励，并判断回合是否结束。
  - 标准返回： 
    - next_observation: 执行动作后的新观测。
    - reward: 智能体因执行该动作获得的奖励（浮点数）。
    - terminated: 布尔值，表示回合是否因达到终止条件而结束（如游戏失败，达到目标）。
    - truncated: 布尔值，表示回合是否因时间限制等非自然终止条件而结束。
    - info: 字典，包含诊断信息（例如调试数据，不应作为智能体的输入）。
- render() 方法 (可选)
    - 用于可视化环境状态，例如在屏幕上显示图形界面。
    - 对于无头（headless）训练场景，此方法通常无需实现。
- close() 方法 (可选)
    - 用于清理环境资源，例如关闭渲染窗口或释放文件句柄。

## 代码示例

### Sokoban 环境：离散动作的经典解谜任务
1. 环境配置 SokobanEnvConfig
```python
class SokobanEnvConfig:
    # 房间的尺寸 (行, 列)
    dim_room: Tuple[int, int] = (6, 6) 
    # 每个回合的最大步数
    max_steps: int = 100 
    # 房间中箱子的数量
    num_boxes: int = 3 
    # 用于生成可解房间时的搜索深度
    search_depth: int = 300 
    # 网格元素的整数ID到字符表示的映射，用于文本渲染
    grid_lookup: Optional[Dict[int, str]] = field(
        default_factory=lambda: {0: "#", 1: "_", 2: "O", 3: "√", 4: "X", 5: "P", 6: "S"}
    )
    # 网格元素的字符到可读名称的映射
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
    # 动作ID到动作名称的映射 (1:上, 2:下, 3:左, 4:右)
    action_lookup: Optional[Dict[int, str]] = field(
        default_factory=lambda: {1: "Up", 2: "Down", 3: "Left", 4: "Right"}
    )
    # 允许通过dim_x, dim_y设置dim_room的兼容性字段
    dim_x: Optional[int] = None
    dim_y: Optional[int] = None
    render_mode: str = "text"
```

2. 环境实现 SokobanEnv
这是一个标准的强化学习环境实现，它继承了框架中的 BaseDiscreteActionEnv（用于离散动作环境的通用接口）和 GymSokobanEnv（Sokoban 游戏的核心逻辑）。
- 定义工作空间：4个离散动作，ID从1开始 (1, 2, 3, 4)
```python
self.ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1)
```

- reset方法：生成一个新的Sokoban房间布局，并重置游戏内部状态
```python
def reset(self, seed=None):
    try:
        # 使用all_seed确保房间生成的可复现性
        with all_seed(seed):
            # 调用generate_room生成新的房间布局
            self.room_fixed, self.room_state, self.box_mapping, action_sequence = generate_room(
                dim=self.dim_room,
                num_steps=self.num_gen_steps, # 房间生成所需的步数
                num_boxes=self.num_boxes,
                search_depth=self.search_depth,
            )
        # 重置回合相关计数器和状态
        self.num_env_steps, self.reward_last, self.boxes_on_target = 0, 0, 0
        self.player_position = np.argwhere(self.room_state == 5)[0] # 找到玩家位置
        
        # 返回初始观测（通过render方法获取）
        return self.render()
    except (RuntimeError, RuntimeWarning) as e:
        # 如果生成房间失败，尝试使用新种子重新生成
        next_seed = abs(hash(str(seed))) % (2**32) if seed is not None else None
        return self.reset(next_seed)
``` 

- step方法：根据智能体执行的动作更新环境状态。
```python
def step(self, action: int):
    # 记录玩家旧位置，用于判断动作是否有效
    previous_pos = self.player_position
    
    # 调用父类GymSokobanEnv的step方法来执行动作
    _, reward, done, _ = GymSokobanEnv.step(self, action)
    
    # 获取执行动作后的新观测
    next_obs = self.render()
    
    # 判断动作是否实际改变了玩家位置
    action_effective = not np.array_equal(previous_pos, self.player_position)
    
    # 构造并返回额外信息字典
    info = {
        "action_is_effective": action_effective, # 动作是否实际移动了玩家或箱子
        "action_is_valid": True, # 传入的动作ID是否合法（即使撞墙）
        "success": self.boxes_on_target == self.num_boxes, # 是否所有箱子都在目标上（游戏胜利）
    }

    # 返回标准的强化学习环境step结果 (next_observation, reward, terminated, info)
    return next_obs, reward, done, info
```

- render方法：将当前环境状态渲染为文本或图像。
```python
def render(self, mode=None):
    # 使用指定模式或默认模式
    render_mode = mode if mode is not None else self.render_mode 
    
    if render_mode == "text":
        # 文本渲染：将内部数字表示的房间状态转换为ASCII字符网格
        room = np.where((self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state)
        return "\n".join("".join(self.GRID_LOOKUP.get(cell, "?") for cell in row) for row in room.tolist())
    elif render_mode == "rgb_array":
        # 图像渲染：委托给父类GymSokobanEnv的get_image方法
        return self.get_image(mode="rgb_array", scale=1)
    else:
        raise ValueError(f"Invalid mode: {render_mode}")
```

3. 模块测试
```python
import matplotlib.pyplot as plt
# 创建一个Sokoban环境配置
config = SokobanEnvConfig(dim_room=(6, 6), num_boxes=1, max_steps=100, search_depth=10)
# 使用该配置创建Sokoban环境实例
env = SokobanEnv(config)
# 循环10次，每次使用不同的种子重置环境，并打印初始状态，以观察不同房间布局。
for i in range(10):
    # 重置环境并传入种子
    print(env.reset(seed=1010 + i))
    print()
# 进入一个交互循环，允许用户通过键盘输入控制智能体。  
while True:
    keyboard = input("Enter action: ")
    if keyboard == "q":
        break
    # 将输入转换为整数动作ID  
    action = int(keyboard)
    assert action in env.ACTION_LOOKUP, f"Invalid action: {action}"
    # 执行动作，获取新的观测、奖励、结束状态和信息
    obs, reward, done, info = env.step(action)
    print(obs, reward, done, info)
# 如果环境支持RGB数组渲染，则获取最终的游戏画面图像  
np_img = env.get_image("rgb_array")
# 保存图像
plt.imsave("sokoban1.png", np_img)
```

### WebShop 环境：自然语言驱动的复杂交互任务

WebShop 是一个模拟在线购物的任务环境，要求智能体根据自然语言指令完成搜索、选择商品、查看详细信息并下单等操作。每个轨迹最多包含 50 步，对模型的上下文理解能力和任务执行效率提出了较高要求。

下面重点讲解和Sokoban不同的地方：

1. WebShop会解析环境中的可用动作，并将其转换为智能体可生成的文本字符串列表。
```python
def get_available_actions(self):
    # 从底层WebShop模拟器获取原始可用操作信息
    # 与Sokoban的固定动作集不同，WebShop的动作空间是动态的。
    orig_available_actions = WebAgentTextEnv.get_available_actions(self) 
    available_actions = []
    # 定义搜索动作的文本格式
    if orig_available_actions["has_search_bar"]:
        available_actions.append("search[<content>]") 
    # 定义点击动作的文本格式
    for clickable in orig_available_actions["clickables"]:
        if clickable != "search":
            available_actions.append(f"click[{clickable}]") 
    # 返回字符串列表，指导Agent生成哪个字符串      
    return available_actions
```

2. WebShop的reset可指定会话ID和初始指令文本。
```python
def reset(
    self, seed=None, session: Optional[Union[str, int]] = None, instruction_text: Optional[str] = None
) -> any:
  
    # 会话ID管理：如果未提供，则随机生成一个
    if session is None:
        with all_seed(seed):
            session = "".join(random.choices(string.ascii_lowercase, k=10))
    
    # 调用父类WebAgentTextEnv的reset，它会返回文本观测
    obs, _ = WebAgentTextEnv.reset(self, session=session, instruction_text=instruction_text)
    
    # 准备渲染缓存：将初始指令添加到缓存，用于render方法
    self.prepare_render_cache(WebAgentTextEnv.get_instruction_text(self))
    return obs
```

3. WebShop的action是一个自然语言文本字符串。
```python
def step(self, action):
    # 调用父类WebAgentTextEnv的step，它解析并执行文本动作
    state, reward, done, info = WebAgentTextEnv.step(self, action)
    
    # 准备渲染缓存：更新缓存的观测
    self.prepare_render_cache(self.observation)
    
    # 构造额外信息字典
    info = {
        "action_is_effective": tuple(self.get_available_actions()) 
        == ("click[back to search]", "click[< prev]", "click[next >]"), 
        "action_is_valid": True,
        "success": done, 
    }
    return self.observation, reward, done, info
```

## 创建自定义Env

### 步骤概述
1. 选择基类：根据您的任务类型（离散动作或语言交互）选择继承 BaseDiscreteActionEnv 或 BaseLanguageBasedEnv

2. 定义 init：初始化环境参数，定义 observation_space 和 action_space

3. 实现 reset()：定义环境的初始状态

4. 实现 step(action)：定义环境如何根据动作更新状态、计算奖励和判断回合结束

5. 实现 render()：定义环境的渲染逻辑

6. 实现 close()：定义资源清理逻辑

### 设计建议
1. 状态表示
   - 离散动作环境：结构化的网格状态、位置信息等。
   - 语言环境：文本观测应包含所有相关上下文（例如完整的网页内容、指令），并考虑上下文窗口限制。冗余信息过多会导致LLM效率下降或无法处理。
2. 动作空间设计
   - 离散动作环境：动作是预定义的整数或枚举值。
   - 语言环境：动作是自然语言文本。这要求智能体具备自然语言生成能力，并且环境需要能够解析和验证这些文本动作。
3. 奖励函数设计
   - 明确的目标：奖励应清晰地引导智能体实现您期望的行为。
   - 稀疏奖励 vs. 密集奖励：
     - 离散动作环境：奖励通常在完成子目标或最终目标时给予。
     - 语言环境：
        - WebShop 可能奖励稀疏，但也可设计中间奖励。
        - Countdown 使用分层奖励（0，格式分，满分）来引导学习。
    - 避免奖励欺骗（Reward Hacking）： 确保智能体无法通过非预期的方式获得高奖励。
    - 格式惩罚项：在语言环境中，对不符合预期格式的文本动作施加惩罚至关重要，它能有效引导 LLM 生成结构化且可解析的输出。
4. 回合终止条件
   - 清晰定义成功、失败或超时等条件，以结束一个训练回合。使用 terminated 和 truncated 分别表示自然终止和非自然终止。
   - WebShop 还有最大步数限制
5. 不确定性/随机性：如果环境包含不确定性（如FrozenLake），确保其行为是可预测的概率分布，并能在 reset 中通过 seed 控制随机性。
6. 可复现性：使用 seed 参数初始化随机数生成器，以确保每次运行环境时其行为都是可复现的。
