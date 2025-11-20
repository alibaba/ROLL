# Prompt生成指南

在基于大语言模型（LLM）的强化学习 Agent 体系中，`Prompt` 是 LLM 与环境进行交互的唯一介质。LLM 不像传统 Agent 那样直接接收数值状态或输出离散动作 ID，而是通过文本形式的 Prompt 来“感知”环境（观测）并“表达”其决策（动作）。

## 核心概念

在我们的框架中，Prompt 的生成遵循以下几个关键原则：
- LLM 输入是文本：无论环境的原始观测是图像、网格还是其他结构，最终都会被转化为 LLM 能够理解的文本格式。
- Prompt 是动态且上下文相关的：Prompt 不仅仅是当前的环境观测，它会包含历史对话、之前的行动、获得的奖励等信息，形成一个连贯的对话上下文。
- Prompt 是结构化的对话格式：Prompt 通常遵循 LLM 的聊天模板（如 System/User/Assistant 角色），以便 LLM 更好地理解不同部分的意图。
- Prompt 指导 LLM 行为：通过精确的指令、输出格式要求和思考链提示，Prompt 能够引导 LLM 按照预期的方式生成响应。

Prompt 的生成主要由 `EnvManager` 类中的 `_format_messages` 方法负责。

## Prompt 生成的步骤与规则

`_format_messages` 方法是 Prompt 生成的核心。它接收 `env_output`（包含当前观测和历史信息），并根据一系列规则将其转化为 LLM 的输入。

### 步骤1：初始化对话与基本指令

Prompt 的生成始于构建一个对话的骨架，包括系统指令和第一个用户指令。
```python
messages = [
    # System Prompt: 定义LLM的角色和目标
    {"role": "system", "content": "You're a helpful assistant. You are a good game player. You are aiming to get high reward in the game."},
    # 第一个User Prompt: 包含环境的整体介绍和初始指令
    {"role": "user", "content": first_user_content}
]
```
- **System Prompt**: 这是一个固定不变的指令，用于设定 LLM 的通用角色（"有帮助的助手"、"优秀的玩家"）和总体目标（"获得高奖励"）。这为 LLM 提供了一个全局的行动准则。
- **第一个 User Prompt** (`first_user_content`): 这是最关键的初始化部分，它会详细介绍当前环境的规则、符号含义、可用动作和响应格式。它的内容由 `EnvManager._init_prefix_lookup` 方法预先生成，并结合了环境配置中的 `env_instruction`, `grid_vocab`, `action_lookup`。

#### Sokoban 示例：第一个 User Prompt 的生成

假设 `SokobanEnvConfig` 的配置如下：
```yaml
env_instruction: "You are solving the Sokoban puzzle. You are the player and you need to push all boxes to targets. When you are right next to a box, you can push it by moving in the same direction. You cannot push a box through a wall, and you cannot pull a box. The answer must be one of action in a turn, format is <answer>Right</answer>"

grid_vocab:
  "#": "wall"
  "_": "empty"
  "O": "target"
  "√": "box on target"
  "X": "box"
  "P": "player"
  "S": "player on target"

action_lookup:
  1: "Up"
  2: "Down"
  3: "Left"
  4: "Right"
```

那么，`first_user_content`（即第一个 User Prompt）将被构建为类似下面的字符串：
```
You are solving the Sokoban puzzle. You are the player and you need to push all boxes to targets. When you are right next to a box, you can push it by moving in the same direction. You cannot push a box through a wall, and you cannot pull a box. The answer must be one of action in a turn, format is <answer>Right</answer>

The meaning of each symbol in the state is:
#: wall, _: empty, O: target, √: box on target, X: box, P: player, S: player on target

Your available actions are:
Up, Down, Left, Right
```
这个 Prompt 块全面地向 LLM 描述了 Sokoban 游戏的规则、视觉符号的含义和可执行的动作，为后续的决策提供了基础认知。

### 步骤2：遍历环境历史，构建多轮对话上下文

在初始 Prompt 之后，`_format_messages` 会遍历 `env_output['history']`，将之前每一步的观测、LLM 的响应和环境的奖励添加到对话中，形成一个连续的上下文。
```python
# 遍历环境历史，构建多轮对话Prompt
for idx, content in enumerate(env_output["history"]):
    # 1. 添加回合编号
    messages[-1]["content"] += f"\nTurn {idx + 1}:\n"

    # 2. 处理环境状态 (State)
    if "state" in content:
        FORMAT_PROMPT = "<think> [Your thoughts] </think> <answer> [your answer] </answer>" if self.pipeline_config.enable_think else "<answer> [your answer] </answer>"
        LENGTH_PROMPT = f"Max response length: {self.env_config_lookup[env_output['env_id']]['max_tokens']} words (tokens)."
        messages[-1]["content"] += (
            f"State:\n{content['state']}\n"
            f"You have {content['actions_left']} actions left. "
            f"Always output: {FORMAT_PROMPT} with no extra text."
            f"Strictly follow this format, history response that do not follow the format will be set as 'INVALID'. {LENGTH_PROMPT}\n"
            f"Decide the next action:\n"
        )
    
    # 3. 处理LLM的响应 (LLM's Response)
    if "llm_raw_response" in content:
        messages.append({"role": "assistant", "content": content["llm_response"]})

    # 4. 处理奖励 (Reward)
    if "reward" in content:
        messages.append({"role": "user", "content": f"Reward:\n{content['reward']}\n"})
```        
- 回合编号：`\nTurn {idx + 1}:\n` 明确标记了对话的当前回合，帮助 LLM 理解时间序列。
- 环境状态 (`State`)：当前回合的环境观测。对于 Sokoban，这是文本形式的网格布局。
- 剩余动作数：`You have {content['actions_left']} actions left`. 提示 LLM 当前回合的行动限制，有助于长期规划。
- 强制输出格式：通常是[Your thoughts] [your answer]（如果 `enable_think` = true）或  [your answer]，这强制 LLM 以结构化的方式返回其思考和最终行动。
- `LENGTH_PROMPT`：提示 LLM 响应的最大长度。 
- `Strictly follow this format...`：强调格式的重要性，并警告格式不符将导致响应无效。
- LLM 响应 (`Assistant` 角色)：LLM 在前一回合生成的动作，以 `Assistant` 消息的形式添加到历史中。 
- 奖励 (`User` 角色)：LLM 前一行动的奖励反馈，以 `User` 消息的形式添加到历史中。这提供了 RL 信号。

#### Sokoban 示例：回合式 Prompt 构造

假设环境初始状态是：
```
#####
#__O#  <- 目标O
#P_X#  <- 玩家P，箱子X
#___#
#####
```

1. 回合 1（LLM 首次接收 Prompt）

在 LLM 生成第一个动作之前，它接收到的 Prompt 可能是这样（简化格式，实际会经过 `apply_chat_template` 转换）：
```
<|im_start|>system
You're a helpful assistant. You are a good game player. You are aiming to get high reward in the game.<|im_end|>
<|im_start|>user
You are solving the Sokoban puzzle. You are the player and you need to push all boxes to targets. When you are right next to a box, you can push it by moving in the same direction. You cannot push a box through a wall, and you cannot pull a box. The answer must be one of action in a turn, format is <answer>Right</answer>

The meaning of each symbol in the state is:
#: wall, _: empty, O: target, √: box on target, X: box, P: player, S: player on target

Your available actions are:
Up, Down, Left, Right

Turn 1:
State:
#####
#__O#  
#P_X#  
#___#
#####
You have 100 actions left. Always output: <answer> [your answer] </answer> with no extra text. Strictly follow this format, history response that do not follow the format will be set as 'INVALID'. Max response length: 100 words (tokens).
Decide the next action:<|im_end|>
<|im_start|>assistant
```
LLM 可能会生成 `<answer>Right</answer>`

2. 回合 2（LLM 接收到新的状态和奖励）

假设 LLM 选择了 Right，环境响应后，箱子被向右推了一格，奖励为 -0.1。新的状态是：
```
#####
#__O#
#_PX#
#___#
#####
```

此时，LLM 将接收到包含第一回合所有交互的 Prompt：
```
<|im_start|>system
You're a helpful assistant. You are a good game player. You are aiming to get high reward in the game.<|im_end|>
<|im_start|>user
You are solving the Sokoban puzzle. You are the player and you need to push all boxes to targets. When you are right next to a box, you can push it by moving in the same direction. You cannot push a box through a wall, and you cannot pull a box. The answer must be one of action in a turn, format is <answer>Right</answer>

The meaning of each symbol in the state is:
#: wall, _: empty, O: target, √: box on target, X: box, P: player, S: player on target

Your available actions are:
Up, Down, Left, Right

Turn 1:
State:
#####
#__O#  
#P_X#  
#___#
#####
You have 100 actions left. Always output: <answer> [your answer] </answer> with no extra text. Strictly follow this format, history response that do not follow the format will be set as 'INVALID'. Max response length: 100 words (tokens).
Decide the next action:<|im_end|>
<|im_start|>assistant
<answer>Right</answer><|im_end|>
<|im_start|>user
Reward:
-0.1
<|im_end|>
<|im_start|>user
Turn 2:
State:
#####
#__O#
#_PX#
#___#
#####
You have 99 actions left. Always output: <answer> [your answer] </answer> with no extra text. Strictly follow this format, history response that do not follow the format will be set as 'INVALID'. Max response length: 100 words (tokens).
Decide the next action:<|im_end|>
<|im_start|>assistant
```
这样，LLM 每次都能看到完整的对话历史，包括它自己的决策和环境的反馈，这对于学习和长程规划至关重要。

### 步骤 3：应用聊天模板并最终生成 Prompt 文本

最后一步是将构建好的messages列表转换成 LLM 实际接受的单个字符串形式的 Prompt。
```python
# 应用聊天模板，生成最终Prompt文本
if self.processor: # 对于多模态模型使用ProcessorMixin
    text = self.processor.apply_chat_template(messages, add_generation_prompt=(not prepare_for_update), tokenize=False)
else: # 对于纯文本模型使用PreTrainedTokenizer
    text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=(not prepare_for_update), tokenize=False)

# 强制LLM生成特定起始词（推理模式下）
if not prepare_for_update:
    if self.pipeline_config.enable_think:
        text += "<think>"  # 强制LLM在回答前思考
    else:
        text += "<answer>" # 强制LLM直接回答

# 清理特殊Token
text = text.replace("<|im_end|>\n", "<|im_end|>")
```
- `apply_chat_template`：这是 Hugging Face transformers 库提供的一个方法，它会根据所用 LLM 的特定格式（例如 Qwen 的 `<|im_start|>role\ncontent<|im_end|>` 结构），将 messages 列表转换为一个扁平的字符串。
- `add_generation_prompt`：在推理模式下（not prepare_for_update），这个参数通常会向 Prompt 末尾添加一个特殊 token，如 `<|im_start|>assistant\n`，明确告诉 LLM 现在轮到它作为 assistant 角色进行生成了。
- 强制生成起始词：在 LLM 进行推理（生成响应）时，为了确保其输出严格遵循预设的格式，我们会在 Prompt 的末尾额外添加一个特定的起始标记，例如 `<think>` 或 `<answer>`。这是一种被称为“Prompt 注入（Prompt Injection）”或“条件生成（Conditional Generation）”的技术。
    - 引导 LLM 续写：LLM 的本质是预测给定文本序列的下一个最可能的 token。当我们把 `<answer>` 这样的标记放在 Prompt 的末尾时，LLM 会将其视为一个未完成的序列，并自然而然地尝试续写这个标记之后的内容。
    - 强制格式遵循：如果 Prompt 中已经明确规定了响应必须是 `<answer>[your answer]</answer>` 这样的格式，那么通过在 Prompt 末尾放置 `</answer> `，我们实际上是在预填充了响应格式的一部分。LLM 接收到这个不完整的格式后，就会被“强制”引导去生成`[your answer]`的部分，并最终续写`</answer>` 。

## 总结：Prompt 生成的完整流程

1. **环境配置**（`SokobanEnvConfig`）：定义了环境的静态信息（指令、符号含义、动作名称）
2. `_init_prefix_lookup`：在 `EnvManager` 初始化时，将环境配置中的静态信息组合成 `first_user_content`
3. `_format_messages`：
    a. 开始一个新回合或接收到新的环境反馈时被调用。
    b. 将 `System Prompt` 和 `first_user_content` 作为对话的开端。
    c. 遍历 `env_output['history']`，依次添加回合编号、环境状态、剩余动作数、LLM 历史响应、奖励等动态信息。
    d. 在每个回合的环境状态之后，重复强制性格式要求。
    e. 使用 `tokenizer.apply_chat_template` 将构建好的结构化 `messages` 列表转化为 LLM 可接受的最终 Prompt 字符串。
    f. 在推理时，额外添加强制生成起始词 `<think>` 或 `<answer>`。
4. **LLM 接收 Prompt**：接收到这个精心构造的 Prompt 字符串，进行推理并生成响应。

通过这种分层、结构化且动态的 Prompt 生成机制，我们的框架能够有效地将复杂环境与 LLM 的强大语言能力相结合，使其能够理解任务、感知环境、学习规则并执行复杂操作。
