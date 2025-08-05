# 自定义Reward Worker

## Reward核心概念

### 强化学习中的奖励函数（Reward Function）

在强化学习（Reinforcement Learning, RL）中，奖励函数是指导智能体（Agent）学习的关键机制，它决定了智能体（Agent）在与环境交互过程中“好”与“坏”行为的判断标准。环境根据状态和智能体采取的动作，产生一个标量信号作为奖励反馈，这个标量信号衡量智能体这一轮动作的好坏。

### RLVR Pipeline中的Reward Worker

在ROLL框架中的RLVR pipeline（Reinforcement Learning with Verifiable Rewards Pipeline）中，Reward workers通过使用可验证的、基于规则的奖励函数，为模型提供明确的二元反馈（正确为1，错误为0），从而优化其性能。RLVR pipeline支持针对不同领域的各种奖励机制：

*   MathRuleRewardWorker – 评估数学推理的正确性和步骤。
    
*   CodeSandboxRewardWorker – 通过执行代码并验证其输出来评估代码生成。
    
*   LLMJudgeRewardWorker – 使用另一个 LLM 作为评判者来评估生成答案的质量。
    
*   GeneralValRuleRewardWorker – 评估通用验证任务的答案正确性。
    
*   CrossThinkQARuleRewardWorker – 评估跨思维问答任务的推理质量。
    
*   MultipleChoiceBoxedRuleRewardWorker – 评估多选题答案的格式和正确性。
    
*   GeneralRuleRewardWorker (IFEval) – 基于IFEval评估文本生成任务的约束满足度。
    
*   DetectionRewardWorker – 评估视觉检测任务的目标检测准确性。
    

## Reward Worker核心功能

RewardWorker继承自基类Worker，Worker基类提供了：分布式计算支持，如rank管理、多GPU并行；网络通信，通过Ray分布式框架集成；状态管理，如模型加载、卸载、检查点；统一的日志记录；配置管理等。RewardWorker继承Worker，一个标准的RewardWorker通常需要实现以下功能：

*   initialize方法
    

*   初始化策略（如果需要）
    

*   compute\_rewards方法（核心方法）
    
    *   标准输入 `data: DataProto`
        
    *   解码响应文本：将模型输出的token ID序列转换回可读的文本`self.tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=True)`
        
    *   获取ground truth
        
    *   计算奖励（通常应用自定义奖励函数）
        
    *   标准输出，必须要返回包含以下tensors的DataProto：
        

```python
output = DataProto.from_dict(
    tensors={
        "token_level_rewards": token_level_rewards, # token级别的奖励（形状与responses相同）
        "response_level_rewards": response_level_rewards, #response级别的奖励
        "scores": scores, # 用于日志记录的分数
    }
)
```

## 代码示例（自定义Reward Worker）

1.  类定义和初始化
    

```python
class CustomRewardWorker(Worker):
    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        # 设置rank信息
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        
        # 初始化tokenizer
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        
        # 初始化策略（如果需要模型推理）
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None
        
        # 自定义奖励函数
        self.custom_reward_fn = self._create_custom_reward_function()
```

2.  initialize方法
    

```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def initialize(self, pipeline_config):
    # 初始化策略（如果需要）
    if self.strategy:
        self.strategy.initialize(model_provider=default_reward_model_provider)
        self.strategy.offload_states()
    #如果不需要，可以直接pass
```

3.  compute\_rewards方法
    

```python
@register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
def compute_rewards(self, data: DataProto):
    """
    核心奖励计算方法
    """
    # 1. 解码响应文本
    response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=False)
    
    # 2. 获取ground truth
    ground_truths = data.non_tensor_batch["ground_truth"]
    
    # 3. 计算奖励
    rewards = []
    for response, ground_truth in zip(response_text_list, ground_truths):
        reward = self._compute_single_reward(response, ground_truth)
        rewards.append(reward)
    
    # 4. 返回结果
    token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
    response_level_rewards = torch.tensor(rewards, dtype=torch.float16)
    
    return DataProto.from_dict(
        tensors={
            "token_level_rewards": token_level_rewards,
            "response_level_rewards": response_level_rewards,
            "scores": response_level_rewards,
        }
    )
```

4.  自定义奖励函数
    

```python
def _compute_single_reward(self, response: str, ground_truth: str) -> float:
    """
    实现具体的奖励计算逻辑
    """
    # 清理响应文本
    response = response.replace("<|endoftext|>", "").replace("<|im_end|>", "").replace("<pad>", "")
    
    # 实现你的奖励计算逻辑
    reward = 0.0
    
    # 示例：检查答案正确性
    if self._check_answer_correctness(response, ground_truth):
        reward += 1.0
    
    # 示例：格式奖励
    if self._check_format_correctness(response):
        reward += 0.5
    
    # 示例：重复惩罚
    repetition_penalty = self._compute_repetition_penalty(response)
    reward += repetition_penalty
    
    return reward

def _check_answer_correctness(self, response: str, ground_truth: str) -> bool:
    # 实现答案正确性检查
    pass

def _check_format_correctness(self, response: str) -> bool:
    # 实现格式正确性检查
    pass

def _compute_repetition_penalty(self, response: str) -> float:
    # 实现重复惩罚计算
    pass
```

装饰器说明：

@register(dispatch\_mode=Dispatch.ONE\_TO\_ALL)

    用于初始化等只需要执行一次的方法

    所有worker都会执行，但只有第一个worker的结果有效

@register(dispatch\_mode=Dispatch.DP\_MP\_COMPUTE, clear\_cache=False)

    用于计算密集型方法（如compute\_rewards）

    支持数据并行和模型并行

    clear\_cache=False表示不清除缓存，提高性能

## 总结

自定义Reward Worker的关键要素：

1.  继承Worker基类：获得分布式计算能力
    
2.  实现compute\_rewards方法：核心奖励计算逻辑
    
3.  使用正确的装饰器：确保分布式执行
    
4.  返回标准格式：符合ROLL框架要求
    
5.  添加错误处理和日志记录