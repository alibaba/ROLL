# RLVR Pipeline

RLVR Pipeline (Reinforcement Learning with Verifiable Rewards Pipeline) 是 ROLL 框架中的核心组件，专门为大型语言模型的强化学习训练而设计的高效分布式训练管道。该管道通过虚拟奖励机制，能够显著提升LLM在复杂推理、代码生成、数学计算等关键任务上的性能表现。

在人工智能领域，强化学习与可验证奖励（RLVR）作为一种创新的训练方法，通过使用可验证的、基于规则的奖励函数，为模型提供明确的二元反馈（正确为1，错误为0），从而优化其性能。与传统的强化学习从人类反馈（RLHF）不同，RLVR避免了主观人类评估或复杂奖励模型的依赖，使训练过程更加透明、高效。这种方法特别适用于数学推理、代码生成等具有明确正确性标准的任务。

## 核心优势

*   **多样化任务支持**：RLVR Pipeline 内置支持多种任务类型，包括数学推理、代码生成、LLM评判和指令跟随，每种任务都配备了专门的奖励评估机制。通过 `MathRuleRewardWorker` 自动评估数学问题的正确性，使用 `CodeSandboxRewardWorker` 通过代码执行验证程序正确性，利用 `LLMJudgeRewardWorker` 进行开放性问题的质量评估。灵活的扩展接口设计使得新任务类型的集成变得简单直接。
    
*   **多任务联合训练**：支持跨领域的同时优化，实现模型在数学、编程、通用推理等多个领域的协同提升。通过 `domain_interleave_probs` 精确控制各领域数据的采样比例，每个领域可配置独立的奖励处理策略和权重系数，避免了单一任务训练可能导致的能力局限。
    
*   **算法友好的强化学习框架**：提供多种强化学习策略选项，支持 PPO、GRPO、Reinforce、TOPR等多种前沿算法。丰富的奖励处理策略包括奖励标准化、奖励裁剪、奖励缩放等，多种优势估计方法，以及灵活的损失函数配置，使研究人员能够轻松实验不同的算法组合。
    
*   **全面的性能监控**：细粒度的指标追踪系统提供全方位的训练过程监控，同时追踪组级别和批次级别的性能指标，按任务领域分别统计和展示性能指标，以及 GPU 使用率、内存占用、训练吞吐量等系统指标，为模型训练过程提供全面的可视化和分析功能**。**
    
*   **高效的分布式计算**：基于 [Ray](https://www.ray.io/) 框架实现的分布式训练架构，通过异构任务调度智能分配不同类型的工作节点，动态资源管理根据任务负载自动调整资源分配，生成、奖励计算、模型更新等阶段并行执行，以及节点故障自动恢复的容错机制，充分利用现代 GPU 集群的计算能力。
    

## 主要属性

####  核心配置
    

*   pipeline\_config: RLVRPipeline类的核心配置对象，类型为RLVRConfig，包含了整个强化学习训练管线的所有配置参数。
    

####  Actor-Critic架构集群
    

*   actor\_train: RLVRPipeline中的策略网络训练集群，负责执行PPO算法的核心训练逻辑。
    
*   actor\_infer: RLVRPipeline中的策略网络推理集群，负责生成响应。
    
*   reference: RLVRPipeline中的参考模型集群，作为策略优化过程中的基准模型，用于计算KL散度。
    
*   critic(optional): 估计状态价值函数(仅在GAE模式下使用)
    
*   reward：RLVRPipeline中的策略网络奖励集群，负责为生成的响应计算奖励分数，支持多领域、多类型的奖励计算：
    
    *   数学规则奖励（`MathRuleRewardWorker`）：评估数学推理的正确性
        
    *   代码沙箱奖励（`CodeSandboxRewardWorker`）：通过执行代码并验证其输出来评估代码
        
    *   LLM判断奖励（`LLMJudgeRewardWorker`）：使用另一个 LLM 作为评估者来评估所生成答案的质量
        

####   数据相关属性
    

*   domain\_datasets: `Dict[str, datasets.Dataset]`, 按领域分组的训练数据集字典
    
*   val\_dataset: 验证数据集
    
*   domain\_batch\_size: 各个领域对应的批次大小配置，根据`domain_interleave_probs`分配各领域的批次大小
    

####   调度器属性
    

*   generate\_schedulers: `Dict[str, DynamicSamplingScheduler]`，各领域的动态采样调度器
    
*   val\_generate\_scheduler: 验证阶段的生成调度器
    

####   控制器和辅助工具
    

*   kl\_ctrl: 自适应调整KL惩罚系数，防止策略更新偏离参考策略过远
    

*   tokenizer: 处理文本的编码和解码
    

*   running: 计算和维护运行时的统计信息
    

## 核心流程

```python
def run():
    初始化 TPS 计时器和指标管理器
    for global_step in range(max_steps):
        # 1. 模型状态管理
        更新模型参数 (actor_train -> actor_infer)
        # 2. 评估阶段 (每隔eval_steps执行)
        if val_dataset and global_step % eval_steps == 0:
            batch = 验证环境rollout(len(val_dataset))
            计算评估指标 (accuracy按tag分组统计)
        # 3. 训练数据收集
        启动推理服务器和奖励集群
        for domain in domains:
            domain_batches[domain] = 调度器.get_batch(domain_batch_size[domain])
        batch = 合并所有domain的batches
        停止推理服务器和奖励集群    
        # 4. 计算关键概率和值  
        ref_log_probs = reference模型.计算对数概率(batch)
        old_log_probs = actor_train模型.计算对数概率(batch)
        if 使用GAE估计器:
            values = critic模型.计算值函数(batch)     
        # 5. 奖励处理和优势计算 (按domain分组处理)
        for domain, domain_batch in batch.group_by("domain"):
            获取sample_level_mask
            按组归一化奖励分数 (running_moments)
            应用KL惩罚 (kl_controller)
            计算token级奖励
            计算优势函数 (GAE或其他方法)
        重新合并并排序batches
        # 6. 模型训练
        if 使用GAE估计器:
            critic模型.训练步骤(batch)
        if global_step > critic_warmup:
            actor模型.训练步骤(batch)
        # 7. 记录和保存
        更新TPS指标
        记录训练指标 (按domain分组)
        保存检查点和调度器状态
        打印样本日志 (每隔logging_steps)
        记录到tracker     
```

#### model\_update

同步训练模型参数到推理模型，确保用于生成rollout数据的推理模型使用最新的训练参数。在PPO算法中，训练模型actor\_train负责参数更新和梯度计算，推理模型actor\_infer负责生成rollout数据，为了确保训练一致性，推理模型需要定期同步最新的训练数据，这样生成的rollout数据才能反应当前策略的真实表现。

```python
#初始化阶段设置同步对
self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
      frequency=self.pipeline_config.actor_train.model_update_frequency,)

#在训练循环中执行同步
model_update_metrics: Dict = self.model_update(global_step)
metrics_mgr.add_metrics(model_update_metrics)
```

#### step\_generate

训练数据收集采用多Domain并行生成的架构。启动推理服务器和奖励计算集群，然后为每个训练域(domain)按配置比例并行生成对应大小的数据batch，每个域通过独立的调度器从各自的数据集中采样prompts并用actor模型生成responses，同时计算对应的奖励分数，最后将所有域的batches合并成一个完整的训练batch，并清理推理资源，完成一轮训练数据的收集过程。

```python
#为每个domain并行启动batch生成
for domain, scheduler in self.generate_schedulers.items():
    scheduler.get_batch.remote(...)
# 收集所有domain的结果
domain_batches = {}
for domain, scheduler_ref in scheduler_refs.items():
    domain_batch = ray.get(scheduler_ref, timeout=rpc_timeout)
    domain_batches[domain] = domain_batch
# 合并所有domain的batches
generate_output = DataProto.concat([domain_batch for domain_batch in domain_batches.values()])
```

#### cal\_ref\_log\_probs

`reference.compute_log_probs`计算reference model对当前batch数据的对数概率。用于后续的KL散度惩罚计算，防止训练策略偏离初始策略太远。

#### cal\_old\_log\_probs\_values

计算当前训练模型对rollout数据的对数概率（旧策略概率）和值函数估计，是PPO算法中计算重要性采样比率(importance sampling ratio)的关键步骤。其中，actor\_train.compute\_log\_probs，使用当前训练模型计算对rollout数据的对数概率。critic.compute\_values，如果使用GAE，同时计算状态值函数。

```python
if self.pipeline_config.adv_estimator == "gae":
  self.critic.compute_values(batch, blocking=False)
self.actor_train.compute_log_probs(batch, blocking=False)
```

#### adv

RLVR训练流水线中的核心数据处理模块，主要负责对模型生成的响应数据进行强化学习训练前的预处理。代码首先为每个样本分配唯一标识符并按任务域（domain）进行分组，然后对每个域的数据执行四个关键步骤：（1）`get_sample_level_mask`应用样本级别掩码策略过滤不合适的样本；（2）`reward_postprocess`对奖励信号进行后处理和标准化；（3）`compute_token_reward`将响应级别的奖励分配到词元级别，并结合KL散度控制防止模型偏离过多；（4）`compute_advantage`使用GAE等方法计算优势函数值用于PPO算法的策略更新。最后`DataProto.concat`将所有域的处理结果合并并恢复原始顺序。这种按域分组的设计允许不同任务类型（如数学推理、代码生成等）使用各自最适合的处理策略，从而提高多域强化学习的训练效果和稳定性。整个过程还包含详细的性能监控和指标收集，确保训练过程的可观测性。

#### step\_train

RLVR训练流水线中的模型训练执行阶段，负责协调Actor-Critic网络的参数更新过程。实现了critic预热机制——只有当训练步数超过预热阈值时才开始更新actor网络，这样设计是为了让critic先稳定学习价值函数再进行策略更新。整个训练过程采用Ray框架进行分布式异步执行以提高效率，同时通过计时器监控训练时间，并收集两个网络的训练指标（如损失值、梯度范数等）添加到监控系统中。

```python
if self.pipeline_config.adv_estimator == "gae":
    self.critic.train_step(batch, blocking=False)

with actor_train_timer:
    #critic预热
    if self.pipeline_config.critic_warmup <= global_step:
        #更新actor网络
        actor_train_metrics_refs = self.actor_train.train_step(batch, blocking=False)

```

#### do\_checkpoint

保存检查点

#### tracker.log

生成文本示例日志

```python
def val():
    初始化验证指标管理器
    # 1. 验证数据生成
    创建空batch，设置验证生成配置
    启动推理服务器 (actor_infer)
    加载奖励集群状态 (所有reward_clusters)
    # 2. 验证环境rollout
    batch = 验证调度器.get_batch(整个验证数据集大小)
    # 3. 清理推理资源
    停止推理服务器
    卸载奖励集群状态
    # 4. 计算验证指标
    计算整体准确率 (scores == 1的比例)
    记录全局验证指标 (val_correct/all/mean)
    # 5. 按tag分组统计
    grouped_batch = batch.group_by("tag") 
    for tag, group_batch in grouped_batch:
        计算该tag的准确率
        记录分组验证指标 (val_correct/{tag}/mean)
        打印分组结果
    # 6. 返回验证结果
    return 验证指标字典
```

val函数是RLVR流水线中的验证评估函数，主要功能是在训练过程中定期评估模型在验证集上的表现。val函数首先启动推理服务器并加载奖励模型状态，然后通过验证调度器对整个验证数据集进行批量生成和评分，接着计算整体验证准确率（通过判断得分是否等于1），并将验证数据按照不同标签（tag）进行分组统计各组的平均正确率，最后返回包含整体和分组验证指标的结果字典，为训练过程提供模型性能的量化评估，帮助监控训练效果和调整训练策略。