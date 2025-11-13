# RLVR Pipeline

RLVR Pipeline (Reinforcement Learning with Verifiable Rewards Pipeline) is a core component in the ROLL framework, specifically designed as an efficient distributed training pipeline for large language model reinforcement learning. Through virtual reward mechanisms, this pipeline can significantly improve LLM performance on key tasks such as complex reasoning, code generation, and mathematical calculations.

In the field of artificial intelligence, Reinforcement Learning with Verifiable Rewards (RLVR) is an innovative training method that uses verifiable, rule-based reward functions to provide models with clear binary feedback (1 for correct, 0 for incorrect), thereby optimizing their performance. Unlike traditional Reinforcement Learning from Human Feedback (RLHF), RLVR avoids dependence on subjective human evaluation or complex reward models, making the training process more transparent and efficient. This method is particularly suitable for tasks with clear correctness standards, such as mathematical reasoning and code generation.

## Core Advantages

*   **Diverse Task Support**: RLVR Pipeline has built-in support for multiple task types, including mathematical reasoning, code generation, LLM judgment, and instruction following, each equipped with dedicated reward evaluation mechanisms. `MathRuleRewardWorker` automatically evaluates the correctness of mathematical problems, `CodeSandboxRewardWorker` verifies program correctness through code execution, and `LLMJudgeRewardWorker` performs quality assessment of open-ended questions. The flexible extension interface design makes integration of new task types simple and direct.
    
*   **Multi-Task Joint Training**: Supports simultaneous optimization across domains, achieving collaborative improvement of models in multiple fields such as mathematics, programming, and general reasoning. Precise control of data sampling ratios for each domain through `domain_interleave_probs`, with independent reward processing strategies and weight coefficients configurable for each domain, avoiding the capability limitations that single-task training might cause.
    
*   **Algorithm-Friendly Reinforcement Learning Framework**: Provides multiple reinforcement learning strategy options, supporting various cutting-edge algorithms such as PPO, GRPO, Reinforce, and TOPR. Rich reward processing strategies include reward normalization, reward clipping, reward scaling, etc., multiple advantage estimation methods, and flexible loss function configuration, enabling researchers to easily experiment with different algorithm combinations.
    
*   **Comprehensive Performance Monitoring**: A fine-grained metrics tracking system provides comprehensive training process monitoring, tracking performance metrics at both group and batch levels, statistically displaying performance metrics by task domain, as well as system metrics such as GPU usage, memory occupation, and training throughput, providing comprehensive visualization and analysis functions for the model training process**.**
    
*   **Efficient Distributed Computing**: A distributed training architecture based on the [Ray](https://www.ray.io/) framework, intelligently allocating different types of worker nodes through heterogeneous task scheduling, dynamically managing resources by automatically adjusting resource allocation based on task load, executing generation, reward calculation, and model update phases in parallel, and featuring automatic fault recovery mechanisms, fully utilizing the computing power of modern GPU clusters.
    
## Main Attributes

#### Core Configuration

*   pipeline_config: The core configuration object of the RLVRPipeline class, of type RLVRConfig, containing all configuration parameters for the entire reinforcement learning training pipeline.

#### Actor-Critic Architecture Clusters

*   actor_train: The policy network training cluster in RLVRPipeline, responsible for executing the core training logic of the PPO algorithm.
*   actor_infer: The policy network inference cluster in RLVRPipeline, responsible for generating responses.
*   reference: The reference model cluster in RLVRPipeline, serving as a baseline model in the policy optimization process for calculating KL divergence.
*   critic (optional): Estimates the state value function (only used in GAE mode)
*   reward: The policy network reward cluster in RLVRPipeline, responsible for calculating reward scores for generated responses, supporting multi-domain and multi-type reward calculation:
    
    *   Mathematical rule rewards (`MathRuleRewardWorker`): Evaluating the correctness of mathematical reasoning
    *   Code sandbox rewards (`CodeSandboxRewardWorker`): Evaluating code by executing it and verifying its output
    *   LLM judgment rewards (`LLMJudgeRewardWorker`): Using another LLM as an evaluator to assess the quality of generated answers

#### Data-Related Attributes

*   domain_datasets: `Dict[str, datasets.Dataset]`, a dictionary of training datasets grouped by domain
*   val_dataset: Validation dataset
*   domain_batch_size: Batch size configuration for each domain, allocating batch sizes for each domain according to `domain_interleave_probs`

#### Scheduler Attributes

*   generate_schedulers: `Dict[str, DynamicSamplingScheduler]`, dynamic sampling schedulers for each domain
*   val_generate_scheduler: Generation scheduler for the validation phase

#### Controllers and Auxiliary Tools

*   kl_ctrl: Adaptively adjusts the KL penalty coefficient to prevent the policy update from deviating too far from the reference policy
*   tokenizer: Handles text encoding and decoding
*   running: Calculates and maintains runtime statistics

## Core Process

```python
def run():
    Initialize TPS timer and metrics manager
    for global_step in range(max_steps):
        # 1. Model state management
        Update model parameters (actor_train -> actor_infer)
        # 2. Evaluation phase (executed every eval_steps)
        if val_dataset and global_step % eval_steps == 0:
            batch = Validation environment rollout(len(val_dataset))
            Calculate evaluation metrics (accuracy statistics grouped by tag)
        # 3. Training data collection
        Start inference server and reward cluster
        for domain in domains:
            domain_batches[domain] = Scheduler.get_batch(domain_batch_size[domain])
        batch = Merge all domain batches
        Stop inference server and reward cluster    
        # 4. Calculate key probabilities and values  
        ref_log_probs = Reference model.calculate log probabilities(batch)
        old_log_probs = Actor_train model.calculate log probabilities(batch)
        if using GAE estimator:
            values = Critic model.calculate value function(batch)     
        # 5. Reward processing and advantage calculation (processed by domain grouping)
        for domain, domain_batch in batch.group_by("domain"):
            Get sample_level_mask
            Normalize reward scores by group (running_moments)
            Apply KL penalty (kl_controller)
            Calculate token-level rewards
            Calculate advantage function (GAE or other methods)
        Re-merge and sort batches
        # 6. Model training
        if using GAE estimator:
            Critic model.training step(batch)
        if global_step > critic_warmup:
            Actor model.training step(batch)
        # 7. Record and save
        Update TPS metrics
        Record training metrics (grouped by domain)
        Save checkpoints and scheduler state
        Print sample logs (every logging_steps)
        Record to tracker     
```

#### model_update

Synchronize training model parameters to the inference model to ensure that the inference model used for generating rollout data uses the latest training parameters. In the PPO algorithm, the training model actor_train is responsible for parameter updates and gradient calculations, while the inference model actor_infer is responsible for generating rollout data. To ensure training consistency, the inference model needs to periodically synchronize the latest training data, so that the generated rollout data can reflect the true performance of the current policy.

```python
# Initialize phase to set synchronization pairs
self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
      frequency=self.pipeline_config.actor_train.model_update_frequency,)

# Execute synchronization in training loop
model_update_metrics: Dict = self.model_update(global_step)
metrics_mgr.add_metrics(model_update_metrics)
```

#### step_generate

Training data collection adopts a multi-domain parallel generation architecture. Start the inference server and reward calculation cluster, then generate corresponding size data batches in parallel for each training domain according to the configuration ratio. Each domain samples prompts from its respective dataset through an independent scheduler and generates responses using the actor model, while simultaneously calculating corresponding reward scores. Finally, all domain batches are merged into a complete training batch, and inference resources are cleaned up, completing one round of training data collection.

```python
# Start batch generation in parallel for each domain
for domain, scheduler in self.generate_schedulers.items():
    scheduler.get_batch.remote(...)
# Collect results from all domains
domain_batches = {}
for domain, scheduler_ref in scheduler_refs.items():
    domain_batch = ray.get(scheduler_ref, timeout=rpc_timeout)
    domain_batches[domain] = domain_batch
# Merge all domain batches
generate_output = DataProto.concat([domain_batch for domain_batch in domain_batches.values()])
```

#### cal_ref_log_probs

`reference.compute_log_probs` calculates the log probabilities of the reference model for the current batch data. Used for subsequent KL divergence penalty calculation to prevent the training strategy from deviating too far from the initial strategy.

#### cal_old_log_probs_values

Calculate the log probabilities (old policy probabilities) and value function estimates of the current training model for rollout data, which is a key step in the PPO algorithm for calculating the importance sampling ratio. actor_train.compute_log_probs uses the current training model to calculate the log probabilities of rollout data. critic.compute_values, if using GAE, simultaneously calculates the state value function.

```python
if self.pipeline_config.adv_estimator == "gae":
  self.critic.compute_values(batch, blocking=False)
self.actor_train.compute_log_probs(batch, blocking=False)
```

#### adv

The core data processing module in the RLVR training pipeline, mainly responsible for preprocessing the response data generated by the model before reinforcement learning training. The code first assigns unique identifiers to each sample and groups them by task domain, then performs four key steps for each domain's data: (1) `get_sample_level_mask` applies sample-level mask strategies to filter unsuitable samples; (2) `reward_postprocess` post-processes and normalizes reward signals; (3) `compute_token_reward` assigns response-level rewards to token level and combines with KL divergence control to prevent excessive model deviation; (4) `compute_advantage` calculates advantage function values using GAE or other methods for PPO algorithm policy updates. Finally, `DataProto.concat` merges all domain processing results and restores the original order. This domain-grouping design allows different task types (such as mathematical reasoning, code generation, etc.) to use their most suitable processing strategies, thereby improving the training effectiveness and stability of multi-domain reinforcement learning. The entire process also includes detailed performance monitoring and metrics collection to ensure the observability of the training process.

#### step_train

The model training execution phase in the RLVR training pipeline, responsible for coordinating the parameter update process of Actor-Critic networks. Implements a critic warm-up mechanism - only starts updating the actor network when the training step count exceeds the warm-up threshold, designed to let the critic first stabilize learning the value function before policy updates. The entire training process uses the Ray framework for distributed asynchronous execution to improve efficiency, while monitoring training time through timers, and collecting training metrics (such as loss values, gradient norms, etc.) from both networks to add to the monitoring system.

```python
if self.pipeline_config.adv_estimator == "gae":
    self.critic.train_step(batch, blocking=False)

with actor_train_timer:
    # Critic warm-up
    if self.pipeline_config.critic_warmup <= global_step:
        # Update actor network
        actor_train_metrics_refs = self.actor_train.train_step(batch, blocking=False)

```

#### do_checkpoint

Save checkpoints

#### tracker.log

Generate text sample logs

```python
def val():
    Initialize validation metrics manager
    # 1. Validation data generation
    Create empty batch, set validation generation configuration
    Start inference server (actor_infer)
    Load reward cluster state (all reward_clusters)
    # 2. Validation environment rollout
    batch = Validation scheduler.get_batch(entire validation dataset size)
    # 3. Clean up inference resources
    Stop inference server
    Unload reward cluster state
    # 4. Calculate validation metrics
    Calculate overall accuracy (proportion of scores == 1)
    Record global validation metrics (val_correct/all/mean)
    # 5. Group statistics by tag
    grouped_batch = batch.group_by("tag") 
    for tag, group_batch in grouped_batch:
        Calculate accuracy for this tag
        Record grouped validation metrics (val_correct/{tag}/mean)
        Print grouped results
    # 6. Return validation results
    return Validation metrics dictionary
```

The val function is the validation evaluation function in the RLVR pipeline, mainly responsible for periodically evaluating model performance on the validation set during training. The val function first starts the inference server and loads the reward model state, then performs batch generation and scoring on the entire validation dataset through the validation scheduler, calculates overall validation accuracy (by judging whether scores equal 1), groups validation data by different tags to statistically calculate average accuracy for each group, and finally returns a result dictionary containing overall and grouped validation metrics, providing quantitative evaluation of model performance for the training process, helping monitor training effectiveness and adjust training strategies.