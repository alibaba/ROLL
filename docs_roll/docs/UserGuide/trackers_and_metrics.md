# Trackers and Metrics

The ROLL framework supports multiple experiment tracking tools to help you monitor and analyze the training process. This document will provide detailed instructions on how to configure and use these trackers.

## Supported Trackers

The ROLL framework currently supports the following trackers:

1. **TensorBoard** - Visualization tool developed by Google
2. **Weights & Biases (WandB)** - Powerful machine learning experiment tracking platform
3. **SwanLab** - Next-generation AI experiment tracking tool
4. **Stdout** - Direct output to standard output

## Configuring Trackers

In the YAML configuration file, trackers are configured through the `track_with` and `tracker_kwargs` parameters:

```yaml
# Using TensorBoard
track_with: tensorboard
tracker_kwargs:
  log_dir: /path/to/tensorboard/logs

# Using Weights & Biases
track_with: wandb
tracker_kwargs:
  api_key: your_wandb_api_key
  project: your_project_name
  name: experiment_name
  notes: "Experiment description"
  tags:
    - tag1
    - tag2

# Using SwanLab
track_with: swanlab
tracker_kwargs:
  login_kwargs:
    api_key: your_swanlab_api_key
  project: your_project_name
  logdir: /path/to/swanlab/logs
  experiment_name: experiment_name
  tags:
    - tag1
    - tag2

# Using Stdout
track_with: stdout
```

## SwanLab Usage Details

### Configuring SwanLab

To use SwanLab in ROLL, configure as follows:

```yaml
track_with: swanlab
tracker_kwargs:
  login_kwargs:
    api_key: your_api_key  # Your SwanLab API key
  project: roll-experiments  # Project name
  logdir: ./swanlog  # Log storage directory
  experiment_name: ${exp_name}  # Experiment name, usually using the exp_name variable
  tags:  # Experiment tags
    - roll
    - rl
    - experiment
```

### Obtaining SwanLab API Key

1. Visit the [SwanLab website](https://swanlab.cn/)
2. Register or log in to your account
3. Go to the user settings page
4. Find the API key and copy it

## Metric Monitoring

The ROLL framework automatically records the following types of metrics:

## Algorithm Performance Metrics

### Validation Phase
- val/score/mean: Average score per episode during validation phase. Reflects the model's average performance on unseen environments.
- val/score/max / val/score/min: Maximum / minimum score per episode during validation phase.

### Value Related
- critic/lr: Learning rate of the value function (Critic). The learning rate is the step size for optimizer updates to model parameters.
- critic/loss: Loss between value network predictions and actual returns.
- critic/value: Mean of value network predictions for batch states at the beginning of data collection or training. These values are typically used as baselines when calculating advantage functions.
- critic/vpred: Mean of value network predictions for batch states in the current optimization. This value updates with training iterations.
- critic/clipfrac: Whether value function clipping (value_clip) was used and the proportion of clipping effectiveness.
- critic/error: Mean squared error between value network predictions and actual returns.

### Reward Related
- critic/score/mean: Mean of raw environment rewards.
- critic/score/max / critic/score/min: Maximum / minimum of raw environment rewards.
- critic/rewards/mean: Mean of normalized/clipped rewards.
- critic/rewards/max / critic/rewards/min: Maximum / minimum of normalized/clipped rewards.
- critic/advantages/mean: Mean of advantages. Reflects how much extra reward taking a specific action in a given state brings compared to the average level.
- critic/advantages/max / critic/advantages/min: Maximum / minimum of advantages.
- critic/returns/mean: Mean of returns. Expected cumulative rewards.
- critic/returns/max / critic/returns/min: Maximum / minimum of returns.
- critic/values/mean: Mean of value function (Value Function) estimates. Reflects the model's estimate of future total rewards for a state.
- critic/values/max / critic/values/min: Maximum / minimum of value function.
- tokens/response_length/mean: Average length of generated responses.
- tokens/response_length/max / tokens/response_length/min: Maximum / minimum length of generated responses.
- tokens/prompt_length/mean: Average length of prompts.
- tokens/prompt_length/max / tokens/prompt_length/min: Maximum / minimum length of prompts.

### Policy Related
- actor/lr: Learning rate of the current policy network (Actor). The learning rate is the step size for optimizer updates to model parameters.
- actor/ppo_ratio_high_clipfrac: High clipping ratio in PPO policy optimization.
- actor/ppo_ratio_low_clipfrac: Low clipping ratio in PPO policy optimization.
- actor/ppo_ratio_clipfrac: Clipping ratio in PPO policy optimization.
- actor/ratio_mean: Mean ratio of the policy network (Actor) (exponential of the ratio of new to old policy log probabilities).
- actor/ratio_max / actor/ratio_min: Maximum / minimum ratio of the policy network (Actor).
- actor/clipfrac: Clipping ratio of the policy network (Actor).
- actor/kl_loss: KL divergence penalty term between current policy and reference policy. Used to prevent the policy from deviating too far from the original model.
- actor/total_loss: Weighted sum of policy gradient loss, KL divergence loss, and entropy loss (if present). This is the actual loss used for model backpropagation.
- actor/approxkl: Approximate KL divergence between current policy and old policy. Measures the step size of each policy update.
- actor/policykl: Exact KL divergence between current policy and old policy.

### Evaluation Metrics
- critic/ref_log_prob/mean: Mean log probability output by the reference model. Used as a performance baseline for measuring old policy or reference policy.
- critic/old_log_prob/mean: Mean log probability output by the old policy (Actor before training). Used to measure differences between new and old policies.
- critic/entropy/mean: Mean entropy of the policy. Entropy measures the randomness or exploratory nature of the policy, with high entropy indicating stronger exploration.
- critic/reward_clip_frac: Proportion of reward clipping. Reflects how many reward values were clipped, and if too high, may require adjusting reward range or clipping thresholds.

#### PPO Loss Metrics
- actor/pg_loss: Policy gradient loss of the PPO algorithm. The goal is to minimize this loss to improve the policy.
- actor/weighted_pg_loss: Weighted value of policy gradient loss.
- actor/valid_samples: Number of valid samples in the current batch.
- actor/total_samples: Total number of samples in the current batch (i.e., batch size).
- actor/valid_sample_ratio: Proportion of valid samples in the current batch.
- actor/sample_weights_mean: Mean of all sample weights in the batch.
- actor/sample_weights_min / actor/sample_weights_max: Minimum / maximum of all sample weights in the batch.

#### SFT Loss Metrics
- actor/sft_loss: Supervised fine-tuning loss.
- actor/positive_sft_loss: Positive sample supervised fine-tuning loss.
- actor/negative_sft_loss: Negative sample supervised fine-tuning loss.

## Framework Performance Metrics

### Global System Metrics
- system/tps: Tokens processed per second. This is a key metric for measuring overall system throughput.
- system/samples: Total number of samples processed.

### Phase Duration Metrics
- time/rollout: Duration of the data collection (Rollout) phase.
- time/ref_log_probs_values_reward: Duration for computing reference model log probabilities and values.
- time/old_log_probs_values: Duration for computing old policy log probabilities and values.
- time/adv: Duration of the advantages calculation phase.

### Execution Phases
In the following time and memory metrics, `{metric_infix}` will be replaced with specific execution phase identifiers, such as:
- train_step: Training phase
- generate: Text generation/inference phase
- model_update: Model parameter update/synchronization phase
- compute_log_probs: Log probability computation phase
- do_checkpoint: Model saving/checkpoint phase
- compute_values: Value computation phase
- compute_rewards: Reward computation phase

#### Time Metrics
- time/`{metric_infix}`/total: Total execution time for the entire operation (from entering state_offload_manager to exiting).
- time/`{metric_infix}`/execute: Execution time for actual business logic (i.e., the yield part, such as model training, generation, etc.).
- time/`{metric_infix}`/onload: Time to load model state (strategy.load_states()) to GPU or memory.
- time/`{metric_infix}`/offload: Time to offload model state (strategy.offload_states()) from GPU or memory.

#### GPU Memory Metrics
- Memory snapshot at the beginning (after model state offloading)
    - memory/`{metric_infix}`/**start/offload**/allocated/`{device_id}`: Currently allocated GPU memory on a specific device_id.
    - memory/`{metric_infix}`/**start/offload**/reserved/`{device_id}`: Currently reserved GPU memory on a specific device_id.
    - memory/`{metric_infix}`/**start/offload**/max_allocated/`{device_id}`: Peak allocated GPU memory from the start of this operation to the current moment on a specific device_id.
    - memory/`{metric_infix}`/**start/offload**/max_reserved/`{device_id}`: Peak reserved GPU memory from the start of this operation to the current moment on a specific device_id.
- Memory snapshot after loading model state (before executing business logic)
    - memory/`{metric_infix}`/**start/onload**/allocated/`{device_id}`: Currently allocated GPU memory on a specific device_id.
    - memory/`{metric_infix}`/**start/onload**/reserved/`{device_id}`: Currently reserved GPU memory on a specific device_id.
    - memory/`{metric_infix}`/**start/onload**/max_allocated/`{device_id}`: Peak allocated GPU memory from the start of this operation to the current moment on a specific device_id.
    - memory/`{metric_infix}`/**start/onload**/max_reserved/`{device_id}`: Peak reserved GPU memory from the start of this operation to the current moment on a specific device_id.
- Memory snapshot after executing business logic (before offloading model state)
    - memory/`{metric_infix}`/**end/onload**/allocated/`{device_id}`: Currently allocated GPU memory on a specific device_id.
    - memory/`{metric_infix}`/**end/onload**/reserved/`{device_id}`: Currently reserved GPU memory on a specific device_id.
    - memory/`{metric_infix}`/**end/onload**/max_allocated/`{device_id}`: Peak allocated GPU memory from the start of this operation to the current moment on a specific device_id.
    - memory/`{metric_infix}`/**end/onload**/max_reserved/`{device_id}`: Peak reserved GPU memory from the start of this operation to the current moment on a specific device_id.
    - memory/`{metric_infix}`/**end/onload**/max_allocated_frac/`{device_id}`: Fraction of peak allocated GPU memory relative to total GPU memory on a specific device_id.
    - memory/`{metric_infix}`/**end/onload**/max_reserved_frac/`{device_id}`: Fraction of peak reserved GPU memory relative to total GPU memory on a specific device_id.
- Memory snapshot after offloading model state (at operation end)
    - memory/`{metric_infix}`/**end/offload**/allocated/`{device_id}`: Currently allocated GPU memory on a specific device_id.
    - memory/`{metric_infix}`/**end/offload**/reserved/`{device_id}`: Currently reserved GPU memory on a specific device_id.
    - memory/`{metric_infix}`/**end/offload**/max_allocated/`{device_id}`: Peak allocated GPU memory from the start of this operation to the current moment on a specific device_id.
    - memory/`{metric_infix}`/**end/offload**/max_reserved/`{device_id}`: Peak reserved GPU memory from the start of this operation to the current moment on a specific device_id.

#### CPU Memory Metrics
- memory/cpu/`{metric_infix}`/start/rss: Actual physical memory (Resident Set Size) occupied by the process at the start of the operation.
- memory/cpu/`{metric_infix}`/start/vms: Virtual memory (Virtual Memory Size) occupied by the process at the start of the operation.
- memory/cpu/`{metric_infix}`/end/rss: Actual physical memory occupied by the process at the end of the operation.
- memory/cpu/`{metric_infix}`/end/vms: Virtual memory occupied by the process at the end of the operation.