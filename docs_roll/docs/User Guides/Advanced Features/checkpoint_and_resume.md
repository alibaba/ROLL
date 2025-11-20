# Checkpoint Saving and Resuming Guide

In the ROLL framework, the checkpoint mechanism allows you to save the model state during training so that you can resume training when needed. This document will provide detailed instructions on how to configure and use the checkpoint saving and resuming functionality.

## Checkpoint Saving Configuration

The ROLL framework configures checkpoint saving settings through the `checkpoint_config` parameter. The following is a typical configuration example:

```yaml
checkpoint_config:
  type: file_system
  output_dir: /data/cpfs_0/rl_examples/models/${exp_name}
```

### Configuration Parameter Details

1. **type**: Specifies the type of checkpoint storage
   - Currently supports `file_system`, which means saving checkpoints to the file system

2. **output_dir**: Specifies the directory path for saving checkpoints
   - Variables can be used, such as `${exp_name}` for the experiment name
   - The framework will automatically create timestamp subdirectories under this directory to distinguish different checkpoints

## Checkpoint Saving Mechanism

The ROLL framework automatically saves checkpoints in the following situations:

1. **Periodic Saving**: Automatically saved at intervals set by the `save_steps` parameter
   ```yaml
   save_steps: 100  # Save checkpoint every 100 steps
   ```

2. **At Training End**: Automatically save the final checkpoint when training is completed

3. **Manual Saving**: Checkpoints can be manually saved by calling the appropriate API in the code

## Resuming Training Configuration

To resume training from a checkpoint, set the `resume_from_checkpoint` parameter:

```yaml
resume_from_checkpoint: false  # Default is not to resume training
```

To enable the resume training feature, set this parameter to the checkpoint path:

```yaml
resume_from_checkpoint: /data/cpfs_0/rl_examples/models/exp_name/checkpoint-500
```

### How Resuming Training Works

1. When `resume_from_checkpoint` is set to a valid checkpoint path, the framework will:
   - Load model parameters
   - Restore optimizer state
   - Restore learning rate scheduler state
   - Restore other training states such as training steps

2. Resuming training continues from the training step at which the checkpoint was saved

## Usage Example

The following is a complete configuration example showing how to set up checkpoint saving and resuming functionality:

```yaml
exp_name: "qwen2.5-7B-rlvr-config"
seed: 42
logging_dir: ./output/logs
output_dir: ./output

# Checkpoint configuration
checkpoint_config:
  type: file_system
  output_dir: /data/cpfs_0/rl_examples/models/${exp_name}

# Resume training configuration
resume_from_checkpoint: false  # Set to checkpoint path to resume training

# Training control parameters
max_steps: 500
save_steps: 100  # Save checkpoint every 100 steps
logging_steps: 1
eval_steps: 10

# Other training configurations...
```

To resume training from a checkpoint, simply set `resume_from_checkpoint` to the corresponding checkpoint path:

```yaml
resume_from_checkpoint: /data/cpfs_0/rl_examples/models/qwen2.5-7B-rlvr-config/checkpoint-300
```

## Best Practices

1. **Periodic Checkpoint Saving**: Reasonably set `save_steps` based on training time and resource consumption
2. **Check Storage Space**: Ensure `output_dir` has sufficient storage space to save checkpoints
3. **Verify Checkpoints**: Verify the integrity and validity of checkpoints before resuming training
4. **Backup Important Checkpoints**: Backup important checkpoints to prevent data loss

By properly configuring the checkpoint saving and resuming functionality, you can ensure the safety and recoverability of the training process, avoiding loss of training progress due to unexpected interruptions.