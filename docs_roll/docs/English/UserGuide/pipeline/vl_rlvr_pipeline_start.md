
# RLVR Pipeline for VLM


**Table of Contents**

- [RLVR Pipeline for VLM](#rlvr-pipeline-for-vlm)
  - [✨️Overview](#️overview)
  - [✨️Core Components](#️core-components)
    - [Main Module (`RLVRPipeline`)](#main-module-rlvrvlmpipeline)
    - [Configuration File (`RLVRConfig`)](#configuration-file-rlvrconfig)
    - [Reward Worker](#reward-worker)
  - [✨️Data Preparation](#️data-preparation)
    - [Data Format](#data-format)
  - [✨️Running the Pipeline](#️running-the-pipeline)
    - [Method 1: Using Python Launcher Script](#method-1-using-python-launcher-script)
    - [Method 2: Using Helper Shell Scripts](#method-2-using-helper-shell-scripts)
  - [✨️Step-by-Step Example](#️step-by-step-example)
    - [Step 1: Configure Settings](#step-1-configure-settings)
    - [Step 2: Prepare Environment and Dependencies](#step-2-prepare-environment-and-dependencies)
    - [Step 3: Launch the Pipeline](#step-3-launch-the-pipeline)
    - [Step 4: Monitoring](#step-4-monitoring)
    - [Step 5: Outputs and Results](#step-5-outputs-and-results)

---

## ✨️Overview

 RLVR pipeline for VLM shares the same advantages with its counterpart for LLM ([doc](./rlvr_pipeline_start.md)), and it has built-in support for both visual reasoning and visual perception tasks, including math (for reasoning) and detection (for perception) currently, each equipped with specialized reward evaluation mechanisms, which enables simultaneous optimization of model capabilities across multiple domains such as math and detection.

---

## ✨️Core Components

### Main Module (`RLVRVLMPipeline`)

`RLVRVLMPipeline` (located in `roll/pipeline/rlvr/rlvr_vlm_pipeline.py`) is the primary coordinator for the entire reinforcement learning process. It manages the complete training workflow, including:

* Initializing and managing distributed workers (actor, critic, reference, and various reward workers).
* Coordinating data collection and processing.
* Executing model training steps (e.g., PPO updates for actor and critic).
* Handling model synchronization and checkpoint saving.
* Validation set evaluation.
* Recording metrics and experiment tracking.

**Source code**: `roll/pipeline/rlvr/rlvr_vlm_pipeline.py`, in which Qwen2.5-VL is supported directly

---

### Configuration File (`RLVRConfig`)

RLVR pipeline for VLM uses the same configuration file (`RLVRConfig`) with LLM, please refer to [docs of RLVR Pipeline for LLM](./rlvr_pipeline_start.md) for configuration details. 

A configuration example can be found in `examples/qwen2.5-vl-7B-rlvr/rlvr_megatron.yaml`, and the difference with LLM mainly exists in rewards settings which include visual specific reward and wold be introduced later.

   **Reward Settings**
   The `rewards` section contains reward worker configurations for different domains:

   * **math**
     * `worker_cls`: Worker class name (e.g., `MathRuleRewardWorker`)
     * `tag_included`: These tags use the reward domain for calculation.
     * `model_args`: Reward model parameters
     * `world_size`: Number of workers

   * **cv_detection**
     * Similar configuration, but for evaluation of detection results

    Note that the domains provided here (math and cv_detection) should be same as listed in `domain_interleave_probs`

---

### Reward Worker

The VLM rlvr pipeline supports reward mechanisms for different rlvr domains, and the example used inlcudes:

* **Mathematical Rule Reward (`MathRuleRewardWorker`)** – Evaluates the correctness and steps of mathematical reasoning, which shares with RLVR pipeline for LLM
* **Detection Reward (`DetectionRewardWorker`)** – Evaluates detection results.
  - The detection verifier used here references to [MiniMax-AI/One-RL-to-See-Them-All](https://github.com/MiniMax-AI/One-RL-to-See-Them-All/blob/main/reward_server/verify.py), which combines IoU and mAP scores with specified weighting coefficients for reward (a format reward is combined addtionally). 
  - For IoU score:
    - the IoU threshold is defined by `DET_IOU_THRESHOLD` environment variable, and it can be set to one of `[0.5, 0.55, 0.6,0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]` or set to one of `average` and `dynamic`. 
    - `average` is the default which averages IoU scores of all of these thresholds
    - `dynamic` is an introduced dynamic IoU reward mechanism that provides adaptive, progressive feedback to improves stability and performance. It uses different thresholds for different trainingsteps. Specifically, the IoU threshold is set to 0.85 for the first 10% training steps, and 0.95 for the next 15% training steps, and 0.99 for the remaining training steps.
    - Additionally, the IoU score for each threshold includes two items with differnet strategies(`greedy_match_by_iou_max_iou_first` and `greedy_match_by_iou_max_label_first`), and completeness (calculated by `1.0 - (FN_ratio + FP_raio) / 2.0`) is also taken into account by a completeness weighting coefficient in each item score.
  
  Please refer to the [paper](https://arxiv.org/pdf/2505.18129) for more details

---

## ✨️Data Preparation

### Data Format

For multi-domain RLVR, we use data files with parquet format from [One-RL-to-See-Them-All/Orsta-Data-47k](https://huggingface.co/datasets/One-RL-to-See-Them-All/Orsta-Data-47k) as example inputs thus following their data schema, and here is a sample of math domain to illustrate the data format:

```python
{
    "data_source": "mm_math",
    "images": [<PIL.PngImagePlugin.PngImageFile image mode=RGB size=252x56 at 0x15EEEA390>],
    "prompt": [
        {
            "content": "<image>As shown in the figure, point D is the midpoint of line segment AC, $BC = \\frac{1}{2}AB$, and $BD = 1$cm. What is the length of AB?",
            "role": "user",
        }
    ],
    "ability": "math",
    "reward_model": {
        "answer": "4",
        "ground_truth": "\\boxed{4}",
        "accuracy_ratio": 1.0,
        "format_ratio": 0.0,
        "verifier": "mathverify",
        "verifier_parm": {
            "det_verifier_normalized": None,
            "det_reward_ratio": {
                "iou_max_label_first": None,
                "iou_max_iou_first": None,
                "iou_completeness": None,
                "map": None,
                "map50": None,
                "map75": None,
            },
        },
    },
    "extra_info": {"id": None, "image_path": "images/51284809.png"},
}
```

The underlying data schema is

```python
{
    'data_source': Value(dtype='string', id=None),
    'images': Sequence(feature=Image(mode=None, decode=True, id=None), length=-1, id=None),
    'prompt': [{'content': Value(dtype='string', id=None), 'role': Value(dtype='string', id=None)}],
    'ability': Value(dtype='string', id=None),
    'reward_model': {
        'answer': Value(dtype='string', id=None),
        'ground_truth': Value(dtype='string', id=None),
        'accuracy_ratio': Value(dtype='float32', id=None),
        'format_ratio': Value(dtype='float32', id=None),
        'verifier': Value(dtype='string', id=None),
        'verifier_parm': {
            'det_verifier_normalized': Value(dtype='bool', id=None),
            'det_reward_ratio': {
                'iou_max_label_first': Value(dtype='float32', id=None),
                'iou_max_iou_first': Value(dtype='float32', id=None),
                'iou_completeness': Value(dtype='float32', id=None),
                'map': Value(dtype='float32', id=None),
                'map50': Value(dtype='float32', id=None),
                'map75': Value(dtype='float32', id=None)
            }
        }
    },
    'extra_info': {'id': Value(dtype='string', id=None), 'image_path': Value(dtype='string', id=None)}
}
```

with the following field description:

- `data_source` (Required): the source of the data, e.g. `object365_train`. **NOTE: It should be one value included in `tag_included` of a domain provided under rewards section in the configuration to indicate this sample belonging to the domain and using corresponding reward worker** . Additionally, it is used in validation to give seperated metrics for different data sources.
- `images` (Required): a list of PIL images
- `prompt` (Required): prompt with chat templates. **NOTE**: use `<image>` as image token and `prompt` should have image tokens with the same number as `images`
- `ability` (Optional): indicating data domain, e.g. `cv_detection`. Data from same domain should have the same `ability` value. While it is not used yet.
- `reward_model` (Required): reward related information, combined with reward  
  - `answer` (Optional): the ground truth, not used yet and `ground_truth` is used instead
  - `ground_truth` (Required): the formated ground truth commonly, and the format is corresponding to answer extraction method in reward worker. For example, `ground_truth` of detection should be included in `"<answer>"` and `"</answer>"` to be consistent with `detection_reward_worker.py` which extracts answer using `"<answer>"` and `"</answer>"`
  - `accuracy_ratio` (Required): coeficient of accuracy for score/reward calculation
  - `format_ratio` (Required): coeficient of format for score/reward calculation
  - `verifier` (Optional): verifier name, e.g. `detection`, not used yet. 
  - `verifier_parm` (Optional): verifier parameters, all sub-fields are optional and only required for detection
    - `det_verifier_normalized`: whether to normalize bounding box iou to height/width sized `1000`, this is needed when ground truth is normalized
    - `det_reward_ratio`: weighting coeficients for various detection score combination
      - `iou_max_label_first`: coeficient for iou score using `greedy_match_by_iou_max_iou_first` as strategy
      - `iou_max_iou_first`: coeficient for iou score using `greedy_match_by_iou_max_label_first` as strategy
      - `iou_completeness`: coeficient for iou completeness 
      - `map`: coeficient for map
      - `map50`: coeficient for map50
      - `map75`: coeficient for map75
- `extra_info` (Optional): extra information, not used yet


A more complicated sample with all fields from detection domain is as following:

```python
{
    "data_source": "v3det_train",
    "images": [<PIL.PngImagePlugin.PngImageFile image mode=RGB size=799x533 at 0x15EF1DF90>],
    "prompt": [
        {
            "content": "<image>\nLocate all objects of the designated class present in the image:\n- dog collar\n- shiba dog\n\nBegin by clearly explaining your thought process enclosed within <think> and </think> tags. Afterward, present your final detection results enclosed within <answer> and </answer> tags.\nFor example:\n<think>\nYour detailed reasoning process here.\n</think>\n<answer>\n[{'bbox_2d': [x1,y1,x2,y2],'label': label_name}]\n</answer>",
            "role": "user",
        }
    ],
    "ability": "cv_detection",
    "reward_model": {
        "answer": "[{'bbox_2d': [484, 227, 818, 998], 'label': 'shiba dog'}, {'bbox_2d': [106, 142, 473, 998], 'label': 'shiba dog'}, {'bbox_2d': [274, 468, 427, 652], 'label': 'dog collar'}, {'bbox_2d': [490, 522, 609, 611], 'label': 'dog collar'}]",
        "ground_truth": "<answer>\n[{'bbox_2d': [484, 227, 818, 998], 'label': 'shiba dog'}, {'bbox_2d': [106, 142, 473, 998], 'label': 'shiba dog'}, {'bbox_2d': [274, 468, 427, 652], 'label': 'dog collar'}, {'bbox_2d': [490, 522, 609, 611], 'label': 'dog collar'}]\n</answer>",
        "accuracy_ratio": 1.0,
        "format_ratio": 0.10000000149011612,
        "verifier": "detection",
        "verifier_parm": {
            "det_verifier_normalized": True,
            "det_reward_ratio": {
                "iou_max_label_first": 1.0,
                "iou_max_iou_first": 0.0,
                "iou_completeness": 0.30000001192092896,
                "map": 0.0,
                "map50": 0.0,
                "map75": 0.0,
            },
        },
    },
    "extra_info": {"id": None, "image_path": "images/a00004438/19_1169_36968567106_1209f085a7_c.jpg"},
}
```


You can organize your own data as the above format and use it in the pipeline


---



## ✨️Running the Pipeline



### Method 1: Using Python Launcher Script

The primary method is to use the `examples/start_rlvr_vl_pipeline.py` script. This script uses Hydra to load and manage configurations.

1. **Select or Create a Configuration File**  
   Start with an example YAML (e.g., `examples/qwen2.5-vl-7B-rlvr/rlvr_megatron.yaml`) or create your own configuration.

2. **Execute the Python Launcher Script**

   ```bash
   # Make sure you are in the root directory of the ROLL (ScaleAligner) project
   # export PYTHONPATH=$(pwd):$PYTHONPATH
   
   python examples/start_rlvr_vl_pipeline.py \
          --config_path examples/qwen2.5-vl-7B-rlvr \
          --config_name rlvr_megatron
   ```

   * `--config_path` – Directory containing your YAML configuration.
   * `--config_name` – Filename (without `.yaml`).



### Method 2: Using Helper Shell Scripts

The `examples` directory typically contains shell scripts that wrap the Python launcher (e.g., `run_rlvr_pipeline.sh`).

Example structure:

```bash
#!/bin/bash
# Example: examples/qwen2.5-vl-7B-rlvr/run_rlvr_pipeline.sh

CONFIG_NAME="rlvr_megatron"            # rlvr_megatron.yaml
CONFIG_PATH="examples/qwen2.5-vl-7B-rlvr"

# Set environment variables and other configurations

python examples/start_rlvr_vl_pipeline.py \
       --config_path $CONFIG_PATH \
       --config_name $CONFIG_NAME \
       "$@"   # Pass any additional parameters
```

Run using:

```bash
bash examples/qwen2.5-vl-7B-rlvr/run_rlvr_pipeline.sh
```

---



## ✨️Step-by-Step Example



### Step 1: Configure Settings

* File: `examples/qwen2.5-vl-7B-rlvr/rlvr_megatron.yaml`  
  Key sections include `exp_name`, `seed`, `output_dir`, model paths, `actor_train`, `actor_infer`, `reference`, PPO parameters, and reward configurations.

* Pay special attention to these configuration sections:
  * Data configuration: `actor_train.data_args.file_name` and `domain_interleave_probs`
  * Model configuration: `pretrain` path
  * Distributed strategies: `strategy_args` and `device_mapping` for each worker
  * Reward configuration: Reward workers for different domains in the `rewards` section

### Step 2: Prepare Environment and Dependencies

* Ensure all necessary dependencies are installed. NOTE: VLLM is the only supported inference engine for VLM pipeline currently, thus use the corresponding requirement files:

  ```bash
  pip install -r requirements_torch260_vllm.txt
  ```

* Verify that all model paths in the configuration are accessible.

* Prepare training and validation datasets, ensuring they conform to the data format requirements described above.

### Step 3: Launch the Pipeline

```bash
python examples/start_rlvr_vl_pipeline.py \
       --config_path examples/qwen2.5-vl-7B-rlvr \
       --config_name rlvr_megatron
```

### Step 4: Monitoring

* **Console Output** – Observe Hydra, Ray, and pipeline logs.
* **Log Files** – Check the `logging_dir` specified in the YAML.
* **TensorBoard**

  ```bash
  tensorboard --logdir <your_log_dir>
  ```

### Step 5: Outputs and Results

* **Trained Models** – Checkpoints are saved in the `output_dir`.
* **Evaluation Metrics** – Recorded in TensorBoard and the console.
* **Generated Examples** – The pipeline periodically outputs generated examples so you can visually assess model improvements.

---

*Happy experimenting!*