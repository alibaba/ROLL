---
sidebar_position: 3
---

# How to Add Support for a New Model

To integrate a new model into **ROLL**, you must supply:

1. at least one **inference** implementation, and  
2. at least one **training** implementation.

| Phase     | Pick ≥ 1 backend |
|-----------|-----------------|
| Inference | `vllm`, `sglang` |
| Training  | `DeepSpeed`, `Megatron` |

---

## 1. Inference Strategies

### 1.1 `vllm`
Follow the official guide:  
https://docs.vllm.ai/en/latest/contributing/model/registration.html#out-of-tree-models

### 1.2 `sglang`
Follow the official guide:  
https://docs.sglang.ai/supported_models/support_new_models.html

---

## 2. Training Strategies

### 2.1 `DeepSpeed`

1. Ensure your model can be loaded by  
   ```python
   transformers.AutoModelForCausalLM.from_pretrained(...)
   ```  
   If not, add the model implementation directly to the ROLL repository.
2. Make the model inherit from `transformers.PreTrainedModel`.
3. Make the model can be loaded in `roll/models/model_providers.py`.

Once these steps are complete, you can:
- train with the `deepspeed_train` strategy for `actor_train` worker, and  
- with `hf_infer` or `deepspeed_infer` strategy for the `reference` worker.

### 2.2 `Megatron`

To integrate a Hugging Face model with the `Megatron` training strategy, you need to provide a conversion template. This template defines how to map the model's configuration and weights from the Hugging Face format to the Megatron-Core format.

#### 1. For Standard Transformer Models

If your model has a standard Transformer architecture compatible with `mcore.GPTModel`, you only need to register a new conversion template. All templates are located in `mcore_adapter/src/mcore_adapter/models/converter/template.py`.

To add a new template, you'll call the `register_template` function at the end of this file. Here’s a detailed guide on how to construct the arguments for this function.

##### Registering a New Template

The core of the integration is the `register_template` function. Let's break down its main parameters:

```python
register_template(
    hf_model_type,
    config_hf_to_mca,
    weight_converters,
    hf_layer_prefix,
    constant_mca_config={},
    hf_invalid_keys=[],
    ...
)
```

**a. `hf_model_type` (str):**
This is the most crucial parameter. It must exactly match the `model_type` field in the model's Hugging Face `config.json` file. The converter uses this string to look up the correct template.

**b. `hf_layer_prefix` (str):**
This specifies the prefix for the transformer layers in the Hugging Face model's state dictionary. For most models, this will be something like `"model.layers."`.

**c. `config_hf_to_mca` (Dict[str, str]):**
This dictionary maps configuration parameter names from the Hugging Face `config.json` to their corresponding names in the Megatron-Core `TransformerConfig`.

**d. `weight_converters` (List[ConverOp]):**
This is a list of converter operations that define how to transform weights from the HF format to the MCA format. Each operation is an instance of a `ConverOp` subclass.

Common converter operations include:
- **`RenameConverOp`**: Used for weights that only need to be renamed.
  ```python
  # Renames 'lm_head.weight' in HF to 'output_layer.weight' in MCA
  RenameConverOp(hf_names="lm_head.weight", mca_names="output_layer.weight")
  ```
- **`StackConverOp`**: Stacks multiple HF tensors into a single MCA tensor. This is commonly used for the gate and up projections in SwiGLU layers.
  ```python
  # Stacks two HF weights into one MCA weight for the first feed-forward layer
  StackConverOp(
      hf_names=[".mlp.gate_proj.weight", ".mlp.up_proj.weight"], 
      mca_names=".mlp.linear_fc1.weight", 
      dim=0
  )
  ```
- **`QKVConverOp`**: Fuses the separate Query, Key, and Value weight tensors from HF into a single, interleaved QKV tensor required by Megatron-Core.
  ```python
  # Fuses Q, K, and V weights into a single QKV weight
  QKVConverOp(
      hf_names=[".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"],
      mca_names=".self_attention.linear_qkv.weight",
  )
  ```

**e. `constant_mca_config` (Dict[str, Any]):**
This dictionary defines Megatron-Core configuration values that are constant for the model and are not available in the HF config.


#### 2. For Models with Custom Components

If the model includes unique components not found in a standard `mcore.GPTModel` (e.g., Vision Transformer blocks in a multimodal model like Qwen2-VL), you will need to:
1.  Implement a new model class that inherits from `mcore.GPTModel` and adds the custom logic. You can use the implementations for `qwen2-vl` and `qwen2.5-vl` in the repository as a reference.
2.  Register a template for the parts of the model that are standard, as described above. The template can also handle renaming for the custom parts (e.g., `RenameConverOp(hf_names="visual.{}", mca_names="vision_model.{}")`).

After completing these steps, you can:
- train with the `megatron_train` strategy for the `actor_train` worker, and
- use the `megatron_infer` strategy for the `reference` worker.
