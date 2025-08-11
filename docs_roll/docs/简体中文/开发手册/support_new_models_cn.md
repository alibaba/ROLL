---
sidebar_position: 3
---

# 如何支持新模型

要在 **ROLL** 中集成一个新模型，需要支持：

1.  至少一种 **推理** 实现，以及
2.  至少一种 **训练** 实现。

| 阶段 | 选择 ≥ 1 个后端 |
| --- | --- |
| 推理 | `vllm`, `sglang` |
| 训练 | `DeepSpeed`, `Megatron` |

---

## 1. 推理策略

### 1.1 `vllm`

参考官方文档： https://docs.vllm.ai/en/latest/contributing/model/registration.html#out-of-tree-models

### 1.2 `sglang`

参考官方文档： https://docs.sglang.ai/supported_models/support_new_models.html

---

## 2. 训练策略

### 2.1 `DeepSpeed`

1.  模型需支持通过以下代码加载：
    ```python
    transformers.AutoModelForCausalLM.from_pretrained(...)
    ```
    或者可将模型实现直接添加到 ROLL 仓库中。
2.  模型需继承自 `transformers.PreTrainedModel`。
3.  在 `roll/models/model_providers.py` 中注册模型。

完成以上步骤后，您可以：
-   使用 `deepspeed_train` 策略来训练 `actor_train` worker，以及
-   使用 `hf_infer` 或 `deepspeed_infer` 策略在 `reference` worker。

### 2.2 `Megatron`

要将 Hugging Face 模型与 `Megatron` 训练策略集成，需要实现一个该模型的转换模板。该模板定义了如何将模型的配置和权重从 Hugging Face 格式映射到 Megatron-Core 格式。

#### 1. 对于标准 Transformer 模型

如果您的模型是标准的 Transformer 架构，且与 `mcore.GPTModel` 兼容，您只需注册一个新的转换模板。所有模板都位于 `mcore_adapter/src/mcore_adapter/models/converter/template.py` 中。

要添加新模板，您需要在该文件末尾调用 `register_template` 函数。以下是有关如何构建此函数参数的详细指南。

##### 注册新模板

集成的核心是 `register_template` 函数。让我们分解一下它的主要参数：

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
这是最关键的参数。它必须与模型 Hugging Face `config.json` 文件中的 `model_type` 字段完全匹配。转换器使用此字符串来查找正确的模板。

**b. `hf_layer_prefix` (str):**
这指定了 Hugging Face 模型状态字典中 Transformer 层的权重前缀。对于大多数模型，这通常是 `"model.layers."`。

**c. `config_hf_to_mca` (Dict[str, str]):**
此字典将 Hugging Face `config.json` 中的配置参数名称映射到 Megatron-Core `TransformerConfig` 中的相应名称。

**d. `weight_converters` (List[ConverOp]):**
这是一个转换器操作列表，定义了如何将权重从 HF 格式转换为 MCA 格式。每个操作都是 `ConverOp` 子类的实例。

常见的转换器操作包括：
- **`RenameConverOp`**: 用于仅需要重命名的权重。
  ```python
  # 将 HF 中的 'lm_head.weight' 重命名为 MCA 中的 'output_layer.weight'
  RenameConverOp(hf_names="lm_head.weight", mca_names="output_layer.weight")
  ```
- **`StackConverOp`**: 将多个 HF 张量堆叠成一个 MCA 张量。这通常用于 SwiGLU 层中的门和上投影。
  ```python
  # 将两个 HF 权重堆叠成一个 MCA 权重，用于第一个前馈层
  StackConverOp(
      hf_names=[".mlp.gate_proj.weight", ".mlp.up_proj.weight"], 
      mca_names=".mlp.linear_fc1.weight", 
      dim=0
  )
  ```
- **`QKVConverOp`**: 将 HF 中独立的查询（Query）、键（Key）和值（Value）权重张量融合成 Megatron-Core 所需的单个交错式 QKV 张量。
  ```python
  # 将 Q、K、V 权重融合成单个 QKV 权重
  QKVConverOp(
      hf_names=[".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"],
      mca_names=".self_attention.linear_qkv.weight",
  )
  ```

**e. `constant_mca_config` (Dict[str, Any]):**
此字典定义了模型固定的、但在 HF 配置中不可用的 Megatron-Core 配置值。

#### 2. 包含自定义组件的模型

如果模型包含标准 `mcore.GPTModel` 中没有的独特组件（例如，像 Qwen2-VL 这样的多模态模型中的 Vision Transformer 模块），您需要：
1.  实现一个新的模型类，该类继承自 `mcore.GPTModel` 并添加自定义逻辑。您可以参考仓库中 `qwen2-vl` 和 `qwen2.5-vl` 的实现。
2.  为模型的标准部分注册一个模板，如上所述。该模板也可以处理自定义部分的重命名（例如 `RenameConverOp(hf_names="visual.{}", mca_names="vision_model.{}")`）。

完成这些步骤后，您可以：
-   使用 `megatron_train` 策略训练 `actor_train` worker，以及
-   使用 `megatron_infer` 策略用于 `reference` worker。
