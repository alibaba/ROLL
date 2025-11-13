# MCoreAdapter 模型转换为 Hugging Face 格式

MCoreAdapter 提供了在 Megatron(McoreAdapter) 和 Hugging Face 模型格式之间进行转换的工具。本文档将指导您如何将训练好的 Megatron(McoreAdapter) 模型转换为 Hugging Face 格式，以便在其他项目中使用。

## 转换工具

MCoreAdapter 包含一个转换工具 `tools/convert.py`，可以将 Megatron(McoreAdapter) 模型转换为 Hugging Face 格式。

## 转换命令

由megatron_strategy训练出来的模型，要Hugging Face 模型，请使用以下命令：

```bash
python tools/convert.py --checkpoint_path path_to_megatron_model --output_path path_to_output_hf_model
```

其中：
- `path_to_megatron_model` 是您要转换的 McoreAdapter 模型的路径
- `path_to_output_hf_model` 是转换后 Hugging Face 模型的输出路径

## 注意事项

1. 转换过程可能需要一些时间，具体取决于模型的大小。
2. 确保有足够的磁盘空间来存储转换后的模型。
3. 转换后的 Hugging Face 模型可以直接在支持 Hugging Face Transformers 库的项目中使用。

## 直接使用 Hugging Face 模型

值得注意的是，MCoreAdapter 直接加载 Hugging Face 模型，不需要显式将模型转换为 Megatron(McoreAdapter) 格式的步骤。这在您想将 Hugging Face 模型用于 Roll 框架的强化学习时特别有用。