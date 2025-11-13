# Converting MCoreAdapter Models to Hugging Face Format

MCoreAdapter provides tools for converting between Megatron(McoreAdapter) and Hugging Face model formats. This document will guide you on how to convert a trained Megatron model to Hugging Face format for use in other projects.

## Conversion Tool

MCoreAdapter includes a conversion tool `tools/convert.py` that can convert Megatron(McoreAdapter) models to Hugging Face format.

## Conversion Command

To convert a model trained with megatron_strategy to a Hugging Face model, use the following command:

```bash
python tools/convert.py --checkpoint_path path_to_megatron_model --output_path path_to_output_hf_model
```

Where:
- `path_to_megatron_model` is the path to the McoreAdapter model you want to convert
- `path_to_output_hf_model` is the output path for the converted Hugging Face model

## Notes

1. The conversion process may take some time depending on the size of the model.
2. Ensure you have sufficient disk space to store the converted model.
3. The converted Hugging Face model can be used directly in projects that support the Hugging Face Transformers library.

## Direct Use of Hugging Face Models

It's worth noting that MCoreAdapter can directly load Hugging Face models without explicitly converting models to Megatron(McoreAdapter) format. This is particularly useful when you want to use Hugging Face models for reinforcement learning in the Roll framework.