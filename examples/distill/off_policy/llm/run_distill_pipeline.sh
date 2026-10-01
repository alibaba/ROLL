
#!/bin/bash
set +x

CONFIG_PATH="distill/off_policy/llm"
python examples/start_distill_pipeline.py --config_path $CONFIG_PATH  --config_name distill_megatron
