
#!/bin/bash
set +x

CONFIG_PATH="distill/off_policy/vlm"
python examples/start_distill_vl_pipeline.py --config_path $CONFIG_PATH  --config_name distill_vl_megatron
