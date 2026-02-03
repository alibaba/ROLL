#!/bin/bash
set +x

# fix cudnn for roll-image
# mv /usr/local/lib/python3.12/dist-packages/torch/lib/../../nvidia/cudnn /usr/local/lib/python3.12/dist-packages/torch/lib/../../nvidia/cudnn_bak

CONFIG_PATH=$(basename $(dirname $0))
python examples/start_rlvr_vl_pipeline.py --config_path $CONFIG_PATH  --config_name rlvr_megatron_80G
