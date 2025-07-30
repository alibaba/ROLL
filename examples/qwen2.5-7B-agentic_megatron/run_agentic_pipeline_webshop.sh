#!/bin/bash
# Run `git submodule update --init --recursive` to init submodules before run this script.
set +x


pip install -r third_party/webshop-minimal/requirements.txt --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
python -m spacy download en_core_web_sm

CONFIG_PATH=$(basename $(dirname $0))
python examples/start_agentic_pipeline.py --config_path $CONFIG_PATH  --config_name agentic_val_webshop
