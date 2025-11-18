#!/bin/bash
set +x
source "examples/scripts/config.sh"

WORKER_COUNT=2
CONFIG_FILE="rlvr_lora_zero3_amd.yaml"
# 替换为mos uri
NEBULA_MODEL=""
ENTRY_FILE="examples/start_rlvr_pipeline.py"

CONFIG_PATH=$(basename $(dirname $0))
CONFIG_NAME="${CONFIG_FILE%.yaml}"
JOB_NAME="$CONFIG_PATH-$CONFIG_NAME"

QUEUE="nebula_test2_308x_gpu_hang"


echo "JOB_NAME: ${JOB_NAME}"
echo "WORKER_COUNT: ${WORKER_COUNT}"
echo "CONFIG_NAME: ${CONFIG_NAME}"
echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "ENTRY_FILE: ${ENTRY_FILE}"

args="--config_name ${CONFIG_NAME} --config_path ${CONFIG_PATH}"


mdl_args="--queue=${QUEUE} \
        --entry=${ENTRY_FILE} \
        --worker_count=${WORKER_COUNT}  \
        --file.cluster_file=examples/scripts/cluster.json \
        --job_name=${JOB_NAME} \
        --algo_name=pytorch260_rocm700rc4 \
        --requirements_file_name=nebula_patch/requirements/requirements_torch260_vllm_amd.txt \
        --oss_appendable=true \
        --_NEBULA_MODEL=${NEBULA_MODEL} \
        --nebula_model=${NEBULA_MODEL} \
        --force \
        "
if [ -n "${OPENLM_TOKEN}" ]; then
    mdl_args="${mdl_args} --env=OPENLM_TOKEN=${OPENLM_TOKEN}"
fi

echo ${args}
echo ${mdl_args}

nebulactl run mdl --user_params="${args}" $mdl_args
