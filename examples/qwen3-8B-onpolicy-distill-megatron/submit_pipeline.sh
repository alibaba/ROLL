#!/bin/bash
set +x
source "examples/scripts/config.sh"

WORKER_COUNT=2
CONFIG_FILE="onpolicy_distill_config.yaml"
# Replace with mos uri
NEBULA_MODEL=""
ENTRY_FILE="examples/start_onpolicy_distill_pipeline.py"

CONFIG_PATH=$(basename $(dirname $0))
CONFIG_NAME="${CONFIG_FILE%.yaml}"
JOB_NAME="$CONFIG_PATH-$CONFIG_NAME"

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
        --oss_access_id=${OSS_ACCESS_ID} \
        --oss_access_key=${OSS_ACCESS_KEY} \
        --oss_bucket=${OSS_BUCKET} \
        --oss_endpoint=${OSS_ENDPOINT} \
        --job_name=${JOB_NAME} \
        --algo_name=pytorch280 \
        --requirements_file_name=nebula_patch/requirements/requirements_torch280_vllm.txt \
        --oss_appendable=true \
        --_NEBULA_MODEL=${NEBULA_MODEL} \
        --nebula_model=${NEBULA_MODEL} \
        "
if [ -n "${OPENLM_TOKEN}" ]; then
    mdl_args="${mdl_args} --env=OPENLM_TOKEN=${OPENLM_TOKEN}"
fi

echo ${args}
echo ${mdl_args}

nebulactl run mdl --user_params="${args}" $mdl_args