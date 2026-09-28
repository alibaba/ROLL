sh examples/cicd_pipelines/submit_pipeline_amd.sh 1 \
  examples/start_agentic_pipeline.py \
  agentic_qwen2.5_0.5B_frozen_lake.yaml \
  pytorch280 \
  requirements_torch280_vllm_amd.txt

sh examples/cicd_pipelines/submit_pipeline_amd.sh 1 \
  examples/start_agentic_pipeline.py \
  agentic_qwen2.5_vl_3B_sokoban.yaml \
  pytorch280 \
  requirements_torch280_vllm_amd.txt

sh examples/cicd_pipelines/submit_pipeline_amd.sh 1 \
  examples/start_agentic_pipeline.py \
  agentic_qwen2.5_0.5B_frozen_lake_partial_rollout.yaml \
  pytorch280 \
  requirements_torch280_vllm_amd.txt

sh examples/cicd_pipelines/submit_pipeline_amd.sh 1 \
  examples/start_rlvr_pipeline.py \
  rlvr_qwen2.5_7B_megatron_vllm.yaml \
  pytorch280 \
  requirements_torch280_vllm_amd.txt

sh examples/cicd_pipelines/submit_pipeline_amd.sh 1 \
  examples/start_rlvr_pipeline.py \
  rlvr_qwen2.5_7B_megatron_lora.yaml \
  pytorch280 \
  requirements_torch280_vllm_amd.txt

sh examples/cicd_pipelines/submit_pipeline_amd.sh 1 \
  examples/start_rlvr_pipeline.py \
  rlvr_qwen2.5_7B_fsdp2_lora.yaml \
  pytorch280 \
  requirements_torch280_vllm_amd.txt

sh examples/cicd_pipelines/submit_pipeline_amd.sh 1 \
  examples/start_rlvr_vl_pipeline.py \
  rlvr_qwen3_vl_4B_megatron.yaml \
  pytorch280 \
  requirements_torch280_vllm_amd.txt

sh examples/cicd_pipelines/submit_pipeline_amd.sh 1 \
  examples/start_rlvr_vl_pipeline.py \
  rlvr_qwen3_vl_4B_fsdp2.yaml \
  pytorch280 \
  requirements_torch280_vllm_amd.txt

sh examples/cicd_pipelines/submit_pipeline_amd.sh 2 \
  examples/start_onpolicy_distill_pipeline.py \
  mopd_qwen3_8B_megatron_vllm.yaml \
  pytorch280 \
  requirements_torch280_vllm_amd.txt
