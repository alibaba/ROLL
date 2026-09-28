# Image Provided
We provide pre-built Docker images for a quick start (Links will be updated):

* `torch2.8.0 + vLLM0.11.0`: roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-25.06-py3-torch280-vllm0110
* `torch2.10.0 + vLLM0.16.0rc2.dev502+gade81f17f + megatron-core core_dev_r0.16.0`: roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-25.11-py3-torch2100-mcore0160dev-vllm016dev
* `torch2.11.0 + vLLM0.23.0 + megatron-core 77c0f8cb3 `: roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-26.03-py3-torch2110-vllm0230

For AMD GPU users, We provided pre-built Docker images for a quick start as well:
* `torch2.10.0 + vLLM0.17.1`: amdagi/roll_env_rocm721:latest
We also provided Dockerfiles under 'docker/' for AMD users as an alternatively plan.

You can also find [Dockerfiles](https://github.com/StephenRi/ROLL/tree/feature/fix-ref-for-docs/docker) under the `docker/` directory to build your own images.
