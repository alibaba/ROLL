# 镜像地址

我们提供了预构建的Docker镜像以便快速开始：

* `torch2.6.0 + SGlang0.4.6`: roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-sglang046
* `torch2.6.0 + vLLM0.8.4`: roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-vllm084
* `torch2.8.0 + vLLM0.10.2`: roll-registry-vpc.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-25.06-py3-torch280-vllm0102
* `torch2.8.0 + vLLM0.11.0`: roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-25.06-py3-torch280-vllm0110

您也可以在`docker/`目录下找到[Dockerfiles](https://github.com/StephenRi/ROLL/tree/feature/fix-ref-for-docs/docker)来构建您自己的镜像。