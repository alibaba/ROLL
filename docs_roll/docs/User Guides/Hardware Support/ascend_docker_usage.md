# ROLL on Ascend NPU with Docker

Last updated: 08/18/2026.

## Quick Reference

- ROLL repository: [alibaba/ROLL](https://github.com/alibaba/ROLL)
- Ascend usage guide: [ROLL x Ascend](ascend_usage.md)
- Ascend NPU examples: [Ascend NPU Examples](ascend_npu_examples.md)
- Available pre-built images: [ROLL images on Quay](https://quay.io/repository/ascend/roll?tab=tags)
- Issue tracker: [GitHub Issues](https://github.com/alibaba/ROLL/issues)

---

## ROLL NPU Image

The ROLL Ascend NPU images provide the runtime and Python dependencies required to run ROLL on Huawei Ascend training-series NPUs. Use a pre-built image when possible. Build from `Dockerfile.A2` or `Dockerfile.A3` when you need to customize the image or rebuild it locally.

The Dockerfiles currently cover Atlas 900 A2 and A3 training-series devices.

---

## Supported Tags and Dockerfile Links

| Hardware | Local tag | Pre-built image | Dockerfile | Base image |
|---|---|---|---|---|
| Atlas 900 A2 PODc / Ascend 910B1 | `roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12` | `quay.io/ascend/roll:main-a2` | [`docker/Dockerfile.A2`](https://github.com/alibaba/ROLL/blob/main/docker/Dockerfile.A2) | `quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.12-devel` |
| Atlas 900 A3 PODc / Ascend 910_9391 | `roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12` | `quay.io/ascend/roll:main-a3` | [`docker/Dockerfile.A3`](https://github.com/alibaba/ROLL/blob/main/docker/Dockerfile.A3) | `quay.io/ascend/cann:9.1.0-a3-ubuntu22.04-py3.12-devel` |

---

## Image Contents

| Component | Version |
|---|---|
| CANN | 9.1.0 |
| Python | 3.12 |
| PyTorch | 2.10.0 |
| torch-npu | 2.10.0.post4 |
| vLLM | 0.23.0 |
| vLLM-Ascend | 0.23.0rc1 |
| Transformers | 4.57.6 |
| triton-ascend | 3.2.1 |

---

## Supported Hardware

| Hardware | SOC_VERSION | Docker support |
|---|---|---|
| Atlas 900 A2 PODc / Ascend 910B1 | `ascend910b1` | Supported by `Dockerfile.A2` and `roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12` |
| Atlas 900 A3 PODc / Ascend 910_9391 | `ascend910_9391` | Supported by `Dockerfile.A3` and `roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12` |

Host requirements for the A2/A3 Docker images:

| Item | Requirement |
|---|---|
| Host OS | Ubuntu 22.04 |
| Docker | >= 20.10 |
| Ascend NPU driver | Installed on the host |

---

## Quick Start

### Get the Image

Pull the image that matches the target hardware and apply the local tag used by the commands below:

For Atlas 900 A2 PODc:

```bash
docker pull quay.io/ascend/roll:main-a2
docker tag quay.io/ascend/roll:main-a2 roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12
```

For Atlas 900 A3 PODc:

```bash
docker pull quay.io/ascend/roll:main-a3
docker tag quay.io/ascend/roll:main-a3 roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12
```

### Build the Image

Clone the repository before building from a Dockerfile:

```bash
git clone https://github.com/alibaba/ROLL.git
cd ROLL
```

Build for Atlas 900 A2 PODc:

```bash
docker build -f docker/Dockerfile.A2 -t roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12 .
```

Build for Atlas 900 A3 PODc:

```bash
docker build -f docker/Dockerfile.A3 -t roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12 .
```

The build compiles vLLM and vLLM-Ascend from source. Reserve at least 50 GB of disk space and ensure that the build host has network access to the required package and source repositories.

To override the SOC version explicitly:

```bash
# A2
docker build -f docker/Dockerfile.A2 \
    --build-arg SOC_VERSION=ascend910b1 \
    -t roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12 .

# A3
docker build -f docker/Dockerfile.A3 \
    --build-arg SOC_VERSION=ascend910_9391 \
    -t roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12 .
```

---

## Run Container

The host must expose the Ascend device files and driver directories to the container. The following example starts an A2 container with eight NPUs:

```bash
docker run -dit \
    --name roll_a2 \
    --ulimit nofile=65536:65536 \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /home/$USER:/home/$USER \
    -v /path/to/models:/path/to/models \
    -v /path/to/data:/path/to/data \
    --ipc=host \
    --net=host \
    roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12 \
    /bin/bash
```

For A3, use `roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12`, change the container name, and add the available device files. A full 16-NPU A3 node also requires `/dev/davinci8` through `/dev/davinci15`:

```bash
docker run -dit \
    --name roll_a3 \
    --ulimit nofile=65536:65536 \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci8 \
    --device /dev/davinci9 \
    --device /dev/davinci10 \
    --device /dev/davinci11 \
    --device /dev/davinci12 \
    --device /dev/davinci13 \
    --device /dev/davinci14 \
    --device /dev/davinci15 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /home/$USER:/home/$USER \
    -v /path/to/models:/path/to/models \
    -v /path/to/data:/path/to/data \
    --ipc=host \
    --net=host \
    roll:v0.3-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12 \
    /bin/bash
```

Adjust the `/dev/davinciX` entries to the number of NPUs available on the host. For multi-NPU training, mount every device required by the training topology.

Enter a running container:

```bash
# A2
docker exec -it roll_a2 /bin/bash

# A3
docker exec -it roll_a3 /bin/bash
```

---

## Verify Environment

Run these checks inside the container:

```bash
# NPU visibility
npu-smi info

# CANN environment variables
env | grep -E "ASCEND|LD_LIBRARY_PATH|PATH"

# Python packages and NPU availability
python -c "import torch; import torch_npu; print(torch.npu.is_available())"
python -c "import vllm; print(f'vllm: {vllm.__version__}')"
python -c "import vllm_ascend; print('vllm_ascend available')"
```

---

## Run ROLL Pipeline on Ascend NPU

ROLL's Ascend NPU examples use **FSDP2** as the training backend. Megatron-LM is not supported by the current Ascend setup. Before launching a pipeline, update the model paths and set `device_mapping` according to the NPU topology.

Example RLVR pipeline:

```bash
python examples/start_rlvr_pipeline.py \
    --config_path examples/ascend_examples \
    --config_name qwen3_8b_rlvr_fsdp2
```

The repository also includes `qwen3_30b_rlvr_fsdp2.yaml` and `run_rlvr_pipeline.sh` under `examples/ascend_examples`. Use a configuration that matches the available NPU memory and topology.

---

## Notes

- Install a compatible Ascend NPU driver on the host before starting the container.
- The A2/A3 Docker images are based on Ubuntu 22.04 and Python 3.12.
- If `vLLM-Ascend` cannot be imported, reload the Ascend environment inside the container:

  ```bash
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  source /usr/local/Ascend/nnal/atb/set_env.sh
  ```

- These commands are added to `/root/.bashrc` during image build. If you switch to a non-root user, source them manually when necessary.
- If an NPU is not visible, check the mounted device files, driver paths, and `npu-smi info` output.

---

## License

ROLL is released under the [Apache License 2.0](https://github.com/alibaba/ROLL/blob/main/LICENSE).

CANN, Ascend driver components, Python packages, system libraries, and other pre-installed dependencies may be subject to their own licenses.
