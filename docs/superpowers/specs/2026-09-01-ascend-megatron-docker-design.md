# Ascend Megatron Docker Images Design

## Goal

Make the A2 and A3 Ascend Docker images ready to run ROLL strategies that use
`megatron_train` or `megatron_infer`, without requiring users to install the
Ascend Megatron stack manually after starting the container.

## Scope

Update `docker/Dockerfile.A2` and `docker/Dockerfile.A3` only. The images will
install the three components used by the current Ascend CI and documentation:

| Component | Source | Default ref |
|---|---|---|
| Megatron-Core | NVIDIA Megatron-LM | `core_r0.17.0` |
| TransformerEngineNPU | Ascend GitCode | `main` |
| MegatronAdaptor | Ascend GitCode | `core_r0.17.0` |

The refs will be Docker build arguments so image builds can override them
without editing the Dockerfiles.

The repository's local `mcore_adapter` package will also be installed after
the external dependencies, so ROLL's Megatron strategies are importable from
the image.

## Installation design

1. Install the small build prerequisites required by the stack (`setuptools<80`,
   `pybind11`, and a recent `packaging`).
2. Clone Megatron-LM, TransformerEngineNPU, and MegatronAdaptor into a
   temporary build directory with shallow clones.
3. Install Megatron-Core first, then TransformerEngineNPU, then
   MegatronAdaptor from the cloned source trees without build isolation. Use
   normal package installs so the temporary source trees can be removed after
   validation.
4. Install the local `mcore_adapter` package after the external stack.
5. Keep the existing ROLL, vLLM, torch-npu, and triton-ascend installation
   flow unchanged.
6. Run an import smoke check during image build for `megatron_adaptor`,
   `megatron.core`, and `transformer_engine.pytorch` after sourcing the Ascend
   runtime environment.
7. Remove cloned source trees and pip caches in the existing cleanup step.

## Compatibility and failure behavior

- A2 and A3 use the same package versions and install order; only the base
  image and `SOC_VERSION` differ.
- Build arguments make version mismatches explicit and reproducible at build
  time.
- A failed clone, package installation, or import smoke check fails the image
  build immediately.
- The Dockerfile will not install NVIDIA CUDA `transformer-engine[pytorch]`;
  TransformerEngineNPU is the Ascend implementation.

## Validation

- Verify both Dockerfiles contain the same stack refs and install order.
- Run Dockerfile syntax/static checks available in the workspace.
- If an Ascend Docker builder is available, build or run the import smoke check
  for both A2 and A3 images. Otherwise, report that runtime validation requires
  the corresponding Ascend environment.
