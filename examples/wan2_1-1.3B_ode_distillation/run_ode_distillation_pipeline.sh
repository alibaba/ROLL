#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python examples/start_ode_distillation_pipeline.py \
  --config_path wan2_1-1.3B_ode_distillation \
  --config_name ode_distillation_config \
  "$@"
