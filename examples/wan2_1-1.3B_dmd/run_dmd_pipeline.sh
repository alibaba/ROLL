#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python examples/start_dmd_pipeline.py \
  --config_path wan2_1-1.3B_dmd \
  --config_name dmd_config "$@"
