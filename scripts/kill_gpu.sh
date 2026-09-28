#!/usr/bin/env bash
# kill_all_gpu_procs.sh
set -euo pipefail

# Collect GPU-related PIDs (compute + any process holding /dev/nvidia*)
pids=$(
  {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true
    fuser /dev/nvidia* 2>/dev/null | tr ' ' '\n' || true
  } | sed '/^$/d' | grep -E '^[0-9]+$' | sort -u || true
)

if [[ -z "${pids}" ]]; then
  echo "No GPU processes found."
  exit 0
fi

echo "Killing GPU processes: ${pids//$'\n'/ }"
# SIGKILL all found PIDs
echo "$pids" | xargs -r kill -9
echo "Done."
