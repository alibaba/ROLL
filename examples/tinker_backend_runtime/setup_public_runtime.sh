#!/usr/bin/env bash
# Inherit the CUDA stack from the torch2110/vllm0230 base image; install CPU helpers only.
set -euo pipefail
ROLL_ROOT="${ROLL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
GPU_VENV="${GPU_VENV:-$ROLL_ROOT/.venv-tinker-runtime}"
PYTHON="${PYTHON:-/usr/bin/python3}"
run_gpu_check=0
while (($#)); do
  case "$1" in
    --roll-root) ROLL_ROOT="$2"; shift 2 ;;
    --venv) GPU_VENV="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --gpu-check) run_gpu_check=1; shift ;;
    -h|--help)
      echo 'Usage: setup_public_runtime.sh [--roll-root PATH] [--venv PATH] [--python BASE_PYTHON]'
      echo 'Requires uv and the Python 3.12 torch2.11/vLLM0.23 CUDA13 base image.'
      echo 'Equivalent environment variables: ROLL_ROOT, GPU_VENV, PYTHON.'
      echo '--gpu-check compiles and runs a tiny Triton math kernel on visible CUDA device 0.'
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
command -v uv >/dev/null || { echo 'uv is required; install it before running this script.' >&2; exit 2; }
[[ -f "$ROLL_ROOT/setup.py" && -f "$ROLL_ROOT/mcore_adapter/pyproject.toml" ]] || {
  echo '--roll-root must point to the ROLL checkout.' >&2; exit 2;
}
ROLL_ROOT="$(cd "$ROLL_ROOT" && pwd)"
mkdir -p "$(dirname "$GPU_VENV")"
GPU_VENV="$(cd "$(dirname "$GPU_VENV")" && pwd)/$(basename "$GPU_VENV")"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

# Protect installed GPU distributions from uv's resolver even through transitive dependencies.
"$PYTHON" - "$scratch" "$ROLL_ROOT" <<'PY'
import importlib.metadata as m, json, pathlib, re, sys
out, root = map(pathlib.Path, sys.argv[1:])
if sys.version_info[:2] != (3, 12):
    raise SystemExit('The tested CUDA base requires Python 3.12')
def protected(name):
    name = name.lower().replace('_', '-')
    return (name in {'torch', 'torchaudio', 'torchvision', 'torchao', 'vllm',
                     'megatron-core', 'transformer-engine', 'transformer-engine-torch',
                     'transformer-engine-cu13', 'triton', 'pytorch-triton'}
            or name.startswith(('nvidia-', 'cuda-', 'flash-attn', 'flash-linear-attention')))
snapshot = {}
for d in m.distributions():
    name = d.metadata['Name'].lower().replace('_', '-')
    if protected(name):
        snapshot[name] = {'version': d.version, 'path': str(d.locate_file(''))}
for name, expected in [('torch', '2.11.0'), ('vllm', '0.23.0')]:
    if name not in snapshot or snapshot[name]['version'].split('+')[0] != expected:
        raise SystemExit(f'Base image must already provide {name}=={expected}')
for name in ['transformer-engine', 'transformer-engine-torch', 'megatron-core']:
    if name not in snapshot:
        raise SystemExit(f'Base image must already provide {name}')
(out/'base.json').write_text(json.dumps(snapshot))
(out/'excludes.txt').write_text('\n'.join(sorted(snapshot)) + '\n')
requirements = root/'requirements_tinker.txt'
lines = []
for line in requirements.read_text().splitlines():
    match = re.match(r'([A-Za-z0-9_.-]+)', line.strip())
    if match and protected(match.group(1)):
        continue
    lines.append(line)
(out/'helpers.txt').write_text('\n'.join(lines) + '\n')
PY

if [[ -e "$GPU_VENV" && ! -f "$GPU_VENV/pyvenv.cfg" ]]; then
  echo '--venv exists but is not a virtual environment; refusing to replace it.' >&2
  exit 2
fi
if [[ ! -f "$GPU_VENV/pyvenv.cfg" ]]; then
  uv --no-config venv --python "$PYTHON" --system-site-packages "$GPU_VENV"
fi
grep -q '^include-system-site-packages = true' "$GPU_VENV/pyvenv.cfg" || {
  echo 'The runtime venv must inherit system site packages.' >&2; exit 2;
}
uv --no-config pip install --python "$GPU_VENV/bin/python" \
  --index-url https://pypi.org/simple --excludes "$scratch/excludes.txt" \
  -r "$scratch/helpers.txt"
uv --no-config pip install --python "$GPU_VENV/bin/python" --no-deps \
  --index-url https://pypi.org/simple -e "$ROLL_ROOT" -e "$ROLL_ROOT/mcore_adapter" 'sglang==0.5.2'

# Imported CUDA libraries must match the base image. Preserve the image's preload;
# add its documented cublas libraries only when no preload was supplied.
if [[ -z "${LD_PRELOAD:-}" ]]; then
  cuda_lib=/usr/local/cuda/targets/x86_64-linux/lib
  if [[ -f "$cuda_lib/libcublasLt.so.13" && -f "$cuda_lib/libcublas.so.13" ]]; then
    export LD_PRELOAD="$cuda_lib/libcublasLt.so.13:$cuda_lib/libcublas.so.13"
  fi
fi
# The newer CUDA user libraries need the base image's forward-compat driver.
# Probe existing image paths; do not download libraries or change the system.
runtime_library_dirs=()
for candidate in /usr/local/cuda/compat/lib.real /usr/local/cuda/compat/lib /usr/local/cuda/compat; do
  if [[ -f "$candidate/libcuda.so.1" ]]; then
    runtime_library_dirs+=("$candidate")
    break
  fi
done
[[ ! -d /usr/local/cuda/targets/x86_64-linux/lib ]] || \
  runtime_library_dirs+=(/usr/local/cuda/targets/x86_64-linux/lib)
if ((${#runtime_library_dirs[@]})); then
  runtime_library_path="$(IFS=:; echo "${runtime_library_dirs[*]}")"
  export LD_LIBRARY_PATH="$runtime_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# NVIDIA's base-image Triton build does not bundle cuda.h. GCC consumes CPATH
# without modifying its generated command or any system/package directories.
cuda_include_dir=""
for candidate in "${CUDA_HOME:-/usr/local/cuda}/targets/x86_64-linux/include" \
                 "${CUDA_HOME:-/usr/local/cuda}/include"; do
  if [[ -f "$candidate/cuda.h" ]]; then
    cuda_include_dir="$candidate"
    break
  fi
done
[[ -n "$cuda_include_dir" ]] || {
  echo 'CUDA toolkit cuda.h is missing; use the documented CUDA development base image.' >&2
  exit 2
}
export CPATH="$cuda_include_dir${CPATH:+:$CPATH}"
pushd "$GPU_VENV" >/dev/null
PYTHONPATH="$ROLL_ROOT/mcore_adapter/src:$ROLL_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
"$GPU_VENV/bin/python" - "$scratch/base.json" "$GPU_VENV" "$run_gpu_check" <<'PY'
import importlib.metadata as m, json, os, pathlib, shutil, subprocess, sys, sysconfig, tempfile
snapshot = json.loads(pathlib.Path(sys.argv[1]).read_text())
for name, expected in snapshot.items():
    d = m.distribution(name)
    actual = {'version': d.version, 'path': str(d.locate_file(''))}
    if actual != expected:
        raise SystemExit(f'Protected base distribution was replaced: {name}: {actual}')
# Compile the actual Triton CUDA utility extension, but never load it or call
# CUDA. Imports alone do not expose a missing CUDA header in this base image.
compiler = shutil.which('gcc')
if not compiler:
    raise SystemExit('gcc is required for the Triton CPU compile preflight')
triton_root = pathlib.Path(m.distribution('triton').locate_file('triton/backends/nvidia'))
cuda_root = pathlib.Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))
tool_versions = {}
tool_env_names = []
for binary in ['ptxas', 'cuobjdump', 'nvdisasm']:
    variable = 'TRITON_'+binary.upper()+'_PATH'
    configured = os.environ.get(variable)
    candidates = [pathlib.Path(configured)] if configured else [
        triton_root/'bin'/binary, cuda_root/'bin'/binary]
    tool = next((p for p in candidates if p.is_file() and os.access(p, os.X_OK)), None)
    if tool is None:
        raise SystemExit(f'Missing executable CUDA tool: {variable}')
    os.environ[variable] = str(tool)
    tool_env_names.append(variable)
    tool_versions[binary] = subprocess.check_output([str(tool), '--version'], text=True).strip()
configured = os.environ.get('TRITON_LIBDEVICE_PATH')
candidates = [pathlib.Path(configured)] if configured else [
    triton_root/'lib/libdevice.10.bc', cuda_root/'nvvm/libdevice/libdevice.10.bc']
libdevice = next((p for p in candidates if p.is_file() and p.stat().st_size), None)
if libdevice is None:
    raise SystemExit('CUDA libdevice.10.bc is missing; set TRITON_LIBDEVICE_PATH')
os.environ['TRITON_LIBDEVICE_PATH'] = str(libdevice)
tool_env_names.append('TRITON_LIBDEVICE_PATH')
with tempfile.TemporaryDirectory(prefix='tinker-cuda-compile-') as work:
    command = [compiler, str(triton_root/'driver.c'), '-O3', '-shared', '-fPIC',
               '-Wno-psabi', '-o', str(pathlib.Path(work)/'cuda_utils.so'),
               '-l:libcuda.so.1', '-I'+str(triton_root/'include'),
               '-I'+sysconfig.get_path('include')]
    command.extend('-L'+p for p in os.environ.get('LD_LIBRARY_PATH', '').split(':')
                   if p and (pathlib.Path(p)/'libcuda.so.1').is_file())
    subprocess.run(command, check=True)
# CPU import contracts only: no ray.init(), CUDA tensors, or model loading.
import torch
import transformer_engine.pytorch
import megatron.core
import vllm
import roll
import mcore_adapter
from transformers import AutoTokenizer
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from roll.pipeline.tinker_backend_runtime import runtime_pipeline, roll_backend
# Exercise the lazy backend initialization import path, including both OTLP exporters.
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPSpanExporter
names = ['torch', 'vllm', 'transformer-engine', 'megatron-core', 'transformers',
         'peft', 'numpy', 'ray', 'tensordict', 'hydra-core', 'sglang', 'accelerate',
         'opentelemetry-api', 'opentelemetry-sdk', 'opentelemetry-proto',
         'opentelemetry-exporter-otlp', 'opentelemetry-exporter-otlp-proto-common',
         'opentelemetry-exporter-otlp-proto-grpc', 'opentelemetry-exporter-otlp-proto-http']
report = {'python': sys.version.split()[0], 'cpu_imports_passed': True,
          'protected_base_unchanged': True, 'versions': {n: m.version(n) for n in names},
          'runtime_env': {n: os.environ.get(n, '') for n in ['LD_LIBRARY_PATH', 'LD_PRELOAD', 'CPATH']+tool_env_names},
          'triton_cpu_compile_passed': True, 'triton_version': m.version('triton'),
          'cuda_tool_versions': tool_versions, 'triton_gpu_check_passed': False}
if sys.argv[3] == '1':
    # A file is required for Triton to inspect the @jit function's source.
    # A fresh cache proves the CUDA launcher, PTX assembler and libdevice path.
    with tempfile.TemporaryDirectory(prefix='tinker-triton-gpu-') as work:
        probe = pathlib.Path(work)/'probe.py'
        probe.write_text('''import json
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def math_probe(x, y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x + offsets, offsets < N, other=0)
    result = libdevice.sin(value) + tl.exp(value)
    tl.store(y + offsets, result, offsets < N)

x = torch.linspace(-1, 1, 257, dtype=torch.float32, device='cuda:0')
y = torch.empty_like(x)
math_probe[(triton.cdiv(x.numel(), 128),)](x, y, x.numel(), BLOCK=128)
torch.cuda.synchronize()
torch.testing.assert_close(y, torch.sin(x) + torch.exp(x), rtol=1e-5, atol=1e-6)
print(json.dumps({'elements': x.numel(), 'device': torch.cuda.get_device_name(0),
                  'math': 'libdevice.sin + tl.exp', 'fresh_cache': True}))
''')
        env = os.environ.copy()
        env['TRITON_CACHE_DIR'] = str(pathlib.Path(work)/'cache')
        result = subprocess.run([sys.executable, str(probe)], env=env, text=True,
                                capture_output=True, timeout=120)
        if result.returncode:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            raise SystemExit('Tiny Triton GPU compile/execution check failed')
        report['triton_gpu_check'] = json.loads(result.stdout.strip())
        report['triton_gpu_check_passed'] = True
path = pathlib.Path(sys.argv[2])/'setup-public-runtime.json'
path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
PY
popd >/dev/null
echo "GPU runtime ready: $GPU_VENV (optional tiny GPU check: $run_gpu_check)"
