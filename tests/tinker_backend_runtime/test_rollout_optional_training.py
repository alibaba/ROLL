"""Prove inference workers can import without optional training dependencies."""
from pathlib import Path
import subprocess
import sys


_WORKERS = Path(__file__).resolve().parents[2] / "roll/pipeline/tinker_backend_runtime/workers.py"
_STUBS = r"""
import importlib.util
import sys
from types import ModuleType, SimpleNamespace


def stub(name, **attrs):
    module = ModuleType(name)
    module.__dict__.update(attrs)
    module.__path__ = []
    sys.modules[name] = module
    return module


for name in (
    "roll", "roll.distributed", "roll.distributed.scheduler", "roll.pipeline",
    "roll.pipeline.tinker_backend_runtime", "roll.utils",
):
    stub(name)
stub("numpy")
stub("torch", Tensor=object)


class DataProto:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to(self, device):
        self.device = device
        return self


class InferWorker:
    pass


class ActorWorker:
    pass


def register(**kwargs):
    return lambda fn: fn


stub("roll.distributed.scheduler.decorator", register=register, Dispatch=SimpleNamespace(
    DP_MP_DISPATCH_FIRST=0, ONE_TO_ALL=1, ONE_TO_ALL_ONE=2, DP_MP_COMPUTE=3,
))
stub("roll.distributed.scheduler.protocol", DataProto=DataProto)
stub("roll.pipeline.base_worker", ActorWorker=ActorWorker, InferWorker=InferWorker)
stub("roll.pipeline.tinker_backend_runtime.vllm_primitives", compute_prompt_logprobs_with_vllm_strategy=object())
stub("roll.platforms", current_platform=object())
stub("roll.utils.context_managers", state_offload_manger=object())
stub("roll.utils.functionals", reduce_metrics=object())
stub("roll.utils.offload_states", OffloadStateType=object())
spec = importlib.util.spec_from_file_location("roll.pipeline.tinker_backend_runtime.workers", sys.argv[1])
workers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workers)
assert issubclass(workers.TinkerInferWorker, InferWorker)
assert "roll.pipeline.tinker_backend_runtime.megatron_primitives" not in sys.modules
assert not any(name == "megatron" or name.startswith("megatron.") for name in sys.modules)
"""


def _run(code):
    subprocess.run([sys.executable, "-c", _STUBS + code, str(_WORKERS)], check=True, capture_output=True, text=True)


def test_inference_workers_import_without_megatron():
    _run("")


def test_actor_checkpoint_load_imports_training_primitive_on_demand():
    _run(r"""
calls = []
stub("roll.pipeline.tinker_backend_runtime.megatron_primitives",
     load_checkpoint_for_tinker=lambda *args, **kwargs: calls.append((args, kwargs)))
actor = workers.TinkerActorWorker()
actor.strategy = object()
result = actor.load_checkpoint_tinker("/tmp/checkpoint", load_optimizer=False)
assert calls == [((actor.strategy,), {"load_dir": "/tmp/checkpoint", "load_optimizer": False})]
assert result.meta_info == {"metrics": {"tinker/load_optimizer": 0.0}}
assert result.device == "cpu"
""")
