import importlib
import traceback
for name in ("torch", "transformer_engine.pytorch", "megatron.core", "fla", "mcore_adapter"):
    try:
        module = importlib.import_module(name)
        print(name, getattr(module, "__version__", ""), getattr(module, "__file__", ""), flush=True)
    except Exception:
        traceback.print_exc()
        raise
