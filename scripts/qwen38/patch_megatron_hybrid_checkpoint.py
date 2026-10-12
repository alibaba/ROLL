"""Correct Hybrid Adam checkpoint serialization in the pinned Megatron checkout.

Apply after patch_megatron_cpu_grad_staging.py. Scalar Adam step is saved in
DistributedOptimizer's common parameter-group state, outside parameter shards.
"""
import argparse
import hashlib
from pathlib import Path


SOURCE_SHA256 = "e7a559b1c4bb94138cda6c52d3bc68df79306b3069345009c0d7d2f4331dd0d5"


def patch_source(source):
    old = '''                if isinstance(self.optimizer, HybridDeviceOptimizer):
                    tensors[k] = self.optimizer.state[sharded_model_param][k]
                    continue
'''
    new = '''                if isinstance(self.optimizer, HybridDeviceOptimizer):
                    # Adam's scalar step is restored from common param-group
                    # metadata; it cannot use a model-shaped tensor shard.
                    if k != "step":
                        tensors[k] = self.optimizer.state[sharded_model_param][k]
                    continue
'''
    old_load = '''                        # Main param & optimizer states.
                        self._set_main_param_and_optimizer_states(model_param, src_tensors)

    @torch.no_grad()
    def load_parameter_state_from_fs_model_space'''
    new_load = '''                        # Padding describes the checkpoint bucket, not Adam state.
                        src_tensors = {k: v for k, v in src_tensors.items() if k != "padding"}
                        self._set_main_param_and_optimizer_states(model_param, src_tensors)

    @torch.no_grad()
    def load_parameter_state_from_fs_model_space'''
    replacements = [(old, new), (old_load, new_load)]
    original = source
    for before, after in replacements:
        if original.count(after) == 1:
            original = original.replace(after, before, 1)
    if hashlib.sha256(original.encode()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Unsupported Megatron checkpoint source; inspect before patching")
    patched = original
    for before, after in replacements:
        if patched.count(before) != 1:
            raise RuntimeError("Megatron checkpoint patch target is ambiguous")
        patched = patched.replace(before, after, 1)
    compile(patched, "distrib_optimizer.py", "exec")
    return patched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("megatron_checkout", type=Path)
    args = parser.parse_args()
    target = args.megatron_checkout / "megatron/core/optimizer/distrib_optimizer.py"
    source = target.read_text()
    patched = patch_source(source)
    if patched != source:
        target.write_text(patched)
    print(f"{target}: {hashlib.sha256(patched.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
