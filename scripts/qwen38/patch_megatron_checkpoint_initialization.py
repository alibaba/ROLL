"""Bound Hybrid optimizer state initialization during distributed cold restore.

Apply after patch_megatron_cpu_grad_staging.py to the isolated Megatron copy.
In bounded mode, initialize each CPU suboptimizer from the reusable CPU gradient
buffer. Do not create a full CUDA gradient image or consume model RNG state.
Reuse the existing master parameter owners when loading a same-layout state;
recreating all CPU masters transiently adds another full FP32 model image.
"""
import argparse
import hashlib
from pathlib import Path


SOURCE_SHA256 = "1688dc5b90e1d73f24a6fcb5ad431c179de7f0b3fbc655896bac3e5752003118"

OLD = '''        for group in self.param_groups:
            for param in group["params"]:
                param.grad = torch.randn_like(param)
        self.step()
        self.zero_grad()
'''

NEW = '''        if self.bounded_cpu_grad_staging:
            if self.state:
                return
            self._sync_hdo_param_groups_to_sub_optimizers()
            self._prepare_bounded_cpu_grad_staging()
            for optimizer in self.sub_optimizers:
                learning_rates = []
                gradients = []
                try:
                    for group in optimizer.param_groups:
                        learning_rates.append((group, group["lr"]))
                        # Let the actual optimizer allocate its state without
                        # changing the master or applying weight decay.
                        group["lr"] = 0.0
                        for param in group["params"]:
                            gradients.append((param, param.grad))
                            if optimizer is self.gpu_optimizer:
                                param.grad = torch.zeros_like(param)
                            else:
                                size = param.numel() * param.element_size()
                                staged = self._cpu_grad_staging_buffer[:size]
                                param.grad = staged.view(param.dtype).view(param.shape).zero_()
                    optimizer.step()
                finally:
                    for group, learning_rate in learning_rates:
                        group["lr"] = learning_rate
                    for param, gradient in gradients:
                        param.grad = gradient
            torch.cuda.current_stream().wait_stream(self._h2d_stream)
            self._sync_sub_optimizers_state_to_hdo()
            return
        for group in self.param_groups:
            for param in group["params"]:
                param.grad = torch.randn_like(param)
        self.step()
        self.zero_grad()
'''

REBUILD_OLD = '''            self._init_sub_optimizers()
            self._sync_hdo_param_groups_to_sub_optimizers()
            self._sync_hdo_state_to_sub_optimizers()
'''

REBUILD_NEW = '''            if self.bounded_cpu_grad_staging:
                # Optimizer.load_state_dict keeps the current parameter objects;
                # the pre-hook's FP32 aliases were restored just above. Reuse
                # these owners instead of cloning another full CPU master image.
                current_params = {p for group in self.param_groups for p in group["params"]}
                if current_params != set(self.param_to_inner_param):
                    raise ValueError("bounded Hybrid restore requires unchanged parameter ownership")
            else:
                self._init_sub_optimizers()
            self._sync_hdo_param_groups_to_sub_optimizers()
            self._sync_hdo_state_to_sub_optimizers()
'''

MASTER_OLD = '''        self._update_fp32_params_by_new_state()
        self._move_new_state_to_right_device()
'''

MASTER_NEW = '''        self._update_fp32_params_by_new_state()
        self._move_new_state_to_right_device()
        if self.bounded_cpu_grad_staging and self.param_update_in_fp32:
            # The loaded master has been copied into the actual optimizer
            # parameter. Keep only that canonical owner in the checkpoint state.
            for original_param, inner_param in self.param_to_inner_param.items():
                if original_param in self.state:
                    self.state[original_param]["master_param"] = inner_param
'''


def patch_source(source):
    replacements = [(OLD, NEW), (REBUILD_OLD, REBUILD_NEW), (MASTER_OLD, MASTER_NEW)]
    original = source
    for before, after in replacements:
        if original.count(after) == 1:
            original = original.replace(after, before, 1)
    if hashlib.sha256(original.encode()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Unsupported Megatron Hybrid source; inspect before patching")
    patched = original
    for before, after in replacements:
        if patched.count(before) != 1:
            raise RuntimeError("Megatron Hybrid initialization target is ambiguous")
        patched = patched.replace(before, after, 1)
    compile(patched, "hybrid_optimizer.py", "exec")
    return patched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("megatron_checkout", type=Path)
    args = parser.parse_args()
    target = args.megatron_checkout / "megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py"
    original = target.read_text()
    patched = patch_source(original)
    if patched != original:
        target.write_text(patched)
    print(f"{target}: {hashlib.sha256(patched.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
