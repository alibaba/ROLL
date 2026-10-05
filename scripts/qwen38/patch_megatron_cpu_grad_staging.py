"""Add opt-in, one-parameter CPU gradient staging to the isolated Megatron copy.

Enable with HybridDeviceOptimizer(..., bounded_cpu_grad_staging=True), or set
that attribute on each Hybrid instance before its first step. This requires
overlap_cpu_optimizer_d2h_h2d=True (one CPU suboptimizer per parameter).
FP32 masters, Adam moments, global clipping, and checkpoint state stay intact.
The retained staging allocation is at most the largest CPU parameter in bytes.
The distributed checkpoint loader also restores the actual Hybrid master tensor
in bounded mode, keeping the tensor consumed by CPU Adam consistent with state.
"""

import argparse
import hashlib
import json
from pathlib import Path


ORIGINAL_SHA256 = "28053c32a2468645215b19f246bd8e86e250ffab49a036e49b3f6940b8996dcb"
PATCHED_SHA256 = "1688dc5b90e1d73f24a6fcb5ad431c179de7f0b3fbc655896bac3e5752003118"
ORIGINAL_DISTRIBUTED_SHA256 = "7382c3de15e7b487aeb3c00c3ba5a460735f0d7f50c1d905d52b5695d1a1e7d3"
PATCHED_DISTRIBUTED_SHA256 = "e7a559b1c4bb94138cda6c52d3bc68df79306b3069345009c0d7d2f4331dd0d5"


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError("Megatron Hybrid source differs; inspect before patching")
    return source.replace(old, new, 1)


def patch_source(source):
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest == PATCHED_SHA256:
        return source
    if digest != ORIGINAL_SHA256:
        raise RuntimeError(f"Unsupported Megatron Hybrid source SHA256: {digest}")

    source = _replace_once(source,
        "        overlap_cpu_optimizer_d2h_h2d: bool = True,\n",
        "        overlap_cpu_optimizer_d2h_h2d: bool = True,\n"
        "        bounded_cpu_grad_staging: bool = False,\n")
    source = _replace_once(source,
        "        super(HybridDeviceOptimizer, self).__init__(\n",
        "        if bounded_cpu_grad_staging and not overlap_cpu_optimizer_d2h_h2d:\n"
        "            raise ValueError(\n"
        "                'bounded_cpu_grad_staging requires overlap_cpu_optimizer_d2h_h2d=True'\n"
        "            )\n"
        "        self.bounded_cpu_grad_staging = bounded_cpu_grad_staging\n"
        "        super(HybridDeviceOptimizer, self).__init__(\n")
    source = _replace_once(source,
        "        # Sync the grads from GPU to CPU.\n",
        "        # Bounded mode stages CPU gradients immediately before their update.\n"
        "        if self.bounded_cpu_grad_staging:\n"
        "            return\n\n"
        "        # Sync the grads from GPU to CPU.\n")
    source = _replace_once(source,
        "    def _register_param_copy_back_gpu_hook(self):\n",
        '''    def _prepare_bounded_cpu_grad_staging(self):
        """Reserve one reusable buffer, never a per-parameter gradient cache."""
        if not self.overlap_cpu_optimizer_d2h_h2d:
            raise ValueError(
                'bounded_cpu_grad_staging requires overlap_cpu_optimizer_d2h_h2d=True'
            )
        largest = 0
        for optimizer in self.cpu_optimizers:
            params = list(_param_generator(optimizer))
            if len(params) != 1:
                raise ValueError('bounded_cpu_grad_staging requires one parameter per CPU optimizer')
            param = params[0]
            param.grad = None
            largest = max(largest, param.numel() * param.element_size())
        self.cpu_copy_map_grad.clear()
        if largest and self._cpu_grad_staging_buffer is None:
            # A byte buffer also supports mixed parameter dtypes without separate pools.
            self._cpu_grad_staging_buffer = torch.empty(
                largest, dtype=torch.uint8, device='cpu', pin_memory=self.pin_cpu_grads
            )
        if largest and self._cpu_grad_staging_buffer.numel() < largest:
            raise RuntimeError('CPU parameter size changed after staging buffer allocation')

    def _stage_bounded_cpu_optimizer_grad(self, optimizer):
        param = next(_param_generator(optimizer))
        gpu_param = self.cpu_copys_map_gpu_param[param]
        grad = getattr(gpu_param, 'decoupled_grad', gpu_param.grad)
        param.requires_grad = False
        if grad is None:
            param.grad = None
            return
        size = param.numel() * param.element_size()
        staged = self._cpu_grad_staging_buffer[:size].view(param.dtype).view(param.shape)
        staged.copy_(grad, non_blocking=True)
        param.grad = staged
        self.cpu_copy_map_grad[param] = staged
        self._cpu_optimizer_map_data_event[optimizer] = self._d2h_stream.record_event()

    def _register_param_copy_back_gpu_hook(self):
''')
    source = _replace_once(source,
        "        self._d2h_stream.wait_stream(torch.cuda.current_stream())\n",
        "        if self.bounded_cpu_grad_staging:\n"
        "            self._prepare_bounded_cpu_grad_staging()\n\n"
        "        self._d2h_stream.wait_stream(torch.cuda.current_stream())\n")
    source = _replace_once(source,
        "        if self.gpu_optimizer:\n            self.gpu_optimizer.step(closure)\n",
        "        if self.gpu_optimizer:\n"
        "            if self.bounded_cpu_grad_staging:\n"
        "                torch.cuda.current_stream().wait_stream(self._d2h_stream)\n"
        "            self.gpu_optimizer.step(closure)\n")
    source = _replace_once(source,
        '''        for cpu_optimizer in self.cpu_optimizers:
            d2h_event = self._cpu_optimizer_map_data_event.pop(cpu_optimizer, None)
            if d2h_event is not None:
                d2h_event.synchronize()
            cpu_optimizer.step(closure)
''',
        '''        for cpu_optimizer in self.cpu_optimizers:
            if self.bounded_cpu_grad_staging:
                with torch.cuda.stream(self._d2h_stream):
                    self._stage_bounded_cpu_optimizer_grad(cpu_optimizer)
            d2h_event = self._cpu_optimizer_map_data_event.pop(cpu_optimizer, None)
            if d2h_event is not None:
                d2h_event.synchronize()
            try:
                cpu_optimizer.step(closure)
            finally:
                if self.bounded_cpu_grad_staging:
                    # CPU Adam consumes the gradient synchronously. H2D uses the
                    # master parameter, so the gradient buffer is now reusable.
                    for param in _param_generator(cpu_optimizer):
                        param.grad = None
                        self.cpu_copy_map_grad.pop(param, None)

        if self.bounded_cpu_grad_staging:
            # Subsequent GPU work must observe all asynchronous master copies.
            torch.cuda.current_stream().wait_stream(self._h2d_stream)
''')
    source = _replace_once(source,
        "        self.cpu_copy_map_grad: Dict[torch.Tensor, torch.Tensor] = defaultdict(torch.Tensor)\n",
        "        self._cpu_grad_staging_buffer = None\n"
        "        self.cpu_copy_map_grad: Dict[torch.Tensor, torch.Tensor] = defaultdict(torch.Tensor)\n")
    compile(source, "hybrid_optimizer.py", "exec")
    return source


def patch_distributed_source(source):
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest == PATCHED_DISTRIBUTED_SHA256:
        return source
    if digest != ORIGINAL_DISTRIBUTED_SHA256:
        raise RuntimeError(f"Unsupported Megatron distributed optimizer source SHA256: {digest}")
    source = _replace_once(source,
        '''                if isinstance(self.optimizer, HybridDeviceOptimizer):
                    if k == "param":
                        k = "master_param"
                    self.optimizer.state[sharded_model_param][k] = v
                    continue
''',
        '''                if isinstance(self.optimizer, HybridDeviceOptimizer):
                    if getattr(self.optimizer, "bounded_cpu_grad_staging", False):
                        # Parameter-state loading follows Hybrid.load_state_dict().
                        # Restore the actual master used by its suboptimizer; a
                        # replacement state entry alone leaves that tensor stale.
                        inner_param = self.optimizer.param_to_inner_param[sharded_model_param]
                        if k == "param":
                            inner_param.data.copy_(v)
                            self.optimizer.state[sharded_model_param]["master_param"] = inner_param
                        else:
                            self.optimizer.state[sharded_model_param][k] = v.to(inner_param.device)
                        continue
                    if k == "param":
                        k = "master_param"
                    self.optimizer.state[sharded_model_param][k] = v
                    continue
''')
    compile(source, "distrib_optimizer.py", "exec")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("megatron_checkout", type=Path)
    args = parser.parse_args()
    plans = []
    for relative, patch, expected in (
        ("megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py", patch_source, PATCHED_SHA256),
        ("megatron/core/optimizer/distrib_optimizer.py", patch_distributed_source, PATCHED_DISTRIBUTED_SHA256),
    ):
        target = args.megatron_checkout / relative
        original = target.read_text()
        patched = patch(original)
        before = hashlib.sha256(original.encode()).hexdigest()
        after = hashlib.sha256(patched.encode()).hexdigest()
        if after != expected:
            raise RuntimeError(f"Generated patch SHA256 differs: {after}")
        plans.append((target, patched, {"path": str(target), "before_sha256": before,
                                       "after_sha256": after, "changed": patched != original}))
    # Validate both files before making either change.
    for target, patched, report in plans:
        if report["changed"]:
            target.write_text(patched)
        print(json.dumps(report))


if __name__ == "__main__":
    main()
