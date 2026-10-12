"""Remap initialized weights before allocating separate DDP gradient storage.

Apply to the isolated Megatron checkout. Tensor values, bucket layout and shared
MXFP8 buffers are unchanged. The reorder removes one parameter-sized transient
allocation from DistributedOptimizer initialization.
"""
import argparse
import hashlib
from pathlib import Path


def patch_source(source):
    marker = "# Remap original weights before allocating separate gradient storage."
    if marker in source:
        return source
    allocation = """                self.grad_data = torch.zeros(
                    self.numel,
                    dtype=self.grad_dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
"""
    begin = "            # For MXFP8 param: we only need to map weight gradients to the buffer.\n"
    end = "            param.main_grad = self._get(\n"
    insertion = "        # Finally, map param.data and param.main_grad fields to buffers.\n"
    for fragment in (allocation, begin, end, insertion, "        self.param_data = None\n"):
        if source.count(fragment) != 1:
            raise RuntimeError("Megatron buffer source differs; inspect before patching")
    start = source.index(begin)
    stop = source.index(end, start)
    remap = source[start:stop]
    source = source[:start] + source[stop:]
    source = source.replace(allocation, "")
    source = source.replace("        self.param_data = None\n", "        self.param_data = None\n        self.grad_data = None\n")
    replacement = (
        f"        {marker}\n"
        "        for param in params[::-1]:\n"
        "            param_start_index, _, _ = self.param_index_map[param]\n"
        + remap
        + "        with mem_alloc_context():\n"
        "            if self.grad_data is None:\n"
        + allocation
        + "\n        # Map gradients and build buckets after original weights are released.\n"
    )
    return source.replace(insertion, replacement)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("megatron_checkout", type=Path)
    args = parser.parse_args()
    file = args.megatron_checkout / "megatron/core/distributed/param_and_grad_buffer.py"
    original = file.read_text()
    patched = patch_source(original)
    compile(patched, str(file), "exec")
    before = hashlib.sha256(original.encode()).hexdigest()
    after = hashlib.sha256(patched.encode()).hexdigest()
    if patched != original:
        file.write_text(patched)
    print(f"{file} before={before} after={after} changed={patched != original}")


if __name__ == "__main__":
    main()
