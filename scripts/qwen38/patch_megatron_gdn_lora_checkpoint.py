"""Preserve GDN stacked-output sharding after MCA LoRA wraps input projection.

Apply after patch_megatron_gdn_decay.py to the isolated Megatron checkout.
Both the base weight and LoRA B contain the stacked Q/K/V/Z/beta/alpha output
dimension; LoRA A is shared by those outputs and must not be split.
"""
import argparse
import hashlib
from pathlib import Path


SOURCE_SHA256 = "c2f4e06014104c1fd698a8b34e5cbff288b52c51f04482b31b9ce9d307385188"

OLD = '''        assert sharded_state_dict[f"{prefix}in_proj.weight"].data.size(0) == in_proj_dim_local_tp, (
            in_proj_dim_local_tp,
            sharded_state_dict[f"{prefix}in_proj.weight"],
        )

        sharded_state_dict[f"{prefix}in_proj.weight"] = _split_tensor_factory(
            sharded_state_dict[f"{prefix}in_proj.weight"],
            [
                self.qk_dim_local_tp,
                self.qk_dim_local_tp,
                self.v_dim_local_tp,
                self.v_dim_local_tp,
                self.num_value_heads // self.tp_size,
                self.num_value_heads // self.tp_size,
            ],
            ["query", "key", "value", "z", "beta", "alpha"],
            0,
        )
'''

NEW = '''        input_prefix = f"{prefix}in_proj."
        base_keys = [key for key in (input_prefix + "weight", input_prefix + "base_layer.weight")
                     if key in sharded_state_dict]
        if len(base_keys) != 1:
            raise ValueError("GDN checkpoint requires exactly one input projection base weight")
        output_keys = base_keys + [key for key in sharded_state_dict
                                  if key.startswith(input_prefix + "lora_B.") and key.endswith(".weight")]
        for key in output_keys:
            tensor = sharded_state_dict[key]
            if tensor.data.size(0) != in_proj_dim_local_tp:
                raise ValueError(f"GDN stacked output shape mismatch for {key}: {tensor.data.shape}")
            sharded_state_dict[key] = _split_tensor_factory(
                tensor,
                [
                    self.qk_dim_local_tp,
                    self.qk_dim_local_tp,
                    self.v_dim_local_tp,
                    self.v_dim_local_tp,
                    self.num_value_heads // self.tp_size,
                    self.num_value_heads // self.tp_size,
                ],
                ["query", "key", "value", "z", "beta", "alpha"],
                0,
            )
'''


def patch_source(source):
    original = source.replace(NEW, OLD, 1) if source.count(NEW) == 1 else source
    if hashlib.sha256(original.encode()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Unsupported Megatron GDN source; inspect before patching")
    if original.count(OLD) != 1:
        raise RuntimeError("Megatron GDN input projection target is ambiguous")
    patched = original.replace(OLD, NEW, 1)
    compile(patched, "gated_delta_net.py", "exec")
    return patched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("megatron_checkout", type=Path)
    args = parser.parse_args()
    target = args.megatron_checkout / "megatron/core/ssm/gated_delta_net.py"
    original = target.read_text()
    patched = patch_source(original)
    if patched != original:
        target.write_text(patched)
    print(f"{target}: {hashlib.sha256(patched.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
