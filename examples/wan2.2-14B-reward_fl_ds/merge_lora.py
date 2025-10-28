import os
import argparse
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def load_state_dict_from_safetensors(file_path, device="cpu"):
    state_dict = {}
    with safe_open(file_path, framework="pt", device=str(device)) as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    return state_dict

def merge_lora_into_state_dict(
    base_state_dict,
    lora_state_dict,
    alpha=1.0,
    device="cpu",
    dtype=torch.bfloat16
):
    lora_name_map = {}
    for key in lora_state_dict:
        if ".lora_B." not in key:
            continue
        clean_key = key.replace("base_model.model.", "")
        parts = clean_key.split(".")
        try:
            lora_b_idx = parts.index("lora_B")
        except ValueError:
            continue
        target_parts = parts[:lora_b_idx] + parts[lora_b_idx + 2:]  # 跳过 lora_B 和 rank idx
        if target_parts[0] == "diffusion_model":
            target_parts = target_parts[1:]
        target_name = ".".join(target_parts)
        lora_A_key = key.replace(".lora_B.", ".lora_A.").replace("base_model.model.", "")
        lora_name_map[target_name] = (key, lora_A_key)

    merged_state_dict = base_state_dict.copy()
    updated = 0

    for target_name, (lora_B_key, lora_A_key) in lora_name_map.items():
        if target_name not in merged_state_dict:
            print(f"Warning: {target_name} not in base model. Skipping.")
            continue

        weight_orig = merged_state_dict[target_name].to(device=device, dtype=dtype)
        weight_up = lora_state_dict[lora_B_key].to(device=device, dtype=dtype)
        weight_down = lora_state_dict[lora_A_key].to(device=device, dtype=dtype)

        if len(weight_up.shape) == 4:
            weight_up = weight_up.squeeze(-1).squeeze(-1)  # [out, r, 1, 1] -> [out, r]
            weight_down = weight_down.squeeze(-1).squeeze(-1)  # [r, in, 1, 1] -> [r, in]
            lora_weight = alpha * (weight_up @ weight_down).unsqueeze(-1).unsqueeze(-1)
        else:
            lora_weight = alpha * (weight_up @ weight_down)

        merged_state_dict[target_name] = weight_orig + lora_weight
        updated += 1

    print(f"Merged {updated} LoRA adapters into base model.")
    return merged_state_dict

parser = argparse.ArgumentParser()
parser.add_argument('--lora_dir', type=str, default="/data/models/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors")
parser.add_argument('--base_model_path', type=str, default="/data/models/Wan22_base/high_noise_model/diffusion_pytorch_model.safetensors")
parser.add_argument('--save_dir', type=str, default="/data/models/Wan22/high_noise_model/diffusion_pytorch_model.safetensors")
parser.add_argument('--alpha', type=float, default=1.0)
args = parser.parse_args()

print("Loading base model state dict...")
base_sd = load_state_dict_from_safetensors(args.base_model_path)

# load lora
lora_sd = load_state_dict_from_safetensors(args.lora_dir)

clean_lora_sd = {}
for k, v in lora_sd.items():
    clean_k = k.replace("base_model.model.", "")
    clean_lora_sd[clean_k] = v

# merge
merged_sd = merge_lora_into_state_dict(
    base_state_dict=base_sd,
    lora_state_dict=clean_lora_sd,
    alpha=args.alpha,
    device="cpu",
    dtype=torch.bfloat16 
)

# save
os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)
save_file(merged_sd, args.save_dir)
print(f"Merged model saved to {args.save_dir}")

