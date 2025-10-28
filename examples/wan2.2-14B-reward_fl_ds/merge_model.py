import json
from safetensors import safe_open
from safetensors.torch import save_file
import torch

shard_files = ['diffusion_pytorch_model-00001-of-00006.safetensors', 'diffusion_pytorch_model-00002-of-00006.safetensors',
"diffusion_pytorch_model-00003-of-00006.safetensors","diffusion_pytorch_model-00004-of-00006.safetensors",
"diffusion_pytorch_model-00005-of-00006.safetensors","diffusion_pytorch_model-00006-of-00006.safetensors"]
print("Shard files to load:", shard_files)

full_state_dict = {}

for shard_file in shard_files:
    print(f"Loading {shard_file}...")
    with safe_open(shard_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            full_state_dict[key] = f.get_tensor(key).to(dtype=torch.bfloat16)

output_file = "diffusion_pytorch_model.safetensors"
print(f"Saving merged model to {output_file}...")
save_file(full_state_dict, output_file)

print("Done! Merged model saved.")