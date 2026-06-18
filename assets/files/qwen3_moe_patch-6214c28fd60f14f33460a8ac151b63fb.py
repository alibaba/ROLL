import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor


# force each expert to participate in computation graph so FSDP could gather all expert outputs
def qwen3_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])

        if top_x.numel() > 0:
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        else:
            dummy_output = expert_layer(hidden_states[:1]) * 0.0
            final_hidden_states[:1] = final_hidden_states[:1] + dummy_output

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


def _iter_convert_fsdps_moe_weights(named_params):
    """
    Lazily convert FSDP MoE weights from FSDP format to HF/vLLM expected format.

    Yields (name, tensor, full_tensor_size) one at a time so that the caller can
    control memory by batching into buffers. For expert parameters stored as
    DTensors (EP mode), full_tensor() is called here to gather the shards, but
    because this is a generator the full tensor is only materialized when the
    caller consumes the item.

    FSDP format:
        - layers.0.mlp.experts.gate_up_proj: [num_experts, 2 * intermediate_dim, hidden_dim]
        - layers.0.mlp.experts.down_proj: [num_experts, hidden_dim, intermediate_dim]

    Expected format:
        - layers.0.mlp.experts.0.gate_proj.weight: [intermediate_dim, hidden_dim]
        - layers.0.mlp.experts.0.up_proj.weight:   [intermediate_dim, hidden_dim]
        - layers.0.mlp.experts.0.down_proj.weight: [hidden_dim, intermediate_dim]
    """
    for name, param in named_params:
        if "experts.gate_up_proj" in name:
            if isinstance(param, DTensor):
                param = param.full_tensor()

            gate_proj, up_proj = torch.chunk(param, 2, dim=1)
            num_experts = gate_proj.shape[0]

            for expert_idx in range(num_experts):
                gate_name = name.replace("gate_up_proj", f"{expert_idx}.gate_proj.weight")
                up_name = name.replace("gate_up_proj", f"{expert_idx}.up_proj.weight")
                yield gate_name, gate_proj[expert_idx]
                yield up_name, up_proj[expert_idx]

            # Allow GC to reclaim the full tensor once all slices are yielded
            del param, gate_proj, up_proj

        elif "experts.down_proj" in name:
            if isinstance(param, DTensor):
                param = param.full_tensor()

            num_experts = param.shape[0]

            for expert_idx in range(num_experts):
                down_name = name.replace("down_proj", f"{expert_idx}.down_proj.weight")
                yield down_name, param[expert_idx]

            del param

        else:
            yield name, param
