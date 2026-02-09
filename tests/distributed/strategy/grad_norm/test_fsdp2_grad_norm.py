import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.nn.utils.clip_grad import _get_total_norm

from roll.platforms import current_platform


class SimpleModel(nn.Module):

    def __init__(self, input_size=128, hidden_size=256, output_size=64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc3 = nn.Linear(hidden_size, output_size, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def manual_compute_grad_norm(model, norm_type=2.0):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            # If it's a DTensor, gather to full tensor first
            if isinstance(param.grad, DTensor):
                grad = param.grad.full_tensor()
            else:
                grad = param.grad
            grads.append(grad)

    if len(grads) == 0:
        return torch.tensor(0.0)

    # Compute total norm
    total_norm = torch.norm(
        torch.stack([torch.norm(g.detach(), norm_type) for g in grads]),
        norm_type,
    )
    return total_norm


def fsdp2_compute_grad_norm(model, norm_type=2.0):
    """
    Compute gradient norm using FSDP2 approach (operating on sharded gradients).
    """
    parameters = list(model.parameters())
    grads = [p.grad for p in parameters if p.grad is not None]

    if not grads:
        return torch.tensor(0.0, device=current_platform.current_device())

    total_norm = _get_total_norm(
        grads, norm_type, error_if_nonfinite=False, foreach=None
    )

    # Convert DTensor to full tensor to get global norm
    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()

    return total_norm


def test_gradient_norm_single_gpu():
    """Test gradient norm computation on a single GPU (no sharding)."""

    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    device = torch.device("cuda")

    # Create model and data
    model = SimpleModel().to(device)
    batch_size = 8
    input_data = torch.randn(batch_size, 128, device=device)
    target = torch.randn(batch_size, 64, device=device)

    # Forward pass
    output = model(input_data)
    loss = ((output - target) ** 2).mean()

    # Backward pass
    loss.backward()

    # Compute gradient norm manually
    manual_norm = manual_compute_grad_norm(model)

    # Compute gradient norm using PyTorch's built-in function
    from torch.nn.utils import clip_grad_norm_

    pytorch_norm = clip_grad_norm_(
        model.parameters(), max_norm=float("inf")
    )

    # They should match
    print(f"Manual norm: {manual_norm.item():.6f}")
    print(f"PyTorch norm: {pytorch_norm.item():.6f}")

    assert torch.allclose(
        manual_norm, pytorch_norm, rtol=1e-4, atol=1e-4
    ), f"Manual norm {manual_norm.item()} != PyTorch norm {pytorch_norm.item()}"

    print("✓ Single GPU gradient norm test passed!")


def test_gradient_norm_fsdp2_distributed():
    """
    Test gradient norm computation with FSDP2 in a distributed setting.
    This test should be run with torchrun or similar launcher.

    Example:
        torchrun --nproc_per_node=2 test_fsdp2_grad_norm.py
    """

    if not dist.is_initialized():
        # Initialize distributed if not already done
        if not torch.cuda.is_available():
            print("CUDA not available, skipping distributed test")
            return

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(
        f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    )
    torch.cuda.set_device(device)

    print(f"[Rank {rank}/{world_size}] Starting FSDP2 gradient norm test")

    # Set seed for reproducibility across ranks
    torch.manual_seed(42)

    # Create device mesh for FSDP2
    mesh = init_device_mesh(
        "cuda" if torch.cuda.is_available() else "cpu",
        (world_size,),
        mesh_dim_names=("fsdp",),
    )

    # Create model directly on device (not meta)
    model = SimpleModel().to(device)

    # Apply FSDP2 configuration using PyTorch's fully_shard
    from torch.distributed._composable.fsdp import fully_shard

    mixed_precision = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=True,
    )

    for module in model.modules():
        if isinstance(module, nn.Linear):
            fully_shard(
                module,
                mesh=mesh,
                reshard_after_forward=True,
                mp_policy=mixed_precision,
            )

    fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=True,
        mp_policy=mixed_precision,
    )

    torch.manual_seed(42 + rank)  # Different data per rank
    batch_size = 4
    input_data = torch.randn(
        batch_size, 128, device=device, dtype=torch.bfloat16
    )
    target = torch.randn(
        batch_size, 64, device=device, dtype=torch.bfloat16
    )

    # Forward pass
    output = model(input_data)
    loss = ((output - target) ** 2).mean()

    print(f"[Rank {rank}] Loss: {loss.item():.6f}")

    # Backward pass
    loss.backward()

    # Compute gradient norm using FSDP2 approach
    fsdp2_norm = fsdp2_compute_grad_norm(model)

    print(f"[Rank {rank}] FSDP2 gradient norm: {fsdp2_norm.item():.6f}")

    all_norms = [torch.zeros_like(fsdp2_norm) for _ in range(world_size)]
    dist.all_gather(all_norms, fsdp2_norm)

    if rank == 0:
        print(f"\n[Rank 0] Gradient norms from all ranks:")
        for r, norm in enumerate(all_norms):
            print(f"  Rank {r}: {norm.item():.6f}")

        for r, norm in enumerate(all_norms):
            assert torch.allclose(
                norm, all_norms[0], rtol=1e-3, atol=1e-5
            ), f"Rank {r} norm {norm.item()} != Rank 0 norm {all_norms[0].item()}"

        print("\n✓ FSDP2 distributed gradient norm test passed!")

    dist.barrier()

    if rank == 0:
        print("\nTest completed successfully!")


def test_gradient_norm_consistency():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    device = torch.device("cuda")

    # Create a very simple model for easy verification
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(
                torch.tensor([1.0, 2.0, 3.0], device=device)
            )

    model = TinyModel()

    loss = (model.w**2).sum()
    loss.backward()

    expected_grad = torch.tensor([2.0, 4.0, 6.0], device=device)
    assert torch.allclose(
        model.w.grad, expected_grad
    ), f"Expected grad {expected_grad}, got {model.w.grad}"

    expected_norm = torch.sqrt(torch.tensor(56.0, device=device))

    from torch.nn.utils import clip_grad_norm_

    pytorch_norm = clip_grad_norm_(
        model.parameters(), max_norm=float("inf")
    )

    print(f"Expected norm: {expected_norm.item():.6f}")
    print(f"PyTorch norm: {pytorch_norm.item():.6f}")

    assert torch.allclose(
        pytorch_norm, expected_norm, rtol=1e-4, atol=1e-4
    ), f"PyTorch norm {pytorch_norm.item()} != expected {expected_norm.item()}"

    print("✓ Gradient norm consistency test passed!")


if __name__ == "__main__":
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        print(
            f"Running in distributed mode (Rank {os.environ['RANK']}/{os.environ['WORLD_SIZE']})"
        )
        test_gradient_norm_fsdp2_distributed()
    else:
        print("Running in single-GPU mode")
        print("\n" + "=" * 60)
        print("Test 1: Gradient Norm Consistency")
        print("=" * 60)
        test_gradient_norm_consistency()

        print("\n" + "=" * 60)
        print("Test 2: Single GPU Gradient Norm")
        print("=" * 60)
        test_gradient_norm_single_gpu()

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        print("\nTo test distributed FSDP2, run:")
        print("  torchrun --nproc_per_node=2 test_fsdp2_grad_norm.py")
