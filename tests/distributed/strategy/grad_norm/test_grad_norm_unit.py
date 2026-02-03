import pytest
import torch
import torch.nn as nn


class TestGradientNormBasic:
    """Basic unit tests for gradient norm computation."""

    def test_simple_parameter_grad_norm(self):
        """Test gradient norm with a single parameter."""
        # Create a parameter with known gradient
        param = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

        # Manually set gradient: [2, 4, 6]
        param.grad = torch.tensor([2.0, 4.0, 6.0])

        # Expected L2 norm: sqrt(4 + 16 + 36) = sqrt(56) ≈ 7.4833
        expected_norm = torch.sqrt(torch.tensor(56.0))

        # Compute using PyTorch
        from torch.nn.utils import clip_grad_norm_

        computed_norm = clip_grad_norm_([param], max_norm=float("inf"))

        assert torch.allclose(
            computed_norm, expected_norm, rtol=1e-5, atol=1e-5
        ), f"Computed norm {computed_norm:.6f} != expected {expected_norm:.6f}"

    def test_multiple_parameters_grad_norm(self):
        """Test gradient norm with multiple parameters."""
        # Create parameters
        param1 = nn.Parameter(
            torch.tensor([3.0, 4.0])
        )  # grad will be [1, 0]
        param2 = nn.Parameter(
            torch.tensor([1.0, 2.0])
        )  # grad will be [0, 1]

        param1.grad = torch.tensor([1.0, 0.0])
        param2.grad = torch.tensor([0.0, 1.0])

        # Expected L2 norm: sqrt(1^2 + 0^2 + 0^2 + 1^2) = sqrt(2) ≈ 1.4142
        expected_norm = torch.sqrt(torch.tensor(2.0))

        from torch.nn.utils import clip_grad_norm_

        computed_norm = clip_grad_norm_(
            [param1, param2], max_norm=float("inf")
        )

        assert torch.allclose(
            computed_norm, expected_norm, rtol=1e-5, atol=1e-5
        ), f"Computed norm {computed_norm:.6f} != expected {expected_norm:.6f}"

    def test_model_grad_norm(self):
        """Test gradient norm computation through a simple model."""

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = nn.Parameter(torch.tensor([1.0, 2.0]))
                self.w2 = nn.Parameter(torch.tensor([3.0]))

        model = TinyModel()

        # Create a simple loss: L = w1[0]^2 + w1[1]^2 + w2[0]^2
        # Gradients: dL/dw1 = [2*w1[0], 2*w1[1]] = [2, 4]
        #            dL/dw2 = [2*w2[0]] = [6]
        loss = (model.w1**2).sum() + (model.w2**2).sum()
        loss.backward()

        # Verify gradients
        assert torch.allclose(
            model.w1.grad, torch.tensor([2.0, 4.0])
        ), f"w1.grad = {model.w1.grad}, expected [2, 4]"
        assert torch.allclose(
            model.w2.grad, torch.tensor([6.0])
        ), f"w2.grad = {model.w2.grad}, expected [6]"

        # Expected norm: sqrt(4 + 16 + 36) = sqrt(56)
        expected_norm = torch.sqrt(torch.tensor(56.0))

        from torch.nn.utils import clip_grad_norm_

        computed_norm = clip_grad_norm_(
            model.parameters(), max_norm=float("inf")
        )

        assert torch.allclose(
            computed_norm, expected_norm, rtol=1e-5, atol=1e-5
        ), f"Computed norm {computed_norm:.6f} != expected {expected_norm:.6f}"

    def test_grad_clipping(self):
        """Test that gradient clipping works correctly."""

        # Create parameter with large gradient
        param = nn.Parameter(torch.tensor([3.0, 4.0]))
        param.grad = torch.tensor([3.0, 4.0])  # norm = 5.0

        max_norm = 2.5
        from torch.nn.utils import clip_grad_norm_

        total_norm = clip_grad_norm_([param], max_norm=max_norm)

        # Total norm before clipping should be 5.0
        assert torch.allclose(
            total_norm, torch.tensor(5.0), rtol=1e-5
        ), f"Total norm {total_norm:.6f} != 5.0"

        # After clipping, gradient should be scaled by max_norm / total_norm = 2.5 / 5.0 = 0.5
        expected_grad = torch.tensor([1.5, 2.0])  # [3, 4] * 0.5
        assert torch.allclose(
            param.grad, expected_grad, rtol=1e-5, atol=1e-5
        ), f"Clipped grad {param.grad} != expected {expected_grad}"

        # Verify clipped norm
        clipped_norm = torch.norm(param.grad)
        assert torch.allclose(
            clipped_norm, torch.tensor(max_norm), rtol=1e-5, atol=1e-5
        ), f"Clipped norm {clipped_norm:.6f} != max_norm {max_norm}"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_grad_norm_cuda(self):
        """Test gradient norm computation on CUDA."""

        device = torch.device("cuda")

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5, bias=True)

        model = SimpleModel().to(device)

        # Forward pass
        x = torch.randn(8, 10, device=device)
        y_target = torch.randn(8, 5, device=device)
        y_pred = model(x)
        loss = ((y_pred - y_target) ** 2).mean()

        # Backward pass
        loss.backward()

        # Compute gradient norm
        from torch.nn.utils import clip_grad_norm_

        grad_norm = clip_grad_norm_(
            model.parameters(), max_norm=float("inf")
        )

        # Verify it's a valid number
        assert torch.isfinite(
            grad_norm
        ), f"Gradient norm is not finite: {grad_norm}"
        assert (
            grad_norm > 0
        ), f"Gradient norm should be positive, got {grad_norm}"

        # Manual computation
        total_norm_sq = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = torch.norm(param.grad)
                total_norm_sq += param_norm**2
        manual_norm = torch.sqrt(total_norm_sq)

        assert torch.allclose(
            grad_norm, manual_norm, rtol=1e-4, atol=1e-5
        ), f"Computed norm {grad_norm:.6f} != manual norm {manual_norm:.6f}"


class TestGradientNormEdgeCases:
    """Test edge cases in gradient norm computation."""

    def test_zero_gradients(self):
        """Test gradient norm with zero gradients."""
        param = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        param.grad = torch.zeros_like(param)

        from torch.nn.utils import clip_grad_norm_

        grad_norm = clip_grad_norm_([param], max_norm=1.0)

        assert torch.allclose(
            grad_norm, torch.tensor(0.0)
        ), f"Zero gradient should have norm 0, got {grad_norm}"

    def test_no_gradients(self):
        """Test gradient norm when no parameters have gradients."""
        param = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        # Don't set grad (None)

        from torch.nn.utils import clip_grad_norm_

        grad_norm = clip_grad_norm_([param], max_norm=1.0)

        assert torch.allclose(
            grad_norm, torch.tensor(0.0)
        ), f"No gradient should have norm 0, got {grad_norm}"

    def test_mixed_gradients(self):
        """Test gradient norm when some parameters have gradients and others don't."""
        param1 = nn.Parameter(torch.tensor([3.0, 4.0]))
        param2 = nn.Parameter(torch.tensor([1.0, 2.0]))

        param1.grad = torch.tensor([3.0, 4.0])  # norm = 5
        # param2.grad is None

        from torch.nn.utils import clip_grad_norm_

        grad_norm = clip_grad_norm_([param1, param2], max_norm=float("inf"))

        expected_norm = torch.tensor(5.0)
        assert torch.allclose(
            grad_norm, expected_norm, rtol=1e-5
        ), f"Computed norm {grad_norm:.6f} != expected {expected_norm:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
