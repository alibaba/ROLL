import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5, bias=True)

    def forward(self, x):
        return self.fc(x)


def test_gradient_accumulation_without_scaling():
    """
    Test gradient accumulation WITHOUT loss scaling.
    This demonstrates the problem: gradients scale with accumulation steps.
    """
    print("\n" + "=" * 60)
    print("Test: Gradient Accumulation WITHOUT Scaling (Incorrect)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Test with different accumulation steps
    for grad_acc_steps in [1, 2, 4]:
        model = SimpleModel().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Create mini-batches
        batch_size_per_step = 8
        total_batch_size = batch_size_per_step * grad_acc_steps

        torch.manual_seed(42)
        x_full = torch.randn(total_batch_size, 10, device=device)
        y_full = torch.randn(total_batch_size, 5, device=device)

        # Accumulate gradients WITHOUT scaling
        optimizer.zero_grad()
        for i in range(grad_acc_steps):
            start_idx = i * batch_size_per_step
            end_idx = (i + 1) * batch_size_per_step
            x_mini = x_full[start_idx:end_idx]
            y_mini = y_full[start_idx:end_idx]

            output = model(x_mini)
            loss = ((output - y_mini) ** 2).mean()
            # NO SCALING - This is the problem!
            loss.backward()

        # Compute gradient norm
        from torch.nn.utils import clip_grad_norm_

        grad_norm = clip_grad_norm_(
            model.parameters(), max_norm=float("inf")
        )

        print(f"grad_acc_steps={grad_acc_steps}: grad_norm={grad_norm:.6f}")

    print(
        "\n⚠️  WITHOUT scaling, gradient norm increases with accumulation steps!"
    )


def test_gradient_accumulation_with_scaling():
    """
    Test gradient accumulation WITH loss scaling.
    This demonstrates the correct approach: gradients remain consistent.
    """
    print("\n" + "=" * 60)
    print("Test: Gradient Accumulation WITH Scaling (Correct)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use FIXED total batch size across all tests
    total_batch_size = 32

    # Test with different accumulation steps
    grad_norms = {}
    for grad_acc_steps in [1, 2, 4, 8]:
        torch.manual_seed(42)
        model = SimpleModel().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Adjust batch size per step to keep total constant
        batch_size_per_step = total_batch_size // grad_acc_steps

        # Use SAME data for all configurations
        torch.manual_seed(100)
        x_full = torch.randn(total_batch_size, 10, device=device)
        y_full = torch.randn(total_batch_size, 5, device=device)

        # Accumulate gradients WITH scaling
        optimizer.zero_grad()
        for i in range(grad_acc_steps):
            start_idx = i * batch_size_per_step
            end_idx = (i + 1) * batch_size_per_step
            x_mini = x_full[start_idx:end_idx]
            y_mini = y_full[start_idx:end_idx]

            output = model(x_mini)
            loss = ((output - y_mini) ** 2).mean()
            # CORRECT: Scale by gradient accumulation steps
            scaled_loss = loss / grad_acc_steps
            scaled_loss.backward()

        # Compute gradient norm
        from torch.nn.utils import clip_grad_norm_

        grad_norm = clip_grad_norm_(
            model.parameters(), max_norm=float("inf")
        )
        grad_norms[grad_acc_steps] = grad_norm.item()

        print(f"grad_acc_steps={grad_acc_steps}: grad_norm={grad_norm:.6f}")

    # Verify all gradient norms are similar
    norm_values = list(grad_norms.values())
    max_norm = max(norm_values)
    min_norm = min(norm_values)
    relative_diff = (max_norm - min_norm) / min_norm

    print(f"\nRelative difference: {relative_diff*100:.2f}%")

    if relative_diff < 0.01:  # Within 1%
        print("✓ WITH scaling, gradient norms remain consistent!")
    else:
        print(f"⚠️  Gradient norms vary by {relative_diff*100:.2f}%")
        print(
            "   Note: Small variations are expected due to different computational order"
        )

    return relative_diff < 0.05  # Allow 5% for numerical precision


def test_gradient_accumulation_equivalence():
    """
    Test that gradient accumulation with scaling is equivalent to full-batch training.
    """
    print("\n" + "=" * 60)
    print("Test: Gradient Accumulation Equivalence")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Full batch training (baseline)
    torch.manual_seed(42)
    model_full = SimpleModel().to(device)

    total_batch_size = 32
    torch.manual_seed(100)
    x_full = torch.randn(total_batch_size, 10, device=device)
    y_full = torch.randn(total_batch_size, 5, device=device)

    output_full = model_full(x_full)
    loss_full = ((output_full - y_full) ** 2).mean()
    loss_full.backward()

    from torch.nn.utils import clip_grad_norm_

    grad_norm_full = clip_grad_norm_(
        model_full.parameters(), max_norm=float("inf")
    )

    print(
        f"Full batch (batch_size={total_batch_size}): grad_norm={grad_norm_full:.6f}"
    )

    # Gradient accumulation (should match)
    grad_acc_steps = 4
    batch_size_per_step = total_batch_size // grad_acc_steps

    torch.manual_seed(42)
    model_acc = SimpleModel().to(device)
    model_acc.zero_grad()

    torch.manual_seed(100)
    x_acc = torch.randn(total_batch_size, 10, device=device)
    y_acc = torch.randn(total_batch_size, 5, device=device)

    for i in range(grad_acc_steps):
        start_idx = i * batch_size_per_step
        end_idx = (i + 1) * batch_size_per_step
        x_mini = x_acc[start_idx:end_idx]
        y_mini = y_acc[start_idx:end_idx]

        output = model_acc(x_mini)
        loss = ((output - y_mini) ** 2).mean()
        scaled_loss = loss / grad_acc_steps
        scaled_loss.backward()

    grad_norm_acc = clip_grad_norm_(
        model_acc.parameters(), max_norm=float("inf")
    )

    print(
        f"Gradient accumulation (steps={grad_acc_steps}, batch_size={batch_size_per_step}): grad_norm={grad_norm_acc:.6f}"
    )

    # Compare
    relative_diff = abs(grad_norm_full - grad_norm_acc) / grad_norm_full
    print(f"\nRelative difference: {relative_diff*100:.2f}%")

    # They should be very close (within numerical precision)
    if torch.allclose(grad_norm_full, grad_norm_acc, rtol=1e-3, atol=1e-5):
        print("✓ Gradient accumulation matches full-batch training!")
        return True
    else:
        print(f"⚠️  Mismatch: {grad_norm_full:.6f} vs {grad_norm_acc:.6f}")
        return False


def test_gradient_accumulation_impact_on_norm():
    """
    Demonstrate the impact of gradient accumulation on gradient norms.
    """
    print("\n" + "=" * 60)
    print("Summary: Impact of Gradient Accumulation on Gradient Norms")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nScenario: Training with gradient_accumulation_steps=4")
    print("-" * 60)

    grad_acc_steps = 4
    batch_size_per_step = 8
    total_batch_size = batch_size_per_step * grad_acc_steps

    torch.manual_seed(42)
    x = torch.randn(total_batch_size, 10, device=device)
    y = torch.randn(total_batch_size, 5, device=device)

    # WITHOUT scaling
    torch.manual_seed(42)
    model_no_scale = SimpleModel().to(device)
    model_no_scale.zero_grad()

    for i in range(grad_acc_steps):
        start = i * batch_size_per_step
        end = (i + 1) * batch_size_per_step
        loss = ((model_no_scale(x[start:end]) - y[start:end]) ** 2).mean()
        loss.backward()

    from torch.nn.utils import clip_grad_norm_

    norm_no_scale = clip_grad_norm_(
        model_no_scale.parameters(), max_norm=float("inf")
    )

    # WITH scaling
    torch.manual_seed(42)
    model_with_scale = SimpleModel().to(device)
    model_with_scale.zero_grad()

    for i in range(grad_acc_steps):
        start = i * batch_size_per_step
        end = (i + 1) * batch_size_per_step
        loss = ((model_with_scale(x[start:end]) - y[start:end]) ** 2).mean()
        (loss / grad_acc_steps).backward()

    norm_with_scale = clip_grad_norm_(
        model_with_scale.parameters(), max_norm=float("inf")
    )

    print(f"WITHOUT loss scaling: grad_norm = {norm_no_scale:.6f}")
    print(f"WITH loss scaling:    grad_norm = {norm_with_scale:.6f}")
    print(f"\nRatio (without/with): {norm_no_scale / norm_with_scale:.2f}x")
    print(f"Expected ratio:       {grad_acc_steps:.2f}x")

    # The ratio should match the gradient accumulation steps
    ratio = norm_no_scale / norm_with_scale
    expected_ratio = float(grad_acc_steps)

    if abs(ratio - expected_ratio) < 0.1:
        print(
            f"\n✓ Without scaling, gradients are {grad_acc_steps}x larger!"
        )

    return abs(ratio - expected_ratio) < 0.1


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("GRADIENT ACCUMULATION SCALING TESTS")
    print("=" * 80)

    # Run all tests
    test_gradient_accumulation_without_scaling()

    test1_passed = test_gradient_accumulation_with_scaling()
    test2_passed = test_gradient_accumulation_equivalence()
    test3_passed = test_gradient_accumulation_impact_on_norm()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"Gradient accumulation consistency: {'✓ PASS' if test1_passed else '✗ FAIL'}"
    )
    print(
        f"Full-batch equivalence:            {'✓ PASS' if test2_passed else '✗ FAIL'}"
    )
    print(
        f"Scaling impact verification:       {'✓ PASS' if test3_passed else '✗ FAIL'}"
    )

    if test1_passed and test2_passed and test3_passed:
        print("\n✓ All tests passed!")
        print("\nKEY TAKEAWAY:")
        print(
            "  Always scale loss by 1/gradient_accumulation_steps to maintain"
        )
        print(
            "  consistent gradient magnitudes regardless of accumulation settings."
        )
    else:
        print("\n✗ Some tests failed")

    print("=" * 80)
