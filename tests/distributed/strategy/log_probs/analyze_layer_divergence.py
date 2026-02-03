import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


def load_layer_states(
    state_dir: Path, prefix: str, global_step: int, batch_idx: int = 0, subdir: str = "layers"
) -> Dict:
    """Load all layer states for a given step and batch."""
    layer_states = {}

    # Look in subdirectory (layers or embeddings)
    search_dir = state_dir / subdir if subdir else state_dir

    if not search_dir.exists():
        return layer_states

    # Find all files matching the pattern
    pattern = f"{prefix}_step{global_step}_batch{batch_idx}_*.pt"
    state_files = list(search_dir.glob(pattern))

    for state_file in state_files:
        # Parse filename patterns:
        # - {prefix}_step{step}_batch{batch}_layer_states_{layer_key}_{state_key}.pt
        # - {prefix}_step{step}_batch{batch}_{direct_key}.pt (e.g., inputs_embeds)
        stem = state_file.stem
        prefix_pattern = f"{prefix}_step{global_step}_batch{batch_idx}_"

        if not stem.startswith(prefix_pattern):
            continue

        # Remove prefix to get the key part
        key_part = stem[len(prefix_pattern) :]

        # Check if it's a layer_states file
        # Pattern: layer_states_layer_{N}_{state_key}
        # Example: layer_states_layer_0_before_attn
        if key_part.startswith("layer_states_"):
            parts = key_part.split("_")
            # parts = ['layer', 'states', 'layer', '0', 'before', 'attn', ...]
            if len(parts) >= 4 and parts[0] == "layer" and parts[1] == "states":
                # parts[2] = "layer", parts[3] = layer number
                layer_key = f"{parts[2]}_{parts[3]}"  # e.g., "layer_0"
                if len(parts) > 4:
                    state_key = "_".join(parts[4:])  # e.g., "before_attn"
                else:
                    state_key = "hidden_state"

                if layer_key not in layer_states:
                    layer_states[layer_key] = {}
                layer_states[layer_key][state_key] = torch.load(state_file)
        else:
            # Direct key (e.g., inputs_embeds, visual_image_embeds)
            layer_states[key_part] = torch.load(state_file)

    return layer_states


def compute_tensor_diff(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict:
    """Compute various difference metrics between two tensors."""
    transposed = None
    if tensor1.shape != tensor2.shape:
        # Common layout mismatch between frameworks:
        # - Megatron often uses [S, B, H]
        # - HF/FSDP often uses [B, S, H]
        # Try swapping the first two dims for 2D/3D tensors.
        if tensor1.dim() in (2, 3) and tensor2.dim() == tensor1.dim():
            if tensor1.transpose(0, 1).shape == tensor2.shape:
                tensor1 = tensor1.transpose(0, 1).contiguous()
                transposed = "tensor1"
                if mask is not None and mask.dim() >= 2 and mask.shape == tensor2.transpose(0, 1).shape:
                    mask = mask.transpose(0, 1).contiguous()
            elif tensor2.transpose(0, 1).shape == tensor1.shape:
                tensor2 = tensor2.transpose(0, 1).contiguous()
                transposed = "tensor2"
                if mask is not None and mask.dim() >= 2 and mask.shape == tensor2.transpose(0, 1).shape:
                    mask = mask.transpose(0, 1).contiguous()

    if tensor1.shape != tensor2.shape:
        return {
            "shape_mismatch": True,
            "shape1": list(tensor1.shape),
            "shape2": list(tensor2.shape),
        }

    # Handle bool tensors (e.g., visual_pos_masks)
    if tensor1.dtype == torch.bool or tensor2.dtype == torch.bool:
        if tensor1.dtype != tensor2.dtype:
            return {
                "dtype_mismatch": True,
                "dtype1": str(tensor1.dtype),
                "dtype2": str(tensor2.dtype),
            }
        # For bool tensors, compute element-wise equality
        equal = tensor1 == tensor2
        if mask is not None:
            if mask.shape != tensor1.shape:
                mask = mask.expand_as(tensor1)
            equal = equal | (~mask)  # Consider masked positions as equal
        num_different = (~equal).sum().item()
        total = equal.numel()
        return {
            "is_bool": True,
            "num_different": num_different,
            "total": total,
            "match_rate": (total - num_different) / total if total > 0 else 1.0,
        }

    diff = tensor1 - tensor2
    abs_diff = diff.abs()

    if mask is not None:
        if mask.shape != tensor1.shape:
            # Try to broadcast mask
            mask = mask.expand_as(tensor1)
        abs_diff_masked = abs_diff * mask
        max_diff = abs_diff_masked.max().item()
        mean_diff = abs_diff_masked[mask > 0].mean().item() if mask.any() else 0.0
        max_abs_value = torch.max(tensor1.abs(), tensor2.abs())[mask > 0].max().item() if mask.any() else 0.0
    else:
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        max_abs_value = torch.max(tensor1.abs(), tensor2.abs()).max().item()

    # Relative error
    relative_error = max_diff / (max_abs_value + 1e-10)

    # Cosine similarity
    tensor1_flat = tensor1.flatten()
    tensor2_flat = tensor2.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(tensor1_flat.unsqueeze(0), tensor2_flat.unsqueeze(0)).item()

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "relative_error": relative_error,
        "cosine_similarity": cos_sim,
        "shape_mismatch": False,
        "transposed": transposed,
    }


def compare_layer_states(fsdp_states: Dict, hf_states: Dict, attention_mask: Optional[torch.Tensor] = None) -> Dict:
    """Compare layer states between FSDP2 and HF.

    Handles both:
    - Nested structure: {layer_0: {before_attn: tensor, ...}, ...}  (layer states)
    - Flat structure: {inputs_embeds: tensor, ...}  (embeddings)
    """
    comparison = {}

    # Get all keys (union of both)
    all_keys = set(fsdp_states.keys()) | set(hf_states.keys())

    for key in sorted(all_keys):
        if key not in fsdp_states or key not in hf_states:
            comparison[key] = {"missing": True}
            continue

        fsdp_value = fsdp_states[key]
        hf_value = hf_states[key]

        # Check if this is a nested structure (layer states) or flat (embeddings)
        if isinstance(fsdp_value, dict) and isinstance(hf_value, dict):
            # Nested structure: layer states
            layer_comparison = {}

            # Compare each state within the layer
            all_state_keys = set(fsdp_value.keys()) | set(hf_value.keys())
            for state_key in sorted(all_state_keys):
                if state_key not in fsdp_value or state_key not in hf_value:
                    layer_comparison[state_key] = {"missing": True}
                    continue

                fsdp_tensor = fsdp_value[state_key]
                hf_tensor = hf_value[state_key]

                if isinstance(fsdp_tensor, torch.Tensor) and isinstance(hf_tensor, torch.Tensor):
                    # Skip comparison for visual_pos_masks (they're metadata, just check if they match)
                    if state_key == "visual_pos_masks":
                        if fsdp_tensor.shape == hf_tensor.shape and fsdp_tensor.dtype == hf_tensor.dtype:
                            match = (fsdp_tensor == hf_tensor).all().item()
                            layer_comparison[state_key] = {
                                "is_mask": True,
                                "match": match,
                                "shape": list(fsdp_tensor.shape),
                            }
                        else:
                            layer_comparison[state_key] = {
                                "is_mask": True,
                                "match": False,
                                "shape_mismatch": True,
                                "shape1": list(fsdp_tensor.shape),
                                "shape2": list(hf_tensor.shape),
                            }
                    else:
                        # Create mask for this state if attention_mask is provided
                        # Note: layer states might have different shapes, so we need to be careful
                        mask = None
                        if attention_mask is not None and state_key not in (
                            "visual_pos_masks",
                            "deepstack_visual_embeds",
                        ):
                            # Try to create appropriate mask based on tensor shape
                            if len(fsdp_tensor.shape) >= 2:
                                # attention_mask is [B, S]
                                if (
                                    fsdp_tensor.shape[0] == attention_mask.shape[0]
                                    and fsdp_tensor.shape[1] == attention_mask.shape[1]
                                ):
                                    # [B, S, ...]
                                    mask = attention_mask
                                elif (
                                    fsdp_tensor.shape[0] == attention_mask.shape[1]
                                    and fsdp_tensor.shape[1] == attention_mask.shape[0]
                                ):
                                    # [S, B, ...]
                                    mask = attention_mask.transpose(0, 1)
                                if mask is not None:
                                    mask = mask.unsqueeze(-1)
                                    while mask.dim() < fsdp_tensor.dim():
                                        mask = mask.unsqueeze(-1)
                                    mask = mask.expand_as(fsdp_tensor)

                        diff_stats = compute_tensor_diff(fsdp_tensor, hf_tensor, mask)
                        layer_comparison[state_key] = diff_stats
                else:
                    layer_comparison[state_key] = {"type_mismatch": True}

            comparison[key] = layer_comparison
        elif isinstance(fsdp_value, torch.Tensor) and isinstance(hf_value, torch.Tensor):
            # Flat structure: direct tensor comparison (embeddings)
            # Skip bool tensors (masks)
            if fsdp_value.dtype == torch.bool or hf_value.dtype == torch.bool:
                if fsdp_value.shape == hf_value.shape and fsdp_value.dtype == hf_value.dtype:
                    match = (fsdp_value == hf_value).all().item()
                    comparison[key] = {
                        "is_mask": True,
                        "match": match,
                        "shape": list(fsdp_value.shape),
                    }
                else:
                    comparison[key] = {
                        "is_mask": True,
                        "match": False,
                        "shape_mismatch": True,
                        "shape1": list(fsdp_value.shape),
                        "shape2": list(hf_value.shape),
                    }
            else:
                # Create mask if attention_mask is provided
                mask = None
                if attention_mask is not None:
                    if len(fsdp_value.shape) >= 2:
                        if (
                            fsdp_value.shape[0] == attention_mask.shape[0]
                            and fsdp_value.shape[1] == attention_mask.shape[1]
                        ):
                            mask = attention_mask
                        elif (
                            fsdp_value.shape[0] == attention_mask.shape[1]
                            and fsdp_value.shape[1] == attention_mask.shape[0]
                        ):
                            mask = attention_mask.transpose(0, 1)
                        if mask is not None:
                            mask = mask.unsqueeze(-1)
                            while mask.dim() < fsdp_value.dim():
                                mask = mask.unsqueeze(-1)
                            mask = mask.expand_as(fsdp_value)

                diff_stats = compute_tensor_diff(fsdp_value, hf_value, mask)
                comparison[key] = diff_stats
        else:
            comparison[key] = {"type_mismatch": True}

    return comparison


def find_divergence_point(comparison: Dict, threshold: float = 1e-5) -> Optional[int]:
    """Find the first point where divergence exceeds threshold.

    Supports both:
    - Nested layer structure: {layer_0: {before_attn: {max_diff: ...}, ...}, ...}
    - Flat tensor structure:  {inputs_embeds: {max_diff: ...}, ...}
      (e.g., if some tensors were saved directly under `layers/` without the `layer_states_` prefix)
    """
    for layer_idx, (layer_key, layer_comp) in enumerate(sorted(comparison.items())):
        # Defensive: some keys may map to non-dicts in malformed/partial outputs.
        if not isinstance(layer_comp, dict):
            continue

        # Flat diff-stats dict (max_diff/mean_diff/...) at top level
        if "max_diff" in layer_comp and isinstance(layer_comp.get("max_diff", None), (int, float)):
            if layer_comp.get("max_diff", 0) > threshold:
                return layer_idx, layer_key, "__tensor__"
            continue

        # Nested layer dict case
        if "missing" in layer_comp:
            continue

        for state_key, state_comp in layer_comp.items():
            if not isinstance(state_comp, dict):
                continue
            if "missing" in state_comp or "type_mismatch" in state_comp:
                continue

            if state_comp.get("max_diff", 0) > threshold:
                return layer_idx, layer_key, state_key

    return None


def analyze_divergence(
    fsdp_dir: Path,
    hf_dir: Path,
    inputs_dir: Path,
    output_file: Path,
    fsdp_prefix: str = "fsdp2",
    hf_prefix: str = "hf",
    fsdp_name: str = "FSDP2",
    hf_name: str = "HF",
    global_step: int = 0,
    batch_idx: int = 0,
    threshold: float = 1e-5,
):
    """Main analysis function."""
    print(f"Analyzing divergence for step {global_step}, batch {batch_idx}")

    # Load embeddings first
    print("Loading embeddings...")
    fsdp_embeddings = load_layer_states(fsdp_dir, fsdp_prefix, global_step, batch_idx, subdir="embeddings")
    hf_embeddings = load_layer_states(hf_dir, hf_prefix, global_step, batch_idx, subdir="embeddings")
    print(f"{fsdp_name} embeddings: {list(fsdp_embeddings.keys())}")
    print(f"{hf_name} embeddings: {list(hf_embeddings.keys())}")

    # Load layer states
    print(f"Loading {fsdp_name} layer states...")
    fsdp_states = load_layer_states(fsdp_dir, fsdp_prefix, global_step, batch_idx, subdir="layers")
    print(f"Loaded {len(fsdp_states)} layers from {fsdp_name}")

    print(f"Loading {hf_name} layer states...")
    hf_states = load_layer_states(hf_dir, hf_prefix, global_step, batch_idx, subdir="layers")
    print(f"Loaded {len(hf_states)} layers from {hf_name}")

    # Load attention mask if available
    attention_mask = None
    mask_file = inputs_dir / f"input_step{global_step}_batch{batch_idx}_attention_mask.pt"
    if mask_file.exists():
        attention_mask = torch.load(mask_file)
        print(f"Loaded attention mask: {attention_mask.shape}")

    # Compare embeddings first
    print("Comparing embeddings...")
    embedding_comparison = compare_layer_states(fsdp_embeddings, hf_embeddings, attention_mask)

    # Compare states
    print("Comparing layer states...")
    comparison = compare_layer_states(fsdp_states, hf_states, attention_mask)

    # Find divergence point
    divergence_point = find_divergence_point(comparison, threshold)

    # Generate summary
    summary = {
        "global_step": global_step,
        "batch_idx": batch_idx,
        "fsdp_prefix": fsdp_prefix,
        "hf_prefix": hf_prefix,
        "fsdp_name": fsdp_name,
        "hf_name": hf_name,
        "num_fsdp_layers": len(fsdp_states),
        "num_hf_layers": len(hf_states),
        "divergence_threshold": threshold,
        "divergence_point": divergence_point,
        "embedding_comparison": embedding_comparison,
        "layer_comparison": comparison,
    }

    # Add per-layer summary
    layer_summaries = []
    for layer_key, layer_comp in sorted(comparison.items()):
        if not isinstance(layer_comp, dict):
            continue

        # Flat diff-stats dict at top level (treat as a "layer" summary entry too)
        if "max_diff" in layer_comp and isinstance(layer_comp.get("max_diff", None), (int, float)):
            layer_summaries.append(
                {
                    "layer": layer_key,
                    "max_diff": float(layer_comp.get("max_diff", 0.0)),
                    "mean_diff": float(layer_comp.get("mean_diff", 0.0)),
                    "max_relative_error": float(layer_comp.get("relative_error", 0.0)),
                    "min_cosine_similarity": float(layer_comp.get("cosine_similarity", 1.0)),
                }
            )
            continue

        if "missing" in layer_comp:
            continue

        layer_max_diff = 0.0
        layer_mean_diff = 0.0
        layer_max_relative_error = 0.0
        layer_min_cosine_sim = 1.0

        for state_key, state_comp in layer_comp.items():
            if not isinstance(state_comp, dict):
                continue
            if "missing" in state_comp or "type_mismatch" in state_comp:
                continue

            layer_max_diff = max(layer_max_diff, state_comp.get("max_diff", 0))
            layer_mean_diff = max(layer_mean_diff, state_comp.get("mean_diff", 0))
            layer_max_relative_error = max(layer_max_relative_error, state_comp.get("relative_error", 0))
            layer_min_cosine_sim = min(layer_min_cosine_sim, state_comp.get("cosine_similarity", 1.0))

        layer_summaries.append(
            {
                "layer": layer_key,
                "max_diff": layer_max_diff,
                "mean_diff": layer_mean_diff,
                "max_relative_error": layer_max_relative_error,
                "min_cosine_similarity": layer_min_cosine_sim,
            }
        )

    summary["layer_summaries"] = layer_summaries

    # Save results
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nAnalysis complete. Results saved to {output_file}")

    # Analyze embedding divergence
    print("\n" + "=" * 80)
    print("EMBEDDING ANALYSIS")
    print("=" * 80)
    if embedding_comparison:
        print("\nEmbedding differences:")
        max_emb_diff = 0.0
        max_emb_rel_err = 0.0
        for emb_key, emb_stats in sorted(embedding_comparison.items()):
            if "is_mask" in emb_stats:
                print(f"  {emb_key}: ✓ Match (metadata)")
                continue
            if "shape_mismatch" in emb_stats and emb_stats["shape_mismatch"]:
                print(f"  {emb_key}: ✗ SHAPE MISMATCH")
                continue
            max_diff = emb_stats.get("max_diff", 0)
            rel_err = emb_stats.get("relative_error", 0)
            cos_sim = emb_stats.get("cosine_similarity", 1.0)
            max_emb_diff = max(max_emb_diff, max_diff)
            max_emb_rel_err = max(max_emb_rel_err, rel_err)

            # Determine severity
            severity = ""
            if max_diff > 0.01 or rel_err > 0.01:
                severity = " ⚠️  HIGH"
            elif max_diff > 0.001 or rel_err > 0.001:
                severity = " ⚠️  MEDIUM"

            print(
                f"  {emb_key}: max_diff={max_diff:.6f}, "
                f"rel_error={rel_err:.6f}, "
                f"cos_sim={cos_sim:.6f}{severity}"
            )

        print(f"\nEmbedding Summary:")
        print(f"  Max absolute difference: {max_emb_diff:.6f}")
        print(f"  Max relative error: {max_emb_rel_err:.6f}")

        if max_emb_diff > 0.01 or max_emb_rel_err > 0.01:
            print(f"\n  ⚠️  WARNING: Significant divergence detected at EMBEDDING phase!")
            print(f"     This is likely the ROOT CAUSE of logprobs differences.")
            print(f"     Possible causes:")
            print(f"     1. Different input_ids or tokenization")
            print(f"     2. Different visual encoder outputs (vision model differences)")
            print(f"     3. Different embedding layer weights (model loading/initialization)")
            print(f"     4. Numerical precision differences in embedding computation")
            print(f"     → Check if input_ids are identical between FSDP2 and HF")
            print(f"     → Check if pixel_values are processed identically")
        elif max_emb_diff > 0.001:
            print(f"\n  ⚠️  Moderate differences at embedding phase")
            print(f"     These may accumulate through layers")
        else:
            print(f"\n  ✓ Embeddings are very similar (differences likely numerical precision)")

    print("\n" + "=" * 80)
    print("LAYER-BY-LAYER ANALYSIS")
    print("=" * 80)
    print(f"\nDivergence point (threshold={threshold}): {divergence_point}")
    print("\nLayer summaries (showing divergence progression):")

    # Analyze embedding divergence
    print("\n" + "=" * 80)
    print("EMBEDDING ANALYSIS")
    print("=" * 80)
    if embedding_comparison:
        print("\nEmbedding differences:")
        for emb_key, emb_stats in sorted(embedding_comparison.items()):
            if "is_mask" in emb_stats:
                continue
            if "shape_mismatch" in emb_stats and emb_stats["shape_mismatch"]:
                print(f"  {emb_key}: SHAPE MISMATCH")
                continue
            max_diff = emb_stats.get("max_diff", 0)
            rel_err = emb_stats.get("relative_error", 0)
            cos_sim = emb_stats.get("cosine_similarity", 1.0)
            print(f"  {emb_key}: max_diff={max_diff:.6f}, " f"rel_error={rel_err:.6f}, " f"cos_sim={cos_sim:.6f}")

        # Check if embeddings show significant divergence
        max_emb_diff = max(
            (
                emb_stats.get("max_diff", 0)
                for emb_stats in embedding_comparison.values()
                if "is_mask" not in emb_stats and not emb_stats.get("shape_mismatch", False)
            ),
            default=0,
        )
        max_emb_rel_err = max(
            (
                emb_stats.get("relative_error", 0)
                for emb_stats in embedding_comparison.values()
                if "is_mask" not in emb_stats and not emb_stats.get("shape_mismatch", False)
            ),
            default=0,
        )

        print(f"\nEmbedding summary:")
        print(f"  Max absolute difference: {max_emb_diff:.6f}")
        print(f"  Max relative error: {max_emb_rel_err:.6f}")

        if max_emb_diff > 1e-3 or max_emb_rel_err > 0.01:
            print(f"  ⚠️  WARNING: Significant divergence detected at embedding phase!")
            print(f"     This suggests differences in:")
            print(f"     - Input token embeddings (check if input_ids are identical)")
            print(f"     - Visual encoder outputs (check vision model implementation)")
            print(f"     - Embedding layer weights (check model initialization/loading)")
        else:
            print(f"  ✓ Embeddings are very similar (differences likely due to numerical precision)")

    print("\n" + "=" * 80)
    print("LAYER-BY-LAYER ANALYSIS")
    print("=" * 80)
    print("\nLayer summaries (showing divergence progression):")
    for i, layer_summary in enumerate(layer_summaries[:20]):  # Show first 20 layers
        layer_name = layer_summary["layer"]
        max_diff = layer_summary["max_diff"]
        rel_err = layer_summary["max_relative_error"]
        cos_sim = layer_summary["min_cosine_similarity"]

        # Mark significant divergence
        marker = ""
        if max_diff > threshold:
            marker = " ⚠️  DIVERGED"
        elif max_diff > threshold / 10:
            marker = " ⚠️  WARNING"

        print(
            f"  [{i:2d}] {layer_name}: max_diff={max_diff:.6e}, "
            f"rel_error={rel_err:.6e}, "
            f"cos_sim={cos_sim:.6f}{marker}"
        )

    if len(layer_summaries) > 20:
        print(f"  ... ({len(layer_summaries) - 20} more layers)")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze layer state divergence between FSDP2 and HF")
    parser.add_argument("--fsdp-dir", type=str, required=True, help="Directory containing FSDP2 layer states")
    parser.add_argument("--hf-dir", type=str, required=True, help="Directory containing HF layer states")
    parser.add_argument("--inputs-dir", type=str, required=True, help="Directory containing input tensors")
    parser.add_argument("--output", type=str, default="divergence_analysis.json", help="Output JSON file")
    parser.add_argument("--step", type=int, default=0, help="Global step to analyze")
    parser.add_argument("--batch", type=int, default=0, help="Batch index to analyze")
    parser.add_argument("--threshold", type=float, default=1e-5, help="Divergence threshold")
    parser.add_argument("--fsdp-prefix", type=str, default="fsdp2", help="Prefix used for FSDP2 saved tensors")
    parser.add_argument(
        "--hf-prefix", type=str, default="hf", help="Prefix used for baseline saved tensors (hf/megatron)"
    )
    parser.add_argument("--fsdp-name", type=str, default="FSDP2", help="Display name for FSDP side")
    parser.add_argument("--hf-name", type=str, default="HF", help="Display name for baseline side")

    args = parser.parse_args()

    analyze_divergence(
        fsdp_dir=Path(args.fsdp_dir),
        hf_dir=Path(args.hf_dir),
        inputs_dir=Path(args.inputs_dir),
        output_file=Path(args.output),
        fsdp_prefix=args.fsdp_prefix,
        hf_prefix=args.hf_prefix,
        fsdp_name=args.fsdp_name,
        hf_name=args.hf_name,
        global_step=args.step,
        batch_idx=args.batch,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
