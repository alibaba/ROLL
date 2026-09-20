"""Compare saved real-model MCA and independent HF artifacts without reloading weights."""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mca", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    metrics = []
    for length in cli.lengths:
        pairs = [(f"layer-{layer}", f"hidden-{length}-{layer}.pt", f"hidden-{length}-{layer}.pt")
                 for layer in range(48)]
        pairs.append(("token-logprobs", f"losses-{length}-rank-0.pt", f"losses-{length}.pt"))
        for name, actual_name, expected_name in pairs:
            actual = torch.load(cli.mca / actual_name, map_location="cpu", weights_only=True).float()
            expected = torch.load(cli.reference / expected_name, map_location="cpu", weights_only=True).float()
            assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
            error = actual - expected
            metrics.append(dict(name=name, length=length,
                                relative_l2=float(error.norm()/expected.norm().clamp_min(1e-12)),
                                mean_abs=float(error.abs().mean()), max_abs=float(error.abs().max())))
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(dict(mca=str(cli.mca), reference=str(cli.reference),
                                         comparisons=metrics), indent=2) + "\n")
    print(json.dumps(dict(worst=max(metrics, key=lambda item: item["relative_l2"]),
                          final=[m for m in metrics if m["name"] in ("layer-47", "token-logprobs")])))
    assert all(m["relative_l2"] < 0.03 for m in metrics), f"Parity gate failed; see {cli.output}"


if __name__ == "__main__":
    main()
