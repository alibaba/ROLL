"""Load the unchanged real checkpoint with native vLLM and compare logprobs.

The model's own loader performs name mapping exactly once. This direct-engine
probe establishes inference numerics; ROLL's update protocol is tested separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--mca-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--expert-parallel", action="store_true")
    cli = parser.parse_args()
    cli.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(cli.input_manifest.read_text())
    from vllm import LLM, SamplingParams, __version__
    started = time.monotonic()

    def record(event, **values):
        value = dict(event=event, elapsed=time.monotonic()-started, **values)
        with (cli.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        print(json.dumps(value), flush=True)

    record("load_start", vllm_version=__version__, model=cli.model,
           expert_parallel=cli.expert_parallel)
    engine = LLM(model=cli.model, tensor_parallel_size=8, dtype="bfloat16",
                 max_model_len=max(cli.lengths)+8, max_num_batched_tokens=max(cli.lengths),
                 max_num_seqs=1, gpu_memory_utilization=0.85, enforce_eager=True,
                 enable_expert_parallel=cli.expert_parallel,
                 enable_prefix_caching=False, trust_remote_code=True, seed=1234)
    record("loaded")
    sampling = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1)
    for domain, fixture in manifest["fixtures"].items():
        ids = torch.load(cli.input_manifest.parent / fixture["tensor_file"], weights_only=True)
        assert hashlib.sha256(ids.numpy().tobytes()).hexdigest() == fixture["tokens_sha256"]
        for length in cli.lengths:
            tokens = ids[0, :length].tolist()
            tick = time.monotonic()
            result = engine.generate([{"prompt_token_ids": tokens}], sampling, use_tqdm=False)[0]
            assert result.prompt_token_ids == tokens
            assert len(result.prompt_logprobs) == length
            reference = torch.tensor([-entry[token].logprob
                                      for token, entry in zip(tokens[1:], result.prompt_logprobs[1:])])
            actual = torch.load(cli.mca_output / domain / f"losses-{length}-rank-0.pt",
                                weights_only=True)[0, :-1].float()
            assert torch.isfinite(reference).all()
            difference = actual-reference
            record("comparison", domain=domain, length=length, seconds=time.monotonic()-tick,
                   vllm_mean=float(reference.mean()), mca_mean=float(actual.mean()),
                   relative_l2=float(difference.norm()/reference.norm()),
                   mean_abs=float(difference.abs().mean()), max_abs=float(difference.abs().max()),
                   p95_abs=float(difference.abs().quantile(0.95)),
                   generated_token=result.outputs[0].token_ids)
            torch.save(reference, cli.output / f"losses-{domain}-{length}.pt")
    record("complete")


if __name__ == "__main__":
    main()
