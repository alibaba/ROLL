import argparse
import os
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from tests.distributed.strategy.standalone.fsdp2_standalone_strategy import (
    StandaloneFSDP2Config,
    StandaloneFSDP2Strategy,
)


def _build_text_batch(
    *,
    tokenizer,
    prompt: str,
    response: str,
    device: torch.device,
    max_length: int,
    model_name_or_path: str,
):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text = prompt + response
    enc = tokenizer(
        [text],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    bsz, seqlen = input_ids.shape
    position_ids = torch.arange(seqlen, dtype=torch.long, device=device).unsqueeze(0).expand(bsz, -1)
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    return input_ids, attention_mask, position_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local model path (preferred for standalone runs).")
    parser.add_argument("--prompt", default="Hello", help="Prompt text.")
    parser.add_argument("--response", default=" world", help="Response text (appended to prompt).")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--fsdp-size", type=int, default=1)
    parser.add_argument("--param-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--reduce-dtype", default="fp32", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--reshard-after-forward", type=int, default=1, choices=[0, 1])
    parser.add_argument("--cpu-offload", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use-remove-padding", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    cfg = StandaloneFSDP2Config(
        model_name_or_path=args.model,
        is_trainable=False,
        ulysses_size=int(args.cp_size),
        fsdp_size=int(args.fsdp_size),
        param_dtype=args.param_dtype,
        reduce_dtype=args.reduce_dtype,
        reshard_after_forward=bool(args.reshard_after_forward),
        cpu_offload=bool(args.cpu_offload),
        use_remove_padding=bool(args.use_remove_padding),
    )
    strat = StandaloneFSDP2Strategy(cfg)
    strat.initialize()

    rank = dist.get_rank()
    device = (
        torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True, padding_side="left"
    )
    input_ids, attention_mask, position_ids = _build_text_batch(
        tokenizer=tokenizer,
        prompt=args.prompt,
        response=args.response,
        device=device,
        max_length=int(args.max_length),
        model_name_or_path=args.model,
    )

    with torch.no_grad(), strat.autocast():
        logits = strat.fsdp2_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            forward_args={"use_cache": False},
        )
        log_probs = strat.compute_log_probs(logits=logits, input_ids=input_ids, attention_mask=attention_mask)

    scalar = log_probs.sum()
    dist.all_reduce(scalar)
    if rank == 0:
        print(
            f"[standalone fsdp2] world_size={dist.get_world_size()} cp_size={strat.rank_info.cp_size} "
            f"fsdp_size={cfg.fsdp_size} remove_padding={cfg.use_remove_padding} "
            f"log_probs_sum(all_reduce)={scalar.item():.4f}"
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
