import os
import re
import json
import argparse
from collections import Counter, defaultdict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def normalize_choice(text: str):
    """Normalize any text to single letter A/B/C/D if possible."""
    if not text:
        return None
    # 原文大小写保留用于后续可能的展示，这里统一转大写做匹配
    t = (text or "").strip()
    t_up = t.upper()

    # 清理常见噪声/标签
    # 去掉 <think>...</think> 以及其它尖括号标签
    t_up = re.sub(r"<THINK>.*?</THINK>", " ", t_up, flags=re.DOTALL)
    t_up = re.sub(r"<[^>]+>", " ", t_up)
    # 去掉常见 role 标记
    t_up = re.sub(r"\b(ASSISTANT|SYSTEM|USER)\b", " ", t_up)

    # 1) 优先匹配“结尾的单个 A-D”
    m = re.search(r"([ABCD])\s*$", t_up)
    if m:
        return m.group(1)

    # 2) 常见显式前缀 Answer/Option/Choice
    m = re.search(r'(?:^|\s)(?:ANSWER|OPTION|CHOICE)\s*[:\-]?\s*["\'`(]*\b([ABCD])\b', t_up)
    if m:
        return m.group(1)

    # 3) 文首的单个字母
    m = re.match(r'^\s*["\'`(]*([ABCD])(?:[\).\s]|$)', t_up)
    if m:
        return m.group(1)

    # 4) 任意位置的独立字母
    m = re.search(r"\b([ABCD])\b", t_up)
    if m:
        return m.group(1)

    # 5) 仅出现唯一一个 A-D 时
    letters = re.findall(r"[ABCD]", t_up)
    if len(letters) == 1:
        return letters[0]

    return None


def load_model_and_tokenizer(base_model_path: str, lora_path: str):
    """Load base model and apply LoRA weights, set tokenizer padding properly."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    # 修复 base 情况：不加载 LoRA
    if lora_path and lora_path != "base":
        model = PeftModel.from_pretrained(base_model, lora_path)
    else:
        model = base_model
    model.eval()
    return model, tokenizer


def load_test_dataset(dataset_path: str):
    """Load test dataset from JSON file. Expect a list of items with keys:
    - messages: list of {role, content}
    - output: expected answer (ideally 'A'/'B'/'C'/'D')
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_prompts_from_messages(tokenizer, messages_batch):
    """Apply chat template to a batch of message lists."""
    prompts = []
    for messages in messages_batch:
        prompt = tokenizer.apply_chat_template(
            messages,
            max_length=8192,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt)
    return prompts


@torch.inference_mode()
def generate_response_batch(model, tokenizer, messages_list, max_new_tokens=6, batch_size=32):
    """Batch generate and robustly strip prompt prefix from decoded outputs."""
    all_responses = []

    for i in range(0, len(messages_list), batch_size):
        batch_messages = messages_list[i : i + batch_size]
        # 构造 prompt（chat 模板）
        prompts = build_prompts_from_messages(tokenizer, batch_messages)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )

        sequences = outputs.sequences  # [B, T_in + T_new]
        attn = inputs["attention_mask"]  # [B, T_in]

        for j in range(sequences.size(0)):
            # 方案A：用“解码后的输入前缀”精确剥离（最稳）
            full_dec = tokenizer.decode(sequences[j], skip_special_tokens=True)
            inp_dec = tokenizer.decode(inputs["input_ids"][j], skip_special_tokens=True)

            if full_dec.startswith(inp_dec):
                gen_text = full_dec[len(inp_dec) :].strip()
            else:
                # 方案B：回退到 attention_mask 切分
                in_len = int(attn[j].sum().item())
                gen_text = tokenizer.decode(sequences[j, in_len:], skip_special_tokens=True).strip()

            all_responses.append(gen_text)

    return all_responses


def evaluate_one_checkpoint(model, tokenizer, test_data, batch_size=32, max_new_tokens=6, sample=None):
    """Evaluate one model+tokenizer on provided dataset with detailed statistics."""
    # Optional sampling
    data = test_data
    if sample is not None and 0 < sample < len(test_data):
        data = test_data[:sample]

    messages_list = [item["messages"] for item in data]
    expected_outputs_raw = [item.get("output", "") for item in data]
    expected_choices = [normalize_choice(x) for x in expected_outputs_raw]

    # Generate responses
    print("Generating responses...")
    all_responses = []
    for i in tqdm(range(0, len(messages_list), batch_size), desc="Batches"):
        batch = messages_list[i : i + batch_size]
        batch_resps = generate_response_batch(
            model,
            tokenizer,
            batch,
            max_new_tokens=max_new_tokens,
            batch_size=len(batch),
        )
        all_responses.extend(batch_resps)

    # Parse predictions and compute metrics
    preds = [normalize_choice(x) for x in all_responses]

    total = len(data)
    valid = 0
    correct = 0

    # confusion[gold][pred] -> count
    confusion = defaultdict(lambda: Counter())
    gold_counter = Counter()
    pred_counter = Counter()
    invalid_pred_indices = []

    detailed = []

    for idx, (gold, pred) in enumerate(zip(expected_choices, preds)):
        gold_counter[gold] += 1
        if pred is None:
            invalid_pred_indices.append(idx)
        else:
            pred_counter[pred] += 1
            confusion[gold][pred] += 1
            valid += 1
            if gold is not None and pred == gold:
                correct += 1

        detailed.append(
            {
                "index": idx,
                "gold_raw": expected_outputs_raw[idx],
                "gold": gold,
                "prediction_raw": all_responses[idx],
                "prediction": pred,
                "is_correct": gold is not None and pred == gold,
            }
        )

    accuracy = (correct / total) if total > 0 else 0.0

    # Per-class accuracy (only where gold exists)
    per_class = {}
    for c in ["A", "B", "C", "D"]:
        g = gold_counter.get(c, 0)
        if g > 0:
            per_class[c] = {
                "gold": g,
                "correct": confusion[c][c],
                "accuracy": confusion[c][c] / g,
            }
        else:
            per_class[c] = {"gold": 0, "correct": 0, "accuracy": None}

    summary = {
        "total": total,
        "valid_predictions": valid,
        "invalid_predictions": len(invalid_pred_indices),
        "correct": correct,
        "accuracy": accuracy,
        "gold_distribution": dict(gold_counter),
        "pred_distribution": dict(pred_counter),
        "per_class": per_class,
        "confusion": {g: dict(confusion[g]) for g in confusion},
    }

    return summary, detailed


def list_lora_checkpoints(lora_dir: str):
    """Return a list of LoRA adapter paths to evaluate:
    - the lora_dir itself (if it is a valid adapter)
    - any subdirectories starting with checkpoint-
    """
    paths = []
    if os.path.isdir(lora_dir):
        # include root dir first
        paths.append(lora_dir)
        # add checkpoint-* subdirs
        subs = [
            os.path.join(lora_dir, d)
            for d in os.listdir(lora_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(lora_dir, d))
        ]
        subs = sorted(
            subs,
            key=lambda p: int(os.path.basename(p).split("-")[-1]) if "-" in os.path.basename(p) else 0,
        )
        paths.extend(subs)
    else:
        # single path
        paths.append(lora_dir)
    return paths


def save_results(save_dir, lora_path, summary, detailed):
    os.makedirs(save_dir, exist_ok=True)
    name = os.path.basename(lora_path.rstrip("/")) or "base"
    out_path = os.path.join(save_dir, f"choice_eval_{name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "lora_path": lora_path,
                "summary": summary,
                "results": detailed,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return out_path


def parse_args():
    ap = argparse.ArgumentParser(description="Robust multi-choice evaluation (LoRA checkpoints supported).")
    ap.add_argument("--base_model", default="/project/hdtaccuracy/models/base/Qwen3-8B", help="Base HF model path")
    ap.add_argument(
        "--lora_dir",
        default="/project/hdtaccuracy/trains/choice-sft/qwen3-8b-lora-sft",
        help="LoRA adapter path or directory containing checkpoints",
    )
    ap.add_argument(
        "--dataset",
        default="/project/hdtaccuracy/Personality-Alignment/choice_ver/four_choices_question_v7/v7_test.json",
        help="Test dataset JSON path",
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=2)
    ap.add_argument("--sample", type=int, default=None, help="Only evaluate the first N samples")
    ap.add_argument("--save_dir", default="choice_eval_outputs", help="Directory to save detailed JSON results")
    return ap.parse_args()


def main():
    args = parse_args()

    print(f"Listing LoRA checkpoints in {args.lora_dir} ...")
    lora_paths = ["base"] + list_lora_checkpoints(args.lora_dir)
    # lora_paths = ["base"]
    print(f"Found {len(lora_paths)} adapters to evaluate:")
    for p in lora_paths:
        print(f"  - {p}")

    print("Loading test dataset...")
    test_data = load_test_dataset(args.dataset)
    print(f"Total samples: {len(test_data)}" + (f" (evaluating first {args.sample})" if args.sample else ""))

    results_summary = []

    for lora_path in lora_paths:
        print("\n" + "=" * 80)
        print(f"Evaluating LoRA: {lora_path}")
        print("Loading model and tokenizer...")

        model, tokenizer = load_model_and_tokenizer(args.base_model, lora_path)

        print("Evaluating...")
        summary, detailed = evaluate_one_checkpoint(
            model=model,
            tokenizer=tokenizer,
            test_data=test_data,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            sample=args.sample,
        )

        print(
            f"Accuracy: {summary['accuracy']:.4f}  "
            f"(correct={summary['correct']}/{summary['total']}, "
            f"invalid={summary['invalid_predictions']})"
        )

        out_path = save_results(args.save_dir, lora_path, summary, detailed)
        print(f"Detailed results saved to: {out_path}")

        results_summary.append(
            {
                "lora_path": lora_path,
                "accuracy": summary["accuracy"],
                "correct": summary["correct"],
                "total": summary["total"],
                "invalid": summary["invalid_predictions"],
            }
        )

        # free VRAM between adapters
        del model
        torch.cuda.empty_cache()

    # Aggregate summary
    agg_path = os.path.join(args.save_dir, "choice_eval_results.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print("\nAll adapters summary saved to:", agg_path)


if __name__ == "__main__":
    main()
