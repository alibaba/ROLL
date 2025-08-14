#!/usr/bin/env python3
import argparse
import json
import os
import sys
import random
from typing import Dict, List, Tuple, Optional

GENERATION_MESSAGE = [
    {
        "role": "system",
        "content": "You are an answer agent for multiple-choice questions. Your task is to output one letter 'A', 'B', 'C', or 'D'. Do not explain, do not copy the option, do not output anything about conversation. Just output the letter.",
    },
    {
        "role": "user",
        "content": "Choose 'A', 'B', 'C', or 'D' for the following question: {question}\n{reference_data}\nYour output should be a single letter 'A', 'B', 'C', or 'D', Do not explain, do not copy the option, do not output anything about conversation.\nNow, your output is:",
    },
]


def load_questions(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("questions file must be a JSON array")
        return data


def load_prompts(path: str) -> Dict[str, str]:
    """Load prompts from JSONL: each line must have {'qid': ..., 'prompt': ...}"""
    prompts: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[warn] prompts line {lineno} JSON parse error: {e}", file=sys.stderr)
                continue
            qid = str(obj.get("qid", "")).strip()
            prompt_text = obj.get("prompt")
            if not qid or not isinstance(prompt_text, str):
                print(f"[warn] prompts line {lineno} missing qid or prompt", file=sys.stderr)
                continue
            prompts[qid] = prompt_text
    return prompts


def extract_profile_conv_from_prompt(prompt: str) -> Tuple[str, str]:
    profile = ""
    conversation_history = ""
    if "[Profile Begin]" in prompt and "[Profile End]" in prompt:
        start = prompt.find("[Profile Begin]") + len("[Profile Begin]")
        end = prompt.find("[Profile End]")
        if end > start:
            profile = prompt[start:end].strip()
    if "[Conversation History Begin]" in prompt and "[Conversation History End]" in prompt:
        start = prompt.find("[Conversation History Begin]") + len("[Conversation History Begin]")
        end = prompt.find("[Conversation History End]")
        if end > start:
            conversation_history = prompt[start:end].strip()
    return profile, conversation_history


def create_four_choice_question_text(question: dict, profile: str, conversation_history: str) -> str:
    parts: List[str] = []
    if profile:
        parts.append(f"[Profile Begin]\n{profile}\n[Profile End]\n")
    if conversation_history:
        parts.append(f"[Conversation History Begin]\n{conversation_history}\n[Conversation History End]\n")
    parts.append("Which response is most appropriate for this person in this context?\n")
    # Add choices sorted by label A-D
    choices = question.get("choices", [])
    for choice in sorted(choices, key=lambda x: x.get("label", "")):
        parts.append(f"{choice.get('label', '').strip()}. {choice.get('text', '').strip()}")
    return "\n".join(parts).rstrip() + "\n"


def get_correct_answer_letter(question: dict) -> Optional[str]:
    # Prefer explicit correct_answer if present
    letter = question.get("correct_answer")
    if isinstance(letter, str) and letter.strip():
        return letter.strip()
    # Fallback: find choice with is_correct == True
    for ch in question.get("choices", []):
        if ch.get("is_correct") is True:
            lbl = ch.get("label")
            if isinstance(lbl, str) and lbl.strip():
                return lbl.strip()
    return None


def build_messages(question_text: str, reference_data: str = "") -> List[dict]:
    return [
        GENERATION_MESSAGE[0],
        {
            "role": "user",
            "content": GENERATION_MESSAGE[1]["content"].format(
                question=question_text, reference_data=reference_data or ""
            ),
        },
    ]


def export_dataset(items: List[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if out_path.lower().endswith(".json"):
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    else:
        # default to JSONL
        with open(out_path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")


def parse_qid(qid: str) -> Tuple[str, int, int]:
    """
    解析qid，返回(user_id, line_idx, msg_idx)
    若解析失败，则将整个qid作为user_id，索引置0。
    """
    try:
        parts = str(qid).split("_")
        if len(parts) >= 3:
            user_id = "_".join(parts[:-2])
            line_idx = int(parts[-2])
            msg_idx = int(parts[-1])
        else:
            user_id, line_idx, msg_idx = str(qid), 0, 0
        return user_id, line_idx, msg_idx
    except Exception:
        return str(qid), 0, 0


def random_split_items(items: List[dict], test_ratio: float = 0.2, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """随机划分 items"""
    rng = random.Random(seed)
    arr = items.copy()
    rng.shuffle(arr)
    n_test = int(len(arr) * test_ratio)
    test_items = arr[:n_test]
    train_items = arr[n_test:]
    return train_items, test_items


def user_based_split_items(items: List[dict], test_ratio: float = 0.2) -> Tuple[List[dict], List[dict]]:
    """
    按用户划分：每个用户末尾 test_ratio 的样本作测试集（按 line_idx, msg_idx 排序）
    """
    buckets: Dict[str, List[Tuple[dict, int, int]]] = {}
    for it in items:
        qid = it.get("qid", "")
        user_id, line_idx, msg_idx = parse_qid(qid)
        buckets.setdefault(user_id, []).append((it, line_idx, msg_idx))

    train_items: List[dict] = []
    test_items: List[dict] = []
    for _, arr in buckets.items():
        arr.sort(key=lambda x: (x[1], x[2]))
        n = len(arr)
        n_test = max(1, int(n * test_ratio)) if n > 0 else 0
        user_test = arr[-n_test:] if n_test > 0 else []
        user_train = arr[:-n_test] if n_test > 0 else arr
        train_items.extend([x[0] for x in user_train])
        test_items.extend([x[0] for x in user_test])
    return train_items, test_items


def user_partial_split_items(
    items: List[dict],
    test_ratio: float = 0.2,
    user_subset_ratio: float = 0.3,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """
    新语义：
    - 随机抽取 user_subset_ratio 比例的用户，这些用户的所有样本全部进入测试集，标记 test_tag="fully"
      （若计算得到数量 >= 总用户且总用户>1，则减少 1，保证不是所有用户都被选为 fully）
    - 其余用户：各自末尾 test_ratio 比例(至少1条若该用户样本数>0) 进入测试集，标记 test_tag="partially"，其余进入训练集
    返回: (train_items, test_items)
    注意：测试集中每条样本都带有字段 "test_tag": "fully" 或 "partially"
    """
    # 分桶
    buckets: Dict[str, List[Tuple[dict, int, int]]] = {}
    for it in items:
        qid = it.get("qid", "")
        user_id, line_idx, msg_idx = parse_qid(qid)
        buckets.setdefault(user_id, []).append((it, line_idx, msg_idx))

    users = list(buckets.keys())
    if not users:
        return items, []

    rng = random.Random(seed)
    rng.shuffle(users)

    k = max(1, int(len(users) * user_subset_ratio))
    if k >= len(users) and len(users) > 1:
        k = len(users) - 1  # 避免全部用户都 fully
    fully_users = set(users[:k])

    train_items: List[dict] = []
    test_items: List[dict] = []

    for uid, arr in buckets.items():
        arr.sort(key=lambda x: (x[1], x[2]))  # 按 (line_idx, msg_idx)
        if uid in fully_users:
            for rec, _, _ in arr:
                rec["test_tag"] = "fully"
                test_items.append(rec)
        else:
            n = len(arr)
            if n == 0:
                continue
            n_test = max(1, int(n * test_ratio))
            tail = arr[-n_test:]
            head = arr[:-n_test]
            for rec, _, _ in head:
                train_items.append(rec)
            for rec, _, _ in tail:
                rec["test_tag"] = "partially"
                test_items.append(rec)

    return train_items, test_items


def derive_split_paths(out_path: str) -> Tuple[str, str]:
    """
    基于 --out 生成 train/test 输出路径：
    - 若以 .json/.jsonl 结尾：追加 _train/_test
    - 否则视为目录：写入 train.jsonl / test.jsonl
    """
    base, ext = os.path.splitext(out_path)
    if ext.lower() in {".json", ".jsonl"}:
        return f"{base}_train{ext}", f"{base}_test{ext}"
    os.makedirs(out_path, exist_ok=True)
    return os.path.join(out_path, "train.jsonl"), os.path.join(out_path, "test.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Build {qid, messages, output} dataset for choice QA.")
    parser.add_argument("--questions", required=True, help="Path to questions JSON array")
    parser.add_argument("--prompts", required=True, help="Path to prompts JSONL")
    parser.add_argument("--out", required=True, help="Output path (.json or .jsonl, or a directory)")
    parser.add_argument("--max_items", type=int, default=0, help="Optional cap on number of items")
    parser.add_argument("--skip_missing_prompt", action="store_true", help="Skip items without prompt")
    parser.add_argument(
        "--split_mode",
        choices=["none", "random", "user", "user_partial"],
        default="none",
        help="Split dataset into train/test by mode",
    )
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for random/user_partial split")
    parser.add_argument(
        "--user_subset_ratio",
        type=float,
        default=0.2,
        help="For user_partial: ratio of users to sample for tail test extraction",
    )
    args = parser.parse_args()

    questions = load_questions(args.questions)
    prompts_map = load_prompts(args.prompts)

    items: List[dict] = []
    skipped_no_prompt = 0
    skipped_no_answer = 0

    for q in questions:
        qid = str(q.get("qid", "")).strip()
        if not qid:
            continue
        prompt_text = prompts_map.get(qid)
        if not prompt_text:
            if args.skip_missing_prompt:
                skipped_no_prompt += 1
                continue
            else:
                prompt_text = ""

        profile, conv = extract_profile_conv_from_prompt(prompt_text)
        question_text = create_four_choice_question_text(q, profile, conv)
        messages = build_messages(question_text, reference_data="")

        correct_letter = get_correct_answer_letter(q)
        if not correct_letter:
            skipped_no_answer += 1
            continue

        items.append(
            {
                "qid": qid,
                "messages": messages,
                "output": correct_letter,
            }
        )

        if args.max_items and len(items) >= args.max_items:
            break

    if args.split_mode == "none":
        export_dataset(items, args.out)
        print(f"Built {len(items)} items -> {args.out}")
    else:
        if args.split_mode == "random":
            train_items, test_items = random_split_items(items, test_ratio=args.test_ratio, seed=args.seed)
        elif args.split_mode == "user":
            train_items, test_items = user_based_split_items(items, test_ratio=args.test_ratio)
        else:  # user_partial
            train_items, test_items = user_partial_split_items(
                items,
                test_ratio=args.test_ratio,
                user_subset_ratio=args.user_subset_ratio,
                seed=args.seed,
            )
        train_out, test_out = derive_split_paths(args.out)
        export_dataset(train_items, train_out)
        export_dataset(test_items, test_out)
        print(f"Built {len(items)} items; train={len(train_items)}, test={len(test_items)}")
        print(f"Saved train -> {train_out}")
        print(f"Saved test  -> {test_out}")

    if skipped_no_prompt:
        print(f"Skipped {skipped_no_prompt} items due to missing prompt", file=sys.stderr)
    if skipped_no_answer:
        print(f"Skipped {skipped_no_answer} items due to missing correct answer", file=sys.stderr)


if __name__ == "__main__":
    main()
