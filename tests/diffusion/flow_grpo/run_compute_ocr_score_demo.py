from roll.pipeline.diffusion.rewards.qwen_vl_judge_reward_worker import QwenVLJudgeRewardWorker


def main():
    judge_texts = [
        "Page 688",
        "SYSTEM\nOVERDOE ACTIVE",
        "Brenda M. G.",
        "\"First Steps",
        "WARNING\nTake With Food",
    ]
    ground_truths = [
        "Page 666",
        "System Override Active",
        "Photosynthesis Process",
        "First Steps",
        "Take With Food",
    ]

    print("=== _compute_ocr_score demo ===")
    for idx, (judge_text, ground_truth) in enumerate(zip(judge_texts, ground_truths), start=1):
        score = QwenVLJudgeRewardWorker._compute_ocr_score(None, judge_text, ground_truth)
        print(f"[{idx}]")
        print(f"judge_text   : {repr(judge_text)}")
        print(f"ground_truth : {repr(ground_truth)}")
        print(f"score        : {score}")
        print("-" * 40)


if __name__ == "__main__":
    main()
