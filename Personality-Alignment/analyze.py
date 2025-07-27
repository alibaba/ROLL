import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict, Counter
from pathlib import Path
import statistics


def load_dataset(file_path: str):
    """加载数据集"""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def parse_qid(qid: str):
    """解析qid，返回(user_id, line_idx, msg_idx)"""
    parts = qid.split("_")
    user_id = "_".join(parts[:-2])
    line_idx = int(parts[-2])
    msg_idx = int(parts[-1])
    return user_id, line_idx, msg_idx


def analyze_length_distribution(records, base_dir, output_dir="analysis_results"):
    """分析数据长度分布"""
    output_path = Path(base_dir) / output_dir
    output_path.mkdir(exist_ok=True)

    # 收集长度数据
    prompt_lengths = []
    output_lengths = []
    user_data = defaultdict(list)

    for record in records:
        prompt = record.get("prompt", "")
        output = record.get("output", "")

        prompt_len = len(prompt)
        output_len = len(output)

        prompt_lengths.append(prompt_len)
        output_lengths.append(output_len)

        # 按用户分组
        user_id, _, _ = parse_qid(record["qid"])
        user_data[user_id].append({"prompt_len": prompt_len, "output_len": output_len})

    # 1. 基础统计信息
    print("=== 数据长度分布分析 ===")
    print(f"总记录数: {len(records)}")
    print(f"用户数: {len(user_data)}")

    # Prompt长度统计
    print("\n--- Prompt长度统计 ---")
    print(f"平均长度: {statistics.mean(prompt_lengths):.2f}")
    print(f"中位数: {statistics.median(prompt_lengths):.2f}")
    print(f"最小长度: {min(prompt_lengths)}")
    print(f"最大长度: {max(prompt_lengths)}")
    print(f"标准差: {statistics.stdev(prompt_lengths):.2f}")
    print(f"25分位数: {np.percentile(prompt_lengths, 25):.2f}")
    print(f"75分位数: {np.percentile(prompt_lengths, 75):.2f}")
    print(f"95分位数: {np.percentile(prompt_lengths, 95):.2f}")
    print(f"99分位数: {np.percentile(prompt_lengths, 99):.2f}")

    # Output长度统计
    print("\n--- Output长度统计 ---")
    print(f"平均长度: {statistics.mean(output_lengths):.2f}")
    print(f"中位数: {statistics.median(output_lengths):.2f}")
    print(f"最小长度: {min(output_lengths)}")
    print(f"最大长度: {max(output_lengths)}")
    print(f"标准差: {statistics.stdev(output_lengths):.2f}")
    print(f"25分位数: {np.percentile(output_lengths, 25):.2f}")
    print(f"75分位数: {np.percentile(output_lengths, 75):.2f}")
    print(f"95分位数: {np.percentile(output_lengths, 95):.2f}")
    print(f"99分位数: {np.percentile(output_lengths, 99):.2f}")

    # 2. 用户级别统计
    user_prompt_means = []
    user_output_means = []
    user_sample_counts = []

    for user_id, user_records in user_data.items():
        user_prompt_lens = [r["prompt_len"] for r in user_records]
        user_output_lens = [r["output_len"] for r in user_records]

        user_prompt_means.append(statistics.mean(user_prompt_lens))
        user_output_means.append(statistics.mean(user_output_lens))
        user_sample_counts.append(len(user_records))

    print("\n--- 用户级别统计 ---")
    print(f"每用户平均样本数: {statistics.mean(user_sample_counts):.2f}")
    print(f"用户样本数中位数: {statistics.median(user_sample_counts):.2f}")
    print(f"最少样本用户: {min(user_sample_counts)} 样本")
    print(f"最多样本用户: {max(user_sample_counts)} 样本")
    print(f"用户间prompt平均长度差异(标准差): {statistics.stdev(user_prompt_means):.2f}")
    print(f"用户间output平均长度差异(标准差): {statistics.stdev(user_output_means):.2f}")

    # 3. 异常值检测
    def detect_outliers(data, method="iqr"):
        if method == "iqr":
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return [x for x in data if x < lower or x > upper]
        elif method == "zscore":
            mean = np.mean(data)
            std = np.std(data)
            return [x for x in data if abs((x - mean) / std) > 3]

    prompt_outliers = detect_outliers(prompt_lengths)
    output_outliers = detect_outliers(output_lengths)

    print("\n--- 异常值分析 ---")
    print(f"Prompt异常值数量: {len(prompt_outliers)} ({len(prompt_outliers)/len(prompt_lengths)*100:.2f}%)")
    print(f"Output异常值数量: {len(output_outliers)} ({len(output_outliers)/len(output_lengths)*100:.2f}%)")

    if prompt_outliers:
        print(f"Prompt异常值范围: {min(prompt_outliers)} - {max(prompt_outliers)}")
    if output_outliers:
        print(f"Output异常值范围: {min(output_outliers)} - {max(output_outliers)}")

    # 4. 生成可视化图表
    plt.style.use("default")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Prompt长度分布直方图
    axes[0, 0].hist(prompt_lengths, bins=50, alpha=0.7, color="blue", edgecolor="black")
    # axes[0, 0].set_title("Prompt长度分布")
    # axes[0, 0].set_xlabel("长度")
    # axes[0, 0].set_ylabel("频次")
    axes[0, 0].set_title("Prompt Length Distribution")
    axes[0, 0].set_xlabel("Length")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].axvline(
        statistics.mean(prompt_lengths),
        color="red",
        linestyle="--",
        # label=f"均值: {statistics.mean(prompt_lengths):.0f}",
        label=f"Mean: {statistics.mean(prompt_lengths):.0f}",
    )
    axes[0, 0].axvline(
        statistics.median(prompt_lengths),
        color="orange",
        linestyle="--",
        # label=f"中位数: {statistics.median(prompt_lengths):.0f}",
        label=f"Median: {statistics.median(prompt_lengths):.0f}",
    )
    axes[0, 0].legend()

    # Output长度分布直方图
    axes[0, 1].hist(output_lengths, bins=50, alpha=0.7, color="green", edgecolor="black")
    # axes[0, 1].set_title("Output长度分布")
    # axes[0, 1].set_xlabel("长度")
    # axes[0, 1].set_ylabel("频次")
    axes[0, 1].set_title("Output Length Distribution")
    axes[0, 1].set_xlabel("Length")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].axvline(
        statistics.mean(output_lengths),
        color="red",
        linestyle="--",
        # label=f"均值: {statistics.mean(output_lengths):.0f}",
        label=f"Mean: {statistics.mean(output_lengths):.0f}",
    )
    axes[0, 1].axvline(
        statistics.median(output_lengths),
        color="orange",
        linestyle="--",
        # label=f"中位数: {statistics.median(output_lengths):.0f}",
        label=f"Median: {statistics.median(output_lengths):.0f}",
    )
    axes[0, 1].legend()

    # 用户样本数分布
    axes[0, 2].hist(user_sample_counts, bins=30, alpha=0.7, color="purple", edgecolor="black")
    # axes[0, 2].set_title("每用户样本数分布")
    # axes[0, 2].set_xlabel("样本数")
    # axes[0, 2].set_ylabel("用户数")
    axes[0, 2].set_title("User Sample Count Distribution")
    axes[0, 2].set_xlabel("Sample Count")
    axes[0, 2].set_ylabel("Number of Users")
    axes[0, 2].axvline(
        statistics.mean(user_sample_counts),
        color="red",
        linestyle="--",
        # label=f"均值: {statistics.mean(user_sample_counts):.1f}",
        label=f"Mean: {statistics.mean(user_sample_counts):.1f}",
    )
    axes[0, 2].legend()

    # Box plot - Prompt vs Output长度
    axes[1, 0].boxplot([prompt_lengths, output_lengths], labels=["Prompt", "Output"])
    # axes[1, 0].set_title("Prompt vs Output长度对比")
    # axes[1, 0].set_ylabel("长度")
    axes[1, 0].set_title("Prompt vs Output Length Comparison")
    axes[1, 0].set_ylabel("Length")

    # 长度相关性散点图
    sample_indices = np.random.choice(len(prompt_lengths), min(1000, len(prompt_lengths)), replace=False)
    sample_prompts = [prompt_lengths[i] for i in sample_indices]
    sample_outputs = [output_lengths[i] for i in sample_indices]

    axes[1, 1].scatter(sample_prompts, sample_outputs, alpha=0.5)
    # axes[1, 1].set_title("Prompt vs Output长度相关性")
    # axes[1, 1].set_xlabel("Prompt长度")
    # axes[1, 1].set_ylabel("Output长度")
    axes[1, 1].set_title("Prompt vs Output Length Correlation")
    axes[1, 1].set_xlabel("Prompt Length")
    axes[1, 1].set_ylabel("Output Length")

    # 计算相关系数
    correlation = np.corrcoef(prompt_lengths, output_lengths)[0, 1]
    axes[1, 1].text(
        0.05,
        0.95,
        # f"相关系数: {correlation:.3f}",
        f"Correlation: {correlation:.3f}",
        transform=axes[1, 1].transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat"),
    )

    # 用户平均长度分布
    axes[1, 2].scatter(user_prompt_means, user_output_means, alpha=0.6)
    # axes[1, 2].set_title("用户平均Prompt vs Output长度")
    axes[1, 2].set_title("User Average Prompt vs Output Lengths")
    # axes[1, 2].set_xlabel("平均Prompt长度")
    axes[1, 2].set_xlabel("Average Prompt Length")
    # axes[1, 2].set_ylabel("平均Output长度")
    axes[1, 2].set_ylabel("Average Output Length")

    plt.tight_layout()
    plt.savefig(output_path / "length_distribution_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 5. 保存详细统计结果
    analysis_results = {
        "basic_stats": {
            "total_records": len(records),
            "total_users": len(user_data),
            "prompt_stats": {
                "mean": statistics.mean(prompt_lengths),
                "median": statistics.median(prompt_lengths),
                "min": min(prompt_lengths),
                "max": max(prompt_lengths),
                "std": statistics.stdev(prompt_lengths),
                "percentiles": {
                    "25": float(np.percentile(prompt_lengths, 25)),
                    "75": float(np.percentile(prompt_lengths, 75)),
                    "95": float(np.percentile(prompt_lengths, 95)),
                    "99": float(np.percentile(prompt_lengths, 99)),
                },
            },
            "output_stats": {
                "mean": statistics.mean(output_lengths),
                "median": statistics.median(output_lengths),
                "min": min(output_lengths),
                "max": max(output_lengths),
                "std": statistics.stdev(output_lengths),
                "percentiles": {
                    "25": float(np.percentile(output_lengths, 25)),
                    "75": float(np.percentile(output_lengths, 75)),
                    "95": float(np.percentile(output_lengths, 95)),
                    "99": float(np.percentile(output_lengths, 99)),
                },
            },
        },
        "user_level_stats": {
            "avg_samples_per_user": statistics.mean(user_sample_counts),
            "median_samples_per_user": statistics.median(user_sample_counts),
            "min_samples": min(user_sample_counts),
            "max_samples": max(user_sample_counts),
            "user_prompt_length_std": statistics.stdev(user_prompt_means),
            "user_output_length_std": statistics.stdev(user_output_means),
        },
        "outlier_analysis": {
            "prompt_outliers": len(prompt_outliers),
            "output_outliers": len(output_outliers),
            "prompt_outlier_ratio": len(prompt_outliers) / len(prompt_lengths),
            "output_outlier_ratio": len(output_outliers) / len(output_lengths),
        },
        "correlation": {"prompt_output_correlation": float(correlation)},
    }

    with open(output_path / "detailed_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 分析完成! 结果保存到 {output_path}")
    print("生成的文件:")
    print("- length_distribution_analysis.png (可视化图表)")
    print("- detailed_analysis.json (详细统计结果)")


if __name__ == "__main__":
    # 分析原始数据
    base_dir = "/project/hdtaccuracy/Personality-Alignment/"
    input_file = base_dir + "dialogue_dataset_all_v5_summarized.jsonl"  # 输入文件
    print("正在分析原始数据集...")
    records = load_dataset(input_file)
    analyze_length_distribution(records, "original_data_analysis")

    # 如果存在划分后的数据，也进行分析
    split_dir = Path(base_dir) / "split_data_v5"
    if split_dir.exists():
        for split_file in [
            "train_random.jsonl",
            "test_random.jsonl",
            "train_user_based.jsonl",
            "test_user_based.jsonl",
        ]:
            if (split_dir / split_file).exists():
                print(f"\n正在分析 {split_file}...")
                split_records = load_dataset(split_dir / split_file)
                analyze_length_distribution(split_records, base_dir, f"analysis_{split_file.replace('.jsonl', '')}")
