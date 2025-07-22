import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple


def load_dataset(file_path: str) -> List[Dict]:
    """加载数据集"""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_dataset(records: List[Dict], file_path: str):
    """保存数据集"""
    with open(file_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_qid(qid: str) -> Tuple[str, int, int]:
    """解析qid，返回(user_id, line_idx, msg_idx)"""
    parts = qid.split("_")
    user_id = "_".join(parts[:-2])  # 处理user_id中可能包含下划线的情况
    line_idx = int(parts[-2])
    msg_idx = int(parts[-1])
    return user_id, line_idx, msg_idx


def random_split(records: List[Dict], test_ratio: float = 0.2, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    随机划分数据集

    Args:
        records: 数据记录列表
        test_ratio: 测试集比例
        seed: 随机种子

    Returns:
        (train_records, test_records)
    """
    random.seed(seed)
    records_copy = records.copy()
    random.shuffle(records_copy)

    test_size = int(len(records_copy) * test_ratio)
    test_records = records_copy[:test_size]
    train_records = records_copy[test_size:]

    return train_records, test_records


def user_based_split(records: List[Dict], test_ratio: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
    """
    按用户划分数据集，每个用户的最后20%的对话作为测试集

    Args:
        records: 数据记录列表
        test_ratio: 测试集比例

    Returns:
        (train_records, test_records)
    """
    # 按用户分组数据
    user_records = defaultdict(list)

    for record in records:
        user_id, line_idx, msg_idx = parse_qid(record["qid"])
        user_records[user_id].append((record, line_idx, msg_idx))

    train_records = []
    test_records = []

    for user_id, user_data in user_records.items():
        # 按msg_idx排序，确保时间顺序
        user_data.sort(key=lambda x: (x[1], x[2]))  # 按line_idx, msg_idx排序

        # 计算该用户的测试集大小
        test_size = max(1, int(len(user_data) * test_ratio))  # 至少1个测试样本

        # 最后的样本作为测试集
        user_test = user_data[-test_size:]
        user_train = user_data[:-test_size] if len(user_data) > test_size else []

        # 提取记录
        train_records.extend([item[0] for item in user_train])
        test_records.extend([item[0] for item in user_test])

    return train_records, test_records


def split_dataset(input_file: str, output_dir: str = ".", test_ratio: float = 0.2, seed: int = 42):
    """
    划分数据集的主函数

    Args:
        input_file: 输入数据文件路径
        output_dir: 输出目录
        test_ratio: 测试集比例
        seed: 随机种子
    """
    print(f"Loading dataset from {input_file}...")
    records = load_dataset(input_file)
    print(f"Total records: {len(records)}")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 方法1: 随机划分
    print("\n=== Random Split ===")
    train_random, test_random = random_split(records, test_ratio, seed)
    print(f"Random split - Train: {len(train_random)}, Test: {len(test_random)}")

    # 保存随机划分结果
    save_dataset(train_random, output_path / "train_random.jsonl")
    save_dataset(test_random, output_path / "test_random.jsonl")

    # 方法2: 按用户划分
    print("\n=== User-based Split ===")
    train_user, test_user = user_based_split(records, test_ratio)
    print(f"User-based split - Train: {len(train_user)}, Test: {len(test_user)}")

    # 统计用户信息
    user_stats = defaultdict(lambda: {"train": 0, "test": 0})
    for record in train_user:
        user_id, _, _ = parse_qid(record["qid"])
        user_stats[user_id]["train"] += 1
    for record in test_user:
        user_id, _, _ = parse_qid(record["qid"])
        user_stats[user_id]["test"] += 1

    print(f"Number of users: {len(user_stats)}")
    print(f"Average train samples per user: {len(train_user) / len(user_stats):.2f}")
    print(f"Average test samples per user: {len(test_user) / len(user_stats):.2f}")

    # 保存按用户划分结果
    save_dataset(train_user, output_path / "train_user_based.jsonl")
    save_dataset(test_user, output_path / "test_user_based.jsonl")

    # 保存统计信息
    stats = {
        "total_records": len(records),
        "test_ratio": test_ratio,
        "random_split": {"train": len(train_random), "test": len(test_random)},
        "user_based_split": {"train": len(train_user), "test": len(test_user), "num_users": len(user_stats)},
        "user_stats": dict(user_stats),
    }

    with open(output_path / "split_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Split completed! Files saved to {output_path}")
    print("Generated files:")
    print("- train_random.jsonl (随机划分训练集)")
    print("- test_random.jsonl (随机划分测试集)")
    print("- train_user_based.jsonl (按用户划分训练集)")
    print("- test_user_based.jsonl (按用户划分测试集)")
    print("- split_stats.json (划分统计信息)")


if __name__ == "__main__":
    # 使用示例
    input_file = "dialogue_dataset_all_v4.jsonl"  # 输入文件
    output_dir = "split_data"  # 输出目录

    split_dataset(input_file, output_dir, test_ratio=0.2, seed=42)
