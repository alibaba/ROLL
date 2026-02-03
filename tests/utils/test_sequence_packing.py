import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict
from tensordict import TensorDict
from roll.distributed.scheduler.protocol import DataProto


def test_load_balance_packer():
    """测试 LoadBalancePacker 并展示哪些样本被打包到一起"""

    # 导入必要的类
    from roll.utils.sequence_packing import LoadBalancePacker, SequencePackingConfig

    # 创建配置
    config = SequencePackingConfig(
        algorithm="load_balance",
        max_packed_sequence_length=4096
    )

    # 创建 packer
    packer = LoadBalancePacker(config)

    # 创建测试数据 - 10个样本，不同的序列长度
    batch_size = 10
    max_seq_len = 2048

    # 创建不同长度的序列
    sequence_lengths = [512, 1024, 256, 2048, 128, 768, 1536, 384, 896, 640]
    print(f"\n{'=' * 80}")
    print(f"原始数据:")
    print(f"{'=' * 80}")
    print(f"总样本数: {batch_size}")
    print(f"最大序列长度配置: {config.max_packed_sequence_length}")
    print(f"\n各样本的序列长度:")
    for idx, length in enumerate(sequence_lengths):
        print(f"  样本 {idx}: {length} tokens")

    # 创建 attention_mask 来模拟真实的序列长度
    attention_masks = []
    input_ids_list = []

    for seq_len in sequence_lengths:
        # 创建 attention_mask: 前 seq_len 个位置为 1，其余为 0
        mask = torch.zeros(max_seq_len, dtype=torch.long)
        mask[:seq_len] = 1
        attention_masks.append(mask)

        # 创建假的 input_ids
        input_ids = torch.randint(0, 1000, (max_seq_len,), dtype=torch.long)
        input_ids_list.append(input_ids)

    # 堆叠成批次
    attention_mask = torch.stack(attention_masks)
    input_ids = torch.stack(input_ids_list)

    # 创建 TensorDict
    batch_dict = TensorDict(
        source={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        batch_size=(batch_size,)
    )

    # 创建 DataProto
    mini_batch = DataProto(
        batch=batch_dict,
        non_tensor_batch={},
        meta_info={}
    )

    # 设置参数
    tp_size = 1
    cp_size = 1
    vp_size = 1

    # 创建一个假的 dp_group（对于测试，我们可以传 None）
    class FakeDPGroup:
        pass

    dp_group = FakeDPGroup()

    # 调用 packer
    print(f"\n{'=' * 80}")
    print(f"开始打包...")
    print(f"{'=' * 80}")

    micro_batches = list(packer.make_micro_batch_iter_for_sequence_packing(
        mini_batch=mini_batch,
        tp_size=tp_size,
        cp_size=cp_size,
        vp_size=vp_size,
        dp_group=dp_group,
        micro_batch_size=None  # LoadBalancePacker 会自动计算
    ))

    # 展示结果
    print(f"\n{'=' * 80}")
    print(f"打包结果:")
    print(f"{'=' * 80}")
    print(f"总共生成了 {len(micro_batches)} 个 micro batches\n")

    print(f"partition_indices_list:   {micro_batches[0].meta_info['partition_indices_list']}")

    total_workload = 0
    for micro_idx, micro_batch in enumerate(micro_batches):
        partition_indices = micro_batch.meta_info.get('partition_indices', [])

        # 获取这个 micro batch 中的序列长度
        batch_seq_lens = []
        for idx in partition_indices:
            seq_len = sequence_lengths[idx]
            batch_seq_lens.append(seq_len)

        # 计算总长度和工作负载
        total_seq_len = sum(batch_seq_lens)
        workload = sum(packer.calculate_workload(seq_len) for seq_len in batch_seq_lens)
        total_workload += workload

        print(f"Micro Batch {micro_idx}:")
        print(f"  包含样本: {partition_indices}")
        print(f"  样本数量: {len(partition_indices)}")
        print(f"  各样本长度: {batch_seq_lens}")
        print(f"  总序列长度: {total_seq_len} tokens")
        print(f"  工作负载: {workload:,.0f}")
        print(f"  平均长度: {total_seq_len / len(partition_indices):.1f} tokens")
        print()

    # 计算负载均衡统计
    workloads = []
    seq_lengths = []
    for micro_batch in micro_batches:
        partition_indices = micro_batch.meta_info.get('partition_indices', [])
        batch_seq_lens = [sequence_lengths[idx] for idx in partition_indices]
        workload = sum(packer.calculate_workload(seq_len) for seq_len in batch_seq_lens)
        workloads.append(workload)
        seq_lengths.append(sum(batch_seq_lens))

    print(f"{'=' * 80}")
    print(f"负载均衡统计:")
    print(f"{'=' * 80}")
    print(f"工作负载分布:")
    print(f"  最大: {max(workloads):,.0f}")
    print(f"  最小: {min(workloads):,.0f}")
    print(f"  平均: {np.mean(workloads):,.0f}")
    print(f"  标准差: {np.std(workloads):,.0f}")
    print(f"  不平衡度: {(max(workloads) - min(workloads)) / np.mean(workloads) * 100:.2f}%")
    print()
    print(f"序列长度分布:")
    print(f"  最大: {max(seq_lengths)} tokens")
    print(f"  最小: {min(seq_lengths)} tokens")
    print(f"  平均: {np.mean(seq_lengths):.1f} tokens")
    print(f"  标准差: {np.std(seq_lengths):.1f} tokens")

    # 可视化（简单的文本条形图）
    print(f"\n{'=' * 80}")
    print(f"工作负载可视化:")
    print(f"{'=' * 80}")
    max_workload = max(workloads)
    bar_width = 50
    for i, workload in enumerate(workloads):
        bar_len = int((workload / max_workload) * bar_width)
        bar = '█' * bar_len
        print(f"Batch {i}: {bar} {workload:,.0f}")

    print(f"\n{'=' * 80}")

    # ============ 测试 restore_results_order ============
    print(f"\n{'=' * 80}")
    print(f"测试 restore_results_order:")
    print(f"{'=' * 80}")

    # 1. 模拟计算结果（已经按照打乱的顺序 concat 在一起）
    # 计算总样本数
    total_samples = sum(len(mb.meta_info['partition_indices']) for mb in micro_batches)

    # 创建模拟的计算结果（按照打乱的顺序）
    shuffled_results = {
        'logits': torch.arange(total_samples).float().unsqueeze(1),  # [total_samples, 1]
        'loss': torch.arange(total_samples).float() * 10,  # [total_samples]
    }

    print(f"模拟计算结果（打乱顺序）:")
    print(f"  logits shape: {shuffled_results['logits'].shape}")
    print(f"  loss shape: {shuffled_results['loss'].shape}")
    print(f"  logits 前5个值: {shuffled_results['logits'][:5].squeeze().tolist()}")
    print(f"  loss 前5个值: {shuffled_results['loss'][:5].tolist()}")

    # 2. 获取 partition_indices_list
    partition_indices_list = mini_batch.meta_info['partition_indices_list']
    print(f"\npartition_indices_list: {partition_indices_list}")

    # 3. 还原顺序
    restored_results = LoadBalancePacker.restore_results_order(
        shuffled_results,
        partition_indices_list
    )

    print(f"\n还原后的结果（原始顺序）:")
    print(f"  logits shape: {restored_results['logits'].shape}")
    print(f"  loss shape: {restored_results['loss'].shape}")
    print(f"  logits 前5个值: {restored_results['logits'][:5].squeeze().tolist()}")
    print(f"  loss 前5个值: {restored_results['loss'][:5].tolist()}")

    # 4. 验证还原是否正确
    # 由于我们的模拟数据是 [0, 1, 2, 3, ...] 按打乱顺序排列
    # 还原后应该对应原始索引的顺序
    print(f"\n验证还原正确性:")

    # 构建期望的结果（按原始顺序）
    current_idx = 0
    expected_order = []
    for partition in partition_indices_list:
        for _ in partition:
            expected_order.append(current_idx)
            current_idx += 1

    # 将期望顺序映射回原始索引
    original_order = [0] * total_samples
    current_idx = 0
    for partition in partition_indices_list:
        for orig_idx in partition:
            original_order[orig_idx] = expected_order[current_idx]
            current_idx += 1

    print(f"  期望的 logits 值（前10个）: {original_order}")
    print(f"  实际的 logits 值（前10个）: {restored_results['logits'][:10].squeeze().tolist()}")

    # 检查是否完全匹配
    is_correct = torch.allclose(
        restored_results['logits'].squeeze(),
        torch.tensor(original_order, dtype=torch.float)
    )
    print(f"  还原结果{'✓ 正确' if is_correct else '✗ 错误'}")

    print(f"\n{'=' * 80}\n")




if __name__ == "__main__":
    # 设置随机种子以便复现
    torch.manual_seed(42)
    np.random.seed(42)

    test_load_balance_packer()
