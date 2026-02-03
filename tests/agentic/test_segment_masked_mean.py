import torch
import pytest
from roll.pipeline.agentic.agentic_actor_worker import compute_segment_masked_mean


def test_single_segment():
    """测试单段连续的1"""
    # mask: [0, 0, 1, 1, 1, 0, 0]
    # tensor: [0, 0, 2, 4, 6, 0, 0]
    # 期望: 第2-4位置的mean是 (2+4+6)/3 = 4.0
    mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0]], dtype=torch.long)
    tensor = torch.tensor([[0, 0, 2, 4, 6, 0, 0]], dtype=torch.float32)
    
    result = compute_segment_masked_mean(tensor, mask)
    
    expected = torch.tensor([[0, 0, 4.0, 4.0, 4.0, 0, 0]], dtype=torch.float32)
    torch.testing.assert_close(result, expected)


def test_multiple_segments():
    """测试多段连续的1，中间有0分隔"""
    # mask: [0, 1, 1, 0, 1, 1, 1, 0]
    # tensor: [0, 1, 2, 0, 3, 4, 5, 0]
    # 第一段(位置1-2): mean = (1+2)/2 = 1.5
    # 第二段(位置4-6): mean = (3+4+5)/3 = 4.0
    mask = torch.tensor([[0, 1, 1, 0, 1, 1, 1, 0]], dtype=torch.long)
    tensor = torch.tensor([[0, 1, 2, 0, 3, 4, 5, 0]], dtype=torch.float32)
    
    result = compute_segment_masked_mean(tensor, mask)
    
    expected = torch.tensor([[0, 1.5, 1.5, 0, 4.0, 4.0, 4.0, 0]], dtype=torch.float32)
    torch.testing.assert_close(result, expected)


def test_starts_with_one():
    """测试以1开头的情况"""
    # mask: [1, 1, 0, 1, 0]
    # tensor: [2, 4, 0, 6, 0]
    # 第一段(位置0-1): mean = (2+4)/2 = 3.0
    # 第二段(位置3): mean = 6.0
    mask = torch.tensor([[1, 1, 0, 1, 0]], dtype=torch.long)
    tensor = torch.tensor([[2, 4, 0, 6, 0]], dtype=torch.float32)
    
    result = compute_segment_masked_mean(tensor, mask)
    
    expected = torch.tensor([[3.0, 3.0, 0, 6.0, 0]], dtype=torch.float32)
    torch.testing.assert_close(result, expected)


def test_ends_with_one():
    """测试以1结尾的情况"""
    # mask: [0, 1, 1, 1]
    # tensor: [0, 2, 4, 6]
    # 期望: 位置1-3的mean是 (2+4+6)/3 = 4.0
    mask = torch.tensor([[0, 1, 1, 1]], dtype=torch.long)
    tensor = torch.tensor([[0, 2, 4, 6]], dtype=torch.float32)
    
    result = compute_segment_masked_mean(tensor, mask)
    
    expected = torch.tensor([[0, 4.0, 4.0, 4.0]], dtype=torch.float32)
    torch.testing.assert_close(result, expected)


def test_all_ones():
    """测试全为1的情况"""
    # mask: [1, 1, 1]
    # tensor: [1, 2, 3]
    # 期望: 所有位置的mean是 (1+2+3)/3 = 2.0
    mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
    tensor = torch.tensor([[1, 2, 3]], dtype=torch.float32)
    
    result = compute_segment_masked_mean(tensor, mask)
    
    expected = torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32)
    torch.testing.assert_close(result, expected)


def test_all_zeros():
    """测试全为0的情况"""
    # mask: [0, 0, 0]
    # tensor: [1, 2, 3]
    # 期望: 所有位置都是0
    mask = torch.tensor([[0, 0, 0]], dtype=torch.long)
    tensor = torch.tensor([[1, 2, 3]], dtype=torch.float32)
    
    result = compute_segment_masked_mean(tensor, mask)
    
    expected = torch.tensor([[0, 0, 0]], dtype=torch.float32)
    torch.testing.assert_close(result, expected)


def test_single_one():
    """测试单个1的情况"""
    # mask: [0, 0, 1, 0, 0]
    # tensor: [0, 0, 5, 0, 0]
    # 期望: 位置2的值是5.0
    mask = torch.tensor([[0, 0, 1, 0, 0]], dtype=torch.long)
    tensor = torch.tensor([[0, 0, 5, 0, 0]], dtype=torch.float32)
    
    result = compute_segment_masked_mean(tensor, mask)
    
    expected = torch.tensor([[0, 0, 5.0, 0, 0]], dtype=torch.float32)
    torch.testing.assert_close(result, expected)


def test_complex_pattern():
    """测试复杂模式：多段，开头和结尾都是1"""
    # mask: [1, 1, 0, 0, 1, 1, 1, 0, 1]
    # tensor: [1, 2, 0, 0, 3, 4, 5, 0, 6]
    # 第一段(位置0-1): mean = (1+2)/2 = 1.5
    # 第二段(位置4-6): mean = (3+4+5)/3 = 4.0
    # 第三段(位置8): mean = 6.0
    mask = torch.tensor([[1, 1, 0, 0, 1, 1, 1, 0, 1]], dtype=torch.long)
    tensor = torch.tensor([[1, 2, 0, 0, 3, 4, 5, 0, 6]], dtype=torch.float32)
    
    result = compute_segment_masked_mean(tensor, mask)
    
    expected = torch.tensor([[1.5, 1.5, 0, 0, 4.0, 4.0, 4.0, 0, 6.0]], dtype=torch.float32)
    torch.testing.assert_close(result, expected)


def test_batch_processing():
    """测试batch处理"""
    # batch_size=2
    # 样本1: mask=[0,1,1,0], tensor=[0,2,4,0] -> mean=3.0
    # 样本2: mask=[1,1,0,1], tensor=[1,3,0,5] -> 第一段mean=2.0, 第二段mean=5.0
    mask = torch.tensor([
        [0, 1, 1, 0],
        [1, 1, 0, 1]
    ], dtype=torch.long)
    tensor = torch.tensor([
        [0, 2, 4, 0],
        [1, 3, 0, 5]
    ], dtype=torch.float32)
    
    result = compute_segment_masked_mean(tensor, mask)
    
    expected = torch.tensor([
        [0, 3.0, 3.0, 0],
        [2.0, 2.0, 0, 5.0]
    ], dtype=torch.float32)
    torch.testing.assert_close(result, expected)


def test_segments_not_multiplied():
    """测试不同段之间不相乘（验证独立性）"""
    # mask: [1, 1, 0, 1, 1]
    # tensor: [1, 1, 0, 10, 10]
    # 第一段(位置0-1): mean = (1+1)/2 = 1.0
    # 第二段(位置3-4): mean = (10+10)/2 = 10.0
    # 如果相乘，结果应该是10.0，但实际应该是各自独立
    mask = torch.tensor([[1, 1, 0, 1, 1]], dtype=torch.long)
    tensor = torch.tensor([[1, 1, 0, 10, 10]], dtype=torch.float32)
    
    result = compute_segment_masked_mean(tensor, mask)
    
    # 验证第一段是1.0，第二段是10.0，不相乘
    assert result[0, 0].item() == pytest.approx(1.0)
    assert result[0, 1].item() == pytest.approx(1.0)
    assert result[0, 3].item() == pytest.approx(10.0)
    assert result[0, 4].item() == pytest.approx(10.0)


if __name__ == "__main__":
    # 运行所有测试
    test_single_segment()
    print("test_single_segment passed")
    
    test_multiple_segments()
    print("test_multiple_segments passed")
    
    test_starts_with_one()
    print("test_starts_with_one passed")
    
    test_ends_with_one()
    print("test_ends_with_one passed")
    
    test_all_ones()
    print("test_all_ones passed")
    
    test_all_zeros()
    print("test_all_zeros passed")
    
    test_single_one()
    print("test_single_one passed")
    
    test_complex_pattern()
    print("test_complex_pattern passed")
    
    test_batch_processing()
    print("test_batch_processing passed")
    
    test_segments_not_multiplied()
    print("test_segments_not_multiplied passed")
    
    print("\n所有测试通过！")

