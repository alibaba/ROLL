# ROLL SEQUENCE PACKING

ROLL框架目前支持了Sequence Packing功能，通过句子打包来避免pad token，提高计算效率。本文档详细介绍该功能的实现思路以及相应使用配置方法。

> **注意**：目前只有 `megatron_strategy` 支持了 `sequence_packing`。

## 1. 简介

在RL训练场景中，rollout数据的分布通常具有长尾效应。而在常规的训练过程中，我们通常将一个micro batch的数据组合为一个batch进行训练，每条样本都会被pad到预设的最大长度，这不仅导致了算力被消耗在了大量pad token上，而且拖慢了训练速度。

为了解决上面的问题，ROLL中提供了Sequence Packing这一特性，其核心思路是：
* 将当前micro batch中长短不同的句子打包在一起以消除pad token
* 使用打包算法优化打包效率，减少micro batch数量，提高训练效率

## 2. 实现原理

### 2.1 数据划分层次结构

在分布式训练中，数据按照以下层次结构进行划分：

```
GLOBAL BATCH (全局批次)
├── DP RANK 0 → BATCH 0
│   └── MINI BATCH 0 (用于一次梯度更新)
│       ├── MICRO BATCH 0 (最小计算单元)
│       ├── MICRO BATCH 1
│       └── ...
├── DP RANK 1 → BATCH 1  
│   └── MINI BATCH 0
│       ├── MICRO BATCH 0
│       └── ...
└── ...
```

- **GLOBAL BATCH**: actor_infer产生的完整rollout结果
- **BATCH**: Global Batch按DP rank划分后的子集
- **MINI BATCH**: Batch中用于单次梯度更新的数据（考虑gradient accumulation）
- **MICRO BATCH**: Mini Batch进一步划分的最小计算单元，参与单次forward/backward

在常规训练中，每个micro batch中的样本都会被padding到固定长度，造成大量计算资源浪费。Sequence Packing通过在micro batch级别进行序列打包来解决这个问题。

### 2.2 序列打包核心机制

Sequence Packing的核心目标是在消除padding token的同时，确保在复杂的分布式训练环境下（特别是Context Parallel和Tensor Parallel）能够正确、高效地运行。为了实现这一目标，打包过程需要满足特定的对齐要求，这些要求直接关系到模型能否正常训练以及训练效率的高低。

#### 2.2.1 对齐要求：2×CP_SIZE×TP_SIZE的倍数

在启用Context Parallel (CP) 和 Tensor Parallel (TP) 的情况下，序列长度必须是 **2 × CP_SIZE × TP_SIZE** 的倍数。

这个对齐要求来源于两个并行策略的需求：

1. **TENSOR PARALLEL (TP) 需求**：当启用Sequence Parallel时，序列会在forward过程中被切分到不同的TP rank上处理，因此序列长度需要能被TP_SIZE整除。

2. **CONTEXT PARALLEL (CP) 需求**：为了实现CP负载均衡，序列需要被切分为2×CP_SIZE个逻辑块，因此序列长度需要能被2×CP_SIZE整除。

综合这两个需求，序列长度必须是 **2 × CP_SIZE × TP_SIZE** 的倍数，这样才能同时满足TP和CP的正确运行要求。

#### 2.2.2 为什么需要因子2？CP负载均衡详解

在Context Parallel (CP) 训练中，因果注意力机制的特殊性会导致严重的负载不均衡问题。

**问题根源 - 因果注意力的不对称性**

考虑一个长度为6的序列 `[0, 1, 2, 3, 4, 5]`，在CP=2的情况下：

```
完整的因果注意力掩码:
     0  1  2  3  4  5
0  [ 1  0  0  0  0  0 ]
1  [ 1  1  0  0  0  0 ]  
2  [ 1  1  1  0  0  0 ]
3  [ 1  1  1  1  0  0 ]
4  [ 1  1  1  1  1  0 ]
5  [ 1  1  1  1  1  1 ]
```

**朴素切分方案的问题**：

如果简单地将序列均分为两部分：
- CP0负责: `[0, 1, 2]`
- CP1负责: `[3, 4, 5]`

那么实际的计算负载为：
- **CP0**: 只需要计算自己负责位置的注意力权重（6个权重计算）
- **CP1**: 需要计算自己负责位置对所有前面位置的注意力权重（15个权重计算）

**负载比例: 6:15 = 2:5**，CP1的计算量是CP0的2.5倍！

**解决方案 - 2×CP交错切分**

Megatron-Core采用的解决方案是将序列切分为 **2×CP** 个块，然后采用交错分配策略：

```
原始序列: [0, 1, 2, 3, 4, 5]
切分为4块: |[0,1]|[2,3]|[4,5]|[p,p]|  (需要padding到4的倍数)

交错分配:
- 块0 [0,1] → CP0
- 块1 [2,3] → CP1  
- 块2 [4,5] → CP1
- 块3 [p,p] → CP0

最终分配:
- CP0: [0,1] + [p,p]
- CP1: [2,3] + [4,5]
```

通过这种精心设计的分配策略，两个CP rank的计算负载变得相对均衡，避免了明显的性能瓶颈。

因此，**因子2是CP负载均衡的核心设计**，确保在因果注意力机制下各个CP rank的工作量基本相等。

#### 2.2.3 完整打包示例

假设当前microbatch包含以下样本（原始序列长度为8）：

| 样本ID | 原始序列 | 有效长度 |
|--------|----------|----------|
| 0 | `[0, 0, p, p, p, p, p, p]` | 2 |
| 1 | `[1, 1, 1, 1, p, p, p, p]` | 4 |
| 2 | `[2, 2, 2, 2, 2, 2, p, p]` | 6 |
| 3 | `[3, p, p, p, p, p, p, p]` | 1 |

配置参数：`CP_SIZE=2`, `TP_SIZE=1`

**步骤1：移除原始padding**
```
样本0: [0, 0]
样本1: [1, 1, 1, 1]  
样本2: [2, 2, 2, 2, 2, 2]
样本3: [3]
```

**步骤2：重新padding到对齐边界**
- 对齐因子 = 2 × CP_SIZE × TP_SIZE = 2 × 2 × 1 = 4

重新padding后的序列：
```
样本0: [0, 0, p, p] → 长度4
样本1: [1, 1, 1, 1] → 长度4  
样本2: [2, 2, 2, 2, 2, 2, p, p] → 长度8
样本3: [3, p, p, p] → 长度4
```

**步骤3：CP切分详细过程**

在CP_SIZE=2的情况下，每个序列会被逻辑上切分为 **2×CP_SIZE = 4** 个部分，然后按照交错规则分配给不同的CP rank。

具体切分和分配规则如下：

对于任意长度为L的序列，在CP_SIZE=2时：
- 序列被划分为4个连续的段：段0、段1、段2、段3
- 每个段的长度为 L/4
- 分配规则：
  - **CP0**: 段0 + 段3
  - **CP1**: 段1 + 段2

应用到我们的例子：

- **样本0** `[0, 0, p, p]` (长度4):
  - 段0: `[0]`, 段1: `[0]`, 段2: `[p]`, 段3: `[p]`
  - CP0获得: 段0 + 段3 = `[0] + [p]` → 实际处理 `[0, p]`
  - CP1获得: 段1 + 段2 = `[0] + [p]` → 实际处理 `[0, p]`

- **样本1** `[1, 1, 1, 1]` (长度4):
  - 段0: `[1]`, 段1: `[1]`, 段2: `[1]`, 段3: `[1]`
  - CP0获得: `[1] + [1]` → `[1, 1]`
  - CP1获得: `[1] + [1]` → `[1, 1]`

- **样本2** `[2, 2, 2, 2, 2, 2, p, p]` (长度8):
  - 段0: `[2, 2]`, 段1: `[2, 2]`, 段2: `[2, 2]`, 段3: `[p, p]`
  - CP0获得: `[2, 2] + [p, p]` → `[2, 2, p, p]`
  - CP1获得: `[2, 2] + [2, 2]` → `[2, 2, 2, 2]`

- **样本3** `[3, p, p, p]` (长度4):
  - 段0: `[3]`, 段1: `[p]`, 段2: `[p]`, 段3: `[p]`
  - CP0获得: `[3] + [p]` → `[3, p]`
  - CP1获得: `[p] + [p]` → `[p, p]`

**步骤4：各CP rank的最终打包结果**

- **CP0的完整输入**: `[0, p, 1, 1, 2, 2, p, p, 3, p]`
- **CP1的完整输入**: `[0, p, 1, 1, 2, 2, 2, 2, p, p]`

**步骤5：累积序列长度计算**

Padded累积长度: `[0, 4, 8, 16, 20]`

### 2.3 LOSS计算流程

在Sequence Packing模式下，loss计算需要特殊的处理流程：

1. **模型输出解包**：使用`_unpack_sequences`函数将packed的输出还原为单个序列
   - 根据`cu_seqlens_padded`计算每个序列在当前CP rank上的起止位置
   - `seq_starts = cu_seqlens_padded[:-1] // cp_size`
   - `seq_ends = cu_seqlens_padded[1:] // cp_size`

2. **逐序列loss计算**：
   - 对每个解包后的序列单独调用loss函数
   - 需要将原始数据调整到对应的序列长度（使用`adjust_sequence_length`）
   - 累加所有序列的loss值

3. **结果聚合**：
   - 将所有序列的loss相加得到总loss
   - 聚合各个序列的metrics
   - 应用loss scaling（如果启用）

这种逐序列计算的方式确保了loss计算的正确性，即使在复杂的CP+TP+packing组合场景下也能准确计算梯度。

### 2.4 负载均衡优化

为了最大化Sequence Packing的效果，ROLL在多个层面应用了**Karmarkar-Karp算法**进行负载均衡优化。

**Karmarkar-Karp算法简介**：
这是一种经典的多路划分算法，用于将一组数字划分为k个子集，使得各子集的和尽可能接近。在Sequence Packing场景中，该算法被用来确保各个计算单元的负载相对均衡，避免性能瓶颈。

主要优化包括：
- **GLOBAL BATCH → DP RANK 负载均衡**：确保每个DP rank获得相似的总token数量
- **MINI BATCH → MICRO BATCH 负载均衡**：确保每个micro batch的计算负载均衡

具体的实现细节和责任分工请参考第3.2节。

## 3. 实现流程

### 3.1 打包与解包核心逻辑

pack部分主要是在strategy中进行处理的，开启`use_sequence_packing`后strategy会自动对microbatch进行pack，并对输出的logits进行unpack并计算loss。

**核心打包函数 `_pack_sequences`** 实现了以下逻辑：
1. 移除原始padding，提取有效token
2. 计算累积序列长度（原始和padded版本）
3. 重新padding到`2*cp_size*tp_size`的倍数
4. 处理CP切分和分配
5. 拼接序列并创建`PackedSeqParams`

**Loss计算**通过`loss_wrapper`实现解包和逐序列loss计算。

### 3.2 负载均衡责任分工

负载均衡在ROLL框架中有明确的责任分工：

1. **GLOBAL BATCH → DP RANK 负载均衡**：
   - **负责模块**: Pipeline层（`batch_balance`函数）
   - **优化目标**: 确保每个DP rank获得相似的总token数量
   - **实现方式**: 在数据分发前使用Karmarkar-Karp算法重排序

2. **MINI BATCH → MICRO BATCH 负载均衡**：
   - **负责模块**: Strategy层（`make_micro_batch_iter_for_sequence_packing`）
   - **优化目标**: 确保每个micro batch的计算负载均衡
   - **实现方式**: 在micro batch生成时应用Karmarkar-Karp算法

3. **随机性保留**：
   - Batch → Mini Batch的划分保持随机性（用于shuffle），因此不进行负载均衡优化

这种分层优化策略确保了从全局到局部的各个层面都能获得良好的负载均衡，最大化硬件利用率。

## 4. 参数配置

### 4.1 如何启用SEQUENCE PACKING

要使用Sequence Packing功能，只需要在配置文件中设置 `use_sequence_packing: true` 即可。

### 4.2 配置参数详解（通俗版）

#### `algorithm`（打包算法）
- **`none`**：默认的简单打包方式，按照数据原有的顺序进行打包
- **`load_balance`**：智能负载均衡打包，会重新排列数据使得每个micro batch的计算量更加均衡，推荐使用

#### `max_packed_sequence_length_train`（训练时最大打包长度）
- 这个参数控制在训练时，打包后的序列最长可以有多长
- 比如设置为8192，意味着打包后的序列总长度不会超过8192个token
- 设置合理的值可以避免内存溢出，同时保证打包效率

#### `max_packed_sequence_length_forward`（推理时最大打包长度）
- 和训练时的参数类似，但专门用于推理阶段
- 通常可以和训练时设置相同的值

#### `min_num_micro_batches_train`（训练时最少micro batch数量）
- 控制每个mini batch至少要分成多少个micro batch
- 设置为1表示不限制，让系统自动决定最优的划分方式
- 如果遇到显存不足的问题，可以适当增大这个值来减少每个micro batch的大小

#### `min_num_micro_batches_forward`（推理时最少micro batch数量）
- 和训练时的参数类似，但用于推理阶段

### 4.3 完整配置示例

```yaml
actor_train:
  # 启用sequence packing功能
  use_sequence_packing: True
  
  # sequence packing的具体配置
  sequence_packing_args:
    # 使用负载均衡算法，效果更好
    algorithm: load_balance
    
    # 训练时打包后的最大序列长度为8192
    max_packed_sequence_length_train: 8192
    
    # 推理时打包后的最大序列长度为8192  
    max_packed_sequence_length_forward: 8192
    
    # 训练时最少分成1个micro batch（即不限制）
    min_num_micro_batches_train: 1
    
    # 推理时最少分成1个micro batch
    min_num_micro_batches_forward: 1
  
  # 必须使用megatron策略才能支持sequence packing
  strategy_args:
    strategy_name: megatron_train
```

### 4.4 使用建议

1. **必选条件**：只能在`megatron_train`或`megatron_infer`策略下使用
2. **推荐配置**：建议使用`load_balance`算法，可以获得更好的性能
3. **长度设置**：`max_packed_sequence_length`应该根据你的GPU显存大小来调整，一般可以设置为模型支持的最大序列长度
4**自定义Loss函数**：如果是自定义loss func使用sequence packing的话，请参考自定义loss func文档，确保正确设置了`apply_loss_scale`参数

通过合理配置Sequence Packing，可以在保持模型性能的同时显著提升训练效率，特别是在处理变长序列的强化学习场景中效果尤为明显。