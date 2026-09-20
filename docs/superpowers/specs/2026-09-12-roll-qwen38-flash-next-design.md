# ROLL 支持 Qwen3.8-Flash-Next 的设计与验收

日期：2026-09-12。状态：用户已批准，正在实现与验证；尚未通过目标模型完整 SFT/RL 验收。

## 1. 目标与当前边界

用户要求研究模型架构、改进 ROLL 及依赖，在 172.16.120.181 上使用 `/data_hdd/Qwen3.8-Flash-Next` 充分验证高效 SFT/RL。前一项 27B 修复已有独立 PR #497，本项另开开发分支，不能把其结果记作 Flash-Next 验证。

第一交付面是文本 SFT 和 GRPO/RLVR；保留原始 vision/MTP 权重及配置，导出时明确其未训练，默认不启用 MTP speculative decoding。视觉后训练、MTP 联合训练和分布式 CP 不在第一验收面内，不能宣传为已支持。不是将模型改名为 Qwen3-Next 或 Qwen3.5，也不是忽略加载失败的张量。

2026-09-17 用户补充：第一交付面还包括 on-policy distillation（OPD）。复用 ROLL 的纯 OPD 和 RL+OPD 路径，LoRA 学生必须调用显式教师，不能用关闭 adapter 后的学生基础模型代替教师。实模验收使用冻结的、确有不同权重的教师，在学生真实 rollout 上检查逐 token 教师概率、KL/优势来源、非零梯度与更新、教师稳定性、版本同步和保存后继续更新。主干全参数优先，N-gram 表继续冻结。构造分支测试通过不能作为实模 OPD 支持结论。

本设计将架构完整性和可训练参数范围分开。冻结 N-gram 表仍须执行正确查表、PLE 注入和反传；它不等于全参数训练。QSA 保留 indexer 前向选择也不等于训练了 indexer。

## 2. 新核验的事实

### 2.1 权重和资源

2026-09-12 在 .181 读取全部 safetensors 文件头，131 个分片、1,658 个张量名与索引一致，无缺片、重复键或索引映射错误。仅核验文件头/形状/字节范围对应，不是全部权重数值校验，也不是逐字节内容校验。

- `model_type=qwen4_exp`，`architectures=[Qwen4ExpForConditionalGeneration]`，输入为 BF16，未含量化配置。
- 数据载荷 359,999,963,128 字节；总元素 179,999,981,459，包含 35 个整数哈希常量。
- 文本浮点参数 176,943,899,520；扣除表后 125,743,653,760；路由专家 120,795,955,200。
- N-gram 表 51,200,245,760 参数，102,400,491,520 字节，即 95.368 GiB。
- 独立 vision 448,931,056 参数；独立 MTP 2,607,150,848 参数。共享权重及模型卡的统计口径可能不同，不直接用营销参数量作内存预算。
- 8 张 H800，每卡总显存 81,559 MiB；诊断前后全部空闲。MemAvailable 2,084,825,120 KiB，约 1,988.24 GiB；NVMe 可用约 4.0 TiB。

完整文本参数仅 FP32 master + Adam m/v 就需 1,977.50 GiB；另有 BF16 权重 329.58 GiB，若所有梯度同时常驻，BF16/FP32 梯度分别 329.58/659.17 GiB。不能把“CPU offload 开关存在”当成可用全参数训练方案。

证据：`output/roll-qwen38-flash-next-research/bf16-model/tensor-census.json`。config SHA256 `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b`；index SHA256 `99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de`。

### 2.2 架构合同

| 部件 | 本模型要求 | 实现约束 |
|---|---|---|
| 主干 | 48 层，H=2560，36 GDN + 12 QSA，重复 GDN/GDN/GDN/QSA | 用全局层号验证 PP/VPP 分片，不能要求每个 stage 持有全部 48 层 |
| GDN | K/V heads=16/48，head_dim=128，conv=4 | 对齐 q/k/v/z/beta/alpha 分量；FLA/FlashQLA 前后向数值核验 |
| QSA core | Q/KV heads=24/2，head_dim=256，输出 gate | Q 和 gate 按每个头交错；TP4 候选已通过 KV 复制下的 gate 数值/梯度回归，真实主干训练仍待验收 |
| QSA indexer | Q/KV heads=4/1，head_dim=128，block=4，budget=2048 | 选最多 512 个完整因果块，再加未满块的尾部；池化 raw K 后才 norm/RoPE |
| MoE | 每层 512 experts，top-10，FFN=640，共享 expert=640 | BF16 checkpoint 专家是 stacked 格式，保持 gate/up 及 EP 全局编号 |
| GR | 4 路残差，宽 10240，低秩 320 | 分组 zero-centered RMSNorm；SiLU(down/4)；sigmoid read 后均值；write=2*sigmoid(inject/4) |
| N-gram | bigram/trigram，16 个哈希头，head_dim=160，128 个存储分片 | 精确保留 int64 hash、槽位 offset、EOS 语义；不得根据配置近似重建不同词表 |
| PLE | 第 2 层前置，key 投影到 10240，value 投影到 2560 | 分支 gate、signed-sqrt、归一化 gated value，再做 dilation=3 的 depthwise conv，最后相加 |
| 输出 | final GR mixer 收窄到 H，直接进入 head（无额外 norm） | 不能多一次普通残差或 norm |

原始 token IDs 必须送到 PLE；不走基于 embeddings 的反查 token 路径。GR 不等价于 Megatron 当前带 Sinkhorn 的 mHC。

## 3. 已有补丁审查结果

服务器 `sugon-rl` 的研究、转换脚本和 GR 算子可提供参考，但不能直接作为完整实现合并。下列结论区分实测与静态审查。

1. **GR 重复残差已在 .181 实测。** 将注意力和 MLP 参数清零后，正确 GR 层应保持输入不变。旧 `HyperConnectionTransformerLayer` 输出相对输入最大误差 2.0703125；仅在该测试进程移除 `self_attn_bda`/`mlp_bda` 的普通残差，误差为 0。原因是其调用 `_forward_attention`/`_forward_mlp` 后又作 GR combine。峰值分配约 192.51 MiB。没有修改原补丁文件。
2. **PLE 真实形状不匹配已核验。** 旧实现 key_proj=[2560,2560]，三个 norm=[2560]，conv=[2560,1,4]；checkpoint 分别为 [10240,2560]、[10240]、[10240,1,4]。旧实现还缺分支门控的 signed-sqrt 和 dilation=3，卷积输入/相加顺序不同。小表反传非零不能证明该模型的 PLE 正确。
3. **模型工厂未连通，静态确认。** 已复制的 qwen4 adapter 只返回 stock hybrid block spec，没有将 GR/PLE 接入其实际模型工厂；单独构造 GR 层的探针没有覆盖这个接口。
4. **QSA indexer 被丢弃，静态确认。** 序列有效长度不超过 2048 时，选择集合可以退化为全部可见 token；超过预算则不等价。短序列 smoke 不能验收 QSA，也不能证明索引器更新。
5. **CPU optimizer 配置丢失，静态确认。** `megatron_strategy.py` 构造 `OptimizerConfig` 未转发 `optimizer_cpu_offload`、offload_fraction 等参数。现有 step 前 reload Adam 状态的补丁需要和真正 CPU optimizer 区分。
6. **旧报告存在过期结论。** 包括“唯一缺 PLE”、把 requires_grad=False 一概视为必分配 Adam 状态，以及用缩小模型 TP 配置推断真实模型。实现将以当前参数组、张量形状和峰值为准。

诊断日志：`output/roll-qwen38-flash-next-research/existing-adapter-probe.log`。复现使用已有镜像 ID `sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`，torch 2.13.0+cu130、mcore 0.16.0rc0、TE 2.18.0、FLA 0.5.2。已有源码不是 git checkout，后续必须保存源文件 hash，不能以包版本代表全部修改。

## 4. 方案选择

| 路径 | 优点 | 代价/限制 | 决策 |
|---|---|---|---|
| Megatron EP + 精确 GR/PLE/QSA + CPU optimizer | 复用 ROLL 调度和 MoE 并行，面向长期高效 SFT/RL | 需要真实模型适配、QSA backward、权重同步与内存生命周期工作 | 推荐主线 |
| Transformers + FSDP2/ZeRO | 便于建立架构数值参考，LoRA 可降低状态内存 | 当前 HF QSA Python 循环及 dense mask、整表 embedding、不高效的专家执行不能直接作为吞吐方案 | 作为正确性参考/LoRA 对照，不宣称已高效 |
| 旧补丁 + dense attention + 冻结表 | 表面上改动少 | 已确认 GR/PLE 错误；长序列改变架构；缺失真实 train/infer 闭环 | 不采用 |

推荐将交付拆成下面三个可独立验收的训练模式。默认先完成 A 与 B，C 是独立扩展验收，不将它们混称全参数训练。

- **A：架构完整的 LoRA SFT/RL。** 文本主干保持正确前向；N-gram 表冻结，adapter 参数按清单训练。明确列出 router/GR/PLE/indexer 是否包含 adapter 或被训练，不依赖模糊的 all-linear 默认。目标是单机先获得可用的高效率闭环。
- **B：主干全参数、N-gram 表冻结。** 训练 125.74B 文本主干（含 QSA indexer 的独立 KL 项）；表在共享 CPU 存储查表，PLE 小件正常训练。不能称为 176.94B 全参数训练。其 CPU optimizer/gradient 峰值必须实测，容量门槛未过不能发布为可用配置。
- **C：可训练 N-gram 表。** 独立 row-sharded CPU embedding backend + 全局合并重复行梯度 + 明确的 sparse optimizer 合同。SparseAdam 和只在命中时更新的 lazy Adam 不等价于原训练配方的 dense Adam；必须 opt-in、写入配置/checkpoint 并测试。要求精确 dense Adam 时另走带足够资源或分层状态存储的路径；本机能否高效实现不作无证据承诺。

## 5. 分层实现设计

### 5.1 模型适配和 checkpoint 合同

在 mcore_adapter 增加 `qwen4_exp` 的 config、template 和 model，并在模型工厂显式选择正确 decoder。GR 和 PLE 放在模型专用模块，避免全局 monkeypatch 普通 TransformerLayer。GR 直接消费 attention/MLP 分支输出；attention/MLP 内不再执行普通 BDA residual。GR norm 取代对应 input/pre-MLP norm。

维护一份按 checkpoint key 定义的处置表：trainable、frozen external、preserved auxiliary、derived constant。每次 load/export 生成 coverage report；未知 missing/unexpected key 为错误。128 个表分片保留独立存储，不拼成 95 GiB 临时张量。没有实现的 indexer/GR/PLE 不得用 DropConverOp 隐藏。

HF ↔ MCA 转换分别处理 GDN 的六段输入投影、Q/gate 交错、SwiGLU 两段与专家 EP/ETP 编号。序列分片的 GR 参数梯度必须正确同步。权重导出切片需独立 storage，防止保存视图导致文件放大。

构造 PP/VPP 本地层时，使用固定 Megatron 版本的 `get_transformer_layer_offset` 和 `get_num_layers_to_build` 取得当前虚拟 stage 的全局起点与层数。混合注意力模式校验和 QSA/GDN 模块替换必须使用同一段全局 `layer_types`，错误报告保留全局层号。回归覆盖四层周期不对齐的六层 stage 和三层虚拟 chunk；构造与资产测试通过不解除 PP/VPP 训练预检限制。

### 5.2 QSA 前后向与索引器目标

先建立 FP32 小规模参考；真实序列路径使用 block 索引和在线 softmax，不分配全局 `[B,S,S]` attention mask。raw K 以 4-token 块池化后 norm，块起始位置施加 64 维 partial RoPE；每个 query 只能选择同一 segment 的完整过去块，并加入因果尾部。

选择器可分 tile 扫描完整块并维护精确 top-k，减少临时存储；计算量仍含 O(S²/4) 的索引打分，不能宣传整体为线性复杂度。选定索引保存在 backward/recompute 中，避免重算时 tie-breaking 改变训练图。

训练核需支持 Q/K/V 梯度和 selected-key scatter accumulation；forward 保存 LSE/索引，backward 重算分数，避免保留所有 attention probabilities。不能直接把 vLLM 的 paged inference kernel 当成支持 autograd 的训练核。

技术报告 §2.1.2 Eq.17–20 明确索引器 KL：teacher 是跨注意力头聚合并 L1 归一化的 token 概率，再按完整微块 max-pool；稀疏训练只在选中完整块上重新归一化，与 indexer score softmax 做 KL。teacher stop-gradient，tail 不进入 block KL，没有完整块时 KL 为 0。记录 `qsa_indexer_kl_coef`，分别报告 SFT/RL 主损失和索引器损失，不能直接复用 RL reference KL 系数。

官方报告 Eq.15 未显示温度，当前 HF 参考 score 除以 sqrt(128)。正比例不影响 top-k，但影响 KL 的温度；因此 selection 按参考确认，KL 实现显式定义 temperature 并以报告公式为默认，不能暗中混用。对已有 post-trained checkpoint 不重新执行 1000-step indexer 初始化配方。

短序列 dense core 快路径仅在全部合法 token 都被选择时启用；它仍计算已配置的 indexer KL。超过预算的长度必须测试真实稀疏选择。

### 5.3 N-gram 与 PLE

节点内存储由唯一 owner 管理表，worker 通过请求行 ID 取得 embedding；prefetch 可以和早期层计算重叠。冻结表允许只读 mmap/共享内存，不能每 worker 复制 95 GiB。只用有上限的 pinned staging buffer，不将整表无条件 pin。checkpoint 记录表、hash 常量、配置的内容 hash/来源及可移植重定位规则；恢复时先校验，不依赖写死绝对路径。

PLE 在宽残差上按分支计算 key/query gate，value 广播到 4 个分支；遵循 signed-sqrt(dot/sqrt(H))、sigmoid、grouped norm 和 dilation=3 的卷积顺序。不得 detach residual。训练模式 C 的 embedding autograd 仅传回命中行，累积阶段先全局 coalesce 再更新，同步 step/version。

第一版关闭跨样本 packing 和 CP；普通 batch/padding 要正确屏蔽。之后 packing 必须同时处理 QSA block 边界、GDN reset、N-gram 前两 token 上下文及 PLE 9-token conv state。仅传一个 attention mask 不够。所有未验收组合在预检阶段报错。

### 5.4 并行与 CPU optimizer

主干训练当前候选布局 TP=4、EP=8、ETP=1、PP=1、CP=1、SP=true，8 个 rank；dense DP=2，microbatch=1，梯度累积=2，全局 batch=4。此前 TP2 候选完成一次真实更新后在第二步触及显存上限。TP4 已通过八卡小模型损失/梯度回归及真实配置预检，100 步训练和续训仍待验收。必须由真实 mcore process groups 验证 EP 与 DP×TP 的约束。不能把 TP×EP 简单当作物理 GPU 数。rollout 用 TP=8，训练到推理的重分片独立验收。

容量估算与完整验收须分开：此前 TP2 每卡专家 BF16 权重约 28.125 GiB，dense 权重约 4.608 GiB。TP4 减少 dense 复制，实际各阶段峰值由 v11 采样确认。若所有 FP32 梯度同时保留，单卡会超 80 GiB；需要分桶梯度 D2H、CPU 端 FP32 累积或其它经过数值验证的状态布局。不能靠只 offload Adam moments 解决全部峰值。

转发并验证 Megatron CPU optimizer 配置，记录实际 optimizer 类型、参数/状态设备和大小。将 GPU Adam 的 phase offload 与真正 CPU optimizer 分成明确分支；CPU 分支不能在 step 前无条件将 Adam 状态拉回 GPU。处理 SFT metadata、RL phase 切换、梯度累积、global clipping、overflow/skip 与 checkpoint state。

CPU phase 更换模型 shard 对象时，必须同步 Hybrid 的正向／反向参数映射、optimizer groups/state，并使 Megatron bucket 的参数／梯度通信缓存失效。只改变 Tensor 的 device 或检查旧引用的指针不足以证明卸载成功；验收需比较当前 CPU master/moment 对象身份、旧 GPU 存储释放及多次切换后的连续更新轨迹。仅加载模型以计算 logprob 时不重建训练梯度。

SFT 的断点恢复必须在 worker 初始化策略后实际加载模型、optimizer、scheduler 和 RNG；不能只恢复 pipeline 的数据步数。Megatron torch_dist 模型加载使用未包装模型的 sharded-state 模板，结束或失败后恢复 DDP 包装。Hybrid Adam 的标量 step 使用公共 parameter-group checkpoint 元数据，不作为模型形状的 tensor shard；bucket padding 标志也不进入 Adam 状态。当前固定依赖补丁 `scripts/qwen38/patch_megatron_hybrid_checkpoint.py` 在 CPU gradient-staging 补丁之后应用，修改前校验完整源文件哈希。PyTorch global-plan validator 兼容层保持已安装版本的 bool 或错误列表返回合同。

有界 CPU optimizer 的冷恢复初始化必须复用 CPU gradient staging，不创建另一整份 CUDA 梯度、不改变模型值或消耗 RNG。相同参数布局的恢复必须保留已有 FP32 master owners，载入后将 master state 归一到实际 optimizer 参数，避免重复创建或保留整个 FP32 主参数镜像。`scripts/qwen38/patch_megatron_checkpoint_initialization.py` 对固定依赖实施上述合同，支持校验后升级和重复应用；小模型的存储身份与精确下一步更新测试不能替代真实全模型峰值验收。

Megatron 从完整参数 Hybrid 构造 distributed optimizer 时会重建分片 Hybrid；旧 CPU optimizer 的 step hook 引用环可能延长完整 FP32 master 的生命周期。ROLL 的 CPU distributed optimizer 工厂在返回前执行 GC，保证这些已替换的 owners 在 Adam moments 初始化前回收。回归检查即使关闭自动 GC，旧 owner 及其 CPU masters 也不再存活，并保留实际训练轨迹；不提高 Ray 内存阈值来掩盖额外状态。

有界 CPU distributed optimizer 完成自身 checkpoint 模板和状态恢复后，模型 checkpoint 的 GDN/MLP factory 合并期间暂时释放 CUDA 训练梯度为 CPU 标量占位。模型合并结果安装并释放后再重建梯度缓冲；异常路径恢复 DDP 包装和原有梯度阶段，避免同时保留梯度与整套合并权重。Optimizer 模板构造仍要求物化 bucket 几何，不能在其之前丢弃梯度。临时上下文不能把完整梯度搬到 CPU，也不能重复注册同一模型的 backward hooks。 模型读取或安装抛错时，在重建梯度前清除已结束 traceback frame 的局部引用（包含异常 cause/context 链），保留原始异常类型、消息与堆栈位置；单个 optimizer leaf 重建失败仍尝试清理其余 leaves，并向调用方报告失败。

Megatron scheduler 的原生加载使用 `step(increment=saved_num_steps)`；ROLL 恢复前重置计数，使 checkpoint 步数具有绝对含义。重复恢复不能把已有步数累加进去而提前降低学习率。实际恢复回归同时比较 scheduler、CPU Adam 外层和子优化器状态、原始梯度及下一步完整参数更新。

恢复任何可变训练状态之前，各 rank 必须共同检查所需 RNG 文件，任一缺失使所有 rank 一致拒绝恢复；RNG 文件在读取前消失同样报错。恢复测试覆盖 Python、NumPy、Torch CPU/CUDA 及 Megatron 命名 CUDA RNG streams。Global-plan validator 的测试从独立加载的已安装原生源码取得参考，避免测试顺序导致补丁与自身比较。

LoRA 训练 checkpoint 的根目录保存 scheduler/RNG/optimizer 和统一的外部资产身份，各 adapter 的 MCA 权重保存在对应名称子目录。恢复按同一目录合同组装 adapter 映射，在加载 optimizer 前校验全部 adapter tensor 的键和形状；PEFT 的非严格加载不能静默接受缺失 adapter。固定 PEFT 依赖中的 adapter 名插入只修改 tensor 后缀，不能改写 GR 模块名内的 `weight`；兼容修复由 `scripts/qwen38/patch_peft_adapter_state_keys.py` 校验完整源哈希后实施。多个 PP/VPP stage 的冻结资产身份须聚合成全局覆盖完整的清单，允许本地无 PLE 层，任一 stage 的身份失败须同步到其它 rank。此资产合同不替代 PP/VPP 前后向及流水线通信验收。

验收 runner 的完成标记只证明所记录的更新次数和 checkpoint 结构检查，不替代真实冷恢复与精确下一步更新。LoRA 检查须覆盖每个 adapter 的迭代 tracker、所有 TP/PP/EP 分片、外部资产 schema/PLE 层覆盖，以及可反序列化且字段完整的 scheduler 和全部 RNG。主干模型加 CPU Adam 的 checkpoint 约 1.77 TB；验证配置将 `rpc_timeout` 设为 14400 秒，覆盖模型保存及慢存储上传。任何 worker 上传超时或缺少 pipeline 状态均不能写入完成标记。

容量预检和运行时采样应覆盖模型初始化、optimizer 首步懒初始化、backward、step、export、rollout warmup、save/resume。规划至少 128 GiB 主存余量和每卡约 8 GiB 未占用空间；达不到则调小 microbatch/采用 A 模式，不自动改写用户指定的训练范围。

Adapter 本地读取或键/形状校验失败，必须在进入 optimizer、scheduler、模型加载前汇总到全部 rank；故障 rank 保留原始异常，其它 rank 的错误包含来源 rank。非零 rank 的第二个 adapter 缺失、损坏、缺键、多键、形状或类型错误均须使所有 rank 保持当前可变状态和 DDP 包装；修复文件后仍能恢复。

大量专家 LoRA 参数的梯度裁剪必须使用一次全局 norm 和同一 clipping coefficient。固定 Megatron 依赖通过 `scripts/qwen38/patch_megatron_gradient_clipping.py` 将原生 in-place 缩放调用限制为每批 2048 个梯度，避免超过 TE 的 NVTETensor 句柄池容量；不能对每批重新计算 norm。补丁只接受明确的前后源文件 SHA256。已安装 TE 的句柄池上限为 20321；当前官方文档的池容量环境变量在该编译库中未生效，不能用未验证配置替代此修复。

vLLM 的 LoRA 加载按实际函数签名选择现代或旧版选项，不以开发版版本号的数值大小推断接口。正则 target_modules 保持字符串含义，列表保持列表，调用方配置不被原地修改。遵循原生加载器的 unstacked weights mapper、skip prefixes、专家加载布局和 3D LoRA 元数据；原始 loader 异常直接传播，不通过捕获 TypeError 重试旧接口。API 兼容测试不替代 GR/PLE/GDN 专有 adapter 的实模 rollout 对账。

#### 有界 RL token statistics

Qwen4 的 SFT 继续使用带 labels 的分块 vocabulary cross entropy。ROLL 的 RL forward 不传 labels，因此另设 `model_config_kwargs.bounded_rl_token_statistics=true` 显式启用的模型输出合同；默认关闭时仍返回普通 logits。启用后，Qwen4 在 output head 前按 token 分块计算所选 next-token logprob 和全词表 entropy，返回真实 FP32 `Tensor[B,S,2]`。ROLL strategy 只按初始化时验证过的 capability flag 解释两个 channel，不根据最后一维大小猜测输出类型。

labels 沿用现有 Megatron strategy 规则：`input_ids[:,1:]`、按 `response_mask[:,1:]` 将 response 外目标置 0、末尾补 dummy 0，strategy 再取 `[:, :-1] * response_mask[:, 1:]`。`.181` 当前 vLLM V1 sampler 的 raw logprobs 在 temperature 和 penalty 前计算，因此该路径固定使用原始 logits 和 `temperature=1`，不提供 processed-logprob 或温度缩放扩展。CP、跨样本 packing、MTP、带 bias/deferred gradient/custom communication 的 head 以及 output-head adapter 在 vocabulary allocation 前报错；transformer adapter 和 frozen plain head 可用。forward 不缓存跨 microbatch 的 labels、mask 或统计量。

### 5.5 vLLM 与 ROLL 权重更新

固定依赖源码/镜像，使用能力探测加限定版本分支，不修改 `vllm.__version__` 伪装为旧版本。已有定制安装用 `models/qwen3_8_flash_next`，当前上游用 `models/qwen4_exp`，不能用名称搜索 miss 推断模型不存在。

接入 points 包括 executor 的 hybrid cache block negotiation、Ray GPU API、worker sleep/wake、模型装载和 quantization patch 的隔离。mapper 只能应用一次：当前上游模型 `load_weights` 已调用带 mapper 的 AutoWeightsLoader，不能无条件叠加旧 worker mapper 补丁。

新权重序列携带 step/version、源/目标布局和 bounded chunks。专家按 global expert ID 流式发送，不聚合整个 512-expert tensor 或整表；converter 用明确 layout capability，替换现有 model_type 白名单判断。初版 BF16→BF16，FP8 rollout 需另验 quantize/scale 更新，不直接载入另一个 FP8 checkpoint 当作同一策略。

状态机：完成/暂停生成 → 排空请求 → 缓存失效 → 流式更新全部 trainable weights → 等待所有 rank ack → 切换 version → 允许新生成。QSA K/index、GDN recurrent/conv、PLE conv/N-gram context、prefix cache 均需要处理。reference 权重保持固定且独立校验。

actor_train/actor_infer/reference 共享 8 卡时不得同时全量驻留。CPU 端也按峰值预算：不能默认在 1.4 TiB Adam 状态之外同时保留多份约 234 GiB actor/reference CPU 副本。参考模型采用独立流式加载/重用存储，先测 I/O 和 phase 时间；若内存门槛不过，B 模式不通过验收。

### 5.6 效率优化顺序

正确性基线 → MoE grouped GEMM/EP → 分桶状态和模型更新 → GDN FLA/FlashQLA 对比 → GR/PLE fusion → QSA 训练核优化。FlashQLA 是 GDN 前后向内核，并非 QSA kernel。选择后端以真实 H800、真实 head geometry 的数值/梯度与吞吐测试为准。

Muon 不作为首个 post-training 支持的前置条件。若后续加入，矩阵 Q/K/V、SwiGLU gate/up、GDN 六段必须独立处理；router、GR 低秩和 embedding 保持明确 optimizer 分类。不能从预训练配方直接移用后训练学习率和 batch size。

## 6. 验证计划：全部在 .181 产生新证据

| 级别 | 必测内容 | 通过标准 |
|---|---|---|
| V0 环境/权重 | 依赖 hash、模型 census、能力及参数范围预检 | 所有输入可复现；未知/不支持组合在启动前报错 |
| V1 组件 | GR 零分支 identity；GR/PLE 前向+梯度；GDN/专家/Qgate 转换；N-gram IDs | IDs/离散布局精确一致；FP32 reference 初始 atol=1e-5, rtol=1e-4；BF16 atol/rtol=2e-2，并记录相对 L2，不能只看 grad 非零 |
| V2 稀疏边界 | 长度 1/3/4/5/2047/2048/2049/2052/4096/8192；padding、EOS、空完整块、top-k tie | 无越界/跨 segment；选集与 reference 对齐；QKV/索引器梯度正确；2049 也不预设必然丢 token |
| V3 分布式 | TP1/2 数值；EP1/8、ETP1；真实 global expert index；SP 梯度 | gather 后 HF/MCA roundtrip 对齐；FP32 master 一步更新对照；层/参数覆盖完整 |
| V4 真实权重前向 | 48 层 + 实际 95 GiB 表；固定中英/代码输入；2K 与 8K | 与独立 HF/reference 的逐层采样和 logits/logprobs 对账；不丢关键模块；显存/内存余量合格 |
| V5 SFT | A/B 分开；固定数据，≥100 optimizer steps，含 2K/8K 窗口；heldout；中途保存恢复 | 无 NaN/OOM；trainable 参数组有梯度和 delta；冻结部分不变；连续/恢复同 seed 轨迹在已定误差内；heldout 无明显崩坏 |
| V6 RL | 同一目标 BF16 模型 actor/reference；≥20 次完整 rollout→reward→logprob→update→reload | 每轮 model version/ack 一致；至少 10 组非零 advantage 且对应非零 actor update；reference 固定；sampled checkpoint 与 vLLM logprob 对账 |
| V7 恢复/故障 | SFT/RL checkpoint 重启；中断分桶更新；表路径失效；缓存污染 | 不以半更新权重生成；清楚报错/可恢复；保存 optimizer/RNG/scheduler/table version；可搬移目录恢复 |
| V8 性能 | 固定硬件、dtype、trainable 范围、seq/batch/recompute；baseline 对优化版 | 5 次 warmup 后 ≥20 个测量 step，3 次重复；报告中位数/P95 tokens/s、step/phase 时间和内存；收益须超过重复实验波动 |

V1 容差为开工基线，任何放宽必须解释对应算子精度误差，不能为迁就错误而更改。V5/V6 是工程稳定性与训练有效更新验收，不代表下游能力提升；另留固定 heldout loss、至少 256 条代表性任务及质量退化检查。模型不产生有方差 reward 时记录数据问题并调整可验证任务，不能把零更新 RL 记为成功。

8K 必须是首个真实稀疏训练窗口；32K 为性能扩展测试，262K/1M 不凭 config 宣称支持。若 8K 超资源边界，必须明确标记未通过并继续调整实现，而不是用 ≤2K dense smoke 替代。

## 7. 代码与交付组织

1. 在独立 ROLL checkout 新建分支，保存现有用户修改；此前 PR #497 保持其原验证范围。
2. mcore_adapter：模型模块、转换、capability 和 tests。Megatron 必需接口修改单独 patch/固定修订；不覆盖 .181 原工作目录。
3. ROLL：optimizer/offload 生命周期、aux loss、模型 weight-update contract、vLLM compatibility、配置预检。
4. 可复现独立容器/源码树、依赖 lock、SFT/RL configs、验证 runner 和完整指标报告。最终按验证结果同步回本地 `framework/ROLL`。
5. 分 PR 提交通用兼容修复和模型支持；未通过真实模型验收不标记 ready。原用户对官方 PR 的授权仍有效，不另外要求一次相同发布授权。

设计阶段不修改现有服务、不运行旧环境脚本中的 restart/清理命令、不复制模型权重到本机。此次诊断已结束，GPU 已释放。后续执行只使用本任务独立路径和容器。

## 8. 参考版本与材料

- 官方技术报告：QwenLM/Qwen3.8-Flash-Next `69885871a64393807d988b27b1b5e380e8f28526`，`tech_report.pdf`，§2.1.2、§2.2、§2.3、§2.4。
- Transformers `df04b012229d50d2b6dfba32c61c3057c3a40ea1`，`src/transformers/models/qwen4_exp/modeling_qwen4_exp.py`。
- vLLM `9d88ceb02694c6e3df182ec3c87201284f50e2c6`，`vllm/models/qwen4_exp/nvidia/`。
- Megatron upstream 当前查询 `4605e3f6a88ef16f408743a207f4414597f6b168`；generic mHC 不作为 Qwen GR 等价实现。
- Context7：已 resolve `/alibaba/roll` 并读取新模型注册、Megatron、转换及 RL 工作流文档。
- 本地官方 checkout baseline：`bd6335b4b946c21fc526deee5e6acea8c23f6d79`。所有下载参考和旧补丁副本置于 `output/roll-qwen38-flash-next-research/`；不修改这些参考副本冒充新的实现。


### 2026-09-13 09:41 UTC: LoRA 实模两步通过，冷恢复与 RL 转换继续

- `.181` LoRA TP1/EP8/ETP1、rank64、CPU Adam 完成两次更新并正常退出（session40234，exit0）。训练 loss 2.414041629 → 2.196113424；heldout loss 2.588216335 → 2.577569250。这是 8 条训练/8 条 heldout 的 smoke，不是长期质量结论。
- `completion.json` 确认 8 份 adapter、16 个优化器文件、8 份 worker RNG、pipeline CUDA RNG、scheduler step2 和冻结资产清单。八 rank 均记录成功更新、有限参数值；GR/PLE/QSA/indexer/GDN/专家/shared expert 的有界采样在两步内均变化。峰值 CUDA 45,970,934,272 B（42.81 GiB）。
- 从该检查点启动实模冷恢复（session13630，driver484235，Ray26403/dashboard28292），沿用同一数据与常量学习率，把总 epoch/step 延长到4，继续两次更新。这不是与无中断轨迹逐张量比较。
- vLLM API 修复独立复审 spec PASS、quality APPROVE；实际安装版本的 CPU tensor loader 精确通过。实模 rollout 未通过，主远端源仍未同步这项修改。
- 实际 MCA CPU 转换复现 GDN/GR adapter 缺分布规则、专家 adapter 缺转换 op、QSA gated LoRA B 输出形状错误。已建立独立转换修复子任务；不能据 SFT 通过宣称 RL 可用。
- 检查点校验器复审确认 tracker 修复，但资产路径/空清单和 RNG 内部结构仍有假阳性，正在按真实 native loader 合同修正。
- 全主干完整恢复后保存、LoRA/主干各100步、RL20轮、2K/8K完整数值门槛、导出迁移故障和性能验收仍未完成。保留 v9 有效主干检查点与 v6 完整 staging；Flash-Next 未同步 vendor、未提交 PR。

证据：`output/sft-training-metrics/full-lora-smoke-v2/{completion,summary}.json`、`output/sft-lora-pipeline-smoke-v2.log`、`output/sft-lora-pipeline-coldresume-v1.log`、`output/dependency-source-vllm/qwen4-lora-conversion-red-v1.log`。


### 2026-09-13 10:07 UTC 更新

LoRA 2K 冷恢复和真实 8K 长文本 smoke 均已正常退出并完成检查点校验。冷恢复继续两步，scheduler 到4；8K 两步 loss 1.382071897→1.339245498，所有 rank 的七类训练参数采样均变化，峰值 CUDA 57,069,223,424 B。证据分别在 `output/sft-training-metrics/full-lora-coldresume-v1/` 与 `full-lora-long-smoke-v1/`。

100 步 8K LoRA SFT 已启动（session33222，driver534002），固定400条训练/256条heldout；已完成首步并继续运行，尚无100步完成结论。完整主干SFT、数值一致性、RL20轮、精确连续/恢复对账、导出/故障/性能门槛仍开放。

校验器新增损坏用例40项通过，root对三个真实检查点复核通过。round3修复由root做范围复审，见 `validator-review4-report.md`；不声称另一子代理复审通过。vLLM API修复已同步主远端，实际本地adapter文件读取探针逐元素一致。RL候选配置及80/16公开GPQA划分通过实际严格配置解析和编码检查；还未启动RL。LoRA转换新回归在真实CPU环境RED9failed1passed，修复进行中。


### 2026-09-13 11:37 UTC continuation

100-step 8K LoRA SFT is still running, with95 completed updates at11:36:07UTC. Both long-input phases contain actual8192-token batches. checkpoint50 is retained and passes the latest native structural validator:8adapter payloads,16optimizer files,8worker RNGs,pipeline CUDA RNG,scheduler51. Heldout loss at step0 was1.47875770 and before step50 was0.81504061; final heldout not measured yet. No100-step completion claim.

Final LoRA template/streaming CPU suite20passed2skipped11.32s. Public adapter export fix adds Qwen4-only direct vLLM2D serialization (not generic HF PEFT reload): RED6failed13.22s -> GREEN26passed2skipped13.46s. Complete real adapter export passed in25.03s, file3,515,704,472B/148808tensors. Native vLLM file loader/EP filtering and CPU packing passed for all48layers on EP0/EP7, each9892projections/276packed modules, actual ranks6/64 with exact scale2, zero CUDA allocation. Full model adapter forward/RL remains unverified. Independent artifact review identified2P1s now fixed and root-reviewed; independent fix re-review remains pending because collaboration control is unavailable. Detailed report: lora-artifact-fix-report.md.

Next GPU job is the prepared identical100-step configuration resumed from checkpoint50 to99, with final heldout evaluation after saving. Existing run must exit and release GPUs first. A full adapter/Adam/RNG checkpoint comparator passed a native small-checkpoint positive/negative check: identical accepted, separate LoRA/Adam changes detected, missing RNG rejected. Evidence output/sft-training-metrics/comparator-selfcheck-v1.log. Main remote production code is not changed by the isolated artifact work; no vendor sync or Flash-Next PR. All full-backbone, exact restore, numerical, RL, export lifecycle and performance gates remain open.


### 2026-09-13 12:08 UTC: 100-step LoRA complete; exact cold-resume comparison running

The continuous real-model LoRA SFT run exited 0 and all eight ranks completed 100 successful updates. The final checkpoint99 passes the latest native structural validator (scheduler100, eight adapters, 16 optimizer files, eight worker RNG files, pipeline CUDA RNG and frozen-asset manifest). Peak CUDA allocation was 57,704,576,512 B (53.74 GiB/card); minimum host available was 1,584,964,308,992 B. All seven trainable groups had finite changed bounded samples. This is mixed effective sequence length under an 8K window: 20 updates had actual8192-token batches; it is not 100 fully populated8K updates or a full frozen-parameter checksum. Heldout loss was1.47875770 atstep0 and0.81504061 beforestep50; the original run has no final heldout measurement. Evidence: output/sft-training-metrics/full-lora-100step-8k-v1/summary.json, checkpoint99-validator-v1.log, and output/sft-lora-pipeline-100step-8k-v1/completion.json.

The identical configuration resumed from retained checkpoint50, keeping total100steps, seed42, constant LR and input hashes. Active driver560995/session54613, Ray26406/dashboard28295; output/sft-lora-pipeline-100step-8k-resume-v1.log. Fresh SSH at12:05UTC confirmed the driver and all8 active GPUs. Newly executed steps51–66 have exactly matching loss, global gradient norm and batch mean length against the continuous trajectory. Earlier rows loaded from checkpoint history are excluded from this evidence. Final full adapter/Adam/scheduler/RNG/pipeline comparison waits for checkpoint99. The output-only launcher adds a native final heldout evaluation after the final save. Do not modify the running source/config, launch anotherGPU job, or stopRay globally.

Artifact serializer fixes remain isolated pending scoped independent review; complete real-file native vLLM CPU load/packing evidence covers all48layers and EP0/EP7 but not GPU forward. Main-backbone100steps and complete re-save, full numerical gates, RL20cycles, deployment/export/fault/performance gates, vendor sync and Flash-Next PR remain open. Existing user authorization remains sufficient; current SSH access works with the required sandbox network escalation.


### 2026-09-13 12:26 UTC: artifact review closed and CPU attribution recorded

Independent artifact scoped re-review is now spec PASS / quality APPROVE with nonblocking follow-ups (`lora-artifact-fix-rereview-v1.md`). Root added public-convert negative coverage for nonuniform rank/alpha, bias/DoRA/rsLoRA/modules-to-save, conflicting duplicate tensors, empty stream and malformed names/dimensions, including no partial output when a later named adapter is invalid. Exact current post_converter source plus tests uploaded to isolated regression directory only; combined CPU artifact/conversion/streaming suite36passed2skipped13.89s, session94493exit0, log output/dependency-source-vllm/lora-artifact-coverage-v2.log includes source hashes. Multi-adapter export still retains all adapters at once (~3.5GB perrealadapter); this limitation remains explicit.

CPU-only real-input GR attribution used exact pinned HF classes and current MCA hyperconnection source with CUDA uninitialized, four threads. For code2K/layer17 and en/zh8K/layer12, grouped sum/div normalization equals native mean exactly; FP32 block relativeL2 is3.33e-8–3.73e-8. Native BF16 gate accumulation differs from MCA FP32 gate accumulation by~.0027; this does not explain the observed full-FP32 cumulative gate failure. No production arithmetic or tolerance changed. Evidence output/dependency-source-vllm/gr-cpu-attribution-v1.{json,log}, sourcehashes included. Prepared output/numerical-layer-replay/hf_routing_attribution.py for selected actual GPU layer replays after the recoveryjob releases GPUs. Forced selections are diagnosis only, never natural parity acceptance.

Recovery actual steps51–84 loss/gradnorm/batch lengths match continuous records exactly; final checkpoint tensors and finalheldout remain pending. Fresh .181 storage:283GiB NVMe free,12TiB HDDfree,~1.45TiB MemAvailable. Main training sources/config unchanged. NativeGPU adapter forward, fullRL and remaining release gates remain unverified.


### SFT pipeline RNG recovery contract (2026-09-13 continuation)

The saved pipeline RNG must be restored before new driver-side data work. Resume skips completed epochs without constructing their DataLoader iterators. For a checkpoint inside an epoch, reconstruct the current iterator and discard consumed batches, then restore the checkpoint RNG before fetching the first new batch; at an epoch boundary the new iterator consumes its seed normally. This preserves Python/NumPy/Torch driver RNG and single-process randomized transforms through subsequent validation and epoch transitions. Missing RNG must fail before new training updates. Multiprocess DataLoader worker-local stochastic state is not captured by the existing checkpoint format, so exact stochastic multi-worker data replay is not claimed. The approved fixed public validation uses num_workers=0.

Actual SFT run/DataLoader/native RNG serialization regression (three checkpoint positions: middle of first epoch, epoch boundary, middle of second epoch) currently fails3cases11.77s before implementation: output/sft-training-metrics/pipeline-rng-red-v1.log. The existing100step recovery uses the unmodified pipeline; its fullcheckpoint comparison remains separate evidence.


### 2026-09-13 12:55 UTC: 100-step recovery complete; pipeline RNG defect isolated

Recovery session54613 exited0 with49newupdates(steps51–99), completecheckpoint99 and finalnativeheldout.9482877074660792(256examples). Initialheldout1.47875770 andstep50heldout.81504061; finalimprovesinitialbutregressesfromstep50. All8rank traces complete andfinite, sevenrequiredtrainablegroupschanged; peakCUDA57,700,410,880B, minhostavailable1,591,381,143,552B. Continuous/recovered49steps match loss,gradnorm andinputlengths exactly. Latestnativevalidator session78685exit0 verifies fullcheckpointstructure/scheduler100/assets/RNG.

Fulltensorcomparison v2 covered408,806tensorleaves/7,475,287,104elements and297,936DCPstorageitems. AllLoRA,Adam,scheduler,eightworkerRNGsandotherpipelineRNGstateexact. Only mismatch: pipeline/rng_state_pipeline.pth/cpu (twoentries,maxabs4). v1comparison failedclosed onincorrectrootdist_optimizerdirectory; v2usesactualiter_*/dist_optimizer andpassesupdatedactualDCP positive/negative selfcheck before realcomparison. Evidence output/sft-training-metrics/resume-checkpoint-exact-v2.{json,log}, full-lora-100step-8k-resume-v1/summary.json and resume-checkpoint-validator-v1.log. This is not fullRNG-exactacceptance.

Root repaired SFTPipeline driverRNGrecovery accordingtothe appendedcontract. ThreeactualDataLoader/randomtransformcontinuationcasesRED3failed11.77s -> combinedGREEN8passed12.10s,includingmissingRNGrefusalandexistingworkerAdamresume. Source/tests onlyisolated; no mainremoteproductiondeploymentorfullmodelrerunyet. Scopedreviewbrief sft-pipeline-rng-review-brief.md exists, but collaborationrouteagainmalformed; noindependentreviewclaim. maxsteps800 earlier suspiciondisproved: actualstrategy dividesbyDP8and logsworker100before schedulerconstruction; no schedulerfixneeded.

NativevLLMGPUprobe v1 exited1beforeengineconstruction duea diagnosticassertionconfusing publicre-exportpathwithclassdefinition. Correctinstalleddefinition isvllm.models.qwen3_8_flash_next.nvidia.model.Qwen3_8FlashNextForConditionalGeneration; fixedprobev2nowrunning(session60112),EngineCore586078,TP8/EP,BF16,eager,rank64LoRA. Log output/dependency-source-vllm/native-lora-gpu-probe-v2.log;remoteJSON output/dependency-source-vllm/native-lora-gpu-probe-results-v2.json. Root addedexplicitcustom_init_workerRPCafter LLMconstruction (nativeLLMdoesnotcallROLLextensionhookautomatically), plus launcher ownenvironmentandidleGPUcheck. Twoearlierlongcommandapprovalstimeoutbeforeexecution; simplifiedreviewablescriptcommandexecutednormally, nocurrentpermissionblock. Do notlaunchotherGPUtasksuntilthisengineexitsandGPUsrelease.

HF routingattributionprobe ispreparedanduploaded output/numerical-layer-replay/hf_routing_attribution.py; stillnotexecuted. Allfullnumerical/RL20/backbone100+re-save/fault/performance/vendor/FlashNextPRgatesremainopen.


### 2026-09-13 13:16 UTC: native LoRA lifecycle passed; routing attribution completed

The real native vLLM TP8/EP LoRA probe v3 exited 0. All ten assertions passed: baseline replay with adapter disabled, real adapter effect (56/68 prompt-token logprobs changed), eight-rank load/removal acknowledgements, exact adapter reload, and exact base/adapter replay after sleep1/wake. Three Chinese/English/code prompts used 23/22/26 input tokens. Evidence: `output/dependency-source-vllm/native-lora-gpu-probe-results-v3.json` and the v3 log. This is a small native inference lifecycle probe, not full ROLL RL, cross-framework parity, 20 weight versions, or performance acceptance. Shutdown logged forced EngineCore termination and a shared-memory warning; fresh SSH at 13:16 UTC confirmed all eight GPUs at 0 MiB.

HF single-layer routing attribution v1 exited 0 and exactly reproduced all three saved HF outputs with FP32 and TF32 disabled. Code2K layer15 changed one token's top-10 expert set across the two input trajectories; English8K layer8 changed two. Aligning selected sets reduced trajectory errors from 2.91e-4 to 1.67e-6 and 3.88e-4 to 9.87e-6, respectively. Chinese8K layer6 changed no expert set between the two HF trajectories, but same-input HF/MCA error remains 7.11e-4 and needs another attribution step. Forced selection is diagnostic only; the complete natural numerical gate remains failed. Records: `output/numerical-layer-replay/hf-routing-attribution-v1/{summary.json,events.jsonl}` and the v1 log.

SFT driver RNG fix remains covered by the earlier 8-pass CPU regression; independent scoped review is now dispatched through the actual agent tool. Corrected full-model resume-v2 launcher is prepared with separate output and Ray ports, preserving continuous checkpoint50/99. It has not started. The latest validated converter/template and pipeline RNG production changes are still isolated, pending deployment before the relevant acceptance run.

Ruling: retain all original continuous and resumed checkpoints as immutable comparison evidence. Close pipeline RNG equality with a fresh identical continuation, and continue routing diagnosis without relaxing numerical tolerances. Full-backbone 100 steps/re-save, full RL, export/fault/performance gates, vendor synchronization and Flash-Next PR remain open.


### 2026-09-13 13:53 UTC: reviewed fixes deployed; real RNG recovery rerun active

The independent SFT driver RNG scoped review is spec PASS / quality APPROVE (`sft-pipeline-rng-review-report.md`). Root deployed the reviewed pipeline plus reviewed Qwen4 template/post-converter to the main validation container after confirming all eight GPUs idle. All three source SHA256 values exactly matched the local `output/sft-training-metrics/acceptance-deploy-v2-manifest.json`. Previous production files are retained in container `/tmp/roll-qwen38-pre-acceptance-v2.tar`. Existing checkpoints are untouched. This deployment also makes the reviewed adapter conversion available to the subsequent ROLL RL run.

The corrected full-model recovery v2 is ACTIVE: local session18041, driver590612, Ray26407/dashboard28296, output `output/sft-lora-pipeline-100step-8k-resume-v2`, checkpoint timestamp `20260913-134252`. All eight workers restored checkpoint50; first new step51 began13:48:18UTC. Steps51–53 now match uninterrupted loss, global gradient norm and effective input length exactly. Keep GPUs exclusive to this job and do not mutate its sources/config. Final99/full tensor-RNG equality is pending.

Single-layer diagnostic v2 exited0(session32984). All three HF saved outputs exactly reproduced again. Switching from full-softmax/topk to logits-topk/selected-softmax changed zero expert sets in these cases and did not resolve Chinese8K layer6 same-input error(.00071133). This rules out that proposed explanation in the observed layer; no production routing change is warranted from it. Record: `output/numerical-layer-replay/hf-routing-attribution-v2/summary.json`. Natural numerical acceptance remains failed.

Root prepared `output/numerical-layer-replay/mca_single_layer_capture.py` plus launcher to instantiate the actual original layer6 spec only, load converted real weights under TP2/SP+EP8/ETP1, and capture GDN/MLP/shared/router components on identical saved input. Syntax/help/shell checks passed; the distributed capture is NOT executed and is queued after SFT. It must first reproduce the historical MCA layer output before attributing those old measurements. No full-model parity claim follows from the prepared script.

Operational note: scp approval review timed out twice; SSH transfer to host /tmp succeeded. The target output directory is container-owned, so files were copied into the existing container with docker cp then unpacked there. No current authorization gap. An attempted additional agent dispatch was misrouted as an Ignore prompt; root resumed the diagnostic preparation locally. Do not use user-input tools for agent control.


### 2026-09-13 14:21 UTC: LoRA admission race reproduced and repaired locally

Static independent preflight (`rl-candidate-preflight-v2.md`) found an unawaited colocated `add_lora.remote` and lost rank replies. Root added an unchanged-production-method test chain through InferWorker, VllmStrategy, CustomAsyncLLM and MegatronWeightUpdater, with real asyncio and a pending Future replacing the external engine/Ray transport. RED v1 failed on the test's wrong owner class name and is not bug evidence. Corrected RED v2 reproduced all eight intended failures: early update completion, swallowed engine error, discarded success replies, and accepted missing/failed rank replies. Minimal fix waits for final admission, propagates the native reply list, and rejects cardinality or boolean failures. Combined local suite:30passed1skipped(.21s), `lora-admission-green-v2.log`; v1 green had two obsolete Receiver fixtures missing the rank contract, now corrected. This verifies the source boundary; real Ray/native GPU repeated-adapter admission still needs validation. Files/hashes are listed in `lora-admission-source-manifest-v1.json`. Changes remain LOCAL ONLY, not deployed into the active SFT source.

Fresh .181 data read verifies the candidate GPQA file exists with exactly80records and SHA256 f5332641040e99df916b9318730953c8240ddbe676c178e2dfc294302c249570. The preflight's missing-local-file issue is not a missing remote dataset. Absolute paths will be used for data/output/checkpoints and rollout dumping. Full RL20/version/reference/advantage evidence remains unimplemented/unexecuted.

Additional static phase review found two issues requiring targeted reproduction before RL: MegatronTrainStrategy load_states/offload_states only invoke the frozen-module helper when include is explicit; include=None currently skips it. Also start_model_update reloads actor model parameters before transfer, whereas custom_add_lora wakes the full inference base before that sender context exits. With a ~40GB frozen actor plus ~44GB inference base percard, simultaneous residence may exceed80GB. These are static/source-budget findings, not fresh full-GPU OOM evidence. Investigate with actual tiny/native offload and admission sequencing; do not launch unattended full RL merely because the new Future tests pass. Keep the active SFT untouched.


### 2026-09-13 15:01 UTC: SFT exact recovery closed; LoRA phase regression in progress

The full 8K LoRA continuation v2 exited 0 after all 49 new updates (steps51–99). Freshly archived rank traces confirm all eight ranks updated successfully, all watched trainable samples finite, and GR/PLE/QSA/indexer/GDN/experts/shared groups changed. Loss, global gradient norm and effective input min/max/mean match uninterrupted training at every new step. Final heldout256 loss is0.9482877074660792; initial1.47875770 and intermediate0.81504061, so no monotonic-improvement or downstream-ability claim. Fresh continuation CUDA peak57,700,410,880B and min host available1,586,957,209,600B. Evidence: output/sft-training-metrics/full-lora-100step-8k-resume-v2/summary.json.

Full checkpoint comparison v3 is exact=true, failure_count0 across408,806tensor leaves/7,475,287,104elements,8adapter payloads,297,936optimizer storage items,8worker RNG files and pipeline RNG/scheduler/step. Structural validator also passed. Evidence: output/sft-training-metrics/resume-checkpoint-exact-v3.json and resume-checkpoint-validator-v2.json. Checkpoints compared are continuous20260913-095906/checkpoint-99 and resume20260913-134252/checkpoint-99. This closes exact deterministic resume for this LoRA configuration (num_workers0,shuffleFalse), not arbitrary multiworker loader recovery or the backbone/full RL gates.

Independent LoRA admission review: spec PASS / quality PASS (lora-admission-review-report.md); Linux31passed evidence matches current six-file hashes. Main validation source not yet deployed with admission fix. Legacy vLLM online-LoRA remains outside verified support.

New real tiny Qwen4/LoRA/CPU Adam phase regression v1 failed before target assertions because launcher TORCH_COMPILE_DISABLE=1 conflicted with FlexAttention; this is harness failure, not production evidence. Removing that launcher override produced intended RED2 in26.08s: default offload retains frozen CUDA weights, and real Worker model-update admission sees frozen actor base on CUDA. Minimal isolated repair adds strategy-specific model-update load kwargs and symmetric include_frozen_parameters handling, restores include=None frozen offload/load. Native checkpoint/next-update GREEN is running session94683. Evidence logs: output/dependency-source-vllm/lora-frozen-phase-red-v{1,2}.log and green-v1.log. No full GPU RL capacity claim.

Natural numerical acceptance remains failed. MCA layer6 component capture is queued after phase regression; no tolerance changes. Backbone100steps/full re-save, RL20/version/reference/advantages, export/fault/performance gates, final review/vendor sync/Flash-Next PR remain open. Preserve valid v9 and all v6 staging; avoid another1.77TB NVMe save.


### 2026-09-13 16:29 UTC: exact LoRA phase recovery repaired; real RL smoke prepared

SFT LoRA100-step8K continuationv2 is fully archived and exact:49new updates, loss/gradnorm/input metrics unchanged, finalcheckpoint comparison408,806tensorleaves/7,475,287,104elements failure_count0 includingalladapter/Adam/scheduler/worker+pipelineRNG. Finalheldout256loss.9482877074660792. ContinuationpeakCUDA57,700,410,880B, minhostavailable1,586,957,209,600B. These completedresults belong to the recorded SFTsource; subsequentphasefixes have only small/native anddistributedregressionevidence.

LoRA phase defects repaired in local+isolatedsource: defaultincludeNone nowhandlesfrozenparams; strategy-specific model-updatekwargs keepLoRAfrozenbaseCPUwhileadaptersCUDA; allchainedoptimizerparams restorebeforeDDPhooksregister; frozenpacking uses256bytealignment. Diagnosticchain separates two independentcauses of nextupdate drift: missingexpertgradientsfromprematurehookregistration (optimizer-only/individual-frozenRED2->GREEN2), and residualroundingfromunalignedfrozenpacking (alignedpackingdiagnosticGREEN1). Earlierstatement thatpackingwasruledout is superseded: itwasnot theonlycause, butwasthesecondcause. No tolerancechange.

Finalactualdefault/admissionnativeLoRAphaseGREEN2in35.43s; CPUbufferalignmentGREEN5in3.23s. Broaderregression14passed44.98s (backbone+LoRAcheckpoint andCPUoptimizer), then4rankDP4/EP2hybridlifecycletests3passedoneachrank35.36–35.71s, session4560exit0. Logs output/dependency-source-vllm/{lora-frozen-phase-green-v3,buffer-alignment-cpu-green-v1,offload-distributed-regression-v1}.log.

Actualadapteradmissionfailure thenreproduced actorCUDAretention (RED1,22.83s). state_offload_manger nowexecutesexistingoffloadinfinally andrestoresprior roll_EXEC_FUNC_NAME. All3nativecases(default,admission,admission_failure) GREEN3in44.31s, session40910exit0; exactcheckpoint/gradients/Adam/RNG/nextupdateallretained. Thiscoversoperationfailureafterload, notpartialload/cleanupitself failing orallRLfaultgates. Source/tests: newtests/third_party/megatron/test_lora_frozen_phase_offload.py andtests/utils/test_offload_buffer_alignment.py. Mainremote training sourcehasnotbeendeployedwiththesephasefixes.

IndependentadmissioncompletionreviewPASSED(lora-admission-review-report.md); independentphase/offloadreviewremainsPENDING becausemultipletoolcallsweremisroutedintoIgnoreprompts. Rootmustnotclaim independentapproval. Reviewbrief/package/rootreportprepared. Ruling: continue reversiblevalidationinanewisolatedsource snapshotwhileindependentreviewisunavailable; finalupstreamPRstillrequirescompletegatesandreview.

Chinese8Klayer6captureandHFcomponentattributionbothEXIT0. All8MCA ranksreproducedhistoricaloutput;HFhistoricaloutputandselfrepeat exact. Samewhole-layerinputerror.00071133 amplifies atonechangedexperttoken6135 (HF499vsMCA311,margin1.43e-6). WithsamecapturedMLPinput,routerlogitsexactandallsetsidentical;MLPerror7.19e-7. SamecapturedGDNinputstill1.24e-6. ForcedMCAselectionreduceslayererrorto4.25e-7, diagnosticonly. Naturalfullmodelgate remainsfailed. Evidence output/numerical-layer-replay/hf-mca-components-zh6-v1/summary.json andzh-layer6-attribution-report.md. No numericalproductionchanges.

PreparedactualRL2step smoke: output/rl-validation-data/run_rl_smoke.py, rlvr-lora-smoke-v1.yaml, run_rl_smoke_v1.sh. Uses80GPQArecordswithknownSHA,seed42,absolute model/data/output/checkpointpaths, TP1/EP8actor,TP8/EPinfer,BF16,CPUAdam,enabledboundedlogprob/entropy. Driverrecordsactualcompute_advantageandmodel_update events; existingrealworkerupdateprofilerrecordsboundedparamdeltas. Doesnotclaimfullfingerprints/referenceor20versions; finaltrainedv2isnotreloadedinthissmoke. Complete636file/~8MBsource snapshotisolatedat /tmp/roll-qwen38-rl-smoke-v1/ROLL; manifestSHA7fe888ffcd930b28f00b22849cb87f6320fec7ae86af4409b536ed909cf0c914. Mainmodelcache/outputassetsreferencedabsolutely. CPUtypedconfigpreflightpendingresult; GPUrunnotstartedasyet. Plannedoutput mainROOT/output/rl-lora-pipeline-smoke-v1,Ray26410/dashboard28299.

Remaining: natural2K/8Knumerical, mainbackbone100steps+completecold-resume re-save, fullRL20/version/reference/advantage/reloadvalidation, export/fault/performance, independentfinalreview/vendor sync/FlashNextPR. Preservevalidv9andfailedv6fullstaging; noadditional1.77TBNVMecheckpoint.


### 2026-09-13 17:31 UTC: real RL Ray initialization failure reproduced and repaired

The v1 real ROLL RL smoke (session92417) exited1 before any RL update. The inner traceback is CustomRayDistributedExecutor._init_workers_ray calling a removed native RayWorkerWrapper.get_node_and_gpu_ids RPC. Fresh SSH confirmed driver620540, InferWorker643409 and EngineCore643871 absent and all eight GPUs at0MiB before scoped regression. This supersedes earlier pending/running entries; v1 did not pass.

Installed vLLM0.1.dev20073+g8e685d198 exposes get_node_and_physical_gpu_ids; WorkerWrapperBase/GPU Worker source also confirms assigned_physical_gpu_ids is optional and the existing single-device CUDA_VISIBLE_DEVICES/local_rank0 path remains valid. Minimal production fix selects the available native RPC and preserves ROLL explicit GPU mapping. Real Ray actor RED:1failed26.04s with the same AttributeError. GREEN:23passed27.13s including the actual two-actor mapping to GPU3/GPU1 verified by UUID, existing compatibility and LoRA admission cases. Native actor creation/rank/environment/discovery are real; heavyweight model init/load are omitted only in this transport regression. Test source: tests/third_party/vllm/test_ray_executor_initialization.py. Logs: output/dependency-source-vllm/ray-initialization-{red,green}-v1.log. Local syntax and git diff whitespace checks passed. These results do not establish full RL success or all older vLLM versions.

New full RL smoke snapshot is isolated at /tmp/roll-qwen38-rl-smoke-v2/ROLL,637sourcefiles, manifestSHA03bca064482492cb4b8c85840568d3d30a87ceb8c4aaf2f300d624c46efbc3cf. Only production delta from v1 is ray_distributed_executor.py; v2 launcher uses new output/Ray26411/dashboard28300/autocache and explicit single-node VLLM_HOST_IP127.0.0.1. Original sources/checkpoints are preserved. RUN_STATE: CPU preflight session36151 exited0; all637sourcehashes and80JSONrecords verified. Real v2 GPU smoke launched session39776, driver648524, Ray26411/dashboard28300, checkpoint timestamp20260913-173313. Logs confirm the eight-GPU Ray cluster and actual actor/reward worker creation; model initialization is still in progress and no RL update has been observed. Active log: output/rl-validation-data/rl-lora-pipeline-smoke-v2.log. Do not modify this isolated source/config or launch overlapping GPU tasks; check session39776 before next continuation. Evidence and launcher: output/rl-validation-data/source-snapshot-v2.json, run_rl_smoke_v2.sh, rl-smoke-preflight-v3.log. Context7 official docs were fetched; CLI rejected its documented --research flag as unsupported. Installed sources were read directly and archived for version-specific comparison.

Current release gates: real LoRA100updates at an8K maximum and exact checkpoint50→99 continuation accepted for the recorded SFT source/config; all49newsteps and full adapter/Adam/scheduler/worker+pipeline RNG compare exactly (408806tensorleaves,7475287104elements). Actual8192-token batches occurred in20of100updates. Resume finalheldout256loss.9482877074660792, initial1.47875770/midpoint.81504061; no monotonic-quality claim. Latest LoRA frozen-phase/hook/alignment/failure-finally fixes passed native tiny/distributed regressions but await full-model revalidation and independent review. Natural full-model numerical tolerance remains unmet; GDN small arithmetic differences can flip a near-tied MoE expert choice (Chinese8K layer6 token6135), as independently replayed in HF/MCA. No tolerance or production routing changed. Main-backbone100steps and complete cold-restore re-save, full20cycle RL/version/reference/advantage/final-reload, export/fault/performance and final review remain open. Flash-Next has no vendor sync, commit or PR. Preserve v9 valid checkpoint and failedv6 complete staging; no further1.77TB checkpoint on NVMe.


### 2026-09-14 06:08 UTC: RL v2 reached model cache validation, then failed on CSA geometry

The real v2 ROLL RL run passed source/data preflight, started the eight-GPU Ray cluster and created actor/inference/reward workers. It then failed during vLLM EngineCore KV-cache initialization with `ValueError: CSA+linear layer 3 cache specs violate CSA geometry`; no `advantages`, `model_update_start`, or `model_update_return` event was recorded. The preceding Ray RPC failure is therefore repaired and independently covered by 23 passing tests, but full RL remains unverified. Native vLLM LoRA probe still passes with max_model_len=512/max_num_seqs=1. ROLL v2 used max_model_len=1536/max_num_seqs=8. The archived vLLM docs identify CSA per-state compression and cache block alignment as the enforced constraint; no production tolerance was changed.

A low-cost v3 boundary configuration is prepared locally: prompt_length=256, response_length=128, inference max_model_len/max_num_batched_tokens=512 and max_num_seqs=1, with separate launcher ports/output. It is not uploaded or run because SSH to 172.16.120.181 has repeatedly been closed after connection establishment. No overlapping GPU task is assumed; the last observed v2 driver/engine had exited, but a fresh inventory is pending remote access recovery. Files: output/rl-validation-data/rlvr-lora-smoke-v3.yaml and run_rl_smoke_v3.sh.


### 2026-09-14 19:40 UTC: CSA root cause fixed; real RL v6 active

ROLL's custom Ray executor omitted the native post-model-load `current_platform.update_block_size_for_backend(worker.vllm_config)` callback. Native Ray and multiprocessing executors both use it after attention registration to align hybrid recurrent-state and attention cache pages. The isolated production repair restores that synchronous all-worker callback when the installed platform exposes it. CSA validation, QSA math, GPU mapping and numerical tolerances are unchanged.

The actual Ray/native CSA fixture first reproduced the exact layer-3 geometry error (red-v3); red-v1/v2 were fixture import/config failures and are not root-cause evidence. After the repair, the .181 focused suite passed 25 tests in 44.74 seconds, including native cache grouping and explicit GPU UUID mapping. Local compatibility/admission coverage passed 22 tests with 1 skip. Evidence: `output/csa-diagnostic-v1/ray-cache-{red-v3,green-v1}.log` and `local-compat-green.log`. Independent scoped review found no blocking correctness issue (`csa-finalization-review-report.md`); absent-hook legacy coverage is a nonblocking gap. These transport/cache tests omit full model weights.

The earlier v6 preflight claim using a launcher that switched to v5 remains withdrawn. The corrected dedicated v6 launcher verified 818 current source files, 80 dataset rows and seed 42 at `/tmp/roll-qwen38-rl-smoke-v6/ROLL`; manifest SHA256 `fab442839fe48772a781718f8e273cf2b4e9b8f3e58e8b6920f8ddb8a29484a2`. Real full-weight two-step RL v6 is now active (Ray 26416/dashboard 28305, EngineCore 134319 and eight Ray workers). Source/config are immutable during this run. Persistent log and exit file are main ROOT `output/rl-validation-data/rl-lora-pipeline-smoke-v6.{log,exit}`; output is `output/rl-lora-pipeline-smoke-v6`. The engine is still initializing; no rollout/advantage/optimizer success is claimed. Do not overlap GPU jobs or impose a short startup timeout. Cache diagnostics are read-only and opt-in.

Next: finish v6, then validate 20 RL cycles with weight-version fingerprints/reference sentinels/final adapter admission. Full-backbone 100-step and complete cold-restore re-save, natural numerical parity, export/fault/performance and final review remain open. Preserve all checkpoint evidence and avoid another 1.77 TB NVMe checkpoint. No Flash-Next vendor synchronization or PR is yet justified.


### CUDA IPC transfer-buffer compatibility contract (2026-09-14)

Full-weight RL v6 completed native CSA grouping/engine initialization and actor loading, then failed at the first LoRA bucket import with `pidfd_getfd: Operation not permitted`. It exited 1 and released all GPUs. The sender uses expandable CUDA segments; the default container has no added capabilities or security overrides. The proposed repair uses ordinary CUDA allocation only for the newly packed IPC bucket and restores the exact prior allocator settings in a finally block. Training tensors retain their allocator policy. CPU-source conversion must also allocate the final CUDA bucket inside this scope. Preserve normal CUDA IPC, tensor values/dtypes/layout metadata, repeated-buffer lifetimes and the existing GPU UUID mapping. Validate with actual sibling sender/receiver processes under the unchanged container permissions before re-running full RL.


### 2026-09-14 20:01 UTC: full-model CSA passed; CUDA IPC repaired; RL v7 started

Full-weight v6 exited 1 at its first LoRA bucket import, after native CSA grouping and engine initialization succeeded (block size 208; cache/warmup 44.14 seconds) and inference sleep freed 58.25 GiB per GPU. Actor initialization succeeded; step-0 model_update_start was recorded at 19:43:19 UTC. The new failure was pidfd_getfd: Operation not permitted in PyTorch _new_shared_cuda. All GPUs were released. Complete log archived at output/csa-diagnostic-v1/rl-lora-pipeline-smoke-v6.log. No RL update passed.

A real sibling-process CUDA IPC regression reproduced this with expandable segments for CUDA and CPU input sources (2 failed, 1 plain-allocation control passed). The isolated fix scopes ordinary allocation to bucket packing/final CUDA conversion, restores the exact runtime allocator settings even on error, and leaves source tensors/training policy intact. GREEN 5 passed in 15.66 seconds includes repeated exact transmission, BF16/FP32 and noncontiguous inputs, CPU source, snapshot getter fallback, and real allocation-error restoration. Container capabilities and security settings are unchanged. Source/tests: roll/utils/send_recv_utils.py and tests/utils/test_cuda_ipc_weight_bucket.py; evidence roll-ipc-{red,green}-v1.log. Full-model revalidation pending.

Independent frozen-phase review completed with no blocking issue (lora-frozen-phase-independent-review.md); its actual admission tests omit tensor transfer and do not establish full RL. New IPC scoped review brief prepared.

RL v7 snapshot contains 819 verified files at /tmp/roll-qwen38-rl-smoke-v7/ROLL, manifest SHA256 f47bf75962d7ed3f4bbc1c96b6677bcc9412adfd82787dea9c1be42665d3d9f2. Dedicated v7 preflight exited 0 and verifies its actual source root, 80 rows, seed 42 and 2 steps. Run launched persistently with Ray26417/dashboard28306; log/exit are mainROOT/output/rl-validation-data/rl-lora-pipeline-smoke-v7.{log,exit}, results mainROOT/output/rl-lora-pipeline-smoke-v7. Keep source/config unchanged and GPUs exclusive while active. Next gate is successful transfer, rollout, advantage computation and optimizer updates; full 20-cycle/fingerprint/reference acceptance still follows. Other release gates remain open.


### 2026-09-14 20:38 UTC: full-model IPC admission passed; RL data truncation confirmed

Fresh v7 driver events establish successful complete native LoRA updates/admissions at version0 (20:12:12 UTC,26.30s) and version1 (20:33:12UTC,25.49s). The CSA and pidfd transport failures are crossed by real full weights. First rollout has32 responses; all score0 and finish_reason=length at128tokens. Driver observation contains0effective/0nonzero advantage tokens. Production RLVRPipeline detects final_response_mask.sum()==0 and skips that optimizer step. This is NOT successful RL training, even if driver completion later exits0. Current v7 remains active and immutable.

Ruling: prepare an immutable answer-only instruction derivative of the approved80GPQA records, preserving every question, option, answer and source ID; keep max_len_mask and the original numerical gates. This fixes the validation prompt/response-budget mismatch without accepting truncated zero-update training. Cost if wrong: another bounded2step run may still yield insufficient reward diversity, and no RL acceptance is granted. Transformation script and source/output hashes are under output/rl-validation-data/prepare_answer_only.py and gpqa-answer-only-v1.manifest.json.

Independent CUDA IPC allocation review completed with no concrete blocker; report ipc-allocation-review-report.md checks current hashes againstv7. Limitations include process-wide allocator settings beyond the ROLL lock, partial older-version coverage, mixed-device exception rather than OOM coverage, and finite IPC lifetime testing. A separate instrumentation preflight reviewer found required sentinel loss_mask_keys metadata absent; that prepared runner has not been deployed and will be repaired before runtime validation.


### 2026-09-14 20:51 UTC: v8 isolated preflight and evidence validator prepared

V8 source snapshot has831files with manifestSHA979e7b6833774ffb559c1f43bff335114a3efe5cacc850b2be84ce04819b0ced. Remote isolated root /tmp/roll-qwen38-rl-smoke-v8/ROLL; typed CPU preflight exited0 and verified all source hashes,80records,seed42,two steps,128response tokens. Data derivativeSHA7b12123041485f2ad6d20c337a31892a6ee3a61b804798ce8ef101f6676e8d5b. GPU run not started;v7 still ownsGPUs.

Root fixed the independent review's missing sentinel loss_mask_keys, selected the new tracing profile by default, rejected reused run evidence, checked sentinel shape, and changed the final-reload field to calls_completed rather than claiming gate acceptance. Exact production token-statistics AST fixture reproduced failure; runner RED7failed, combined runner/full-hash GREEN13passed. This is CPU boundary evidence, not full cluster acceptance.

New fail-closed validate_rl_versions.py checks all8source/receiver ranks and0..Nversions, full content/schema/count/bytes equality, native adapter membership, generation request IDs joined to actual rollout IDs, fixed disabled-reference equality, changing enabled reference and weights, all optimizer steps, valid nonzero groups and sleep/wake states. Fault fixtures: RED14missing-validator failures; GREEN14passed, including missingrank, mismatched content/schema, false admission, stalegeneration, unstable reference, unchanged weights, zeroadvantage, missingupdate, observererror, duplicateproducer, missingfinal and missingsleep. This validator is local-only and not independently reviewed or validated on a successful real run yet. It does not grant full release acceptance.

Independent expected adapter schema was read from prior native-validated export header:148808tensors,3494297600payloadbytes,schemaSHA136af55cfb028a7b6b8f78f22fe5563ad34fba4c5bc7b0804c4f1e9864272126. Matching verifies names/shapes/dtypes/bytes independently of new transfer telemetry. TensorInventory now records both full-content and schema hashes.

Review clarification: direct Megatron updater does not inspect admission booleans, but VllmStrategy.add_lora does reject missing/false rank acknowledgements. The observer validator still independently checks native replies; no new production admission defect is established by that review wording. Agent routing again misdirected into Ignore; root performed fixes locally. Do not invoke user-input tools for agent control.


### 2026-09-14 21:02 UTC: first-update checkpoint initialization design

V7 EXIT1, both32-response rollouts had0valid/0nonzero advantage tokens and no optimizer steps. At final checkpoint, DistributedOptimizer._get_main_param_and_optimizer_states raised master_param because Hybrid exposes only suboptimizer states created lazily by AdamW. Actual installed CPUAdam is torch.optim.AdamW; CPUoffload constructor sets init_state_fn=None. Existing bounded dummy_step is for cold loading and performs an lr0 optimizer step; using it directly before save would advance Adam's bias-correction counter and is unsuitable for a pristine checkpoint.

Ruling: initialize missing all-CPU AdamW state before serialization through the installed AdamW initialization routine, one parameter at a time, without calling optimizer.step, changing weights, consuming RNG or advancing step counters. Reuse existing FP32 master owners and restore any prior gradients. Synchronize initialized native suboptimizer state into Hybrid before DistributedOptimizer serializes common metadata and tensor shards. Scope is the target all-CPU AdamW path; partial GPU offload and other optimizer types retain their existing behavior. Regression must compare subsequent training against an untouched lazy AdamW, exercise save/reload atstep0 and verify states already initialized remain unchanged. Full native distributed checkpoint test is queued after current v8, which remains immutable.


### 2026-09-14 21:07 UTC: all-CPU Adam initialization regression passed on .181

New ROLL helper initialize_cpu_adamw_state uses native AdamW._init_group one parameter at a time; no optimizer.step is executed. prepare_cpu_adam_for_checkpoint traverses chained/distributed wrappers, handles all-CPU Hybrid AdamW and syncs native master owners before serialization. Mixed GPU and non-AdamW paths are unchanged. Strategy calls this only when CPUoffload and optimizer save are enabled. Changes are LOCAL plus standalone CPU test snapshot; activev8 and shared Megatron unchanged.

Local RED6missing-helper failures; local GREEN6(.54s) cover fused/nonfused and AMSGrad, unchanged parameters/RNG/gradients/step0, pristine save/reload, exact three subsequent AdamW updates, already-trained state no-op, failure gradient restoration and rejecting SGD reinterpretation. Fresh .181 actualtorch2.13+Megatron CPU suite7passed7.42s includes exact native Hybrid synchronization and DistributedOptimizer._get_main_param_and_optimizer_states: reproduces original master_param KeyError before helper then validates master identity/zero-step/shard keys after helper. This fixture assembles native CPU optimizer ownership and does not invoke CUDA model construction or full distributed checkpoint IO. Full real strategy save/resume tests now include pristine checkpoint cases for LoRA and backbone and are queued afterv8. No production deployment claim yet.

V8 now loaded all131weights and renegotiated block size208 at21:05:52UTC. Reference/version/optimizer evidence still pending.


### 2026-09-14 21:19 UTC: true RL update reached; rollout metadata loses request identity

V8 first update succeeded on all8actor ranks, with3nonzero advantage groups and30valid tokens. Five of32responses finished naturally; most still hit128tokens. Disabled-adapter sentinel version1 is exactly equal to version0. Version0 serialized and received inventories match all8ranks and independent148808tensor/3494297600byte schema. Generation events correctly carry eight distinct request IDs and native admitted adapter ID1913755634.

A separate existing dump defect is now reproduced: rollout_dump_to_specific_path repeats DataProto.concat's first meta_info across every sample. V8 first dump contains the same request ID ending_0 for all32rows, whereas actual native generation IDs are_0.._7. This corrupts dump provenance, not the observed training masks: actual valid tokens30 match the five natural completions. Current strict validator must reject this missing association; do not rewrite historical artifacts or silently accept them.

Ruling: capture request-specific sampling metadata in non_tensor_batch at postprocess_output_data, preserving it through expansion/concatenation/reordering, then have the dump use these per-sample records. Keep only JSON-safe generation fields (request ID, generation config, finish reasons, token IDs and token identifiers), excluding large router tensors and unrelated runtime state. Preserve legacy dump fallback when custom rollout loops do not supply these records. Validate two distinct requests with different return lengths through actual DataProto/postprocessing/concat/reorder/file-write on .181 CPU before sustained RL. Activev8 remains unchanged; its native generation events retain separate evidence.


### 2026-09-14: v8 finished; native pristine checkpoint fixed; v9 prepared

V8 exited0 at21:27:47UTC. Both optimizer steps0/1 succeeded on all8ranks, final adapterversion2 admission/wake/sleep completed, and GPUs released. Fixed disabled-reference SHA stays01d81515cc895b7fa9649478bd1c02eed0e091f629e56c90b8892745619278bb across0/1/2; enabledversion2 differs (maxabs1.51725578). Full transfer/receiver content+schema hashes match all8ranks forversions0/1/2. Three valid nonzero advantage groups exist only atstep0 (30validtokens);step1 has36validtokens but0nonzero groups. Do not claim bothsteps have policy advantage. Archived original evidence output/rl-lora-pipeline-smoke-v8 and log output/rl-validation-data/rl-lora-pipeline-smoke-v8.log. Original strict summary v8-lifecycle-summary.json rejects incorrect request counts for bothrollouts and unexpected generation requests. Exit0 is not acceptance. Python shutdown cleanup ImportError also appears after recorded completion; no training failure was reported.

Rollout provenance fix captures whitelisted per-request JSON generation metadata in non_tensor_batch before concatenation/reordering and uses it at dump; custom loops without records retain fallback. Corrected nativeCPU RED2failed (wrongID/TensorJSON), GREENv1 2passed12.06s, independent review approved production with coveragegap. Added exact generation_config/finishreasons assertions and real legacyfallback writer: GREENv2 3passed12.01s. Review report rollout-metadata-review-report.md; localfix changes no historicalv8evidence.

ActualGPU pristine checkpoint regression completed in isolated /tmp/roll-initial-checkpoint-regression/ROLL: RED2failed19.19s withoriginalmaster_param KeyError, GREEN2passed42.00s. Real tiny LoRA and backbone models exercised strategy distributed save/reload, scheduler/RNG and subsequentupdate. Full nativeCPU fixture7passed retained. This establishes small native checkpoint IO, not new fullmodel pristine-save/coldrestore. Independent helperreview pending.

Independent lifecycle-validator review reproduced five fail-open families. Added14fault cases and actualmask_elements instrumentation. RED14failed then combinedGREEN42passed1.18s. Enforce32samplingrecords/fourperrequest/payloadconsistency, valid integercounters bounded by observedmask, positivePID tiedtofilename, hexSHA and currentadapterset/order onwake/sleep. Historicalv8 predatesmask_elements and its archived originalvalidatorresult is retained without modifyingdata. Scopedre-review remains pending.

Prepared20step v9 snapshot1099files SHA71bc8c83f5879555937e3489643b15f9ee09920d071118b5440db8873579c014 at /tmp/roll-qwen38-rl-20step-v9/ROLL. Same80-record answer-only derivativeSHA7b12123041485f2ad6d20c337a31892a6ee3a61b804798ce8ef101f6676e8d5b, seed42, batch8/fourreturns,128response tokens, EP8/TP8, save_steps10. Outputwillbe mainROOT/output/rl-lora-pipeline-20step-v9; Ray26419/dashboard28308. Upload/preflight underway; not yet launched. Twentyupdates and>=10validnonzero groups plus finalversion20admission are required. V4naturalnumerical2K/8K, fullbackbone100+restore, checkpoint-vLLMlogprob comparison, RLcoldrestore, exportmigration/fault/performance gates remain open. NoFlashNextvendor sync/PR.


### 2026-09-14 21:55 UTC: v9 running; next numerical attribution

V9 preflight exited0 with all1099sourcefiles verified. Persistent supervisor202185/driver202186 launched; Ray26419/dashboard28308. Initialactor/imports advancing, no optimizerupdate yet. Immutable source /tmp/roll-qwen38-rl-20step-v9/ROLL; mainROOT output/rl-validation-data/rl-lora-pipeline-20step-v9.{log,exit}; results output/rl-lora-pipeline-20step-v9. Check local /tmp/roll-inspect-v9.py via ssh docker exec python; do not overlapGPUjobs or mutateactivecode.

Independent initial-adam review now PASS/APPROVE with no actionablefinding. Its evidence boundary remains tinyworldsize1 nativeGPU plusCPUstate tests; privateAdamW_init_group versioncoupling and mixedGPU/nonAdam staticcoverage recorded. Rolloutreview's requested testgaps are covered by nativeCPU GREEN3. Validatorfix root-reviewed and faultGREEN42; independent scopedre-review remains unconfirmed after toolingmisroute, not an approvalsblock.

Ruling: continue naturalnumerical rootcause attribution at GDN internalboundaries. Existing same-input layer6 capture demonstrates1.24e-6 GDNrelativeL2 which canflip aMoE expert atmargin1.43e-6, but doesnot locate its arithmetic source. Actualinstalled Megatron usesFLAl2norm beforethecore and explicittriangularinverse/forwardsubstitution; pinnedHF usesPyTorch l2norm insidecore andtorch.linalg.solve_triangular. DenseTP projectionreductions also changeaccumulationorder. Prepare an isolatedcapture/probe tocompare actualqkv, normalizedq/k, decay/beta, coreoutput, gatednorm and projection under identicalrealinputs. Testone boundaryintervention atatime; neverforceexpertselection or relaxacceptancetolerances. CPUalgorithmchecks arediagnostics only; freshGPUcapture isqueuedafterv9 and must firstreproduce previousMCAoutput. NoGDNproductionchange authorized by anunverifiedhypothesis.


### RL cold-resume driver RNG boundary

Static read found BasePipeline saves rng_state_pipeline.pth but RLVRPipeline.run never restores it; SFT already restores it at the first unseen batch. Scheduler input progress restores its dataset_iter_count by replaying the seeded permutation. Native vLLM uses an engine seed but does not save generation RNG, so stochastic continuation must not be claimed bitwise identical from driver restoration alone. Currentv9 uses fixedKL; adaptiveKL state is a separate boundary still unaudited.

Ruling: add a focused RLdriver RNG restore before firstnew work, requiring the already-saved pipeline RNG file when resuming. Reproduce with actualRLVRPipeline.run and WorkerState serializers in CPU mode, stopping only at the first external actor call; verify Python/NumPy/Torch nextdraws, missingfile rejection, and fresh-run nochange. Keepactivev9 immutable. Coldrestore acceptance will separately compare savedactor/optimizer/scheduler/workerRNG and fixed-token reference/adapter logits before furthertraining; do not infer stochastic rollout replay equivalence. V9 save_steps10 saves atglobalstep10 and19 (not9), so a checkpoint10 restore beginsat11.


### 2026-09-14 22:14 UTC: first v9 update and request-provenance check passed

V9 step0 finished22:11:01UTC; all8ranks recorded successfuloptimizerupdate. Independently checkedfirstrollout has32samplingrecords/32responses,8requestIDs exactlyfourtimes each, nativeoutputtokenlengths/finishreasons matchperrequest, alladvantagecounters bounded byactualmask_elements. It has54effective tokens and5validnonzero groups. Version1 disabledreferencehash remainsexact; enabled maxabsdelta1.31982803. Source/transfers andremaining19cycles are stillrunning; thisprefixobservation is not lifecycleacceptance. Evidence v9-prefix-evidence-1.json.

RLdriver RNG regression: firstREDv1 includeda CPUplatformfixtureerror (current_platform.randomNone), which isnot RNGbug evidence. CorrectedREDv2 has2failed1passed11.84s: actualnextPython/NumPy/Torch draws differfromsavedstate, and missingRNGfile reachesactorinstead ofrejecting. MinimalRLVRPipeline.run nowrequires andrestores pipeline RNG before firstexternalactorphase. CombinednativeCPU GREEN10passed11.71s includesRL3/SFT4/requestmetadata3. Activev9 unchanged; fixlocal+isolatedCPUonly. No claim of restoredvLLM stochasticgeneratorstate orfullRLcoldrestore.

PreparedGDN diagnostics at /tmp/roll-gdn-attribution-v2/ROLL, basedonv9immutablemanifest plus8diagnostic/sourcefiles manifestSHA42a8f55dcf25fb077dc1d28ae7da18debf6b9c0a48fc470cd0353e34900f6da5. Sequentialafterv9: run_gdn_intermediate_capture_v2.sh on8GPUs (mustfirstreproducehistoricalMCA layer6), thenrun_gdn_attribution_v2.sh onGPU0. BothrefusebusyGPUs. Capturer profilesfirstGDNreturn onTP ranks0/1, savingnativeinputprojection/conv/normalizedqk/decay/beta/core/norm tensors withoutchangingarithmetic; attribution comparespinnedHF onidenticalrealinput andsame-inputindividualboundaries. Scriptscompileand shellsyntaxpass; GPUdiagnostics NOTexecuted. LocalCPUtorch2.12 fixedalgorithmcomparisonoutputrelL2~1.24e-7; native .181torch2.13CPU relL2~1.1e-7 at64/128/512tokens. Thesearealgorithm-onlydiagnostics, notsupportevidence.


### 2026-09-14 22:30 UTC: actual storage budget and synchronous upload errors

Hostdf confirms NVMe mount14Tused/166GiBfree; HDD12TiBfree. Containerparent /data_nvme isoverlayonHDD; only /data_nvme/workspace/roll-qwen38-validation isNVMe bindmount. Earlierparent-directoryshutil.disk_usage outputs were not the checkpoint filesystem and mustnot beused forNVMe budgeting. V9 LoRA checkpointsare~23.6GB each, localstagingremovedaftersuccessfulupload; currenttwo-snapshot budgetfits. No additional1.77TBbackboneNVMe copy. Currentcontainerimageb31944f2eca1b7d0b9f52946ace624ecb757a5aff7c13591870667b545aa453d; inspectdependencychanges beforepreparing anequivalentHDD-mounted validationcontainer.

Ruling: reproduce CheckpointManager.upload swallowing FileSystemUploader errors usingactualtemporaryfilesystemcollision. Onfailure preserve localcheckpointand re-raise so synchronouscaller cannotcontinueasifsavecompleted. Validate successfuldefaultcleanup andkeep_local_file too. This targets synchronous saving usedbytheacceptance configs; asynchronousfutures and atomicmulti-rank completionmarkers remain separate behavior, not claimedfixed. Activev9 isimmutable.


### 2026-09-14: checkpoint upload fixed and HDD container prepared

CheckpointManager.upload now re-raises the original exception after logging, preserving failed local staging. Actual .181 CPU filesystem regression moved from RED 1 failed/2 passed to GREEN 3 passed in 0.63s. This establishes synchronous error propagation only; async future handling and atomic checkpoint completion remain separate. Active v9 is unchanged.

Created roll-qwen38-hdd-validation from the exact source image, network none/private IPC/16 GiB shared memory, all original mounts and their read-only flags retained, plus host /data_hdd/roll-qwen38-validation mounted read-write. Verified both PEFT and vLLM patched Python files by SHA256. Available HDD bytes 12293567442944 at creation. No GPU work started; import equivalence remains pending. Evidence output/cold-restore-memory/hdd-container-created.json.

Ruling: cold RL restore uses the final v9 checkpoint (step19), runs the real restored pipeline with no new updates, re-saves before any additional sentinel or inference call, then compares all adapter/optimizer/scheduler/worker and driver RNG state plus pipeline scheduler KV. Fixed-input actor sentinels must exactly match v9 version20; inference transfer and checkpoint-to-vLLM logprob comparison remain separate. No bitwise stochastic rollout continuation claim.

Scoped validator re-review closed PID and digest findings but independently reproduced three remaining fail-open inputs: three outputs masquerading as four, boolean scalar advantage values, and cross-version lifecycle rollback. Ruling: enforce all three underlying invariants. To associate actual output tokens with dumped response positions without changing immutable v9 telemetry, decode recorded output IDs with the model tokenizer and compare per-request response multisets; identical legitimate responses remain allowed. Tests use an explicit deterministic decoder. Production CLI must supply/load the original tokenizer. This is evidence validation, not a change to rollout generation.


### 2026-09-14: cold snapshot verified; persistent validation queue started

New HDD container CPU preflight matches the original exactly across346distributionversions and8modulepresence/sourcefingerprints; CUDA remaineduninitialized. The optionalflash_attn import was absent inbothcontainers; firstpreflightfailedonthatexpectation, v2recordsitsabsenceandpassesactualequivalence. Evidence output/cold-restore-memory/hdd-container-equivalence.json.

Validatorround2 closes four-outputcardinality, scalarbool/maximumcountconsistency, cross-versionorder andtoken/responseassociation. CorrectedRED8failed0.09s; combinedGREEN52passed1.22s, includingdistinctresponse/reordering/duplicationcontrols. CLI nowrequires --tokenizer withoriginallocalmodel. Actualv9prefixsteps0-8passednativedecodingand4outputassociation; cumulative23nonzerogroups. Full20cycleacceptance remains pending. Scopedround2independentre-review stillpending; no reviewerapprovalinferred.

Cold restore runner restores finalcheckpoint19 through the actualRLVRPipeline.run (zero newupdates), checks step/sourceidentity andfreshsynchronousdestination, and re-saves before additionalreference/inferencecalls. CPUnegativecasesrejectstateadvancement/wrongsource/wrongstep/asyncupload/overwritingbaseline. Comparisonusesactualexistingfulladapter/Adam/scheduler/RNGreader, addsdriverschedulerKV andexactdisabled/enabledactorreferencehashes. Thisdoesnotclaimresumedoptimizerupdate, stochasticvLLMreplay orcheckpoint-to-vLLMnumericparity. Independentproductionboundaryreview approveddriverRNG andsynchronousuploadfixes; reportresume-boundary-review-report.md.

Cold snapshot1107files at /data_hdd/roll-qwen38-validation/sources/rl-cold-v10/ROLL hasmanifestSHAee53b5a81cf769cc892e1f14c780399bd8089d2b28ba9fae12cb6b40193631b9; everyremotehashverified. FreshnativeCPU18passed13.95s. Results target /data_hdd/roll-qwen38-validation/output/rl-lora-cold-v10. Sourceisnowfixed.

HostsupervisorPID1061260 runs /data_hdd/roll-qwen38-validation/environment/queue_validation_v1.py. Queueevents/logs under environment/queued-validation-v1. It waitsforv9exit0andGPUidle, thenruns latestv9lifecyclevalidator, GDNcapturev2, GDNattributionv2, coldpreflight, coldrestore, coldcomparison insequence. Anyfailure stops thequeuewithoriginalevidencepreserved. Do notstartoverlappingGPUjobsormutatequeuedsnapshots/dependencies. NoFlashNextPR/vendor syncoroverallacceptance yet.


### Checkpoint-to-vLLM diagnostic during the next cold load

Ruling: prepare a new v11 snapshot that adds optional fixed-input prompt-logprob observation to the v10 cold re-save flow, without changing v9 or v10. Use the existing fixed 256-record public SFT heldout split (SHA256 4d1b9421cc1c48786d5da6a5360a2717e8180a58dc544a01bbb000b0c7f839e2), first en/zh/code example each, native nonthinking encoding, at most384tokens. Compare actor-selected next-token logprobs against native vLLM prompt_logprobs for identical supplied IDs and the reloaded adapter. Observe native RequestOutput at the existing generation boundary; do not alter generated tokens or returned payload. Record finite values, token identity, native adapter ID and per-token differences using the existing BF16 atol/rtol2e-2 baseline. This is a short checkpoint numerical diagnostic; natural2K/8K acceptance remains separate. Preserve cold re-save before these calls so extra probes cannot pollute the RNG comparison. Replace the waiting queue only after new snapshot CPU checks/hash verification, retaining its old source/evidence and checking no queued GPU command has started.


### 2026-09-14 23:32 UTC: v11 cold/parity snapshot and queue v2

The final next-run snapshot is now v11 at /data_hdd/roll-qwen38-validation/sources/rl-cold-v11/ROLL (1114 manifest files; SHA256 67be1e522c38dc89081de7423dbbd5cac0dc42d90c28422dc2e076bc659c3d6f). It preserves v10 and adds opt-in prompt-logprob observations to the cold re-save run. Fresh .181 CPU suite:24 passed13.99s; separate native VllmStrategy/RequestOutput async boundary with scripted engine passed, preserving returned payload and capturing one correctly aligned prompt record without CUDA initialization. This is not GPU-generation evidence.

Fixed heldout input preflight produced en149tokens, zh210tokens, code384tokens with hashes in cold-v11-input-preflight.json (raw output contains logger preamble). The comparison checks all eight actor rows against native chosen prompt logprobs using BF16 atol/rtol0.02, records adapter identity, and writes logprob-parity/comparison.json. Cold state comparison remains separately reported; the final validator exits nonzero if requested short numerical probes fail. Natural2K/8K gate remains open.

Queue v1 was stopped only after confirming its sole event was waiting_for_rl, RL exit file absent, and PID command identity. It had started no GPU command. Its evidence is preserved with superseded.json. New host supervisor1154102 runs environment/queue_validation_v2.py; events/logs under /data_hdd/roll-qwen38-validation/environment/queued-validation-v2. Order remains v9lifecycle -> GDNcapture -> GDNattribution -> coldpreflight -> coldrestorev11 -> coldcomparison. Do not overlap GPU jobs or mutate v9/GDN/v11 sources and shared dependencies. v11 output: /data_hdd/roll-qwen38-validation/output/rl-lora-cold-v11.

V9 checkpoint10 passed native structure validation: scheduler11,8adapterpayloads (6,927,002,584bytes),16optimizerfiles (16,551,884,604bytes),8workerRNGs,pipelineRNG and externalassets. Path: mainROOT/output/rl-lora-pipeline-20step-v9/checkpoints/20260914-215136/checkpoint-10. Evidence v9-checkpoint10-structure.json. Latest observed all8ranks completedsteps0-12 at23:21UTC; subsequentsteps remain live, not yet finalacceptance.


### Preserve midpoint checkpoints for full-backbone acceptance

The SFT acceptance runner hardcodes max_ckpt_to_keep=1, which removes the midpoint checkpoint after the final save. Prior LoRA continuation retained a separate copy. For the forthcoming full-backbone100-step run, retain two checkpoints in the runner so checkpoint50 remains available for cold continuation without another1.77TB copy. The HDD budget already includes both midpoint/final plus temporary staging. Existing checkpoints and queued snapshots remain unchanged.


### 2026-09-14 23:55 UTC: queue v3 launcher correction

Confirmed queue v2 SOURCE was cold-v11 while its launcher still named run_rl_cold_v10.sh. The old v10 launcher inside that snapshot would change into the superseded v10 source. Corrected the single launcher reference in a new queue v3 (SHA256 0eecad07ee9591d2d37873a8fe88a8af84c1b4b133a4423638a0e232f513d750). Verified old supervisor command identity, paused it, confirmed only waiting_for_rl and absent v9 exit file, then terminated only that waiting supervisor. Preserved evidence in queued-validation-v2/superseded.json. New host PID1261041 runs environment/queue_validation_v3.py; events under environment/queued-validation-v3. Readback verifies it is waiting. Active v9 and all immutable sources are unchanged.

At23:57UTC all8actor ranks have19successful updates (steps0-18), final step19 generation is active. Disabled reference throughversion19 remains unchanged. NVMe144GiBfree; HDD12TiBfree.

Validator round2 independent scoped re-review is PASS for four-output/decoded-response association, numeric bool/max consistency, and complete chronological lifecycle. Reviewer checked recordedRED8failed/GREEN52passed and found no blocking fix regression. Report rl-validator-round2-rereview-report.md. Decoded-text multiset association cannot establish raw-token per-position identity when distinct tokenizations decode identically; preserve this telemetry limitation. Full20cycle validation remains pending.


### 2026-09-15 00:06 UTC: real 20-step RL lifecycle accepted; cold v11 started

V9 exited0, then the independently reviewed strict validator exited0 at00:04:51UTC: lifecycle_passed=true, problems=[],20optimizersteps onall8ranks,21adapterversions,58validnonzero advantagegroups. Disabledreferencehashes remain exact. Evidence output/rl-validation-data/v9-lifecycle-final.{log,json}; this is LoRA RL lifecycle only, not fullbackbone/numerical/coldresume acceptance.

GDN capturev2 exited0 at00:05:40UTC and reproducedhistorical outputs onall8ranks. Attributionv2 exited0 at00:05:55UTC. Sameinput fullGDN relativeL2=1.238135e-6. QKV/z projections, convolution and values arebitwiseequal. a/b projectiondifferences are~8e-7 relative; normalization~6.2e-8; sameinput core~1.4-1.5e-7. The largest isolateddifference isoutputprojection: denseHF vsTP2sum relativeL2=1.168789e-6; reproducing theTP2sum isbitwiseequal toMegatron. This locatesdominant rounding inmatmul partition/reduction, not a demonstrated weightmappingerror. Natural2K/8K gate remainsopen; no tolerances/routing/sourcechanged.

Queuev3 passedcoldpreflight with1114hashes and started realcoldv11 at00:05:58UTC inHDDcontainer, driverPID1243, Ray26421/dashboard28310. Checkpoints go to /data_hdd/roll-qwen38-validation/output/rl-lora-cold-v11/checkpoints/20260915-000600. Do not overlapGPUjobs.


### 2026-09-15 00:18 UTC: full-backbone comparator and 100-step source prepared

New bounded comparator reads native model/optimizer DCP storage entries individually and reuses recursive exact payload comparison. LocalRED19failed due missing feature -> GREEN19passed1.14s; .181nativeCPU19passed2.56s. Real historical metadata confirms largest stored model item635700777B andoptimizer item1271399977B; selected extra state objects are supported lists/tensors/dicts. This establishes diagnostic coverage, not full1.77TB state equality. Independentreview ispending. Reportbackbone-dcp-comparator-report.md.

Full SFTv7 isolated1119files manifestSHA c9047b2c41bd4e35ba2538f08ad387d7fef3c489870f55813323854336026605; remoteHDD sources/sft-backbone-v7/ROLL. NativeCPUpreflight passed00:18:23UTC withTP2/EP8/ETP1/SP,denseDP4,400train/256heldout,100updates/max8K,CPUoptimizerandfrozenN-gram. Launcher output/cold-restore-memory/run_backbone_100step_v7.sh; freshoutput sft-backbone-100step-8k-v7, resumeoutput sft-backbone-100step-8k-resume-v7. Savesmidpoint50andfinal99toHDD withmax_ckpt_to_keep2;8TiBfresh/4TiBresume diskheadroom. NoGPUrun yet. ColdRLv11 continues inHDDcontainer; noactivesource/shareddependencychange.


### 2026-09-15 00:25 UTC: persistent full-backbone sequence queued

Host supervisor1421486 runs environment/queue_backbone_v1.py (SHA2078ac6ccd2224562721223a226cd3279c79728fe3d09eae87b8ddf92cfc02ea). Readbackconfirms queued-backbone-v1/events.jsonl hasonly waiting_for_cold_state_comparison. Itwaitsforv3coldcomparison, requirescold_roundtrip_passed=true, acquiressamegpu-validation-queue.lock andchecksGPUidle. ItthenrunsSFT100 -> resumecheckpoint50for49newupdates -> wholemodel/optimizer/RNG/statecomparison. Anyfailure stopsandpreserveslogs. Ashortcheckpointlogprob diagnosticfailure isrecordedseparatelyanddoesnotpreventindependentbackbonecapacity/statevalidation whencoldstateisexact; releasegate remainsfailedinthatcase. NoGPUoverlapandnochange torunningv11/v7sources.

Cold v11 actualactor/optimizerloadhasreturned andcheckpoint19re-saveisactive at00:25UTC. Fullcoldstate/logprobresultsstillpending.


### 2026-09-15 00:49 UTC: cold state equality confirmed; overall cold run failed

Live SSH readback confirms cold-v11 completed checkpoint19 re-save but exited1 at00:35:20UTC during native vLLM prompt-logprob token decoding (OverflowError). The actual offending token ID has not been captured; root cause is unproven. There is no successful overall completion marker or inference probability comparison.

The separately completed CPU state comparison reports exact=true, failure_count=0, cold_state_phase_passed=true and overall_cold_run_passed=false. All408806 tensor leaves /7475287104 elements match, including adapter/Adam/scheduler/worker and pipeline RNG, pipeline step/KV and disabled/enabled actor sentinels. This establishes restore-and-resave state equality with zero new optimizer updates, not resumed RL training or inference parity. Remote report: /data_hdd/roll-qwen38-validation/environment/cold-v11-state-phase.json; archived locally as output/rl-validation-data/cold-v11-state-phase.json.

Both queued-validation-v3 and queued-backbone-v1 have exited on the upstream failure. At00:49:06UTC all8GPUs show4MiB and0percent utilization, and neither queue nor training launcher is running. Full-backbone100-step GPU acceptance has not started. Previous statements that these queues are waiting/running are superseded. Preserve their original failure events and immutable sources.

Next work: instrument native prompt-logprob IDs before detokenization and isolate native-vLLM versus ROLL/Ray behavior, then revalidate the complete cold path and a resumed RL optimizer update. Full-backbone SFT capacity/restore validation can be scheduled independently against the explicit state-phase report, with overall inference/numerical release gates remaining failed; never synthesize an overall cold-success marker. Natural2K/8K numerical acceptance, full-backboneSFT/RL, export/fault/performance, final review/vendor synchronization and Flash-NextPR remain incomplete.


### 2026-09-15: native prompt-logprob boundary diagnostic

Use a separate diagnostic module with frozen cold-v11 production sources and existing vLLM dependencies. Reuse all three saved cold-v11 actor input token sequences. Observe LogprobsProcessor tensors before detokenization and the actual worker runner return after its existing synchronization, including all IDs/probabilities/ranks, request offsets and target IDs. Do not change returned tensors, skip decoding, force routing or relax numeric tolerances. Compare native TP8/EP eager inference before/after loading the previously validated smoke adapter and sleep/wake. That adapter is diagnostic only and cannot establish parity for the final RL checkpoint. Preserve the original cold-v11 failure and snapshots.


### 2026-09-15: reject equally incomplete or malformed DCP checkpoints

Independent comparator review reproduced two false acceptance cases with real PyTorch DCP payloads and native load failures: both checkpoints omit the same declared tensor chunk, or both store a tensor whose shape differs from the declared chunk. The existing full inspect_checkpoint structure check also accepted these fixtures. Require a shared logical/storage inventory validator used by the structure inspector and comparator: every declared tensor chunk/bytes item has exactly one storage entry; tensor chunks remain in bounds, do not overlap and cover the logical tensor. The streaming comparator must additionally validate each loaded tensor type/dtype/shape against its chunk before comparing values. Preserve per-item bounded memory and same-topology scope. Cover pristine native-loadable multi-chunk tensors and identical corruption on both sides in RED/GREEN tests. Frozen sft-backbone-v7 and cold-v11 snapshots are unchanged; a new snapshot will carry the corrected acceptance code.


### 2026-09-15: schedule independent backbone validation from the exact cold-state phase

GPU availability is intermittent because an unrelated evaluation service uses all8GPUs. The diagnostic waits without stopping that service. After prompt-ID diagnosticv2 returns and GPUs are idle, the independent backbone SFT sequence may start from a newv8 snapshot. Its prerequisite is the explicit cold-v11-state-phase.json exact=true/cold_state_phase_passed=true report; preserve overall_cold_run_passed=false and log any diagnostic failure separately. This is a capacity/save/restore experiment, not overall model-support acceptance. v8 carries the reviewed DCP inventory/payload fix and standalone comparator CLI fix; oldv7 remains immutable. Run100updates, retaincheckpoint50/99, resume49updates and compare complete state. All checkpoints stay on HDD with the existing8TiB/4TiB headroom checks. Preserve fail-stop behavior within the backbone sequence.


### 2026-09-15: final RL adapter export and native parity diagnosticv3

Public LoRAHFConverter CPU export of actualv9checkpoint19 exited0:148808BF16tensors,1747148800elements,3515704472bytes,ranks6/64,allfinite; weightsSHAaf06fcf7adc7cdaa72ac2799470ee40349a191e1590ac2311f308d3cb1ffcfe3. NoCUDA initialized. Artifact /data_hdd/roll-qwen38-validation/output/rl-v9-final-adapter-v1. Export integrity does not establish native reload or parity.

Prepare a new nativeprompt diagnosticv3 using this exact exported finaladapter and savedcold-v11 actor149/210/384token probabilities. Compare everytoken/all8actorrows at unchangedBF16atol/rtol0.02 before/after sleep. Nativebase, finaladapter, IDs and rawworker/frontend tensors remain observed. Numerical mismatch is reported only after preserving all successful requests; decoding failures remain originalerrors. This still does not replace actual ROLLRay coldrestore or natural2K/8K gates. Replace only waitingv2diagnostic/backbonesupervisors after newfiles/hash/CPUchecks and proving neither has startedGPUwork; retain superseded events. v8training source unchanged.


### 2026-09-15 09:55 UTC: integrity fix accepted, final RL export complete, persistent v3 queues active

DCP missing-chunk and wrong-payload false acceptance cases are fixed through shared metadata coverage validation and streamed payload type/dtype/shape checks. Local68passed2.03s; .181PyTorch2.13CPU68passed6.37s. Independent scoped re-review approved v4 hashes, repeated both original corrupt full-checkpoint fixtures and2400small coverage oracle cases. Standalone CLI works without PYTHONPATH. Real model124924/optimizer153258 chunk inventories pass; full1.77TB payload comparison remains unexecuted. Reports/logs: backbone-integrity-review-report.md and output/cold-restore-memory/backbone-integrity-*.

Backbone v8 frozen snapshot:1122files, manifestSHA2044499f969f2709a51582cf4300d668eb218c65e0d8d59791fe5648e8839084, /data_hdd/roll-qwen38-validation/sources/sft-backbone-v8/ROLL. CPUpreflight passed actualTP2/EP8/ETP1/SP,denseDP4,400train/256heldout andHDD8TiBheadroom. No model training update in this session.

Actualv9checkpoint19 publicLoRAHFConverter CPU export succeeded:148808BF16tensors,1747148800elements,3515704472bytes,ranks6/64,allfinite,37.76seconds,noCUDAinit. Artifact /data_hdd/roll-qwen38-validation/output/rl-v9-final-adapter-v1; weightsSHAaf06fcf7adc7cdaa72ac2799470ee40349a191e1590ac2311f308d3cb1ffcfe3. Local evidence output/rl-validation-data/export-v9-final-adapter-v1.{json,log}. Native GPU load/parity is still unverified.

Native CPU vLLM codec/decoder with actual149/210/384token IDs and explicitly synthetic probabilities passes all3cases; injectednegativeID is captured before native OverflowError. This verifies observation plumbing, not the real cold-v11 offendingID or rootcause. Probev1 loaded the engine but failed before generation because callableRPC serialization is disabled; v2/v3 use a named worker extension method. No insecure serialization setting enabled.

GPU use by an unrelated qwen38-base/qwen38-sft-final evaluation container prevents immediate probes. We did not stop or alter that service. Both own v2 supervisors were verified only waiting, paused and rechecked, then superseded with preserved markers. At09:55:54UTC activated promptv3supervisorPID3657791 and backbonev3supervisorPID3657792. Scripts under environment/prompt-id-v3, events under environment/prompt-id-queue-v3 and environment/queued-backbone-v3. Promptv3 waitsforidleGPUs, observesnativeworker/frontendIDs and compares exported finaladapter with savedcoldactor at unchanged0.02atol/rtol, before/after sleep. Backbonev3 waitsforpromptv3return (records any failure), validates cold state evidence SHA2e853c72786be73817c81d13fd181d96d4627c2d3b6d3bf866d1def1b89e5a19, then independently runsv8SFT100 ->checkpoint50resume49 ->fullstatecomparison. It explicitly retains overall_cold_run_passed=false/release_gate_passed=false. All GPU work checks sharedlock/idle; checkpoints remainHDD. Do not edit these queued snapshots/dependencies.

Still open: real offending promptID/rootcause, complete ROLLRay cold lifecycle and resumedRLupdate, natural2K/8Knumerics, fullbackboneSFT/RL, finalexportload/fault/performance, fullbranchreview/vendor sync/FlashNextPR. No overall support or PR readiness claim.


### 2026-09-16: CPU prompt boundaries and real RL continuation preparation

At04:09UTC both existing v3 supervisors still waited; all8GPUs were occupied at78339-78343MiB, even with0percent utilization. No GPU runs, service termination or queued-source changes occurred.

Native vLLM CPU prompt assembly/sampler/Msgpack/decoder diagnostic passed255exhaustive partitions of lengths1..8,15real-token patterns (en149/zh210/code384), and1reordered mixed batch with an unscheduled request. Synthetic logits and deterministic unfilled-memory canaries were used, with process-local PIN_MEMORY=False; CUDA remained uninitialized. This does not exercise GPU copies or Ray transport, and does not reproduce the real cold-v11 invalidID. Probev1 hit a CPU-only pinned-memory environment failure; v2 exposed an incorrect diagnostic rank oracle (installed batched_count_greater_than counts >= includingties); v3 uses the installed definition and passes. Evidence output/prompt-logprobs-debug/chunked-cpu-v3.{json,log}. No production algorithm change. Official Context7 and installedsource both show_sync_device before returning completedpromptlogprobs.

Added opt-in RL resume-training runner, offset-aware lifecyclevalidator and baseline-bound resumevalidator. Root integrationreview reproduced timestamped-checkpoint misbinding, empty/stale destination acceptance, missingoptimizer/stalescheduler acceptance, duplicate baseline source and malformedstep failures. Real eight-rank-layout tinyDCPfixtures exercise the existing structural inspector. Final local98passed2.48s. Native transfer/admission indices restart0in a freshprocess whiledriver/sentinel stepsremainabsolute; newvalidator preserves originalobservations and maps the ranges explicitly. Native-ordinalRED2failed54passed; earlier integrationRED9failed18passed. Independent review of this new toolwave remains pending; no GPU continuation accepted.

Ruling: prepare a fresh candidate resuming v9checkpoint19 to max_steps25 (5newupdates), retaining the existing10validnonzerogroup minimum and all8rank weight/reference/admission checks. This replaces the initial2-update preparation only; if rewarddiversity is insufficient it must fail rather than weakening the gate. Fullstateequality remains a separatecoldstate report; stochasticrollout bitwise continuation is not claimed. Candidate is not inserted into the existingGPUqueues and must await GPU and numerical prerequisites.


### 2026-09-16 05:14 UTC: native CPU acceptance passed; continuation supervisor waiting

Independent source snapshot /data_hdd/roll-qwen38-validation/sources/rl-resume-tools-v1/ROLL has1127files, source-manifestSHAca719c6ef14c1b0330a2d413ff65ad90b71cafc77ebb79433d86d0798d6d0953. Native PyTorch2.13CPU98passed4.65s. Actual typedRLVR preflight confirms baselineRUN/checkpoints/20260914-215136/checkpoint-19, savedstep19, restoredscheduler cursor160, firstupdate20,5remainingupdates andfreshHDDcheckpoint24 destination. Updatedvalidator reran historicalv9 successfully:20updates,21versions,58validnonzerogroups. Evidence archived in output/rl-validation-data/native-resume-tools-v1/{results.json,cpu-tests.log,preflight.log,v9-lifecycle.log}. No newGPUupdate occurred.

Added a separatepersistent supervisorPID4018908 at05:13:39UTC running environment/queue_rl_resume_v1.py (SHA427d81b07274dbfae61c675302b939e38c025080e16136a2160d03ef738528cf). Events environment/queued-rl-resume-v1/events.jsonl. It waits for promptv3exit0andshortparitypassed, backbonev3queue_completeandexactstatecomparison, then obtains the sharedGPUlock and waits for allGPUcompute processes torelease. It rerunspreflight, executes5resumedRLupdates andstrictresumeacceptance, failingwithoutGPUretries onanyprerequisite/training/validationfailure. Overallreleasegate remainsfalse. Originalv3/v8queues andsources unchanged.

05:14:39UTC livecheck: oldsupervisors3657791/3657792 andnew4018908 allalive. Oldevents stillwaiting_for_idle_gpus/waiting_for_prompt_diagnostic; neweventwaiting_for_prompt_and_backbone. All8GPUs remain78339-78343MiB and0percentutilization underexistingservice. Respectuserdecision: no service interruption.

Knownmissinggates remain: realpromptinvalidID/rootcause, fullROLLRaycoldlogprobcycle, natural2K/8Knumerics, fullbackboneSFT/RL, exportGPUload/fault/performance, independentfinalreview/vendor sync/FlashNextPR. The new queue is independent resumedLoRARL acceptance, not stochasticreplay or releasecompletion.


### 2026-09-16 06:38 UTC: GPU released; container-stop interruption and fresh v4 queues

The existing evaluation released GPUs and promptv3autostarted06:24:23UTC. It loaded8rankmodelweights but was interrupted by a Docker containerstop at06:26:04UTC (SIGTERM), thenSIGKILL10seconds later. Both ownvalidationcontainers were stopped; DockerOOMKilled=false. Promptv3exit137beforeanyprobabilityrequest; this is not an OverflowError reproduction. Backbonev3CPUpreflight failedbecausecontainerwasstopped; RLresumev1stoppedonpromptfailure. Preserve alloriginalfailureevents.

Root preservedstopstate/events in environment/container-stop-20260916.json, verifiedexactHDDcontaineridentity/sleepentrypoint andGPUidle, thenrestarted onlyroll-qwen38-hdd-validation. User subsequentlyconfirmedGPUavailabilityandcontinuation. Noothercontainerwasstoppedorrestarted.

Preparedfresh environment/prompt-id-v4 (6scripts,manifestSHA4a93f6074769465bbc23e6aecc2c79d8205042809510be563e0fcd08bdb758ac). Probeandworkerextensionarebyte-identicaltov3; newsupervisor/outputpaths only. Sourcecold-v11, finalexportedadapter, tolerancesandalltraining/v8/resume-tools-v1sourcesunchanged. At06:37:34UTCactivatedPID147084promptv4,147085backbonev4,147086RLresumev2. Events underenvironment/prompt-id-queue-v4,queued-backbone-v4,queued-rl-resume-v2. Promptoutputoutput/prompt-id-v4. Remainingbackbone/resumeoutputsunchangedbecauseGPUrunsneverstarted.

At06:38:17UTCpromptengineinitializing; backbonewaiting_for_prompt_diagnostic,RLwaiting_for_prompt_and_backbone. NoGPUtrainingstepclaim.


### 2026-09-16 07:38 UTC: ROLL bounded-loss fix and fresh v9 backbone run

The v4 native prompt diagnostic exited before generation: the installed runner is
`vllm.v1.worker.gpu.model_runner.GPUModelRunner`, whereas the observer targeted the
older `gpu_model_runner._get_prompt_logprobs_dict`. This is an instrumentation
failure, not a reproduction or resolution of the original prompt token overflow.

Backbone v8 started at 06:39:56 UTC and exited at 06:46:26 with CUDA OOM in initial
held-out validation, before any optimizer update. ROLL's global MTP `_postprocess`
patch materialized full vocabulary logits and bypassed the Qwen chunked-loss hook.
The failure requested another 3.79 GiB while only about 3.7 GiB was free.

Added a Qwen-only labeled postprocess branch that calls its bounded vocabulary
loss in training and eval, preserving tied/untied projection weights and keeping
unsupported inference-context handling explicit. Added actual ROLL MTP-patched
GPU model regression for train/eval, TP1/TP2, and tied/untied weights. RED2 observed
320 projected rows despite chunk size 256. Both `.181` ranks then passed the full
20-test integration suite (155 seconds on the reported rank). Evidence:
`output/cold-restore-memory/backbone-v9-{red2,green}.log`. Existing v8 sources remain
unchanged. The first test-observer attempt was invalid because a global dispatch
mode interfered with FlexAttention; RED2 scopes observation to postprocess.

Frozen source `/data_hdd/roll-qwen38-validation/sources/sft-backbone-v9/ROLL` has
1123 manifest entries, SHA256
`971295c65d05d64422968508d5842af073a6c8be67c649d56ca209092e11d22e`.
Supervisor PID 341035 uses `environment/queue_backbone_v5.py`, with events/logs in
`environment/queued-backbone-v5`. CPU preflight passed and the full run started
07:31:23 UTC. At 07:38:18 UTC all 64 initial validation batches had completed in
101.93 seconds; no successful optimizer update was yet confirmed. It retains the
100-step / checkpoint-50 continuation / complete state-comparison acceptance.
HDD outputs are `sft-backbone-100step-8k-v9`, `sft-backbone-100step-8k-resume-v9`, and
`backbone-continuation-exact-v9.json`.

Prompt v5 instruments native V2 target-ID, top-k, worker-result, async CPU-result,
and frontend decode boundaries without changing arithmetic, values, or error
propagation. Tensor observations synchronize CUDA and are diagnostic only.
Actual V2 class installation and a five-token native GPU Triton fixture passed;
this is not full-model probability parity. Files live in
`output/prompt-logprobs-debug/v5`, remote `environment/prompt-id-v5`.

New queued supervisors at 07:35:40 UTC: PID 381478 waits for the entire backbone
queue to terminate before running prompt v5; PID 381479 launches the existing
five-update RL continuation only after prompt v5 parity AND backbone v9 exact
state equality pass. Shared GPU lock and idle checks preserve other workloads.
Queue script manifest SHA256
`25f30fd8f2ec2b7585f4cd69daf71392b9981ea2f3b8e2df6655805d5949450c`.
No failed one-shot queue was restarted. Overall release remains incomplete.


### 2026-09-16 08:25 UTC: first-gradient NCCL memory fix; v10 running

Backbone v9 completed all 64 initial validation batches, then failed the first
backward gradient reduce-scatter (exit 1 at 07:41:04 UTC). NCCL could not allocate
128 MiB outside PyTorch's allocator. Rank2 recorded 73,353,307,136 bytes allocated,
79,010,201,600 reserved, peak 78,229,739,008; update_successful=false. No successful
backbone update is established by v9. Preserved log:
`output/cold-restore-memory/backbone-v9-nccl-oom.log`.

A two-rank standalone NCCL experiment reproduced OOM with almost all GPU memory
held in unused PyTorch cache; freeing the cache before first reduce-scatter made
both ranks pass and return exact expected values. RED/GREEN logs are
`output/cold-restore-memory/nccl-cache-{red,green}.log`. Added
`FirstGradSyncCacheRelease` and gated its use to CPU-offloaded optimizers with
synchronous gradient reduction. It releases unused cache before the first
successful grad-finalization call, without altering tensors, return values or
exception propagation. Two local helper regressions passed; the actual production
wrapper also passed the two-rank GPU pressure test (`nccl-cache-production-green.log`).
Independent scoped reviews found no blocking defects in either the bounded-loss
integration or the cache-release integration; reports are in the SDD directory.
This does not establish full-model completion or overlapping-gradient support.

Frozen v10 source: `/data_hdd/roll-qwen38-validation/sources/sft-backbone-v10/ROLL`,
1127 manifest entries, SHA256
`7cd4448946c68d83bb6b14b2b065e5ef6d80f932613b806b748115d3993746a2`.
Supervisor PID535887 uses `environment/queue_backbone_v6.py` and
`environment/queued-backbone-v6/{events.jsonl,backbone-100step.log}`.
Preflight passed; v10 started 08:25:00 UTC. Output suffixes are v10, and the queue
retains full 100 steps, checkpoint50 -> 49-update continuation, complete comparison.
No active/old snapshot or shared dependency was modified.

Prompt v5 ran after v9 exited and finished all 9 requests (base, exported final
adapter, post-sleep adapter; en/zh/code). No invalid token IDs were observed.
Native worker, async CPU, frontend and selected-token outputs agree exactly;
post-sleep probabilities are bitwise identical. The instrumented run synchronizes
CUDA, so the original uninstrumented overflow remains unresolved. Adapter-vs-actor
short parity FAILED: en max_abs4.618779/mean_abs0.494292, zh0.853354/0.098915,
code2.329251/0.219183, unchanged atol/rtol0.02. Prompt exit1 was a numerical mismatch,
not an observer/API failure. New RL queue v3 stopped on failed prerequisites;
no RL updates were launched.

Local analysis corrected provenance: frozen cold-v11 actor config is TP1/EP8,
not TP2, while native vLLM uses TP8/EP8. The actor's eight rows are repeated input
samples returned by DP_MP_DISPATCH_FIRST, not eight TP-rank dumps. The384-token code
case contains no answer token (its prompt alone is394tokens). Existing short-probe
artifacts therefore must not be described as answer-token or rankwise acceptance.
See `output/prompt-logprobs-debug/v5/parity-analysis.md`. Next numerical control is
restored-actor adapter-enabled/disabled on identical inputs; no export defect or
scale correction has been established. RL tool tests reran locally:98passed2.57s.


### 2026-09-16: v10 first backbone update; TP4 candidate

Archived the v10 queue and all eight rank metrics in
`output/cold-restore-memory/backbone-v10-evidence.tar.gz`; the derived update
receipt is `backbone-v10-update-summary.json`. Every rank reports step0
`update_successful=true`; rank6 reports step1=false. The run ended at08:35:39UTC
on a632MiB MoE permutation allocation with617MiB free. It did not complete
100updates or produce continuation acceptance. This is the first successful
full-text-backbone update, with the N-gram table frozen, not a complete SFT pass.

TP4/EP8/ETP1 with accumulation2 preserves global batch4 while reducing dense
parameter/gradient replication per GPU. The existing v11 candidate is not yet
frozen. Its original eight-rank regression exposed a QSA output gate left
unsharded when KV groups are fewer than TP ranks. A direct global-projection
regression also failed on all ranks: gate heads8 versus local query heads2.
`backbone-v11-tp4-gate-red.log` preserves that failure. The QSA override now uses
the same within-replicated-group TP rank to slice the gate, and preserves gates
already sliced by newer Megatron versions. Eight-rank value/gradient and full
model loss regressions are running; no TP4 training acceptance is claimed.

A separate actor-only paired adapter diagnostic is being prepared locally for
enabled-before -> disabled -> enabled-after on the same checkpoint19 and token
inputs. It must preserve zero updates and no checkpoint writes. Prompt-v5 parity
remains failed, and all final release gates remain open.


### 2026-09-16 10:27 UTC: TP4 distributed regressions and preflight passed

The actual eight-rank TP4/EP8 run exited0:9tests passed on each rank. Gate values
and QKV gradients match an independently assembled global projection exactly for
KV group counts1/2/4; ROLL-patched bounded loss and full DDP gradient comparisons
also pass (tied/untied, train/eval). Evidence:
`output/cold-restore-memory/backbone-v11-tp4-green.log` and
`backbone-v11-regression-receipt.json`. These are tiny-model GPU regressions,
not full125.74B training acceptance.

Frozen v11 has1129source hashes; manifestSHA256
`9af5dc75156db2b751e993c939084254a826bb06ab1e44016d958a40591130ca`.
CPU preflight verifies all hashes, TP4/DP2/EP8/ETP1 and frozen external N-gram
scope. The production candidate SFT config now usesTP4 withaccumulation2,
preserving globalbatch4. Full100steps/checkpoint50 continuation remain pending.

Actor paired diagnostic has8local CPU harness passes. First real CPU preflight
failed because BaseConfig replaces configured GPU counts with visible counts
while CUDA is hidden; the diagnostic now keeps requested topology checks in CPU
preflight and requires actual8GPU/one-node discovery for execution. The first
source/output remain preserved; a new actor-pair-v2 source is copied from all1114
verified cold-v11 files plus the isolated probe. No optimizer updates or model
acceptance are implied by these harness/preflight checks.


### 2026-09-16 10:49 UTC: actor diagnostic running, v11 queued

Actor source `/data_hdd/roll-qwen38-validation/sources/actor-pair-v3/ROLL` is
verified against1114cold-v11 hashes plus the isolated diagnostic. The original
checkpoint19 and old sources remain unchanged. Native CPU preflight passed;
local diagnostic controls now9passed after preserving nonfinite historical
tracker fields explicitly in valid JSON. Newly measured probabilities still
require finite values, and before/after adapter probabilities require exact equality.

Persistent actor supervisorPID1008638 uses `environment/actor-pair-v3/queue_remote.py`
with events/logs in`environment/queued-actor-pair-v1`; output is
`output/actor-pair-v1-20260916-run03`. Backbone supervisorPID1008639 uses
`environment/actor-pair-v3/queue_backbone_v7.py` and waits for the diagnostic to
return, then for idle GPUs under the shared lock. Its v11 sequence retains
100updates, checkpoint50 ->49updates, and full checkpoint state comparison.
The actor diagnostic result is not a backbone or release acceptance gate; all
previous numerical failure evidence is preserved. No new successful update
from v11 is yet established.


### 2026-09-16 11:01 UTC: backbone initial validation passed; actor retry queued

Backbone v11 began at10:53:21UTC and completed64initial validation batches in
178.2232s. It is now executing the first training update; no successful v11
update is yet established. Live GPU usage is about72,896–73,532MiB.

Actor run03 returned1at10:53:06UTC before restore/probability collection: its
mutation guard read `ChainedOptimizer.optimizer`, which asserts for multiple
children. This was a diagnostic-helper error, not a measured model mismatch.
The guard now traverses `chained_optimizers` without reading that property.
Native CPU check on the installed Megatron blocks root plus both children and
keeps CUDA uninitialized; local control tests9passed. Frozen v4 diagnostic
source uses the same1114verified cold-v11 source files plus the fixed probe.
Its CPU preflight passed11:00:52UTC. SupervisorPID1099042 waits for the entire
backbone-v7 queue to return, then obtains the shared GPU lock. Events are
`environment/queued-actor-pair-v2/events.jsonl`; source/launcher are
`sources/actor-pair-v4/ROLL` and`environment/actor-pair-v4/run_remote.sh`; output
is`output/actor-pair-v1-20260916-run04`. No existing source/run was overwritten.

QSA activation arithmetic was also checked against the pinned HF source and
native vLLM source: both multiply by BF16 gate.sigmoid. The TP4 fix deliberately
preserves that operation; generic Megatron's separate FP32 gate implementation
is not evidence for changing this model's arithmetic.


### 2026-09-16 11:19 UTC: five globally completed TP4 updates

Live check confirms all8ranks completed steps0–4 successfully; step5 is in
progress (four ranks already reported success). No rank has recorded a failed
update. Cumulative PyTorch CUDA peak is72,576,902,656B per rank(67.5925GiB).
Last globally completed step4 took159.181s; loss1.6293753833. All inputs have
shape[2,8192] per worker, but step4's maximum valid sequence is595tokens.
Padded8K execution is not evidence of completion on full8K valid text.
The100-update, checkpoint50 continuation and complete comparison gates remain open.
Receipt:`output/cold-restore-memory/live-20260916-111901.json`.

Supervisor1008639 continues the full backbone-v7 queue; supervisor1099042 waits
for that entire queue to finish before running the repaired actor-pair-v4
diagnostic under the shared lock. No actor-pair probabilities have yet been
measured. The source of the previous diagnostic failure and the native CPU guard
regression are archived under`output/prompt-logprobs-debug/actor-pair-v1/`.

A fresh static full-backbone RL capacity follow-up uses the current eight-rank
parameter census:270.4568GB BF16 role storage, versus408.8816GB observed host
headroom. Actor+reference private CPU copies do not fit that SFT headroom; this
is an estimate, not a measured RL OOM. See
`output/cold-restore-memory/backbone-rl-capacity-followup.md`. Sleeplevel2 also
requires explicitly proving immutable native N-gram table retention/restoration.
No RL memory policy was changed, and full-backbone RL is not launch-ready.
Numerical parity, long-text/full SFT, real RL continuation, backbone RL, export,
fault/performance, vendor synchronization and Flash-Next PR remain incomplete.
