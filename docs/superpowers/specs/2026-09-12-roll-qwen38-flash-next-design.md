# ROLL 支持 Qwen3.8-Flash-Next 的设计与验收

日期：2026-09-12。状态：供审阅的设计；尚未实现本次完整模型适配，尚未通过目标模型 SFT/RL 验收。

## 1. 目标与当前边界

用户要求研究模型架构、改进 ROLL 及依赖，在 172.16.120.181 上使用 `/data_hdd/Qwen3.8-Flash-Next` 充分验证高效 SFT/RL。前一项 27B 修复已有独立 PR #497，本项另开开发分支，不能把其结果记作 Flash-Next 验证。

第一交付面是文本 SFT 和 GRPO/RLVR；保留原始 vision/MTP 权重及配置，导出时明确其未训练，默认不启用 MTP speculative decoding。视觉后训练、MTP 联合训练和分布式 CP 不在第一验收面内，不能宣传为已支持。不是将模型改名为 Qwen3-Next 或 Qwen3.5，也不是忽略加载失败的张量。

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
| QSA core | Q/KV heads=24/2，head_dim=256，输出 gate | Q 和 gate 按每个头交错；训练 TP=2 为首选，TP>2 需另外验证 KV 复制及梯度归并 |
| QSA indexer | Q/KV heads=4/1，head_dim=128，block=4，budget=2048 | 选最多 512 个完整因果块，再加未满块的尾部；池化 raw K 后才 norm/RoPE |
| MoE | 每层 512 experts，top-10，FFN=640，共享 expert=640 | BF16 checkpoint 专家是 stacked 格式，保持 gate/up 及 EP 全局编号 |
| GR | 4 路残差，宽 10240，低秩 320 | 分组 zero-centered RMSNorm；SiLU(down/4)；sigmoid read 后均值；write=2*sigmoid(inject/4) |
| N-gram | bigram/trigram，16 个哈希头，head_dim=160，128 个存储分片 | 精确保留 int64 hash、槽位 offset、EOS 语义；不得根据配置近似重建不同词表 |
| PLE | 第 2 层前置，key 投影到 10240，value 投影到 2560 | 分支 gate、signed-sqrt、归一化 gated value，再做 dilation=3 的 depthwise conv，最后相加 |
| 输出 | final GR mixer 收窄到 H，再 norm/head | 不能多一次普通残差或 norm |

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

主干训练首选候选布局 TP=2、EP=8、ETP=1、PP=1、CP=1、SP=true，8 个 rank；dense DP=4。必须由真实 mcore process groups 验证 EP 与 DP×TP 的约束。不能把 TP×EP 简单当作物理 GPU 数。rollout 用 TP=8，训练到推理的重分片独立验收。

这不是已验证配置：只算专家 BF16 权重，每卡已需约 28.125 GiB，再加 dense 权重约 4.608 GiB。若所有 FP32 梯度同时保留，单卡会超 80 GiB；需要分桶梯度 D2H、CPU 端 FP32 累积或其它经过数值验证的状态布局。不能靠只 offload Adam moments 解决全部峰值。

转发并验证 Megatron CPU optimizer 配置，记录实际 optimizer 类型、参数/状态设备和大小。将 GPU Adam 的 phase offload 与真正 CPU optimizer 分成明确分支；CPU 分支不能在 step 前无条件将 Adam 状态拉回 GPU。处理 SFT metadata、RL phase 切换、梯度累积、global clipping、overflow/skip 与 checkpoint state。

容量预检和运行时采样应覆盖模型初始化、optimizer 首步懒初始化、backward、step、export、rollout warmup、save/resume。规划至少 128 GiB 主存余量和每卡约 8 GiB 未占用空间；达不到则调小 microbatch/采用 A 模式，不自动改写用户指定的训练范围。

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
