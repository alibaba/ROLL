# 昇腾 NPU 常见问题

最后更新：2026/04/27。

本文档汇总了在华为昇腾 NPU 上运行 ROLL 时可能遇到的常见问题及解决方案。

## Docker 与环境

### 容器内 NPU 不可见

**现象：** 容器内执行 `npu-smi info` 无设备返回或报错。

**解决方案：** 确保所有必需的设备和管理路径已正确挂载，检查以下项：

1. `docker run` 命令中包含所有 `--device /dev/davinciX` 条目。
2. 管理设备（`/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc`）已挂载。
3. 宿主机驱动路径已挂载：`/usr/local/Ascend/driver`、`/usr/local/Ascend/add-ons`、`/usr/local/dcmi`。
4. 宿主机上已安装昇腾 NPU 驱动，且宿主机上 `npu-smi info` 可正常工作。

### vLLM-Ascend 导入错误

**现象：** `import vllm_ascend` 失败，或 vLLM 无法检测到 NPU 设备。

**解决方案：** 验证 CANN 环境是否正确加载：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

这些命令在 Docker 镜像构建时已自动添加到 `/root/.bashrc`。如果切换到非 root 用户，可能需要手动执行。

### torch_npu 无法使用

**现象：** `torch.npu.is_available()` 返回 `False`，或无法创建 NPU 张量。

**解决方案：**

1. 确认 `torch_npu` 已安装：`pip show torch_npu`
2. 检查 CANN 环境：`echo $ASCEND_HOME_PATH`
3. 如未加载 CANN 环境，手动执行：
   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```
4. 验证 NPU 可见性：`npu-smi info`
5. 查询`torch`和`torch_npu`版本是否匹配：`pip list | grep torch`

### SOC 版本不匹配

**现象：** 安装或运行 vLLM-Ascend 时出现 `SOC_VERSION not supported` 或 `Ascend device not found` 等错误。

**解决方案：** 确保使用了与硬件匹配的预构建镜像：

- **Atlas 900 A2 PODc** → 使用 `roll:ascend-a2`（`ascend910b1`）
- **Atlas 900 A3 PODc** → 使用 `roll:ascend-a3`（`ascend910_9391`）

当前仓库包含用于构建自定义镜像的 `docker/Dockerfile.A2` 和 `docker/Dockerfile.A3`。如果维护自定义镜像，请确保 SOC 版本与目标硬件匹配。

### 禁用 FRACTAL_NZ模式

**现象：** 在强化学习中开启NZ优化模式很有可能导致精度问题，vllm_ascend中存在该校验，若开启会出现 `ValueError: FRACTAL_NZ mode is enabled. This may cause model parameter precision issues in the RL scenarios.`错误

**解决方案：** 启动脚本前，添加环境变量，禁用NZ：
   ```bash
   export VLLM_ASCEND_ENABLE_NZ=0
   ```

### HCCL参数面端口绑定失败

**现象：** 当前rank或进程在通信算子参数面建链时绑定device侧网卡端口失败，端口被占用，出现 `The IP address XXXX and port XXXX have already been bound`错误

**解决方案：** 

1. HCCL使用device侧网卡的端口时默认需绑定16666端口，因此若有多个进程执行在同一个device上，且均会调用HCCL的通信算子接口，那么就会出现端口已被其他进程绑定导致失败的问题。
2. 此时可先从业务上排查多个进程跑在同一个device上是否符合任务预期，若符合任务预期结果，可通过配置HCCL_NPU_SOCKET_PORT_RANGE环境变量使能多进程场景，如：
   ```bash
   export HCCL_NPU_SOCKET_PORT_RANGE="auto"
   ```


## 依赖冲突

### triton 导入错误

**现象：** `import triton` 失败，或与 `triton-ascend` 冲突。

**解决方案：** 预构建昇腾镜像使用 `triton-ascend`，不使用标准 `triton` 包。如果误装了错误的 triton 包，请执行：

```bash
pip uninstall -y triton triton-ascend
pip install triton-ascend==3.2.1 --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

## 训练配置

### 不支持 Megatron 策略

**现象：** 在 NPU 上使用 `strategy: megatron` 配置时报错。

**解决方案：** 昇腾 NPU 上不支持 Megatron-LM 训练，请使用 FSDP2 作为训练后端：

```yaml
strategy_args:
  strategy_name: fsdp2_train
```

### HCCL 通信超时或失败

**现象：** 多 NPU 分布式训练时出现 `Hccl execute failed`、`LINK_ERROR_INFO`、`EI0006` 建链超时，或 HCCL 初始化失败等错误。单卡训练正常，多卡或多机训练报错。

**解决方案：** 按以下步骤逐一排查：

1. **检查 NPU 卡间链路状态**：
   ```bash
   for i in {0..7}; do hccn_tool -i $i -link -g; done
   ```
   输出应为 `up`，若为其他状态说明链路异常，可尝试重置异常卡：
   ```bash
   npu-smi set -t reset -i <RankId> -c 0 -m 1
   ```

2. **检查 NPU 卡 IP 配置**：
   ```bash
   for i in {0..7}; do hccn_tool -i $i -ip -g; done
   ```
   确保各卡 IP 已配置且无冲突。

3. **检查多节点 TLS 配置一致性**：
   ```bash
   for i in {0..7}; do hccn_tool -i $i -tls -g; done | grep switch
   ```
   所有卡的 TLS 开关状态必须一致，建议统一关闭：
   ```bash
   for i in {0..7}; do hccn_tool -i $i -tls -s enable 0; done
   ```

4. **增大 HCCL 建链超时时间**（默认 120 秒，大模型场景可能不够）：
   ```bash
   export HCCL_CONNECT_TIMEOUT=3600
   ```

5. **检查跨节点网络连通性**：
   ```bash
   # 在节点 B 上 ping 节点 A 的 device IP
   hccn_tool -i 0 -ping -g address <对端节点IP>
   ```
   若 ping 不通，检查防火墙、子网掩码和交换机 VLAN 配置。

6. **关闭防火墙**（多机训练场景）：
   ```bash
   sudo systemctl stop firewalld
   sudo systemctl disable firewalld
   ```

## Ray 集群与多机

### Ray 集群节点无法加入

**现象：** 工作节点无法加入 Ray 集群。主节点日志持续显示 `N nodes have joined so far, waiting for X`，工作节点显示连接错误。

**解决方案：**

1. **检查节点间网络连通性：**
   ```bash
   ping <HEAD_IP>
   ```

2. **检查主节点 MASTER_PORT 是否开放：**
   ```bash
   # 在主节点上检查端口是否在监听
   ss -tlnp | grep 6379
   
   # 在工作节点上测试连通性
   nc -zv <HEAD_IP> 6379
   ```

3. **在所有节点上关闭防火墙或开放端口：**
   ```bash
   sudo systemctl stop firewalld
   sudo systemctl disable firewalld
   ```
   
   需要开放的端口：
   - `MASTER_PORT`（默认 6379）：Ray 集群通信
   - `DASHBOARD_PORT`（默认 8265）：Ray Dashboard
   - `HCCL_IF_BASE_PORT`（默认 23456）：HCCL 跨节点通信
   - `MASTER_PORT` 以上的一段端口用于 Ray 内部服务（通常 10002-19999）

4. **确认 RANK、WORLD_SIZE、MASTER_ADDR 设置正确：**
   ```bash
   echo "RANK=$RANK WORLD_SIZE=$WORLD_SIZE MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
   ```

5. **检查主节点防火墙规则** — 确保从工作节点 IP 到 Ray 端口的入站连接被允许。

### 工作节点启动后立即退出

**现象：** 工作节点启动、加入 Ray 集群后立即退出，未执行任何训练。

**解决方案：** 这是预期行为。在 ROLL 自动启动模式下，工作节点（`RANK>0`）在 Ray 集群初始化完成后会自动调用 `sys.exit(0)`。仅主节点（`RANK=0`）执行训练流水线。工作节点的 Ray 进程保持运行并为训练任务提供服务。在主节点上执行 `ray status` 确认工作节点处于活动状态。

### 跨节点 NPU 通信超时

**现象：** 单机训练正常，多机时出现 HCCL 错误，即使 `hccn_tool -ping` 正常。

**解决方案：**

1. **确认 HCCL_SOCKET_IFNAME 正确且一致：**
   ```bash
   # 检查 NPU 设备 IP 在哪个网卡上
   ip route get <npu_device_ip>
   ```
   网卡名称必须在所有节点上保持一致。

2. **确认 HCCL_IF_BASE_PORT 未被防火墙拦截。**

3. **检查交换机/路由器是否允许 HCCL 流量。** HCCL 使用 RoCEv2（RDMA over Converged Ethernet）。确保交换机配置了 PFC（优先级流控）和 ECN（显式拥塞通知）。

4. **进一步增大 HCCL 超时时间：**
   ```bash
   export HCCL_CONNECT_TIMEOUT=7200
   export HCCL_EXEC_TIMEOUT=7200
   ```

### 共享存储无法访问

**现象：** 训练失败，因为工作节点找不到模型权重或数据文件。

**解决方案：** 所有节点必须能在相同路径访问相同文件。挂载共享文件系统：

```bash
# 示例：在容器内挂载 NFS
mount -t nfs <nfs_server>:/roll /shared/storage

# 或在容器启动时挂载：
docker run ... \
    -v /shared/storage:/data \
    ...
```

确保共享存储有足够带宽用于加载模型权重（每次加载操作数 GB）。

## 资源与性能

### ulimit 不足

**现象：** 出现 `OSError: [Errno 24] Too many open files`、`RuntimeError: Unable to open file` 等错误，或多 NPU 分布式训练时 Ray worker 进程意外崩溃。

**解决方案：** Docker 容器内默认的 `ulimit`（打开文件描述符上限）通常为 1024，对于多 NPU 分布式训练来说不够。在 `docker run` 命令中添加 `--ulimit nofile=65536:65536` 来提高限制：

或在容器运行时手动设置：

```bash
ulimit -n 65536
```

持久化配置可在容器内的 `/etc/security/limits.conf` 中添加：

```
* soft nofile 65536
* hard nofile 65536
```

### NPU 显存不足

**现象：** 训练或推理过程中出现 OOM（Out of Memory）错误而崩溃。

**解决方案：**

1. 在配置文件中减小 `rollout_batch_size` 或 `num_return_sequences_in_group`。
2. 减小 `per_device_train_batch_size`，同时相应增大 `gradient_accumulation_steps`。
3. 在配置中启用 FSDP2 + CPU Offloading：
   ```yaml
   strategy_args:
     strategy_name: fsdp2_train
     strategy_config:
       offload_policy: true
   ```
4. 使用更小的模型或应用 LoRA 以降低显存占用。

### NPU 上 vLLM 推理速度慢

**现象：** vLLM 推理吞吐量明显低于预期。

**解决方案：**

1. 确保 CANN 和 vLLM-Ascend 版本兼容（当前镜像使用 CANN 9.0.0 与 vLLM-Ascend v0.18）。
2. 检查 SOC 版本是否与硬件匹配。
3. 调整配置中 vLLM 的 `gpu_memory_utilization` 和 `max_model_len` 参数。
4. 确认已安装 `triton-ascend`（而非 `triton`），错误的 triton 后端会导致算子编译回退。

## 声明

ROLL 中提供的 Ascend 支持代码皆为参考样例，生产环境使用请通过官方正式途径沟通。
