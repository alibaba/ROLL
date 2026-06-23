# Ascend NPU FAQ

Last updated: 04/27/2026.

This document compiles common issues encountered when running ROLL on Huawei Ascend NPU and their solutions.

## Docker & Environment

### NPU Not Visible Inside Container

**Symptom:** `npu-smi info` returns no devices or an error inside the container.

**Solution:** Ensure all required devices and driver paths are mounted correctly. Check the following:

1. All `--device /dev/davinciX` entries are present in the `docker run` command.
2. Management devices (`/dev/davinci_manager`, `/dev/devmm_svm`, `/dev/hisi_hdc`) are mounted.
3. Host driver paths are mounted: `/usr/local/Ascend/driver`, `/usr/local/Ascend/add-ons`, `/usr/local/dcmi`.
4. The host Ascend NPU driver is installed and `npu-smi info` works on the host.

### vLLM-Ascend Import Error

**Symptom:** `import vllm_ascend` fails or vLLM cannot detect NPU devices.

**Solution:** Verify that the CANN environment is properly sourced:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

These commands are automatically added to `/root/.bashrc` during the Docker image build. If you switch to a non-root user, you may need to source them manually.

### torch_npu Not Working

**Symptom:** `torch.npu.is_available()` returns `False`, or NPU tensors cannot be created.

**Solution:**

1. Verify `torch_npu` is installed: `pip show torch_npu`
2. Check CANN environment: `echo $ASCEND_HOME_PATH`
3. Source the CANN environment if not already done:
   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```
4. Verify NPU visibility: `npu-smi info`
5. Check if `torch` and `torch_npu` versions match: `pip list | grep torch`

### SOC Version Mismatch

**Symptom:** Errors like `SOC_VERSION not supported` or `Ascend device not found` during vLLM-Ascend installation or runtime.

**Solution:** Make sure you are using the correct pre-built image for your hardware:

- **Atlas 900 A2 PODc** → Use `roll:ascend-a2` (`ascend910b1`)
- **Atlas 900 A3 PODc** → Use `roll:ascend-a3` (`ascend910_9391`)

The current repository includes `docker/Dockerfile.A2` and `docker/Dockerfile.A3` for building custom images. If you maintain a custom image, ensure its SOC version matches the target hardware.

### Disable FRACTAL_NZ Mode

**Symptom:** Enabling NZ optimization mode during reinforcement learning is likely to cause precision issues. vLLM-Ascend includes a check for this, and if NZ mode is enabled, it may raise the following error: `ValueError: FRACTAL_NZ mode is enabled. This may cause model parameter precision issues in the RL scenarios.`

**Solution:** Before running the startup script, add the following environment variable to disable NZ mode:

```bash
export VLLM_ASCEND_ENABLE_NZ=0
```

### HCCL Parameter Plane Port Binding Failure

**Symptom:** When the current rank or process establishes a communication operator on the parameter plane, binding the device-side NIC port fails because the port is already occupied. The error may look like: `The IP address XXXX and port XXXX have already been bound`.

**Solution:**

1. HCCL uses the device-side NIC port and binds to port 16666 by default. Therefore, if multiple processes run on the same device and all call HCCL communication operator APIs, the port may already be bound by another process, causing the failure.
2. First check whether running multiple processes on the same device is expected for your workload. If it is expected, enable multi-process scenarios by configuring the `HCCL_NPU_SOCKET_PORT_RANGE` environment variable, for example:
   ```bash
   export HCCL_NPU_SOCKET_PORT_RANGE="auto"
   ```

## Dependency Conflicts

### triton Import Error

**Symptom:** `import triton` fails or conflicts with `triton-ascend`.

**Solution:** The pre-built Ascend images use `triton-ascend` instead of the standard `triton` package. If you accidentally installed the wrong triton package, fix it with:

```bash
pip uninstall -y triton triton-ascend
pip install triton-ascend==3.2.1 --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

## Training Configuration

### Megatron Strategy Not Supported

**Symptom:** Errors when using `strategy: megatron` in configuration on NPU.

**Solution:** Megatron-LM is not supported on Ascend NPU. Use FSDP2 as the training backend:

```yaml
strategy_args:
  strategy_name: fsdp2_train
```

### HCCL Communication Timeout or Failure

**Symptom:** During multi-NPU distributed training, errors such as `Hccl execute failed`, `LINK_ERROR_INFO`, `EI0006` link establishment timeout, or HCCL initialization failure appear. Single-card training works fine, but multi-card or multi-node training fails.

**Solution:** Follow these steps to troubleshoot:

1. **Check NPU inter-card link status**:
   ```bash
   for i in {0..7}; do hccn_tool -i $i -link -g; done
   ```
   The output should be `up`. If any other status is shown, the link is abnormal. Try resetting the affected card:
   ```bash
   npu-smi set -t reset -i <RankId> -c 0 -m 1
   ```

2. **Check NPU card IP configuration**:
   ```bash
   for i in {0..7}; do hccn_tool -i $i -ip -g; done
   ```
   Ensure all cards have IP addresses configured and there are no IP conflicts.

3. **Check TLS configuration consistency across nodes**:
   ```bash
   for i in {0..7}; do hccn_tool -i $i -tls -g; done | grep switch
   ```
   The TLS switch status must be consistent across all cards. It is recommended to disable TLS uniformly:
   ```bash
   for i in {0..7}; do hccn_tool -i $i -tls -s enable 0; done
   ```

4. **Increase HCCL link establishment timeout** (default is 120 seconds, which may be insufficient for large model scenarios):
   ```bash
   export HCCL_CONNECT_TIMEOUT=3600
   ```

5. **Check cross-node network connectivity**:
   ```bash
   # On node B, ping node A's device IP
   hccn_tool -i 0 -ping -g address <peer_node_IP>
   ```
   If the ping fails, check firewall settings, subnet masks, and switch VLAN configurations.

6. **Disable firewall** (for multi-node training scenarios):
   ```bash
   sudo systemctl stop firewalld
   sudo systemctl disable firewalld
   ```

## Ray Cluster & Multi-Node

### Ray Cluster Nodes Not Joining

**Symptom:** Worker nodes fail to join the Ray cluster. The head node logs show `N nodes have joined so far, waiting for X` indefinitely, and worker nodes show connection errors.

**Solution:**

1. **Verify network connectivity between nodes:**
   ```bash
   ping <HEAD_IP>
   ```

2. **Check that MASTER_PORT is open on the head node:**
   ```bash
   # On the head node, verify the port is listening
   ss -tlnp | grep 6379
   
   # On a worker node, test connectivity
   nc -zv <HEAD_IP> 6379
   ```

3. **Ensure firewall is disabled or ports are open on all nodes:**
   ```bash
   sudo systemctl stop firewalld
   sudo systemctl disable firewalld
   ```
   
   Required ports:
   - `MASTER_PORT` (default 6379): Ray cluster communication
   - `DASHBOARD_PORT` (default 8265): Ray dashboard
   - `HCCL_IF_BASE_PORT` (default 23456): HCCL cross-node communication
   - A range of ports above `MASTER_PORT` for Ray internal services (typically 10002-19999)

4. **Verify RANK, WORLD_SIZE, and MASTER_ADDR are set correctly:**
   ```bash
   echo "RANK=$RANK WORLD_SIZE=$WORLD_SIZE MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
   ```

5. **Check firewall rules on the head node** — ensure inbound connections to the Ray ports are allowed from worker node IPs.

### Worker Nodes Exit Immediately

**Symptom:** Worker nodes start, join the Ray cluster, then exit immediately without running any training.

**Solution:** This is expected behavior. In ROLL's auto-launch mode, worker nodes (`RANK>0`) automatically call `sys.exit(0)` after the Ray cluster is initialized. Only the head node (`RANK=0`) executes the training pipeline. The worker nodes' Ray processes remain running and serve the training workload. Check `ray status` on the head node to confirm workers are active.

### Cross-Node NPU Communication Timeout

**Symptom:** Training is fine single-node but fails with HCCL errors when going multi-node, even though `hccn_tool -ping` works.

**Solution:**

1. **Verify HCCL_SOCKET_IFNAME is correct and consistent:**
   ```bash
   # Check which interface the NPU device IPs are on
   ip route get <npu_device_ip>
   ```
   The interface name must be the same across all nodes.

2. **Verify HCCL_IF_BASE_PORT is not blocked by firewall** between nodes.

3. **Check if switch/router allows HCCL traffic.** HCCL uses RoCEv2 (RDMA over Converged Ethernet). Ensure the switch is configured to pass PFC (Priority Flow Control) and ECN (Explicit Congestion Notification) traffic.

4. **Increase HCCL timeouts further:**
   ```bash
   export HCCL_CONNECT_TIMEOUT=7200
   export HCCL_EXEC_TIMEOUT=7200
   ```

### Shared Storage Not Accessible

**Symptom:** Training fails because model weights or data files cannot be found on worker nodes.

**Solution:** All nodes must have access to the same files at the same paths. Mount a shared filesystem:

```bash
# Example: Mount NFS inside each container
mount -t nfs <nfs_server>:/roll /shared/storage

# Or mount at container start:
docker run ... \
    -v /shared/storage:/data \
    ...
```

Ensure the shared storage has sufficient bandwidth for loading model weights (several GB per load operation).

## Resource & Performance

### ulimit Too Low

**Symptom:** Errors like `OSError: [Errno 24] Too many open files`, `RuntimeError: Unable to open file`, or Ray worker processes crashing unexpectedly during multi-NPU training.

**Solution:** The default `ulimit` (open file descriptor limit) in Docker containers is typically 1024, which is insufficient for multi-NPU distributed training. Add `--ulimit nofile=65536:65536` to your `docker run` command to increase the limit:

Or set it inside the container at runtime:

```bash
ulimit -n 65536
```

To make it persistent, add the following line to `/etc/security/limits.conf` inside the container:

```
* soft nofile 65536
* hard nofile 65536
```

### Out of NPU Memory

**Symptom:** Training or inference crashes with OOM (Out of Memory) errors.

**Solution:**

1. Reduce `rollout_batch_size` or `num_return_sequences_in_group` in your configuration file.
2. Reduce `per_device_train_batch_size` and increase `gradient_accumulation_steps` accordingly.
3. Enable FSDP2 with CPU offloading in your config:
   ```yaml
   strategy_args:
     strategy_name: fsdp2_train
     strategy_config:
       offload_policy: true
   ```
4. Use a smaller model or apply LoRA to reduce memory footprint.

### Slow vLLM Inference on NPU

**Symptom:** vLLM inference throughput is significantly lower than expected.

**Solution:**

1. Ensure CANN and vLLM-Ascend versions are compatible (current images use CANN 9.0.0 with vLLM-Ascend v0.18).
2. Check that the SOC version matches your hardware.
3. Adjust vLLM parameters such as `gpu_memory_utilization` and `max_model_len` in your config.
4. Verify that `triton-ascend` is installed (not `triton`), as the wrong triton backend can cause kernel compilation fallbacks.

## Disclaimer

The Ascend support provided in ROLL is intended as a reference example. For production use, please consult official channels.
