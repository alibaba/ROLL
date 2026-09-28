import os
from dataclasses import asdict

import ray
import torch
import transformers
from packaging import version
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from roll.configs.base_config import PPOConfig
from roll.configs.worker_config import is_actor_infer_overlapping_with_any_cluster
from roll.platforms import current_platform
from roll.utils.collective import collective
from roll.utils.logging import get_logger
from roll.utils.network_utils import collect_free_port, get_node_ip
from roll.utils.send_recv_utils import serialize_named_weights
from roll.third_party.fsdp2.qwen3_moe_patch import _iter_convert_fsdps_moe_weights

logger = get_logger()


def _gather_dtensor_full(param_data):
    """Gather full tensor from DTensor shard.

    If shard is on CPU, move it to GPU first so that the all-gather
    collective uses NCCL instead of gloo.
    """
    if not isinstance(param_data, DTensor):
        return param_data
    if param_data.device.type == "cpu" and current_platform.device_type != "cpu":
        param_data = param_data.to(current_platform.current_device())
    return param_data.full_tensor()


def _iter_bucketed_weights(named_params, buffer_size):
    """Yield batches of (name, full_tensor) pairs whose total size stays under buffer_size."""
    waiting_params, waiting_params_size = [], 0
    for name, param in named_params:
        full_tensor_size = param.numel() * param.element_size()
        if waiting_params and waiting_params_size + full_tensor_size > buffer_size:
            yield [(n, _gather_dtensor_full(p.data)) for n, p in waiting_params]
            waiting_params, waiting_params_size = [], 0

        waiting_params_size += full_tensor_size
        waiting_params.append((name, param))

    if waiting_params:
        yield [(n, _gather_dtensor_full(p.data)) for n, p in waiting_params]


def _base_weight_target_name(name: str) -> str:
    """Map a frozen LoRA-injected parameter name to the infer-side name.

    Two train-side conventions exist, distinguished by the PeftModel wrapper
    prefix:
    - PeftModel-wrapped LLM (get_peft_model; names start with
      "base_model.model."): the colocated vLLM worker keeps its own LoRA
      injection, so its base parameters stay peft-style and are keyed WITH
      ".base_layer." (e.g. model.layers.N.self_attn.qkv_proj.base_layer.
      weight). Drop only the wrapper prefix and keep ".base_layer."
      untouched; q/k/v and gate/up are NOT fused here because the worker's
      update_parameter_in_bucket maps unfused members (q_proj/k_proj/v_proj
      -> qkv_proj, gate_proj/up_proj -> gate_up_proj) and concatenates them
      itself. Sending pre-fused qkv_proj names makes its member-rule match
      inside "qkv_proj" and double-apply (qkqkv_proj KeyError).
    - In-place adapter injection (no wrapper prefix, e.g. the Qwen-Image
      diffusion transformer via inject_adapter_in_model): strip
      ".base_layer." so names resolve through vLLM's LoRA wrappers, matching
      QwenImageAdapter.load_transformer_base_weights, which keys its
      params_dict with ".base_layer." stripped.
    """
    if name.startswith("base_model.model."): # for LLM 
        name = name[len("base_model.model."):]
        if name.startswith(("layers.", "embed_tokens.", "norm.")):
            # Bare-inner-model naming; the infer side is the full causal LM.
            name = "model." + name
        return name
    return name.replace(".base_layer.", ".") # for diffusion


def gather_fsdp2_base_weights(model, buffer_size):
    """Gather frozen base weights from a LoRA-injected FSDP2 model (LoRA mode).
    Used to restore the infer-side transformer base after a level-2 sleep.

    Names are mapped by _base_weight_target_name to the receiving infer
    backend's convention (colocated vLLM for PeftModel-wrapped LLMs; the
    Qwen-Image pipeline for in-place-injected diffusion transformers); see
    its docstring for the contract.
    """
    named_params = [
        (_base_weight_target_name(name), param)
        for name, param in model.named_parameters()
        if "lora_" not in name
    ]
    logger.info(
        "gather_fsdp2_base_weights: num_params=%s sample_names=%s",
        len(named_params),
        [n for n, _ in named_params[:3]],
    )
    yield from _iter_bucketed_weights(named_params, buffer_size)


def gather_fsdp2_lora_weights(model, buffer_size, adapter_name="default"):
    """Gather one LoRA adapter's weights from a LoRA-injected FSDP2 model.

    For DiffNFT (adapter_name="ema_lora"), gathers the EMA shadow adapter's
    weights instead of the live "default" adapter so the rollout samples
    under the off-policy EMA weights, matching the paper's design.
    """
    from peft.utils import get_peft_model_state_dict

    lora_state_dict = get_peft_model_state_dict(model, adapter_name=adapter_name)
    named_params = [(name, param) for name, param in lora_state_dict.items()]
    logger.info(
        "gather_fsdp2_lora_weights(adapter_name=%s): num_params=%s, "
        "sample_names=%s, all_contain_lora=%s",
        adapter_name,
        len(named_params),
        [n for n, _ in named_params[:5]],
        all("lora_" in n for n, _ in named_params),
    )
    yield from _iter_bucketed_weights(named_params, buffer_size)


def gather_fsdp2_full_weights(model, buffer_size):
    """Gather all model weights (plain full-parameter model update).

    For FSDP2, we need to get the full tensor from the sharded parameters.
    Also converts FSDP MoE weights from FSDP format to HF/vLLM expected format.
    """
    named_params = [(name, param) for name, param in model.named_parameters()]
    need_convert_moe = (
        version.parse(transformers.__version__) >= version.parse("5.2.0")
        and getattr(model.config, "model_type", "") in ("qwen3_moe", "qwen3_next")
    )
    params_iter = _iter_convert_fsdps_moe_weights(named_params) if need_convert_moe else named_params
    yield from _iter_bucketed_weights(params_iter, buffer_size)


class FSDP2WeightUpdater:
    def __init__(
        self,
        pipeline_config: PPOConfig,
        infer_cluster,
        worker_config,
        model_update_name: str,
        model,
        is_lora,
        adapter_name: str = "default",
    ):
        self.pipeline_config = pipeline_config
        self.worker_config = worker_config
        self.model_update_name = model_update_name
        self.model = model
        self.model_update_infer_workers = infer_cluster.workers
        self._model_update_buffer_size = (
            pipeline_config.model_update_buffer_size_mb * 1024 * 1024
        )  # Convert MB to bytes
        self.is_lora = is_lora
        self.adapter_name = adapter_name
        self.infer_worker_config = infer_cluster.worker_config
        self.infer_cluster = infer_cluster
        self.is_colocated = is_actor_infer_overlapping_with_any_cluster(
            infer_cluster.worker_config, actor_train=worker_config
        )

        # Colocated mode attributes
        self._infer_parallel_cpu_group = None
        self._co_infer_worker = None
        self._buffer_num = None
        self._broadcast_workers = None

        # Separated mode attributes
        self.model_update_group_name = None
        self._model_update_locker = None

        if self.is_colocated:
            self._setup_colocated_model_update()
        else:
            self._setup_separated_model_update()

    def model_update(self):
        if self.is_colocated:
            return self._colocated_model_update()
        return self._separated_model_update()

    def _setup_colocated_model_update(self):
        logger.info(f"RANK {dist.get_rank()} Setup colocated model update")
        infer_worker_devices_num = self.infer_worker_config.num_gpus_per_worker
        train_world_size = dist.get_world_size()

        device_start_diff = min(self.worker_config.device_mapping) - min(self.infer_worker_config.device_mapping)
        device_end_diff = max(self.worker_config.device_mapping) - max(self.infer_worker_config.device_mapping)

        assert device_start_diff % infer_worker_devices_num == 0
        assert device_end_diff % infer_worker_devices_num == 0

        for start_rank in range(0, train_world_size, infer_worker_devices_num):
            end_rank = start_rank + infer_worker_devices_num
            assert end_rank <= train_world_size
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._infer_parallel_cpu_group = new_group
        infer_worker_idx = (dist.get_rank() // infer_worker_devices_num) + (
            device_start_diff // infer_worker_devices_num
        )
        self._co_infer_worker = None
        self._co_infer_worker_rank = None
        if 0 <= infer_worker_idx < len(self.model_update_infer_workers):
            self._co_infer_worker = self.model_update_infer_workers[infer_worker_idx]
            self._co_infer_worker_rank = infer_worker_idx

        # rank0 broadcast to mismatch workers
        if dist.get_rank() == 0 and (device_start_diff > 0 or device_end_diff < 0):
            self._broadcast_workers = []
            if device_start_diff > 0:
                self._broadcast_workers.extend(
                    self.model_update_infer_workers[: device_start_diff // infer_worker_devices_num]
                )
            if device_end_diff < 0:
                self._broadcast_workers.extend(
                    self.model_update_infer_workers[device_end_diff // infer_worker_devices_num :]
                )
            self._setup_broadcast_group()

    def _get_local_visible_gpu_rank(self) -> int:
        """Return the first visible GPU rank from CUDA_VISIBLE_DEVICES.

        In colocated mode (CUDA IPC), the serialized CUDA tensor must be rebuilt
        on the exact same physical GPU as the sender rank used. We use the
        physical GPU id (gpu_rank) to align TP-ranks between train and vLLM.
        """
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not cuda_visible:
            return 0
        return int(cuda_visible.split(",")[0].strip())

    def _get_local_global_gpu_id(self) -> int:
        """Return global GPU id for current train rank based on device_mapping.

        device_mapping uses global ids: global_id = node_rank * gpu_per_node + gpu_rank.
        This is the only stable identifier to align tensors across nodes.
        """
        return int(self.worker_config.device_mapping[dist.get_rank()])

    def _get_co_infer_gpu_rank_order(self) -> list[int] | None:
        """Get per-TP-rank GPU order as seen by the colocated infer worker."""
        if self._co_infer_worker is None:
            return None
        cached = getattr(self, "_co_infer_gpu_rank_order", None)
        if cached is not None:
            return cached
        devices_info = ray.get(self._co_infer_worker.get_devices_info.remote())
        order = [int(d["gpu_rank"]) for d in devices_info]
        setattr(self, "_co_infer_gpu_rank_order", order)
        return order

    def _get_co_infer_global_gpu_id_order(self) -> list[int] | None:
        """Get per-TP-rank global GPU id order for the colocated infer worker.

        vLLM indexes `serialized_named_tensors` by its internal worker rank, which
        follows `resource_placement_groups` order, which in turn follows the
        infer worker's device_mapping slice order.
        """
        if self._co_infer_worker_rank is None:
            return None
        num = self.infer_worker_config.num_gpus_per_worker
        start = int(self._co_infer_worker_rank) * num
        end = start + num
        return [int(x) for x in self.infer_worker_config.device_mapping[start:end]]

    def _setup_separated_model_update(self):
        if dist.get_rank() != 0:
            return

        self._broadcast_workers = self.model_update_infer_workers
        self._setup_broadcast_group()

    def _setup_broadcast_group(self):
        if not self._broadcast_workers:
            return
        self.model_update_group_name = f"{self.model_update_name}_fsdp2"
        num_gpus_per_infer_worker = self.infer_worker_config.num_gpus_per_worker
        infer_device_num = num_gpus_per_infer_worker * len(self._broadcast_workers)
        master_address, master_port = get_node_ip(), collect_free_port()

        refs = [
            infer_worker.setup_collective_group.remote(
                master_address=master_address,
                master_port=master_port,
                group_name=self.model_update_group_name,
                rank_offset=i * num_gpus_per_infer_worker + 1,
                world_size=infer_device_num + 1,
            )
            for i, infer_worker in enumerate(self._broadcast_workers)
        ]
        collective.init_collective_group(
            infer_device_num + 1,
            0,
            group_name=self.model_update_group_name,
            master_addr=master_address,
            master_port=master_port,
        )
        ray.get(refs)

        logger.info(f"Init weights update group {self.model_update_group_name}")

    def _colocated_model_update(self):
        if self.is_lora:
            # LoRA mode sends two streams: the frozen transformer base first
            # (so the infer side can restore it after a level-2 sleep without
            # disk reads), then the adapter weights consumed by custom_add_lora.
            self._colocated_stream_weights(
                gather_fsdp2_base_weights(self.model, buffer_size=self._model_update_buffer_size),
                is_lora=False,
            )
            self._colocated_stream_weights(
                gather_fsdp2_lora_weights(
                    self.model, buffer_size=self._model_update_buffer_size, adapter_name=self.adapter_name
                ),
                is_lora=True,
            )
        else:
            self._colocated_stream_weights(
                gather_fsdp2_full_weights(self.model, buffer_size=self._model_update_buffer_size),
                is_lora=False,
            )
        self._add_lora_to_infer_workers()
        torch.cuda.empty_cache()
        return {}

    def _colocated_stream_weights(self, named_weights_iter, is_lora: bool):
        refs = []
        infer_parallel_size = dist.get_world_size(self._infer_parallel_cpu_group)
        co_infer_rank = dist.get_rank(self._infer_parallel_cpu_group)
        for named_weights in named_weights_iter:
            if self._co_infer_worker is not None:
                serialized_tensors = serialize_named_weights(
                    named_weights, infer_strategy=self.infer_worker_config.strategy_args.strategy_name
                )
                send_global_gpu_id = self._get_local_global_gpu_id()
                send_obj = {"global_gpu_id": send_global_gpu_id, "payload": serialized_tensors}
                infer_parallel_tensors = [serialized_tensors]  # tensors for each infer parallel rank
                if infer_parallel_size > 1:
                    infer_parallel_tensors = [None] * infer_parallel_size if co_infer_rank == 0 else None
                    global_dst_rank = dist.get_global_rank(self._infer_parallel_cpu_group, 0)
                    dist.gather_object(
                        serialized_tensors, infer_parallel_tensors, global_dst_rank, group=self._infer_parallel_cpu_group
                    )
            if refs:
                ray.get(refs)
                refs = []
            if co_infer_rank == 0 and self._co_infer_worker is not None:
                # Align gathered per-train-rank payloads with vLLM TP-rank GPU order.
                if infer_parallel_size > 1:
                    assert isinstance(infer_parallel_tensors, list)
                    infer_global_gpu_id_order = self._get_co_infer_global_gpu_id_order()
                    if infer_global_gpu_id_order is not None and len(infer_global_gpu_id_order) == infer_parallel_size:
                        global_id_to_idx = {gid: i for i, gid in enumerate(infer_global_gpu_id_order)}
                        reordered = [None] * infer_parallel_size
                        extras = []
                        for item in infer_parallel_tensors:
                            if not isinstance(item, dict) or "global_gpu_id" not in item or "payload" not in item:
                                # Backward compatibility: old format was the raw payload.
                                extras.append(item)
                                continue
                            idx = global_id_to_idx.get(int(item["global_gpu_id"]))
                            if idx is None:
                                extras.append(item)
                                continue
                            reordered[idx] = item["payload"]
                        # Fill holes with any extras to avoid hard crash; vLLM side will still
                        # error if GPU mismatch, but this gives best-effort compatibility.
                        for i in range(infer_parallel_size):
                            if reordered[i] is None and extras:
                                extra = extras.pop(0)
                                reordered[i] = (
                                    extra["payload"] if isinstance(extra, dict) and "payload" in extra else extra
                                )
                        if any(x is None for x in reordered):
                            missing = [i for i, x in enumerate(reordered) if x is None]
                            raise RuntimeError(
                                "FSDP2 colocated model update failed to align TP-ranks to GPUs. "
                                f"Missing indices={missing}, infer_global_gpu_id_order={infer_global_gpu_id_order}, "
                                f"gathered={infer_parallel_tensors}"
                            )
                        infer_parallel_tensors = reordered
                    else:
                        infer_parallel_tensors = [
                            (x["payload"] if isinstance(x, dict) and "payload" in x else x)
                            for x in infer_parallel_tensors
                        ]
                else:
                    infer_parallel_tensors = [serialized_tensors]
                refs.append(
                    self._co_infer_worker.update_parameter_in_bucket.remote(
                        infer_parallel_tensors, is_lora=is_lora
                    )
                )
            if self._broadcast_workers:
                refs.extend(self._broadcast_to_infer_workers(named_weights, is_lora=is_lora))
        if refs:
            ray.get(refs)

    def _broadcast_to_infer_workers(self, named_weights, is_lora: bool | None = None) -> list[ray.ObjectRef]:
        if not self._broadcast_workers:
            return []
        if is_lora is None:
            is_lora = self.is_lora
        refs = [
            worker.broadcast_parameter.remote(
                group_name=self.model_update_group_name,
                names=[n for n, _ in named_weights],
                dtypes=[w.dtype for _, w in named_weights],
                shapes=[w.shape for _, w in named_weights],
                is_lora=is_lora,
            )
            for worker in self._broadcast_workers
        ]
        handles = []
        # Keep references to tensors moved to device to prevent premature deallocation
        device_tensors = []

        for _, weight in named_weights:
            # Ensure weight is on the correct device (e.g. NPU) if using HCCL/NCCL
            if weight.device.type == "cpu" and current_platform.device_type != "cpu":
                weight_device = weight.to(current_platform.device_type)
                device_tensors.append(weight_device)
                weight = weight_device

            handles.append(
                collective.broadcast(tensor=weight, src_rank=0, group_name=self.model_update_group_name, async_op=True)
            )
        for handle in handles:
            handle.wait()
        return refs

    def _separated_model_update(self):
        logger.info(f"start broadcast model update {self.model_update_group_name}")
        if self.is_lora:
            # Same two-stream order as the colocated path: base first, then adapter.
            self._separated_stream_weights(
                gather_fsdp2_base_weights(self.model, buffer_size=self._model_update_buffer_size),
                is_lora=False,
            )
            self._separated_stream_weights(
                gather_fsdp2_lora_weights(
                    self.model, buffer_size=self._model_update_buffer_size, adapter_name=self.adapter_name
                ),
                is_lora=True,
            )
        else:
            self._separated_stream_weights(
                gather_fsdp2_full_weights(self.model, buffer_size=self._model_update_buffer_size),
                is_lora=False,
            )
        self._add_lora_to_infer_workers()
        torch.cuda.empty_cache()
        return {}

    def _separated_stream_weights(self, named_weights_iter, is_lora: bool):
        for named_weights in named_weights_iter:
            refs = self._broadcast_to_infer_workers(named_weights, is_lora=is_lora)
            ray.get(refs)

    def _add_lora_to_infer_workers(self):
        if self.is_lora and self.is_colocated:
            dist.barrier()
        if dist.get_rank() != 0 or not self.is_lora:
            return
        peft_config = self.model.peft_config.get("default", None)
        ray.get(
            [worker.add_lora.remote(peft_config=asdict(peft_config)) for worker in self.model_update_infer_workers]
        )
