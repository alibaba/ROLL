"""Rebuild disposable model shards from authoritative CPU optimizer masters."""

import torch


def cpu_master_shard_pairs(optimizer):
    """Validate every shard before any model buffer is discarded.

    Hybrid owns one FP32 master per local DP shard. This deliberately rejects
    partial CPU offload and ordinary FP32 model parameters, whose originals
    could themselves be authoritative optimizer storage.
    """
    if any(optimizer.shard_fp32_groups):
        raise ValueError("CPU master model offload does not support FP32 model parameters")
    parameters = [p for group in optimizer.shard_float16_groups for p in group]
    hybrid = optimizer.optimizer
    originals = [p for group in hybrid.param_groups for p in group["params"]]
    if len(originals) != len(parameters) or {id(p) for p in originals} != {id(p) for p in parameters}:
        raise ValueError("CPU master model offload requires complete Hybrid shard ownership")
    pairs = []
    for parameter in parameters:
        master = hybrid.param_to_fp32_param.get(parameter)
        if (
            not isinstance(master, torch.Tensor)
            or master.device.type != "cpu"
            or master.dtype != torch.float32
            or master.shape != parameter.shape
            or not master.is_contiguous()
            or parameter.dtype not in (torch.bfloat16, torch.float16)
            or hybrid.gpu_params_map_cpu_copy.get(parameter) is not master
        ):
            raise ValueError("CPU master model offload requires an authoritative CPU FP32 master for every shard")
        pairs.append((parameter, master))
    return pairs


@torch.no_grad()
def restore_cpu_master_shards(pairs, chunk_bytes=64 * 1024 * 1024):
    """Cast directly into current model shards, with bounded transfer staging."""
    if chunk_bytes < 4:
        raise ValueError("chunk_bytes must fit at least one FP32 element")
    chunk_elements = chunk_bytes // 4
    for parameter, master in pairs:
        destination, source = parameter.view(-1), master.view(-1)
        for start in range(0, master.numel(), chunk_elements):
            # Blocking transfers make ownership explicit before DP all-gather.
            destination[start:start + chunk_elements].copy_(source[start:start + chunk_elements])


def empty_cpu_parameter_buffer(numel, dtype):
    """Retain shape metadata only; values are unreadable until model reload."""
    return torch.zeros(1, dtype=dtype, device="cpu").expand(numel)


class CpuMasterWeightProvider:
    """Reconstruct one model weight without reloading its complete DDP buffer.

    The model may contain zero-storage CPU placeholders. Only the authoritative
    Hybrid FP32 masters supply parameter values. Disjoint DP slices are summed
    on the communication device before the ordinary TP/EP converter runs.
    No reconstructed tensor is cached; memory is bounded by the caller's bucket
    and its largest individual parameter. Construct anew for each update because
    normal offload/reload rebinds the optimizer's shard objects.
    """

    def __init__(self, models, optimizer, device="cuda"):
        self.device = torch.device(device)
        error = None
        try:
            partitions = self._local_partitions(models, optimizer)
        except Exception as exc:
            error = exc
        self._agree_validation(error)
        # Finish every child group's collectives before rejecting coverage.
        # Expert-DP subgroups eventually join the common TP/EP converter.
        error = None
        for group, parameters, intervals in partitions:
            world = torch.distributed.get_world_size(group) if group is not None else 1
            all_intervals = [intervals]
            if world > 1 and intervals:
                local = torch.tensor(intervals, dtype=torch.int64, device=self.device)
                gathered = [torch.empty_like(local) for _ in range(world)]
                torch.distributed.all_gather(gathered, local, group=group)
                all_intervals = [value.cpu().tolist() for value in gathered]
            try:
                self._validate_coverage(parameters, all_intervals)
            except ValueError as exc:
                error = exc
        self._agree_validation(error)

    def _agree_validation(self, error):
        """All actor ranks reject before entering the next collective phase."""
        failed = torch.tensor(int(error is not None), dtype=torch.int32, device=self.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
        if failed.item():
            if error is not None:
                raise ValueError(str(error)) from error
            raise ValueError("CPU master weight streaming validation failed on another rank")

    def _local_partitions(self, models, optimizer):
        if len(models) != 1 or models[0].config.pipeline_model_parallel_size != 1:
            raise ValueError("CPU master weight streaming currently requires PP1 with one model")
        self.parameters = dict(models[0].named_parameters())
        self.buffers = dict(models[0].named_buffers())
        self.entries = {}
        names = {id(p): name for name, p in self.parameters.items()}
        leaves = getattr(optimizer, "chained_optimizers", [optimizer])
        partitions = []
        for leaf in leaves:
            if not getattr(leaf.config, "offload_model_from_cpu_master", False):
                raise ValueError("CPU master weight streaming requires complete CPU master offload")
            masters = {id(p): master for p, master in cpu_master_shard_pairs(leaf)}
            owned = {}
            for index, group in enumerate(leaf.opt_group_ranges):
                shards = leaf.shard_float16_groups[index]
                if len(shards) != len(group["params"]):
                    raise ValueError("CPU master shard count does not match optimizer ranges")
                for parameter, shard in zip(group["params"], shards):
                    buffer, dtype, bucket = leaf.model_param_gbuf_map[parameter]
                    interval = leaf.gbuf_ranges[buffer][dtype][bucket]["param_map"][parameter]["param"]
                    master = masters[id(shard)]
                    if not (0 <= interval.start < interval.end <= parameter.numel()
                            and interval.end - interval.start == master.numel()):
                        raise ValueError("CPU master shard range does not match its parameter")
                    owned[id(parameter)] = (interval.start, interval.end, master)
            parameters = [p for b in leaf.buffers for p in b.params]
            if set(owned) - {id(p) for p in parameters}:
                raise ValueError("CPU master shard has no matching model parameter buffer")
            intervals = []
            for parameter in parameters:
                name = names.get(id(parameter))
                if name is None or name in self.entries:
                    raise ValueError("CPU master parameter ownership is missing or ambiguous")
                segment = owned.get(id(parameter))
                intervals.append(segment[:2] if segment is not None else (-1, -1))
                self.entries[name] = (leaf.data_parallel_group, segment)
            partitions.append((leaf.data_parallel_group, parameters, intervals))
        if self.parameters.keys() != self.entries.keys():
            raise ValueError("CPU master weight streaming requires every model parameter to be owned")
        return partitions

    @staticmethod
    def _validate_coverage(parameters, all_intervals):
        # Megatron buffers use the same parameter order within each DP group.
        for index, parameter in enumerate(parameters):
            ranges = sorted(row[index] for row in all_intervals if row[index][0] >= 0)
            end = 0
            for start, stop in ranges:
                if start != end or stop <= start:
                    raise ValueError("CPU master DP shards overlap or leave a gap")
                end = stop
            if end != parameter.numel():
                raise ValueError("CPU master DP shards do not cover the complete parameter")

    @torch.no_grad()
    def __call__(self, name, metadata):
        parameter = self.parameters.get(name)
        if parameter is None:
            buffer = self.buffers.get(name)
            if buffer is None or buffer.shape != metadata.shape or buffer.dtype != metadata.dtype:
                raise ValueError(f"Unknown or mismatched CPU master export key: {name}")
            return buffer.to(device=self.device, copy=True)
        if parameter.shape != metadata.shape or parameter.dtype != metadata.dtype:
            raise ValueError(f"CPU master export metadata mismatch: {name}")
        group, segment = self.entries[name]
        result = torch.zeros(parameter.shape, dtype=parameter.dtype, device=self.device)
        if segment is not None:
            start, stop, master = segment
            restore_cpu_master_shards([(result.view(-1)[start:stop], master)])
        if group is not None and torch.distributed.get_world_size(group) > 1:
            torch.distributed.all_reduce(result, group=group)
        return result
