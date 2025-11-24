import json
from collections import defaultdict
from typing import Any
import ray
import torch
from codetiming import Timer

from roll.utils.functionals import reduce_metrics
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.executor.cluster import Cluster

'''
Logits transfer pipeline:
    1) Teacher pp last-stage tp rank 0 sends logits to the corresponding
       Student pp last-stage tp rank 0.
    2) Student then broadcasts the logits to all ranks in its pp last stage.

Two execution phases:
    (1) Teacher-to-Student P2P transfer
    (2) Student internal broadcast

Supported P2P configurations:
    - Ray: direct P2P via Ray.
    - IPC+NCCL: use IPC for same-GPU transfers, NCCL for cross-GPU.
    - NCCL-only: NCCL cannot directly handle same-GPU transfers, so:
        • If teacher.tp_size > 1 → use teacher tp rank 1 as sender for same-GPU.
        • If teacher.tp_size == 1 → adjust dispatch order (offset) to avoid
          same-GPU transfers. 
          Example (dp=4, offset=1):
              original: (data0, data1, data2, data3)
              shifted:  (data1, data2, data3, data0)
'''

class LogitsTransferGroup:
    VALID_BACKENDS = {"ipc+nccl", "nccl-only", "ray"}

    def __init__(self, src_cluster, tgt_cluster, backend="ipc+nccl"):
        if backend not in self.VALID_BACKENDS:
            raise ValueError(f"Invalid backend '{backend}'")
        self.src_cluster = src_cluster
        self.tgt_cluster = tgt_cluster
        self.backend = backend

        # get tensor list from src cluster
        self.tensor_name_list_for_transfer = ray.get(src_cluster.workers[0].get_tensor_name_list_for_transfer.remote())

        self.broadcast_comm_pan = defaultdict(lambda: defaultdict(list))
        self.p2p_comm_plan = defaultdict(lambda: defaultdict(list))
        self.dp_mapping = []
        self.edge_info = {}
        self.phases_cache = {}   # cache phases
        self.phases_dict = {}
        self.need_offset = False
        self.offset = 0

        # global name of the logits transfer task
        self.model_update_name = f"logits_transfer/{self.src_cluster.cluster_name}_2_{self.tgt_cluster.cluster_name}"

        self.make_comm_plan()
        self.make_collective_group()

        self.print_full_plan()

    def _relation(self, dev_t, dev_s):
        if dev_t["node_rank"] == dev_s["node_rank"]:
            if dev_t["gpu_rank"] == dev_s["gpu_rank"]:
                return "SAME_GPU"
            return "SAME_NODE"
        return "CROSS_NODE"

    def logits_transfer_name(self, src_rank, tgt_entry_list):
        tgt_names = [f"({t['rank']})" for t in tgt_entry_list]
        return f"logits_transfer_{self.src_cluster.cluster_name}_{src_rank}_to_{self.tgt_cluster.cluster_name}_{'-'.join(tgt_names)}"

    def make_comm_plan(self):
        src_dp = self.src_cluster.dp_size
        tgt_dp = self.tgt_cluster.dp_size

        if src_dp > tgt_dp:
            slice_type = "student_receive_slice"
        elif src_dp < tgt_dp:
            slice_type = "teacher_send_slice"
        else:
            slice_type = "full"

        self.dp_mapping = [[] for _ in range(tgt_dp)]
        if src_dp >= tgt_dp:
            ratio = src_dp // tgt_dp
            for s_dp in range(tgt_dp):
                for i in range(ratio):
                    t_dp = s_dp * ratio + i
                    self.dp_mapping[s_dp].append({
                        "t_dp": t_dp, "slice_index": i, "total_slices": ratio, "slice_type": slice_type
                    })
        else:
            ratio = tgt_dp // src_dp
            for s_dp in range(tgt_dp):
                t_dp = s_dp // ratio
                slice_index = s_dp % ratio
                self.dp_mapping[s_dp].append({
                    "t_dp": t_dp, "slice_index": slice_index, "total_slices": ratio, "slice_type": slice_type
                })

        teacher_dp_tp0 = {ri.dp_rank: r for r, ri in enumerate(self.src_cluster.worker_rank_info)
                          if ri.is_pipeline_last_stage and ri.tp_rank == 0 and ri.cp_rank == 0}
        teacher_dp_tp1 = {ri.dp_rank: r for r, ri in enumerate(self.src_cluster.worker_rank_info)
                          if ri.is_pipeline_last_stage and ri.tp_rank == 1 and ri.cp_rank == 0}
        student_dp_tp0 = {ri.dp_rank: r for r, ri in enumerate(self.tgt_cluster.worker_rank_info)
                          if ri.is_pipeline_last_stage and ri.tp_rank == 0 and ri.cp_rank == 0}

        if self.backend == "ray":
            for s_dp, mapping in enumerate(self.dp_mapping):
                s_rank = student_dp_tp0[s_dp]
                s_dev = self.tgt_cluster.rank2devices[s_rank][0]
                for m in mapping:
                    t_dp = m["t_dp"]
                    t_rank = teacher_dp_tp0[t_dp]
                    t_dev = self.src_cluster.rank2devices[t_rank][0]
                    entry = {**m, "rank": s_rank, "device": s_dev}
                    self.broadcast_comm_pan[0][t_rank].append(entry)
                    self.edge_info[(t_rank, s_rank)] = self._relation(t_dev, s_dev)

        elif self.backend == "ipc+nccl":
            for s_dp, mapping in enumerate(self.dp_mapping):
                s_rank = student_dp_tp0[s_dp]
                s_dev = self.tgt_cluster.rank2devices[s_rank][0]
                for m in mapping:
                    t_dp = m["t_dp"]
                    t_rank = teacher_dp_tp0[t_dp]
                    t_dev = self.src_cluster.rank2devices[t_rank][0]
                    rel = self._relation(t_dev, s_dev)
                    entry = {**m, "rank": s_rank, "device": s_dev}
                    self.edge_info[(t_rank, s_rank)] = rel
                    if rel == "SAME_GPU":
                        self.p2p_comm_plan[0][t_rank].append(entry)
                    else:
                        self.broadcast_comm_pan[0][t_rank].append(entry)

        elif self.backend == "nccl-only":
            if self.src_cluster.tp_size > 1:
                for s_dp, mapping in enumerate(self.dp_mapping):
                    s_rank = student_dp_tp0[s_dp]
                    s_dev = self.tgt_cluster.rank2devices[s_rank][0]
                    for m in mapping:
                        t_dp = m["t_dp"]
                        t_rank = teacher_dp_tp0[t_dp]
                        t_dev = self.src_cluster.rank2devices[t_rank][0]
                        if self._relation(t_dev, s_dev) == "SAME_GPU" and t_dp in teacher_dp_tp1:
                            t_rank = teacher_dp_tp1[t_dp]
                            t_dev = self.src_cluster.rank2devices[t_rank][0]
                        rel = self._relation(t_dev, s_dev)
                        if rel == "SAME_GPU":
                            raise RuntimeError(f"NCCL-only SAME_GPU conflict {t_rank} → {s_rank}")
                        entry = {**m, "rank": s_rank, "device": s_dev}
                        self.broadcast_comm_pan[0][t_rank].append(entry)
                        self.edge_info[(t_rank, s_rank)] = rel
            else:
                has_conflict = False
                for s_dp, mapping in enumerate(self.dp_mapping):
                    s_rank = student_dp_tp0[s_dp]
                    s_dev = self.tgt_cluster.rank2devices[s_rank][0]
                    for m in mapping:
                        t_dp = m["t_dp"]
                        t_rank = teacher_dp_tp0[t_dp]
                        t_dev = self.src_cluster.rank2devices[t_rank][0]
                        if self._relation(t_dev, s_dev) == "SAME_GPU":
                            has_conflict = True
                            break
                    if has_conflict: break
                if not has_conflict:
                    for s_dp, mapping in enumerate(self.dp_mapping):
                        s_rank = student_dp_tp0[s_dp]
                        s_dev = self.tgt_cluster.rank2devices[s_rank][0]
                        for m in mapping:
                            t_dp = m["t_dp"]
                            t_rank = teacher_dp_tp0[t_dp]
                            t_dev = self.src_cluster.rank2devices[t_rank][0]
                            entry = {**m, "rank": s_rank, "device": s_dev}
                            self.broadcast_comm_pan[0][t_rank].append(entry)
                            self.edge_info[(t_rank, s_rank)] = self._relation(t_dev, s_dev)
                else:
                    found = False
                    for offset in range(1, src_dp):
                        safe = True
                        for s_dp, mapping in enumerate(self.dp_mapping):
                            s_rank = student_dp_tp0[s_dp]
                            s_dev = self.tgt_cluster.rank2devices[s_rank][0]
                            for m in mapping:
                                t_dp_off = (m["t_dp"] + offset) % src_dp
                                t_rank = teacher_dp_tp0[t_dp_off]
                                t_dev = self.src_cluster.rank2devices[t_rank][0]
                                if self._relation(t_dev, s_dev) == "SAME_GPU":
                                    safe = False
                                    break
                            if not safe: break
                        if safe:
                            self.offset = offset
                            found = True
                            break
                    if not found:
                        raise RuntimeError("NCCL-only: Cannot find safe offset")
                    for s_dp, mapping in enumerate(self.dp_mapping):
                        s_rank = student_dp_tp0[s_dp]
                        s_dev = self.tgt_cluster.rank2devices[s_rank][0]
                        for m in mapping:
                            t_dp_off = (m["t_dp"] + self.offset) % src_dp
                            t_rank = teacher_dp_tp0[t_dp_off]
                            t_dev = self.src_cluster.rank2devices[t_rank][0]
                            entry = {**m, "rank": s_rank, "device": s_dev}
                            self.broadcast_comm_pan[0][t_rank].append(entry)
                            self.edge_info[(t_rank, s_rank)] = self._relation(t_dev, s_dev)

    def _group_phases(self, comm_plan_for_pp):
        """
        Split a communication plan in the format {src_rank: tgt_entry_list} into multiple phases (list[dict]),
        ensuring that within the same phase, target student ranks for different src_rank tasks do not conflict.

        comm_plan_for_pp: dict
            { src_rank: [entry_dict, entry_dict, ...], ... }
            Each entry_dict must contain at least the 'rank' field (student rank); other fields are kept as is.

        Returns:
            phases: list[dict]
                Each phase: { src_rank: tgt_entry_list }
        """

        phases = []

        for src_rank, tgt_list in comm_plan_for_pp.items():
            tgt_students = {t["rank"] for t in tgt_list}
            placed = False

            for phase in phases:
                # get all the student ranks of current phase
                phase_students = {t["rank"] for devs in phase.values() for t in devs}
                if tgt_students.isdisjoint(phase_students):
                    # if src_rank already exists, then merge the dict
                    if src_rank in phase:
                        existing_students = {t["rank"] for t in phase[src_rank]}
                        new_targets = [t for t in tgt_list if t["rank"] not in existing_students]
                        phase[src_rank].extend(new_targets)
                    else:
                        phase[src_rank] = list(tgt_list)  # keep original entry
                    placed = True
                    break

            # If no suitable phase exists, create a new one.
            if not placed:
                phases.append({src_rank: list(tgt_list)})

        return phases

    def make_collective_group(self):
        """
        Iterate over phase_id to build collective groups,
        and generate self.phases_dict for convenient model_update-style iteration within each phase later.
        """
        self.phases_cache = self._group_phases(self.broadcast_comm_pan[0])

        # iterate over phase
        for phase_id, phase in enumerate(self.phases_cache):
            refs = []
            # initialize phases_dict[phase_id]
            self.phases_dict[phase_id] = {}

            for src_rank, tgt_entry_list in phase.items():
                tgt_devices = [t['device'] for t in tgt_entry_list]
                group_name = self.logits_transfer_name(src_rank, tgt_entry_list) + f"_phase{phase_id}"

                # group master worker（teacher src_rank worker）
                group_master_worker = self.src_cluster.rank2worker[src_rank]
                master_addr = ray.get(group_master_worker.get_node_ip.remote())
                master_port = ray.get(group_master_worker.get_free_port.remote())

                comm_plan_args = {
                    "group_name": group_name,
                    "master_addr": master_addr,
                    "master_port": master_port,
                    "tgt_devices": tgt_entry_list,
                    "src_pp_rank": phase_id,  # use phase_id as pp_rank
                    "src_rank": src_rank,
                    "slice_info": [
                        (t['rank'], t['slice_index'], t['total_slices'], t['slice_type'])
                        for t in tgt_entry_list
                    ]
                }

                # For later use by logits_transfer (retain only tgt_devices).
                self.phases_dict[phase_id][src_rank] = comm_plan_args

                # Teacher master worker setup group
                if self.backend != "ray":
                    ref = group_master_worker.setup_collective_group.remote(
                        model_update_name=self.model_update_name,
                        comm_plan={src_rank: comm_plan_args},
                        mode="sender"
                    )
                    refs.append(ref)

            print(
                f"phase_id: {phase_id} pp_comm_plan_args: {json.dumps(self.phases_dict[phase_id], indent=4, ensure_ascii=False, sort_keys=True)}")

            if self.backend != "ray":
                # Student workers setup group
                for tgt_worker in self.tgt_cluster.workers:
                    ref = tgt_worker.setup_collective_group.remote(
                        model_update_name=self.model_update_name,
                        comm_plan=self.phases_dict[phase_id],
                        mode="receiver"
                    )
                    refs.append(ref)

            ray.get(refs)

    def apply_offset_by_dp(self, dp: DataProto) -> DataProto:
        """
        Apply block-level circular shift to DataProto according to teacher dp_size and offset.

        Each block corresponds to the samples for one dp_rank.
        Block size = len(dp) // dp_size.

        This is in-place reordering, returns the same DataProto object.
        """
        dp_size = self.src_cluster.dp_size
        offset = getattr(self, "offset", 0)

        if len(dp) == 0 or offset == 0 or dp_size <= 1:
            return dp

        N = len(dp)
        assert N % dp_size == 0, \
            f"[apply_offset_by_dp] Batch size {N} must be divisible by dp_size {dp_size}."

        block_len = N // dp_size
        shift = offset % dp_size

        import torch
        # [dp0_block, dp1_block, ...]  → shift blocks by offset
        indices = torch.arange(N).view(dp_size, block_len)
        # move last `shift` blocks to front
        indices = torch.cat([indices[-shift:], indices[:-shift]], dim=0).reshape(-1)

        new_dp = dp.clone()
        new_dp.reorder(indices)  # in-place modify
        return new_dp

    def logits_transfer(self):
        full_metrics = {}
        for tensor_name_for_transfer in self.tensor_name_list_for_transfer:
            metrics = self.logits_transfer_impl(tensor_name_for_transfer=tensor_name_for_transfer)
            full_metrics.update(metrics)
        return full_metrics

    def logits_transfer_impl(self, tensor_name_for_transfer):
        """
        Execute logits transfer in order of phase_id.
        Phase-first approach, but the structure within each phase is the same as in model_update.
        """
        print(f"\nTensor name for transfer: {tensor_name_for_transfer}")
        print("\n[Logits Transfer Execution - Phase First Style]")
        print(f"\n backend: {self.backend}")
        with Timer("logits_transfer_teacher2student") as teacher2student_timer:
            broadcast_src_set = set()
            for phase_id, phase_plan in sorted(self.phases_dict.items()):
                print(f"\n=== Phase {phase_id} ===")
                refs = []
                for src_rank, broadcast_comm_plan_args in phase_plan.items():
                    broadcast_src_set.add(src_rank)
                    src_worker = self.src_cluster.rank2worker[src_rank]
                    broadcast_tgt_entry_list = broadcast_comm_plan_args['tgt_devices']
                    broadcast_comm_plan_args['tgt_workers'] = [
                        self.tgt_cluster.rank2worker[entry['rank']]
                        for entry in broadcast_tgt_entry_list
                    ]
                    ref = src_worker.logits_transfer.remote(
                        tensor_name_for_transfer=tensor_name_for_transfer,
                        model_update_name=self.model_update_name,
                        broadcast_comm_plan_args=broadcast_comm_plan_args,
                        p2p_tgt_workers=[
                            self.tgt_cluster.rank2worker[entry['rank']]
                            for entry in self.p2p_comm_plan[0][src_rank]],  # if p2p
                        p2p_entry_list=self.p2p_comm_plan[0][src_rank],
                        backend=self.backend
                    )
                    refs.append(ref)
                ray.get(refs)

            # remaining p2p
            refs = []
            for src_rank, p2p_tgt_entry_list in self.p2p_comm_plan[0].items():
                if src_rank in broadcast_src_set:
                    continue
                src_worker = self.src_cluster.rank2worker[src_rank]
                p2p_tgt_workers = [self.tgt_cluster.rank2worker[tgt_entry['rank']] for tgt_entry in p2p_tgt_entry_list]
                ref = src_worker.logits_transfer.remote(
                    tensor_name_for_transfer=tensor_name_for_transfer,
                    model_update_name=self.model_update_name,
                    broadcast_comm_plan_args=None,
                    p2p_tgt_workers=p2p_tgt_workers,
                    p2p_entry_list=p2p_tgt_entry_list,
                    backend=self.backend
                )
                refs.append(ref)
            ray.get(refs)

        print("\n[Logits Transfer Done]\n")

        print("\n[Logits Broadcast Execution]")

        with Timer("student_internal_broadcast") as student_internal_broadcast_timer:
            refs = []
            for tgt_worker in self.tgt_cluster.workers:
                ref = tgt_worker.broadcast_logits.remote(tensor_name_for_transfer=tensor_name_for_transfer, tp=True, cp=False)
                refs.append(ref)
            ray.get(refs)

            refs = []
            for tgt_worker in self.tgt_cluster.workers:
                ref = tgt_worker.broadcast_logits.remote(tensor_name_for_transfer=tensor_name_for_transfer, tp=False,
                                                         cp=True)
                refs.append(ref)
            ray.get(refs)

        print("\n[Logits Broadcast Done]\n")

        metrics = {
            f"logits_transfer/{tensor_name_for_transfer}/logits_transfer_total_time": teacher2student_timer.last + student_internal_broadcast_timer.last,
            f"logits_transfer/{tensor_name_for_transfer}/teacher2student_time": teacher2student_timer.last,
            f"logits_transfer/{tensor_name_for_transfer}/student_internal_broadcast_time": student_internal_broadcast_timer.last,
        }
        return metrics

    # ===== Summary =====
    def print_comm_plan_summary(self):
        print("\n=== LogitsTransferGroup Summary ===")
        print(f"Backend: {self.backend}")
        print(f"DP mapping: {self.dp_mapping}")
        if self.backend == "nccl-only":
            print(f"need_offset={self.need_offset}, offset={self.offset}")
        print("===================================\n")

    def print_broadcast_plan(self):
        print("\n--- Broadcast Communication Plan ---")
        for pp_rank, pp_comm_plan in self.broadcast_comm_pan.items():
            print(f"PP-Rank {pp_rank}:")
            for src_rank, tgt_list in pp_comm_plan.items():
                students = [t['rank'] for t in tgt_list]
                rels = [self._relation(self.src_cluster.rank2devices[src_rank][0], t['device']) for t in tgt_list]
                print(f"  Teacher {src_rank} -> Students {students} | Relations {rels}")
        print("------------------------------------\n")

    def print_p2p_plan(self):
        print("\n--- P2P (IPC) Communication Plan ---")
        for pp_rank, pp_comm_plan in self.p2p_comm_plan.items():
            print(f"PP-Rank {pp_rank}:")
            for src_rank, tgt_list in pp_comm_plan.items():
                students = [t['rank'] for t in tgt_list]
                print(f"  Teacher {src_rank} -> Students {students} [SAME_GPU]")
        print("-------------------------------------\n")

    def print_phases(self):
        """print cached phases"""
        print("\n===== Cached Phases =====")
        for phase_id, phase in enumerate(self.phases_cache):
            for src_rank, tgt_entry_list in phase.items():
                print(f"Phase{phase_id} src_rank {src_rank}: {tgt_entry_list}")
        print("=========================\n")

    def print_full_plan(self):
        self.print_comm_plan_summary()
        self.print_broadcast_plan()
        self.print_p2p_plan()
        self.print_phases()

