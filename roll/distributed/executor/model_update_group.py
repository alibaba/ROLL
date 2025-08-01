import itertools
import json
import time
from collections import defaultdict
from typing import List, Any

import ray

from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.utils.functionals import reduce_metrics


class ModelUpdateGroup:

    def __init__(self, src_cluster: Cluster, tgt_cluster: Cluster, frequency: int = 1):
        self.src_cluster: Any = src_cluster
        self.tgt_cluster: Any = tgt_cluster
        self.frequency = frequency
        self.model_update_name = f"model_update/{self.src_cluster.cluster_name}_2_{self.tgt_cluster.cluster_name}"

        # 存src actor -> tgt actors的映射 (src_actor, tgt_actors)
        # 相同pp_rank的comm_plan是可以并发执行的，全部并发执行需要探索一下
        # Dict[pp_rank, Dict[src_actor_rank, List[tgt_actor_rank]]]
        self.broadcast_comm_pan = defaultdict(lambda: defaultdict(list))

        # 用于相同gpu的actor发送
        self.p2p_comm_plan = defaultdict(lambda: defaultdict(list))

        self.make_comm_plan()
        self.make_collective_group()

    def make_comm_plan(self):
        """
        comm_plan demo:
        {
        "0":
            {
                "0": [
                        {"rank": 0, "device": {"rank": 1, "node_rank": 0, "gpu_rank": 1}},
                        {"rank": 1, "device": {"rank": 0, "node_rank": 0, "gpu_rank": 2}},
                        {"rank": 2, "device": {"rank": 0, "node_rank": 0, "gpu_rank": 4}},
                        {"rank": 3, "device": {"rank": 0, "node_rank": 0, "gpu_rank": 6}}],
                "1": [
                        {"rank": 0, "device": {"rank": 0, "node_rank": 0, "gpu_rank": 0}},
                        {"rank": 1, "device": {"rank": 1, "node_rank": 0, "gpu_rank": 3}},
                        {"rank": 2, "device": {"rank": 1, "node_rank": 0, "gpu_rank": 5}},
                        {"rank": 3, "device": {"rank": 1, "node_rank": 0, "gpu_rank": 7}}]
            },
        "1": {
                "2": [
                        {"rank": 0, "device": {"rank": 0, "node_rank": 0, "gpu_rank": 0}},
                        {"rank": 1, "device": {"rank": 1, "node_rank": 0, "gpu_rank": 3}},
                        {"rank": 2, "device": {"rank": 0, "node_rank": 0, "gpu_rank": 4}},
                        {"rank": 3, "device": {"rank": 0, "node_rank": 0, "gpu_rank": 6}}],
                "3": [
                        {"rank": 0, "device": {"rank": 1, "node_rank": 0, "gpu_rank": 1}},
                        {"rank": 1, "device": {"rank": 0, "node_rank": 0, "gpu_rank": 2}},
                        {"rank": 2, "device": {"rank": 1, "node_rank": 0, "gpu_rank": 5}},
                        {"rank": 3, "device": {"rank": 1, "node_rank": 0, "gpu_rank": 7}}
                    ]
            }
        }
        """
        src_pp_ranks: List[int] = [rank_info.pp_rank for rank_info in self.src_cluster.worker_rank_info]
        group_by_pp_rank = defaultdict(list)
        for i, pp_rank in enumerate(src_pp_ranks):
            group_by_pp_rank[pp_rank].append(i)

        tgt_devices = []
        for rank in range(self.tgt_cluster.world_size):
            for device in self.tgt_cluster.rank2devices[rank]:
                tgt_devices.append(dict(rank=rank, device=device))

        for pp_rank, src_ranks in group_by_pp_rank.items():
            for src_rank in src_ranks:
                self.broadcast_comm_pan[pp_rank][src_rank] = []
            src_rank_iter = itertools.cycle(src_ranks)
            i = 0
            while i < len(tgt_devices):
                tgt_device = tgt_devices[i]
                src_rank = next(src_rank_iter)
                # 如何src_rank和tgt_rank位于同一个设备上，再取一个，如果两个相同，则无法分配当前tgt，加入p2p
                src_device = self.src_cluster.rank2devices[src_rank][0]
                if (src_device["node_rank"], src_device["gpu_rank"]) == (
                    tgt_device["device"]["node_rank"],
                    tgt_device["device"]["gpu_rank"],
                ):
                    src_rank_next = next(src_rank_iter)
                    if src_rank_next == src_rank:
                        self.p2p_comm_plan[pp_rank][src_rank].append(tgt_device)
                    else:
                        i += 1
                        self.broadcast_comm_pan[pp_rank][src_rank_next].append(tgt_device)
                        if i >= len(tgt_devices):
                            break
                        tgt_device_next = tgt_devices[i]
                        self.broadcast_comm_pan[pp_rank][src_rank].append(tgt_device_next)
                else:
                    self.broadcast_comm_pan[pp_rank][src_rank].append(tgt_device)
                i += 1
        print(f"broadcast_comm_pan: {json.dumps(self.broadcast_comm_pan)}")
        print(f"p2p_comm_plan: {json.dumps(self.p2p_comm_plan)}")
        if len(self.p2p_comm_plan) > 0:
            print("p2p comm does not suggest, please change your config")

    def model_update_group_name(self, src_rank, tgt_devices):
        tgt_names = [f"({tgt_device['rank']},{tgt_device['device']['rank']})" for tgt_device in tgt_devices]
        return f"model_update_{self.src_cluster.cluster_name}_{src_rank}_to_{self.tgt_cluster.cluster_name}_{'-'.join(tgt_names)}"

    def make_collective_group(self):
        for pp_rank, pp_comm_plan in self.broadcast_comm_pan.items():
            refs = []
            pp_comm_plan_args = {}
            for src_rank, tgt_devices in pp_comm_plan.items():
                comm_plan_args = {}
                group_name = self.model_update_group_name(src_rank, tgt_devices)
                group_master_worker = self.src_cluster.rank2worker[src_rank]
                group_master_addr = ray.get(group_master_worker.get_node_ip.remote())
                group_master_port = ray.get(group_master_worker.get_free_port.remote())
                comm_plan_args["group_name"] = group_name
                comm_plan_args["master_addr"] = group_master_addr
                comm_plan_args["master_port"] = group_master_port
                comm_plan_args["tgt_devices"] = tgt_devices
                comm_plan_args["src_pp_rank"] = pp_rank
                comm_plan_args["src_rank"] = src_rank
                pp_comm_plan_args[src_rank] = comm_plan_args
                ref = group_master_worker.setup_collective_group.remote(model_update_name=self.model_update_name,
                                                                        comm_plan={src_rank: comm_plan_args})
                refs.append(ref)

            print(f"pp_rank: {pp_rank} pp_comm_plan_args: {json.dumps(pp_comm_plan_args)}")
            for tgt_worker in self.tgt_cluster.workers:
                ref = tgt_worker.setup_collective_group.remote(model_update_name=self.model_update_name,
                                                               comm_plan=pp_comm_plan_args)
                refs.append(ref)
            ray.get(refs)

    def model_update(self, step=None):
        metrics_list = {}
        if step % self.frequency == 0:
            for pp_rank, pp_comm_plan in self.broadcast_comm_pan.items():
                # 一个pp rank 内的要一起更新，目标是更新这一pp rank(pp stage part)内的参数
                # 具体model_update由src role自行实现，这样不需要显式更新模型参数
                refs = []
                for src_rank, tgt_devices in pp_comm_plan.items():
                    src_cluster = self.src_cluster.rank2worker[src_rank]
                    ref = src_cluster.start_model_update.remote(
                        model_update_name=self.model_update_name,
                        tgt_workers=self.tgt_cluster.workers,
                        broadcast_tgt_devices=tgt_devices,
                        p2p_tgt_devices=self.p2p_comm_plan[pp_rank][src_rank],
                    )
                    refs.append(ref)
                data = ray.get(refs)

                metrics_list.update(reduce_metrics(DataProto.concat(data).meta_info.pop("metrics", {})))
        return metrics_list
