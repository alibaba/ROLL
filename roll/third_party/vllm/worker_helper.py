# Copyright (c) 2025, ALIBABA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
from typing import Tuple, Iterable

import torch
import torch.distributed as dist

from roll.utils.collective import collective
from roll.utils.functionals import get_dist_info_from_comm_plan
from roll.utils.logging import get_logger
from roll.utils.send_recv_utils import RecvBucketManager
from roll.third_party.vllm.vllm_utils import patch_vllm_moe_model_weight_loader

logger = get_logger()


class WorkerHelper:

    def reload_model(self):
        if not self.model_runner.model:
            self.model_runner.model_config.load_format = "dummy"
            self.model_runner.load_model()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # before updating the parameters, we need to reinitialize the previously released model
        self.reload_model()
        patch_vllm_moe_model_weight_loader(self.model_runner.model)
        self.model_runner.model.load_weights(weights=weights)

    def load_states(self):
        self.reload_model()
        self.model_runner.model.to('cuda')

    def offload_states(self):
        self.model_runner.model.to('cpu')
        self.cache_engine = None
        self.gpu_cache = None
        if hasattr(self, 'recv_manager'):
            self.recv_manager.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def setup_collective_group(self, comm_plan, backend, rank_in_cluster):
        self.model_update_comm_plan = getattr(self, "model_update_comm_plan", {})
        rank, comm_plan_args = get_dist_info_from_comm_plan(comm_plan, rank_in_cluster=rank_in_cluster,
                                                            rank_in_worker=dist.get_rank())
        if rank is None:
            logger.info(f"no comm_plan found for rank {rank_in_cluster}/{dist.get_rank()}")
            return
        group_name = comm_plan_args["group_name"]
        master_addr = comm_plan_args["master_addr"]
        master_port = comm_plan_args["master_port"]
        world_size = len(comm_plan_args["tgt_devices"]) + 1
        src_pp_rank = comm_plan_args["src_pp_rank"]
        collective.init_collective_group(world_size, rank, backend=backend, group_name=group_name,
                                         master_addr=master_addr, master_port=master_port)
        # A small all_reduce for warmup.
        collective.allreduce(torch.zeros(1).cuda(), group_name=group_name)
        self.model_update_comm_plan[src_pp_rank] = dict(rank=rank,
                                                        world_size=world_size,
                                                        src_pp_rank=src_pp_rank,
                                                        group_name=group_name,
                                                        comm_plan=comm_plan,
                                                        comm_plan_args=comm_plan_args)
        logger.info(f"warmup setup_collective_group: {group_name} rank: {rank} world_size: {world_size}")

    def broadcast_bucket(self, src_pp_rank, meta_infos, bucket_size):
        if src_pp_rank not in self.model_update_comm_plan:
            return
        comm_plan = self.model_update_comm_plan[src_pp_rank]
        buffer = torch.empty(bucket_size, dtype=torch.int8, device="cuda")
        collective.broadcast(tensor=buffer, src_rank=0, group_name=comm_plan["group_name"])
        WorkerHelper.update_parameter_in_bucket(self, meta_infos, buffer, [dist.get_rank()])

    def broadcast_parameter(self, src_pp_rank, dtype, shape, parameter_name):
        if src_pp_rank not in self.model_update_comm_plan:
            return
        comm_plan = self.model_update_comm_plan[src_pp_rank]
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        collective.broadcast(tensor=weight, src_rank=0, group_name=comm_plan["group_name"])
        WorkerHelper.update_parameter(self, parameter_name, weight, [dist.get_rank()])

    def update_parameter(self, parameter_name, weight, ranks_in_worker):
        if dist.get_rank() not in ranks_in_worker:
            return
        self.load_weights([(parameter_name, weight)])
        del weight

    def update_parameter_in_bucket(self, meta_infos, buffer, ranks_in_worker):
        if dist.get_rank() not in ranks_in_worker:
            return
        self.recv_manager = getattr(self, "recv_manager", RecvBucketManager())
        named_params = self.recv_manager.process_bucket(meta_infos, buffer)
        del buffer
        self.load_weights([(name, weight) for name, weight in named_params.items()])