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
import asyncio
import logging
import multiprocessing as mp
import os
import threading
from typing import Dict, Tuple
import atexit

from transformers import AutoModel, AutoProcessor, AutoImageProcessor
ori_model_register = AutoModel.register
ori_processor_register = AutoProcessor.register
ori_image_processor_register = AutoImageProcessor.register

# these are classmethod bounded with cls
def model_register_patch(config_class, model_class, exist_ok=False):
    exist_ok = True
    return ori_model_register(config_class, model_class, exist_ok)

def processor_register_patch(config_class, model_class, exist_ok=False):
    exist_ok = True
    return ori_processor_register(config_class, model_class, exist_ok)

def image_processor_register_patch(config_class,
                                    image_processor_class=None,
                                    slow_image_processor_class=None,
                                    fast_image_processor_class=None,
                                    exist_ok=False):
    exist_ok = True
    return ori_image_processor_register(config_class,
                                        image_processor_class,
                                        slow_image_processor_class,
                                        fast_image_processor_class,
                                        exist_ok)

# to avoid register conflict when import
AutoModel.register = model_register_patch
AutoProcessor.register = processor_register_patch
AutoImageProcessor.register = image_processor_register_patch

# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

from sglang.srt.managers.data_parallel_controller import (
    run_data_parallel_controller_process,
)
from sglang.srt.managers.detokenizer_manager import run_detokenizer_process
from sglang.srt.openai_api.adapter import load_chat_template_for_openai_api
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import (
    configure_logger,
    launch_dummy_health_check_server,
    prepare_model_and_tokenizer,
)
from sglang.srt.entrypoints.engine import Engine, _set_envs_and_config

from roll.third_party.sglang.v043post4_patch.io_struct import (
    SetupCollectiveGroupReqInput,
    BroadcastBucketReqInput,
    BroadcastParameterReqInput,
    UpdateParameterInBucketReqInput,
    UpdateParameterReqInput,
)
from roll.third_party.sglang.v043post4_patch.tokenizer_manager import TokenizerManagerSA
from roll.third_party.sglang.v043post4_patch.scheduler import run_scheduler_process

logger = logging.getLogger(__name__)

import sglang.srt.entrypoints.engine as engine_module

class EngineSA(Engine):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # normalize_batch_and_arguments called in tokenizer_manager which is in
        # the main process, thus can be patched directly
        from sglang.srt.managers.io_struct import GenerateReqInput
        ori_normalize_batch_and_arguments = GenerateReqInput.normalize_batch_and_arguments
        def normalize_batch_and_arguments_patch(self):
            ori_normalize_batch_and_arguments(self)
            # self.is_single=False
            if self.parallel_sample_num == 1:
                num = self.batch_size
            else:
                num = self.batch_size * self.parallel_sample_num
            if not self.modalities:
                self.modalities = [None] * num
            elif not isinstance(self.modalities, list) or len(
                    self.modalities) == 1 and isinstance(
                        self.modalities[0], str):
                self.modalities = [self.modalities] * num
            elif isinstance(self.modalities, list):
                pass

        GenerateReqInput.normalize_batch_and_arguments = normalize_batch_and_arguments_patch

    def setup_collective_group(
        self,
        comm_plan: str,
        backend: str,
        rank_in_cluster: int,
    ):
        obj = SetupCollectiveGroupReqInput(
            comm_plan=comm_plan,
            backend=backend,
            rank_in_cluster=rank_in_cluster,
        )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.setup_collective_group(obj, None)
        )
    
    def broadcast_bucket(
        self,
        src_pp_rank: int, 
        meta_infos: dict, 
        bucket_size: int,
    ):
        obj = BroadcastBucketReqInput(
            src_pp_rank=src_pp_rank,
            meta_infos=meta_infos,
            bucket_size=bucket_size,
        )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.broadcast_bucket(obj, None)
        )
    
    def broadcast_parameter(
        self,
        src_pp_rank, 
        dtype, 
        shape, 
        parameter_name
    ):
        obj = BroadcastParameterReqInput(
            src_pp_rank=src_pp_rank,
            dtype=dtype,
            shape=shape,
            parameter_name=parameter_name,
        )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.broadcast_parameter(obj, None)
        )
    
    def update_parameter(
        self,
        parameter_name, 
        weight, 
        ranks_in_worker
    ):
        obj = UpdateParameterReqInput(
            parameter_name=parameter_name,
            weight=weight,
            ranks_in_worker=ranks_in_worker,
        )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.update_parameter(obj, None)
        )
    
    def update_parameter_in_bucket(
        self,
        meta_infos, 
        buffer, 
        ranks_in_worker
    ):
        """Initialize parameter update group."""
        obj = UpdateParameterInBucketReqInput(
            meta_infos=meta_infos,
            buffer=buffer,
            ranks_in_worker=ranks_in_worker,
        )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.update_parameter_in_bucket(obj, None)
        )

def _launch_subprocesses(server_args: ServerArgs) -> Tuple[TokenizerManagerSA, Dict]:
    """
    Launch the TokenizerManager in the main process, the Scheduler in a subprocess, and the DetokenizerManager in another subprocess.
    """
    # Configure global environment
    configure_logger(server_args)
    server_args.check_server_args()
    _set_envs_and_config(server_args)

    # Allocate ports for inter-process communications
    port_args = PortArgs.init_new(server_args)
    logger.info(f"{server_args=}")

    # If using model from www.modelscope.cn, first download the model.
    server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(
        server_args.model_path, server_args.tokenizer_path
    )

    scheduler_procs = []
    if server_args.dp_size == 1:
        # Launch tensor parallel scheduler processes
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )

        scheduler_pipe_readers = []
        tp_size_per_node = server_args.tp_size // server_args.nnodes
        tp_rank_range = range(
            tp_size_per_node * server_args.node_rank,
            tp_size_per_node * (server_args.node_rank + 1),
        )
        for tp_rank in tp_rank_range:
            reader, writer = mp.Pipe(duplex=False)
            gpu_id = (
                server_args.base_gpu_id
                + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
            )
            proc = mp.Process(
                target=run_scheduler_process,
                args=(server_args, port_args, gpu_id, tp_rank, None, writer),
            )
            with memory_saver_adapter.configure_subprocess():
                proc.start()
            scheduler_procs.append(proc)
            scheduler_pipe_readers.append(reader)
    else:
        # Launch the data parallel controller
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_readers = [reader]
        proc = mp.Process(
            target=run_data_parallel_controller_process,
            args=(server_args, port_args, writer),
        )
        proc.start()
        scheduler_procs.append(proc)

    if server_args.node_rank >= 1:
        # In multi-node cases, non-zero rank nodes do not need to run tokenizer or detokenizer,
        # so they can just wait here.

        for reader in scheduler_pipe_readers:
            data = reader.recv()
            assert data["status"] == "ready"

        if os.getenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN") == "0":
            # When using `Engine` as a Python API, we don't want to block here.
            return None, None

        launch_dummy_health_check_server(server_args.host, server_args.port)

        for proc in scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
        return None, None

    # Launch detokenizer process
    detoken_proc = mp.Process(
        target=run_detokenizer_process,
        args=(
            server_args,
            port_args,
        ),
    )
    detoken_proc.start()

    # Launch tokenizer process
    tokenizer_manager = TokenizerManagerSA(server_args, port_args)
    if server_args.chat_template:
        load_chat_template_for_openai_api(
            tokenizer_manager, server_args.chat_template, server_args.model_path
        )

    # Wait for the model to finish loading
    scheduler_infos = []
    for i in range(len(scheduler_pipe_readers)):
        try:
            data = scheduler_pipe_readers[i].recv()
        except EOFError:
            logger.error(
                f"Rank {i} scheduler is dead. Please check if there are relevant logs."
            )
            scheduler_procs[i].join()
            logger.error(f"Exit code: {scheduler_procs[i].exitcode}")
            raise

        if data["status"] != "ready":
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )
        scheduler_infos.append(data)

    # Assume all schedulers have the same scheduler_info
    scheduler_info = scheduler_infos[0]
    tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]
    return tokenizer_manager, scheduler_info

engine_module._launch_subprocesses = _launch_subprocesses

import psutil
import signal
import setproctitle

from sglang.utils import get_exception_traceback
from sglang.srt.utils import kill_itself_when_parent_died
from sglang.srt.managers.detokenizer_manager import DetokenizerManager


def run_detokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
):
    # to avoid register conflict when import
    AutoModel.register = model_register_patch
    AutoProcessor.register = processor_register_patch
    AutoImageProcessor.register = image_processor_register_patch

    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang::detokenizer")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        manager = DetokenizerManager(server_args, port_args)
        manager.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DetokenizerManager hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)

