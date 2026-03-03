import asyncio
import atexit
import copy
import gc
import io
import os
import pathlib
import random
import setproctitle

import ray
import grpc
import httpx
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import set_seed
from fastapi import Request
from fastapi.responses import Response
from fastapi.routing import APIRoute

from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.strategy import InferenceStrategy
from roll.third_party.sglang import patch as sglang_patch
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightsFromDistributedReqInput,
    InitWeightsUpdateGroupReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.utils import kill_process_tree
from roll.utils.functionals import concatenate_input_and_output, gather_unpadded_input_ids
from roll.utils.logging import get_logger
from roll.utils.network_utils import collect_free_port
from roll.utils.offload_states import OffloadStateType
from roll.platforms import current_platform

try:
    from sglang.srt.hf_transformers_utils import get_tokenizer
except:
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer


logger = get_logger()


class SglangSlaveActor:
    async def initialize(self, sglang_config):
        grpc_mode = sglang_config.get("grpc_mode", None)
        if grpc_mode is None:
            self.model = SglangEngine()
        elif grpc_mode == True:
            raise NotImplementedError(f"grpc_mode is not supported now")
        else:
            self.model = SglangHttpEngine()

        # [no_return]
        # sglang will join scheduler process at node other than 0
        await self.model.initialize(sglang_config)

class SgLangStrategy(InferenceStrategy):
    strategy_name = "sglang"

    def __init__(self, worker: Worker):
        super().__init__(worker)
        self.model = None
        self.slave_list = []
        self.is_model_in_gpu = True
        self.is_kv_cache_in_gpu = True

    async def initialize(self, model_provider):
        set_seed(seed=self.worker.pipeline_config.seed)

        sglang_config = copy.deepcopy(self.worker_config.strategy_args.strategy_config)
        tp_size = sglang_config.pop("tensor_parallel_size", len(self.worker_config.resource_placement_groups))
        pp_size = sglang_config.pop("pipeline_parallel_size", 1)
        gpu_per_worker = current_platform.device_count()

        assert (tp_size * pp_size) % gpu_per_worker == 0
        nnodes = (tp_size * pp_size) // gpu_per_worker

        assert self.worker.rank_info.dp_rank == self.worker.rank
        assert self.worker.rank_info.dp_size == self.worker.world_size
        logger.info(f"[sglang][local]: dp_rank={self.worker.rank} dp_size={self.worker.world_size} {tp_size=}")

        if self.worker_config.model_args.dtype == "fp32":
            dtype = "float32"
        elif self.worker_config.model_args.dtype == "fp16":
            dtype = "float16"
        elif self.worker_config.model_args.dtype == "bf16":
            dtype = "bfloat16"
        else:
            dtype = "auto"

        sglang_config.setdefault("enable_memory_saver", True)
        sglang_config.update(
            {
                "model_path": self.worker_config.model_args.model_name_or_path,
                "dtype": dtype,
                "random_seed": self.worker.pipeline_config.seed,
                "skip_tokenizer_init": True,
                "mem_fraction_static": sglang_config["mem_fraction_static"],
                "trust_remote_code": True,
                "tp_size": tp_size,
                "log_level": sglang_config.get("log_level", "info"),
                "log_level_http": sglang_config.get("log_level_http", "warning"),
                # socket collects free port [32768 - 65535]，allocate sglang port to random [20000-30000] + sglang dp_rank
                "port": random.randint(20000, 30000) + self.worker.rank * 8, # nccl_port = port + random(100, 1000)
                # 'disable_cuda_graph': True,
                "disable_custom_all_reduce": sglang_config.get("disable_custom_all_reduce", True),
                'nnodes': nnodes,
                'node_rank': 0,
            }
        )

        if nnodes > 1:
            sglang_config['dist_init_addr'] = f'{ray.util.get_node_ip_address()}:{collect_free_port()}'

        logger.info(f"[sglang][sglang_config]: {sglang_config}")

        sglang_args_list = []
        for i in range(nnodes):
            sglang_config_tmp = copy.deepcopy(sglang_config)
            sglang_config_tmp['node_rank'] = i
            sglang_args_list.append(sglang_config_tmp)

        if nnodes > 1:
            node_index = 0
            sglang_pg_list = []
            node_index_list = list(range(self.worker.rank * nnodes, (self.worker.rank + 1) * nnodes))
            for item in self.worker_config.resource_placement_groups:
                if item['node_rank'] in node_index_list and item['gpu_rank'] == 0:
                    sglang_pg_list.append(item['placement_group'])
                    node_index += 1

            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
            from roll.utils.constants import RAY_NAMESPACE
            for i in range(1, nnodes):
                sglang_ray_option = {
                    'scheduling_strategy': PlacementGroupSchedulingStrategy(sglang_pg_list[i]), 
                    'name': f'sglang-slave-{node_index_list[i]}',
                    'namespace': RAY_NAMESPACE,
                    'runtime_env': 
                    {'env_vars': 
                        {'WORLD_SIZE': str(nnodes), 
                        'RANK': str(i), 
                        'WORKER_NAME': f'sglang-slave-{node_index_list[i]}',
                        'CUDA_VISIBLE_DEVICES': ','.join(map(str, list(range(gpu_per_worker)))), 'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES': '1', 
                        'ROLL_LOG_DIR': os.getenv("ROLL_LOG_DIR", "./output/logs/")
                        }
                    }, 
                    'num_cpus': 0.01,
                    'num_gpus': 0.01
                }
                sglang_worker = ray.remote(SglangSlaveActor).options(**sglang_ray_option).remote()
                sglang_worker.initialize.remote(sglang_args_list[i])
                self.slave_list.append(sglang_worker)

        # grpc_mode is supported from v0.4.10
        grpc_mode = sglang_config.get("grpc_mode", None)
        if grpc_mode is None:
            self.model = SglangEngine()
        elif grpc_mode == True:
            raise NotImplementedError(f"grpc_mode is not supported now")
        else:
            self.model = SglangHttpEngine()
        await self.model.initialize(sglang_args_list[0])

    def get_url(self):
        return self.model.get_url()

    def op_compute_log_probs(self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        pass

    async def abort_requests(self, request_ids=None):
        # assert isinstance(self.model, SglangEngine)
        # TODO sglang support abort_all and abort parallel sampling request from v0.4.9
        # https://github.com/sgl-project/sglang/pull/6698
        if request_ids is None: # temporary solution to abort rquest with parallel sampling
            request_ids = self.model.engine.tokenizer_manager.rid_to_state
        for rid in request_ids:
            self.model.engine.tokenizer_manager.abort_request(rid)

    async def generate_request(self, payload: dict):
        # assert isinstance(self.model, SglangEngine)
        from sglang import __version__ as version
        if version < '0.5' and payload["sampling_params"]["n"] > 1: # fixed in https://github.com/sgl-project/sglang/pull/7508
            payload["rid"] = None
        obj = GenerateReqInput(
            input_ids=payload["input_ids"],
            sampling_params=payload["sampling_params"],
            rid=payload["rid"],
            return_logprob=payload["return_logprob"],
        )
        generator = self.model.engine.tokenizer_manager.generate_request(obj, None)
        chunks = None
        async for chunks in generator:
            chunks = chunks
        assert chunks is not None
        chunks = chunks if isinstance(chunks, list) else [chunks]

        return postprocess_generate(chunks)

    async def generate(self, batch: DataProto, generation_config):
        # assert isinstance(self.model, SglangEngine)
        assert self.is_model_in_gpu
        sampling_params = create_sampling_params_for_sglang(gen_kwargs=generation_config)
        logger.info(f"sampling_params: {sampling_params}")

        input_ids = batch.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = batch.batch["attention_mask"]  # left-padded attention_mask

        image_data = None
        if "multi_modal_data" in batch.non_tensor_batch:
            prompt_token_ids = []
            image_data = []
            # sglang enforce str(path or url)/bytes image data currently
            # TODO: path image_processor.load_image with hash according to:
            # https://github.com/sgl-project/sglang/pull/4915
            for data in batch.non_tensor_batch["multi_modal_data"]:
                # bug exists in sglang, it only puts image str (standing for path
                # or url) into list and leaves out image bytes. Thus when using
                # image bytes, put it into list mannully
                prompt_token_ids.append(data["prompt_token_ids"])
                # for text and multi-modal mixed data
                if (
                    "multi_modal_data" not in data
                    or "image" not in data["multi_modal_data"]
                    or not data["multi_modal_data"]["image"]
                ):
                    image_data.append(None)
                    continue
                image_per_sample = []
                for image in data["multi_modal_data"]["image"]:
                    byte_stream = io.BytesIO()
                    image.save(byte_stream, "png")
                    image_per_sample.append(byte_stream.getvalue())
                    byte_stream.close()
                image_data.append(image_per_sample)
        else:
            prompt_token_ids = gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)
        return_logprob = sampling_params.pop("return_logprob", False)
        sglang_outputs = await self.model.engine.async_generate(
            input_ids=prompt_token_ids, image_data=image_data, sampling_params=sampling_params, return_logprob=return_logprob
        )

        # (bs * num_return_sequences, max_response_len)
        output_ids = gather_outputs_to_pad_tensor(
            request_outputs=sglang_outputs,
            pad_token_id=self.model.tokenizer.pad_token_id,
            device=input_ids.device,
        )

        # (bs * num_return_sequences, input_len + max_response_len)
        output = concatenate_input_and_output(
            input_ids=input_ids, output_ids=output_ids, num_return_sequences=sampling_params["n"]
        )
        return output

    async def setup_collective_group(self, master_address, master_port, rank_offset, world_size, group_name, backend=None):
        logger.info(f"setup_collective_group {group_name=}")
        payload = {
            "master_address": master_address,
            "master_port": master_port,
            "group_name": group_name,
            "rank_offset": rank_offset,
            "world_size": world_size,
            "backend": backend if backend is not None else current_platform.communication_backend,
        }
        return await self.model.init_weights_update_group(payload)

    async def broadcast_parameter(self, names, dtypes, shapes, group_name, is_lora=False):
        await self._reload_model()
        assert not is_lora, "lora training is not supported with sglang"
        payload = {"names": names, "dtypes": dtypes, "shapes": shapes, "group_name": group_name, "flush_cache": False}
        return await self.model.update_weights_from_distributed(payload)

    async def update_parameter_in_bucket(self, serialized_named_tensors, is_lora=False):
        await self._reload_model()
        assert not is_lora, "lora training is not supported with sglang"
        # required above sglang 0.5
        payload = {
            "load_format": "flattened_bucket",
            "flush_cache": False,
            "serialized_named_tensors": serialized_named_tensors,
        }
        return await self.model.update_weights_from_tensor(payload)

    async def _reload_model(self):
        if self.is_model_in_gpu:
            return
        self.is_model_in_gpu = True
        tags = ["weights"]
        payload = {"tags": tags}
        await self.model.resume_memory_occupation(payload)
        logger.info(f"self.model.resume_memory_occupation {tags=} exec ....")

    async def flush_cache(self):
        await self.model.flush_cache()

    async def load_states(self, *args, **kwargs):
        await self.flush_cache()
        tags = []
        if not self.is_model_in_gpu:
            tags.append("weights")
        if not self.is_kv_cache_in_gpu:
            tags.extend(["kv_cache", "cuda_graph"])
        if tags:
            payload = {"tags": tags}
            await self.model.resume_memory_occupation(payload)
            logger.info(f"self.model.resume_memory_occupation {tags=} exec ....")
        self.is_model_in_gpu, self.is_kv_cache_in_gpu = True, True

    async def offload_states(self, include=None, non_blocking=False):
        if include is None or OffloadStateType.model_params in include:
            if self.worker.pipeline_config.is_actor_infer_colocated and self.is_model_in_gpu:
                await self.model.release_memory_occupation()
                logger.info("self.model.release_memory_occupation exec ....")
                # always release all
                self.is_model_in_gpu, self.is_kv_cache_in_gpu = False, False

        gc.collect()
        current_platform.empty_cache()

class SglangEngine:
    def __init__(self):
        self.engine = None
        self.tokenizer = None

    async def initialize(self, sglang_config):
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        os.environ["FLASHINFER_WORKSPACE_BASE"] = os.path.join(
            pathlib.Path.home().as_posix(), ".cache", os.environ.get("WORKER_NAME", ""))
        self.engine = sglang_patch.engine.engine_module.Engine(**sglang_config)
        self.engine.tokenizer_manager.auto_create_handle_loop() # some rpc of tokenizer_manager will not create handle_loop automatically

        self.tokenizer = get_tokenizer(sglang_config["model_path"], trust_remote_code=True)

    async def init_weights_update_group(self, payload):
        return await self.engine.tokenizer_manager.init_weights_update_group(InitWeightsUpdateGroupReqInput(**payload))

    async def update_weights_from_distributed(self, payload):
        return await self.engine.tokenizer_manager.update_weights_from_distributed(UpdateWeightsFromDistributedReqInput(**payload))

    async def update_weights_from_tensor(self, payload):
        return await self.engine.tokenizer_manager.update_weights_from_tensor(UpdateWeightsFromTensorReqInput(**payload))

    async def resume_memory_occupation(self, payload):
        await self.engine.tokenizer_manager.resume_memory_occupation(ResumeMemoryOccupationReqInput(**payload))

    async def release_memory_occupation(self):
        await self.engine.tokenizer_manager.release_memory_occupation(ReleaseMemoryOccupationReqInput(), None)

    async def flush_cache(self):
        await self.engine.tokenizer_manager.flush_cache()

def shutdown():
    kill_process_tree(os.getpid(), include_parent=False)

class SglangHttpEngine:
    @staticmethod
    async def dummy_health_generate(request: Request) -> Response:
        return Response(status_code=200)

    @staticmethod
    def remove_route(routes, path):
        for index, route in enumerate(routes):
            if isinstance(route, APIRoute) and route.path_format == path:
                del routes[index]
                break

    @staticmethod
    def launch_server(sglang_config):
        setproctitle.setproctitle("sglang::server")
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.entrypoints.http_server import launch_server, app, health_generate
        server_args = ServerArgs(**sglang_config)
        SglangHttpEngine.remove_route(app.routes, "/health")
        SglangHttpEngine.remove_route(app.routes, "/health_generate")
        app.get("/health")(SglangHttpEngine.dummy_health_generate)
        app.get("/health_generate")(SglangHttpEngine.dummy_health_generate)
        app.get("/health_generate_original")(health_generate)
        launch_server(server_args)

    @staticmethod
    async def wait_worker_healthy(worker_process, url, client):
        while True:
            await asyncio.sleep(10)
            try:
                response = await client.get(f"{url}/health_generate_original")
                if response.status_code == 200:
                    break
                elif response.status_code != 503:
                    response.raise_for_status()
                else:
                    logger.info(f"Waiting for sglang worker {url} ready...")
                assert worker_process.is_alive()
            except httpx.ConnectError:
                logger.info(f"Waiting for sglang worker {url} to start...")

    def __init__(self):
        self.worker_process = None
        self.url = None
        self.client = None

    def get_url(self):
        return self.url

    async def initialize(self, sglang_config):
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        os.environ["FLASHINFER_WORKSPACE_BASE"] = os.path.join(
            pathlib.Path.home().as_posix(), ".cache", os.environ.get("WORKER_NAME", ""))
        import multiprocessing

        multiprocessing.set_start_method("spawn")
        atexit.register(shutdown)

        sglang_config["host"] = Worker.get_node_ip()
        assert sglang_config["port"] > 0
        self.worker_process = multiprocessing.Process(
            target=SglangHttpEngine.launch_server,
            args=(sglang_config,),
        )
        self.worker_process.start()
        self.url = f"http://{sglang_config['host']}:{sglang_config['port']}"
        logger.info(f"start sglang server url={self.url}")

        self.client = httpx.AsyncClient(timeout=httpx.Timeout(None))
        await SglangHttpEngine.wait_worker_healthy(worker_process=self.worker_process, url=self.url, client=self.client)

    async def init_weights_update_group(self, payload):
        response = await self.client.post(f"{self.url}/init_weights_update_group", json=payload)
        response.raise_for_status()
        response = response.json()
        return response["success"], response["message"]

    async def update_weights_from_distributed(self, payload):
        payload["dtypes"] = [str(dtype).removeprefix("torch.") for dtype in payload["dtypes"]]
        response = await self.client.post(f"{self.url}/update_weights_from_distributed", json=payload)
        response.raise_for_status()
        response = response.json()
        return response["success"], response["message"]

    async def update_weights_from_tensor(self, payload):
        response = await self.client.post(f"{self.url}/update_weights_from_tensor", json=payload)
        response.raise_for_status()

    async def resume_memory_occupation(self, payload):
        response = await self.client.post(f"{self.url}/resume_memory_occupation", json=payload)
        response.raise_for_status()

    async def release_memory_occupation(self):
        response = await self.client.post(f"{self.url}/release_memory_occupation", json={})
        response.raise_for_status()

    async def flush_cache(self):
        response = await self.client.post(f"{self.url}/flush_cache", json={})
        response.raise_for_status()

class SglangGrpcEngine:
    @staticmethod
    def launch_server(sglang_config):
        setproctitle.setproctitle("sglang::server")
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.entrypoints.grpc_server import serve_grpc
        server_args = ServerArgs(**sglang_config)
        asyncio.run(serve_grpc(server_args))

    @staticmethod
    async def wait_worker_healthy(worker_process, url, client):
        from sglang.srt.grpc import sglang_scheduler_pb2
        request = sglang_scheduler_pb2.HealthCheckRequest()
        while True:
            await asyncio.sleep(10)
            try:
                response = await client.HealthCheck(request)
                if response.healthy:
                    break
                assert worker_process.is_alive()
            except Exception as e:
                logger.info(f"Waiting for sglang worker {url} to start ...")

    async def initialize(self, sglang_config):
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        os.environ["FLASHINFER_WORKSPACE_BASE"] = os.path.join(
            pathlib.Path.home().as_posix(), ".cache", os.environ.get("WORKER_NAME", ""))
        import multiprocessing
        from sglang.srt.grpc import sglang_scheduler_pb2_grpc

        multiprocessing.set_start_method("spawn")
        atexit.register(shutdown)

        sglang_config["skip_tokenizer_init"] = False

        sglang_config["host"] = Worker.get_node_ip()
        assert sglang_config["port"] > 0
        self.worker_process = multiprocessing.Process(
            target=SglangHttpEngine.launch_server,
            args=(sglang_config,),
        )
        self.worker_process.start()
        self.url = f"grpc://{sglang_config['host']}:{sglang_config['port']}"
        logger.info(f"start sglang server url={self.url}")

        self.channel = grpc.aio.insecure_channel(
            f"{sglang_config['host']}:{sglang_config['port']}",
            options=[
                ("grpc.max_send_message_length", 1024 * 1024 * 256),
                ("grpc.max_receive_message_length", 1024 * 1024 * 256),
            ],
        )
        self.client = sglang_scheduler_pb2_grpc.SglangSchedulerStub(self.channel)
        await SglangHttpEngine.wait_worker_healthy(worker_process=self.worker_process, url=self.url, client=self.client)

def gather_outputs_to_pad_tensor(request_outputs, pad_token_id, device=None) -> torch.Tensor:
    if device is None:
        device = current_platform.device_type
    token_ids_list_of_lists = [
        torch.tensor(request_output["output_ids"], device=device) for request_output in request_outputs
    ]
    output_tensor = pad_sequence(token_ids_list_of_lists, batch_first=True, padding_value=pad_token_id)
    return output_tensor


def create_sampling_params_for_sglang(gen_kwargs: dict):
    return dict(
        max_new_tokens=gen_kwargs["max_new_tokens"],
        temperature=gen_kwargs["temperature"],
        top_p=gen_kwargs["top_p"],
        top_k=gen_kwargs["top_k"],
        stop_token_ids=gen_kwargs["eos_token_id"],
        repetition_penalty=gen_kwargs["repetition_penalty"],
        n=gen_kwargs["num_return_sequences"],
        stop=gen_kwargs["stop_strings"],
        no_stop_trim=gen_kwargs.get("include_stop_str_in_output", True),
    )

def postprocess_generate(chunks):
    output_data = {}
    output_token_ids = [chunk.get("output_ids", []) for chunk in chunks]
    output_logprobs = [chunk["meta_info"].get("output_token_logprobs", None) for chunk in chunks]
    has_logprobs = any(logprobs is not None for logprobs in output_logprobs)
    if has_logprobs:
        lens = [min(len(ids), len(logprobs)) for ids, logprobs in zip(output_token_ids, output_logprobs)]
        output_token_ids = [ids[:l] for ids, l in zip(output_token_ids, lens)]
        output_logprobs = [logprobs[:l] for logprobs, l in zip(output_logprobs, lens)]
        output_logprobs = [[prob_info[0] for prob_info in logprobs] for logprobs in output_logprobs]
        output_data["output_logprobs"] = output_logprobs
        assert all([len(ids) == len(logprobs) for ids, logprobs in zip(output_token_ids, output_logprobs)]), (
            "output_token_ids and output_logprobs length not match"
        )
    output_data["output_token_ids"] = output_token_ids
    output_data["finish_reasons"] = []
    for chunk in chunks:
        finish_reason = chunk["meta_info"]["finish_reason"]
        if isinstance(finish_reason, dict):
            finish_reason = finish_reason["type"]
            output_data["finish_reasons"].append(finish_reason)
        else:
            output_data["finish_reasons"].append(finish_reason)
    assert len(output_data["finish_reasons"]) == len(output_data["output_token_ids"])
    return output_data
