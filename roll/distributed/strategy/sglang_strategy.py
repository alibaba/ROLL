import copy
import gc
import io
import os
import dataclasses
import pathlib
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from transformers import set_seed

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
from roll.utils.functionals import concatenate_input_and_output
from roll.utils.logging import get_logger
from roll.utils.network_utils import collect_free_port
from roll.utils.offload_states import OffloadStateType
from roll.platforms import current_platform

try:
    from sglang.srt.hf_transformers_utils import get_tokenizer
except:
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer


logger = get_logger()


def launch_server_process(server_args):
    import multiprocessing
    from sglang.srt.entrypoints.http_server import launch_server
    p = multiprocessing.Process(target=launch_server, args=(server_args,))
    p.start()
    return p


class SglangSlaveActor:
    def __init__(self, sglang_config):
        self.sglang_config = sglang_config
        self.running_process = None
    
    def initialize(self):
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        os.environ["FLASHINFER_WORKSPACE_BASE"] = os.path.join(
            pathlib.Path.home().as_posix(), ".cache", os.environ.get("WORKER_NAME", ""))
        from sglang.srt.server_args import ServerArgs
        sargs = ServerArgs(**self.sglang_config)
        self.running_process = launch_server_process(sargs)
    

class SgLangStrategy(InferenceStrategy):
    strategy_name = "sglang"

    def __init__(self, worker: Worker):
        super().__init__(worker)
        self.model
        self.slave_list = []

    async def initialize(self, model_provider):
        import ray
        set_seed(seed=self.worker.pipeline_config.seed)

        dist.init_process_group(backend=current_platform.communication_backend, timeout=timedelta(minutes=self.worker_config.backend_timeout))
        dist.all_reduce(torch.zeros(1).to(current_platform.device_type))

        sglang_config = copy.deepcopy(self.worker_config.strategy_args.strategy_config)
        tp_size = sglang_config.pop("tensor_parallel_size", len(self.worker_config.resource_placement_groups))
        pp_size = sglang_config.pop("pipeline_parallel_size", 1)
        gpu_per_worker = current_platform.device_count()

        assert (tp_size * pp_size) % gpu_per_worker == 0
        nnodes = (tp_size * pp_size) // gpu_per_worker

        dp_rank = dist.get_rank()
        dp_size = dist.get_world_size()
        self.worker.rank_info.dp_rank = dp_rank
        self.worker.rank_info.dp_size = dp_size
        logger.info(f"[sglang][local]: {dp_rank=} {dp_size=} {tp_size=}")

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
                "port": 30000 + dp_rank * 500,
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
            for item in self.worker_config.resource_placement_groups:
                if item['node_rank'] == node_index and item['gpu_rank'] == 0:
                    sglang_pg_list.append(item['placement_group'])
                    node_index += 1
            
            sglang_ray_option_list = []
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
            from roll.utils.constants import RAY_NAMESPACE
            for i in range(nnodes):
                sglang_ray_option = {
                    'scheduling_strategy': PlacementGroupSchedulingStrategy(sglang_pg_list[i]), 
                    'name': f'sglang-slave-{i}', 
                    'namespace': RAY_NAMESPACE,
                    'runtime_env': 
                    {'env_vars': 
                        {'WORLD_SIZE': str(nnodes), 
                        'RANK': str(i), 
                        'WORKER_NAME': f'sglang-slave-{i}', 
                        'CUDA_VISIBLE_DEVICES': ','.join(map(str, list(range(gpu_per_worker)))), 'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES': '1', 
                        'ROLL_LOG_DIR': os.getenv("ROLL_LOG_DIR", "./output/logs/")
                        }
                    }, 
                    'num_cpus': 0.01, 
                    'max_concurrency': 1000, 
                    'num_gpus': 0.01
                }
                sglang_ray_option_list.append(sglang_ray_option)
         
            SglangSlaveActor_ray = ray.remote(SglangSlaveActor)
            for i in range(1, nnodes):
                _sglang_worker = SglangSlaveActor_ray.options(**sglang_ray_option_list[i]).remote(sglang_args_list[i])
                ray.get([_sglang_worker.initialize.remote()])
                self.slave_list.append(_sglang_worker)


        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        os.environ["FLASHINFER_WORKSPACE_BASE"] = os.path.join(
            pathlib.Path.home().as_posix(), ".cache", os.environ.get("WORKER_NAME", ""))
        self.model = sglang_patch.engine.engine_module.Engine(**sglang_args_list[0])        
        self.is_model_in_gpu = True
        self.is_kv_cache_in_gpu = True
        self.has_run = False

        self.tokenizer = get_tokenizer(self.worker_config.model_args.model_name_or_path, trust_remote_code=True)

        additional_special_tokens = self.tokenizer.additional_special_tokens
        special_tokens = [
            add_token
            for add_token in self.tokenizer.added_tokens_decoder.values()
            if add_token.special and add_token.content not in additional_special_tokens
        ]
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}, replace_additional_special_tokens=False
        )
        logger.info(f"add {special_tokens} to additional_special_tokens: {self.tokenizer.additional_special_tokens}")

    def op_compute_log_probs(self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        pass

    async def abort_requests(self, request_ids=None):
        if request_ids is None: # temporary solution to abort rquest with parallel sampling
            request_ids = self.model.tokenizer_manager.rid_to_state
        for rid in request_ids:
            self.model.tokenizer_manager.abort_request(rid)

    async def generate_request(self, data: DataProto):
        self.has_run = True

        input_ids = data.batch["input_ids"]
        assert input_ids.size(0) == 1, f"data['input_ids'] must have exactly one batch dimension"
        attention_mask = data.batch["attention_mask"]
        rid = data.meta_info["request_id"]
        assert isinstance(rid, str)
        generation_config = data.meta_info.get("generation_config")
        collect_unfinished = data.meta_info.get("collect_unfinished", False)
        max_new_tokens = data.meta_info.get("max_new_tokens", generation_config["max_new_tokens"])
        max_new_tokens = min(max_new_tokens, generation_config["max_new_tokens"])
        sampling_params = create_sampling_params_for_sglang(
            gen_kwargs={**generation_config, "max_new_tokens": max_new_tokens}
        )
        input_ids = gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)
        assert isinstance(input_ids, list) and isinstance(input_ids[0], list)
        if sampling_params['n'] > 1:
            assert not collect_unfinished, "collect_unfinished is not supported in parallel sampling"
            rid = None # sglang does not support using rid with parallel sampling

        obj_init_kw = {}  # return logprobs may be in GenerateReqInput not SamplingParams
        for field in dataclasses.fields(GenerateReqInput):
            if field.name in sampling_params:
                obj_init_kw[field.name] = sampling_params.pop(field.name)
        from sglang import __version__ as version
        if version >= '0.4.6.post4':
            sampling_params['stream_interval'] = 50
        obj = GenerateReqInput(
            input_ids=input_ids[0],
            sampling_params=sampling_params,
            rid=rid,
            stream=True,
            **obj_init_kw,
        )
        chunks: list[dict] = [None for _ in range(sampling_params['n'])]
        generator = self.model.tokenizer_manager.generate_request(obj, None)
        async for chunk in generator:
            index = chunk.get("index", 0)
            chunks[index] = chunk

        output_data = DataProto(meta_info=data.meta_info)

        if not all(chunk is not None for chunk in chunks):
            output_data.meta_info["finish_reasons"] = ["abort"]
            return output_data

        output_token_ids = [chunk.get("output_ids", []) for chunk in chunks]
        output_logprobs = [chunk["meta_info"].get("output_token_logprobs", None) for chunk in chunks]
        has_logprobs = any(logprobs is not None for logprobs in output_logprobs)
        if has_logprobs:
            lens = [min(len(ids), len(logprobs)) for ids, logprobs in zip(output_token_ids, output_logprobs)]
            output_token_ids = [ids[:l] for ids, l in zip(output_token_ids, lens)]
            output_logprobs = [logprobs[:l] for logprobs, l in zip(output_logprobs, lens)]
            output_logprobs = [[prob_info[0] for prob_info in logprobs] for logprobs in output_logprobs]
            output_data.meta_info["output_logprobs"] = output_logprobs
            assert all([len(ids) == len(logprobs) for ids, logprobs in zip(output_token_ids, output_logprobs)]), (
                "output_token_ids and output_logprobs length not match"
            )
        output_data.meta_info["output_token_ids"] = output_token_ids
        output_data.meta_info["finish_reasons"] = []
        for chunk in chunks:
            if isinstance(chunk["meta_info"].get("finish_reason"), dict):
                finish_reason = chunk["meta_info"]["finish_reason"]["type"]
                output_data.meta_info["finish_reasons"].append(finish_reason)
            else:
                # convert finish_reason None to 'abort'
                output_data.meta_info["finish_reasons"].append("abort")
        assert len(output_data.meta_info["finish_reasons"]) == len(output_data.meta_info["output_token_ids"])
        return output_data

    async def generate(self, batch: DataProto, generation_config):
        self.has_run = True
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
        sglang_outputs = await self.model.async_generate(
            input_ids=prompt_token_ids, image_data=image_data, sampling_params=sampling_params, return_logprob=return_logprob
        )

        # (bs * num_return_sequences, max_response_len)
        output_ids = gather_outputs_to_pad_tensor(
            request_outputs=sglang_outputs,
            pad_token_id=self.tokenizer.pad_token_id,
            device=input_ids.device,
        )

        # (bs * num_return_sequences, input_len + max_response_len)
        output = concatenate_input_and_output(
            input_ids=input_ids, output_ids=output_ids, num_return_sequences=sampling_params["n"]
        )
        return output

    async def setup_collective_group(self, master_address, master_port, rank_offset, world_size, group_name, backend=None):
        logger.info(f"setup_collective_group {group_name=}")
        return await self.model.tokenizer_manager.init_weights_update_group(
            InitWeightsUpdateGroupReqInput(
                master_address=master_address,
                master_port=master_port,
                group_name=group_name,
                rank_offset=rank_offset,
                world_size=world_size,
                backend=backend if backend is not None else current_platform.communication_backend,
            )
        )

    async def broadcast_parameter(self, names, dtypes, shapes, group_name, is_lora=False):
        await self._reload_model()
        assert not is_lora, "lora training is not supported with sglang"
        obj = UpdateWeightsFromDistributedReqInput(
            names=names, dtypes=dtypes, shapes=shapes, group_name=group_name, flush_cache=False
        )
        return await self.model.tokenizer_manager.update_weights_from_distributed(obj)

    async def update_parameter_in_bucket(self, serialized_named_tensors, is_lora=False):
        await self._reload_model()
        assert not is_lora, "lora training is not supported with sglang"
        # required above sglang 0.5
        obj = UpdateWeightsFromTensorReqInput(
            load_format="flattened_bucket",
            flush_cache=False,
            serialized_named_tensors=serialized_named_tensors,
        )
        return await self.model.tokenizer_manager.update_weights_from_tensor(obj, None)

    async def _reload_model(self):
        if self.is_model_in_gpu:
            return
        self.is_model_in_gpu = True
        tags = ["weights"]
        await self.model.tokenizer_manager.resume_memory_occupation(ResumeMemoryOccupationReqInput(tags=tags), None)
        logger.info(f"self.model.resume_memory_occupation {tags=} exec ....")

    async def load_states(self, *args, **kwargs):
        if self.has_run:  # flush cache can't be called as the first request for tokenizer_manager
            await self.model.tokenizer_manager.flush_cache()
        tags = []
        if not self.is_model_in_gpu:
            tags.append("weights")
        if not self.is_kv_cache_in_gpu:
            tags.extend(["kv_cache", "cuda_graph"])
        if tags:
            await self.model.tokenizer_manager.resume_memory_occupation(ResumeMemoryOccupationReqInput(tags=tags), None)
            logger.info(f"self.model.resume_memory_occupation {tags=} exec ....")
        self.is_model_in_gpu, self.is_kv_cache_in_gpu = True, True

    async def offload_states(self, include=None, non_blocking=False):
        if include is None or OffloadStateType.model_params in include:
            if self.worker.pipeline_config.is_actor_infer_colocated and self.is_model_in_gpu:
                await self.model.tokenizer_manager.release_memory_occupation(ReleaseMemoryOccupationReqInput(), None)
                logger.info("self.model.release_memory_occupation exec ....")
                # always release all
                self.is_model_in_gpu, self.is_kv_cache_in_gpu = False, False

        gc.collect()
        current_platform.empty_cache()


def gather_unpadded_input_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    gathered_input_ids = [ids[mask.bool()].tolist() for ids, mask in zip(input_ids, attention_mask)]
    return gathered_input_ids


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
        return_logprob=gen_kwargs.get("logprobs", 0) is not None,
        stop=gen_kwargs["stop_strings"],
        no_stop_trim=gen_kwargs.get("include_stop_str_in_output", True),
    )
