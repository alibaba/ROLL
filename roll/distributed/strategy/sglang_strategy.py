import copy
import gc
import io
import queue
import time
import os
from concurrent import futures
from datetime import timedelta
from typing import Optional, List

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from transformers import set_seed
import sglang as sgl

from roll.third_party.sglang import patch as sglang_patch

from sglang.srt.hf_transformers_utils import get_tokenizer
from mcore_adapter.models.converter.convert_utils import RecvBucketManager
from roll.utils.collective import collective


from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.strategy import InferenceStrategy
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType
from roll.utils.functionals import concatenate_input_and_output, GenerateRequestType


logger = get_logger()


class SgLangStrategy(InferenceStrategy):
    strategy_name = "sglang"

    def __init__(self, worker: Worker):
        super().__init__(worker)
        self.model
        self.async_model = None
        self.sampling_params = None
        self.use_gpu_executor = True
        self.executor: futures.ThreadPoolExecutor = futures.ThreadPoolExecutor(max_workers=1)
        self.sglang_outputs_list: List = []
        self.input_ids_list: List = []
        self.recv_manager = RecvBucketManager()
        self.command_queue: Optional[queue.Queue] = None

        self.request_ids = set()
        self.generation_config = None

        self.group_name = "sglang_worker_default"
        collective.init_collective_group(
            world_size=self.worker.world_size,
            rank=self.worker.rank,
            group_name=self.group_name,
            master_addr=self.worker.master_addr,
            master_port=self.worker.master_port,
        )
        self.running = False

    def initialize(self, model_provider):
        set_seed(seed=self.worker.pipeline_config.seed)
        self.command_queue = queue.Queue()

        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=self.worker_config.backend_timeout))
        dist.all_reduce(torch.zeros(1).cuda())

        sglang_config = copy.deepcopy(self.worker_config.strategy_args.strategy_config)
        tp_size = sglang_config.pop("tensor_parallel_size", torch.cuda.device_count())

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

        sglang_config.update(
            {
                "model_path": self.worker_config.model_args.model_name_or_path,
                "dtype": dtype,
                "random_seed": self.worker.pipeline_config.seed,
                "skip_tokenizer_init": True,
                "mem_fraction_static": sglang_config["mem_fraction_static"],
                "trust_remote_code": True,
                "tp_size": tp_size,
                "log_level": "info",
                "enable_memory_saver": True,
                "random_seed": self.worker.pipeline_config.seed,
                "port": 30000 + dp_rank * 500,
                # 'disable_cuda_graph': True,
                "disable_custom_all_reduce": sglang_config.get("disable_custom_all_reduce", True),
            }
        )
        logger.info(f"[sglang][sglang_config]: {sglang_config}")

        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        self.model = sglang_patch.engine.EngineSA(**sglang_config)

        # 用于标记显存是否释放
        self.model.is_model_in_gpu = True

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

        import asyncio

        self.event_loop = asyncio.get_event_loop()

    def op_compute_log_probs(self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        pass

    def start_server(self, data: DataProto, request_complete_callback):
        collective.barrier(group_name=self.group_name)
        self.running = True
        sglang_patch.async_engine.start_async_sglang(
            self.event_loop, self.model, request_complete_callback, self.command_queue
        )

    def add_request(self, command, data: DataProto):
        if command == GenerateRequestType.ADD:
            input_ids = data.batch["input_ids"]
            attention_mask = data.batch["attention_mask"]
            request_id = data.meta_info["request_id"]
            self.request_ids.add(request_id)
            generation_config = data.meta_info.get("generation_config")
            max_new_tokens = data.meta_info.get("max_new_tokens", generation_config["max_new_tokens"])
            max_new_tokens = min(max_new_tokens, generation_config["max_new_tokens"])
            sampling_params = create_sampling_params_for_sglang(
                gen_kwargs={**generation_config, "max_new_tokens": max_new_tokens}
            )
            prompt_token_ids = gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)
            sglang_patch.async_engine.add_request(
                self.command_queue, ([request_id], prompt_token_ids, sampling_params, data.meta_info)
            )

        elif command == GenerateRequestType.ABORT:
            request_id = data.meta_info["request_id"]
            sglang_patch.async_engine.abort_request(self.command_queue, rid=request_id)

        elif command == GenerateRequestType.STOP:
            for abort_rid in self.request_ids:
                sglang_patch.async_engine.abort_request(self.command_queue, rid=abort_rid)
            self.command_queue.put(None)
            self.request_ids.clear()
            self.running = False

    def generate(self, batch: DataProto, generation_config):
        if self.sampling_params is None:
            self.sampling_params = create_sampling_params_for_sglang(gen_kwargs=generation_config)
            old_sampling_params = self.sampling_params
            logger.info(f"sampling_params: {self.sampling_params}")
        else:
            new_sampling_params = create_sampling_params_for_sglang(gen_kwargs=generation_config)
            old_sampling_params = self.sampling_params
            if not compare_sampling_params(new_sampling_params, self.sampling_params):
                self.sampling_params = new_sampling_params
                logger.info(f"switch sampling_params: {self.sampling_params}")

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
        sglang_outputs = self.model.generate(
            input_ids=prompt_token_ids, image_data=image_data, sampling_params=self.sampling_params
        )

        # (bs * num_return_sequences, max_response_len)
        output_ids = gather_outputs_to_pad_tensor(
            request_outputs=sglang_outputs,
            pad_token_id=self.tokenizer.pad_token_id,
            device=input_ids.device,
        )

        # (bs * num_return_sequences, input_len + max_response_len)
        output = concatenate_input_and_output(
            input_ids=input_ids, output_ids=output_ids, num_return_sequences=self.sampling_params["n"]
        )

        # 回归初始采样参数
        self.sampling_params = old_sampling_params

        return output

    def resume_model(self):
        if self.model.release_memory_state:
            self.model.resume_memory_occupation()
            self.model.release_memory_state = False

    def release_model(self):
        if not self.model.release_memory_state:
            self.model.release_memory_occupation()
            self.model.release_memory_state = True

    # 参数同步相关接口
    def setup_collective_group(self, model_update_name, comm_plan, backend="nccl"):
        self.model.setup_collective_group(comm_plan=comm_plan, backend=backend, rank_in_cluster=self.worker.rank)

    def broadcast_parameter(self, model_update_name, src_pp_rank, dtype, shape, parameter_name, is_lora=False):
        self.model.broadcast_parameter(src_pp_rank, dtype, shape, parameter_name)

    def broadcast_bucket(self, model_update_name, src_pp_rank, meta_infos, bucket_size):
        self.model.broadcast_bucket(src_pp_rank, meta_infos, bucket_size)

    def update_parameter(self, model_update_name, parameter_name, weight, ranks_in_worker):
        self.model.update_parameter(parameter_name, weight, ranks_in_worker)

    def update_parameter_in_bucket(self, model_update_name, meta_infos, buffer, ranks_in_worker):
        self.model.update_parameter_in_bucket(meta_infos, buffer, ranks_in_worker)

    def load_states(self, *args, **kwargs):
        if not self.model.is_model_in_gpu:
            self.model.resume_memory_occupation()
            logger.info("self.model.resume_memory_occupation exec ....")
            self.model.is_model_in_gpu = True

    def offload_states(self, include=None, non_blocking=False):
        if include is None or OffloadStateType.model_params in include:
            if self.model.is_model_in_gpu:
                self.model.release_memory_occupation()
                logger.info("self.model.release_memory_occupation exec ....")
                self.model.is_model_in_gpu = False
        self.recv_manager.clear()
        gc.collect()
        torch.cuda.empty_cache()


def gather_unpadded_input_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    gathered_input_ids = [ids[mask.bool()].tolist() for ids, mask in zip(input_ids, attention_mask)]
    return gathered_input_ids


def gather_outputs_to_pad_tensor(request_outputs, pad_token_id, device="cuda") -> torch.Tensor:
    token_ids_list_of_lists = [
        torch.tensor(request_output["output_ids"], device=device) for request_output in request_outputs
    ]
    output_tensor = pad_sequence(token_ids_list_of_lists, batch_first=True, padding_value=pad_token_id)
    return output_tensor


def concatenate_input_and_output(input_ids, output_ids, num_return_sequences):
    batch_size, input_seq_len = input_ids.size()
    _, output_seq_len = output_ids.size()
    repeated_input_ids = (
        input_ids.unsqueeze(1)
        .repeat(1, num_return_sequences, 1)
        .view(batch_size * num_return_sequences, input_seq_len)
    )
    sequences = torch.cat((repeated_input_ids, output_ids), dim=1)
    return sequences


def create_sampling_params_for_sglang(gen_kwargs):
    return dict(
        max_new_tokens=gen_kwargs["max_new_tokens"],
        temperature=gen_kwargs["temperature"],
        top_p=gen_kwargs["top_p"],
        top_k=gen_kwargs["top_k"],
        stop_token_ids=gen_kwargs["eos_token_id"],
        repetition_penalty=gen_kwargs["repetition_penalty"],
        n=gen_kwargs["num_return_sequences"],
        return_logprob=gen_kwargs.get("logprobs", 0) > 0,
        stop=gen_kwargs["stop_strings"],
    )


def compare_sampling_params(params1: dict, params2: dict) -> bool:
    # 只比较采样参数的配置
    param_attrs = [
        "temperature",
        "top_p",
        "top_k",
        "max_new_tokens",
        "n",
        "stop_token_ids", 
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "min_p",
        "stop",
        "ignore_eos",
    ]

    # 比较每个采样参数
    for attr in param_attrs:
        if attr in params1 and attr in params2:
            if params1[attr] != params1[attr]:
                print(f"采样参数 {attr} 不同: {params1[attr]} != {params1[attr]}")
                return False
    return True
