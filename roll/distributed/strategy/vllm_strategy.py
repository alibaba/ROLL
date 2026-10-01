import asyncio
import atexit
import copy
import gc
import os
import random
from collections import deque
from typing import Dict, List, Optional
from packaging.version import Version

import torch
import torch.distributed as dist
import ray
from ray.runtime_env import RuntimeEnv
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.nn.utils.rnn import pad_sequence
from transformers import set_seed
import vllm
from vllm import RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import RequestOutputKind, BeamSearchParams
from vllm.inputs import TokensPrompt
from vllm.utils import random_uuid

from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto, list_of_dict_to_dict_of_list
from roll.distributed.strategy.strategy import InferenceStrategy
from roll.distributed.strategy.vllm_topology import resolve_vllm_mp_topology
from roll.third_party.vllm import create_async_llm
from roll.third_party.vllm.compat import call_maybe_await
from roll.utils.functionals import (
    concatenate_input_and_output,
    reduce_metrics,
    gather_unpadded_input_ids,
)
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType, clear_memory
from roll.platforms import current_platform
from roll.utils.constants import RAY_NAMESPACE


logger = get_logger()

# vllm accepts user provided per item ids since vllm==0.10.2,
# see: https://github.com/vllm-project/vllm/pull/23394
SUPPORT_MM_UUIDS = "multi_modal_uuids" in TokensPrompt.__annotations__


def build_tokens_prompt(mm_inputs: Dict) -> TokensPrompt:
    """Build TokensPrompt from infer inputs produced by `DataCollatorWithPaddingForMM`."""
    prompt = TokensPrompt(
        prompt_token_ids=mm_inputs["prompt_token_ids"],
        multi_modal_data=mm_inputs.get("multi_modal_data"),
        mm_processor_kwargs=mm_inputs.get("mm_processor_kwargs"),
    )
    # stable per item ids let vllm hash them instead of the raw media content and reuse its
    # multi-modal processor cache across requests sharing the same items
    mm_uuids = mm_inputs.get("multi_modal_uuids")
    if mm_uuids and SUPPORT_MM_UUIDS:
        prompt["multi_modal_uuids"] = mm_uuids
    return prompt


class VllmMPHeadlessActor:
    async def initialize(self, vllm_config):
        async def run_headless():
            await asyncio.to_thread(
                lambda: asyncio.run(
                    create_async_llm(
                        resource_placement_groups=[], headless=True, **vllm_config
                    )
                )
            )

        self._task = asyncio.create_task(run_headless())
        self._task.add_done_callback(self._on_exit)
        await asyncio.sleep(0)
        if self._task.done():
            await self._task

    @staticmethod
    def _on_exit(task):
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.exception("vLLM mp headless executor exited unexpectedly")
        else:
            logger.error("vLLM mp headless executor stopped unexpectedly")
        os._exit(1)


class VllmStrategy(InferenceStrategy):
    strategy_name = "vllm"

    def __init__(self, worker: Worker):
        super().__init__(worker)

        # Metrics snapshot infrastructure
        self._metrics_snapshots = deque(maxlen=3600)
        self._metrics_snapshot_interval = 1.0  # Snapshot every 1 second
        self._metrics_task = None
        self._headless_workers = []
        atexit.register(self._shutdown_headless_workers)

    def _shutdown_headless_workers(self):
        for worker in self._headless_workers:
            try:
                ray.kill(worker, no_restart=True)
            except Exception:
                pass
        self._headless_workers.clear()

    async def _start_headless_workers(self, node_groups, vllm_config):
        init_refs = []
        for node_offset, node_group in enumerate(node_groups[1:], start=1):
            env_vars = current_platform.get_custom_env_vars()
            env_vars.update(self.worker_config.system_envs)
            env_vars.update(
                {
                    "WORLD_SIZE": str(self.worker.world_size),
                    "RANK": str(self.worker.rank),
                    "LOCAL_RANK": "0",
                    "CLUSTER_NAME": self.worker.cluster_name,
                    "WORKER_NAME": f"{self.worker.worker_name}-headless-{node_offset}",
                    "VLLM_USE_V1": os.environ["VLLM_USE_V1"],
                }
            )
            if "ROLL_LOG_DIR" in os.environ:
                env_vars["ROLL_LOG_DIR"] = os.environ["ROLL_LOG_DIR"]
            gpu_ranks = sorted(placement["gpu_rank"] for placement in node_group)
            current_platform.update_env_vars_for_visible_devices(env_vars, gpu_ranks)

            actor_options = {
                "scheduling_strategy": PlacementGroupSchedulingStrategy(
                    placement_group=node_group[0]["placement_group"]
                ),
                "namespace": RAY_NAMESPACE,
                "runtime_env": RuntimeEnv(env_vars=env_vars),
                "num_cpus": 0.01,
            }
            if current_platform.ray_device_key == "GPU":
                actor_options["num_gpus"] = 0.01
            else:
                actor_options["num_gpus"] = 0
                actor_options["resources"] = {
                    current_platform.ray_device_key: 0.01
                }

            headless_config = copy.deepcopy(vllm_config)
            headless_config["node_rank"] += node_offset
            actor = ray.remote(VllmMPHeadlessActor).options(**actor_options).remote()
            self._headless_workers.append(actor)
            init_refs.append(actor.initialize.remote(headless_config))

        if init_refs:
            await asyncio.gather(*init_refs)

        # Engine stats logging infrastructure
        self._log_stats_interval = 10.0  # Log engine stats every 10 seconds
        self._log_stats_task = None


    def get_free_port_for_rank(self) -> int:
        VLLM_PORT_START = 20000
        PORT_RANGE = 500
        MAX_LOCAL_WORKER_COUNT = 16

        rank = self.worker.rank
        effective_rank = rank % MAX_LOCAL_WORKER_COUNT
        
        range_start = VLLM_PORT_START + effective_rank * PORT_RANGE
        range_end = range_start + PORT_RANGE
        
        for _ in range(PORT_RANGE):
            port = random.randint(range_start, range_end - 1)
            if self.worker.is_port_available(port):
                return port
        
        raise RuntimeError(
            f"Cannot allocate free port for rank {rank} (effective_rank={effective_rank}) in range [{range_start}, {range_end}]"
        )

    async def initialize(self, model_provider):
        set_seed(seed=self.worker.pipeline_config.seed)
        vllm_config = copy.deepcopy(self.worker_config.strategy_args.strategy_config)

        # Apply GDN attention patch for mixed decode/spec-decode bug fix
        # This patches vLLM versions < v0.17.2 that lack the fix
        from roll.third_party.vllm.gdn_patcher import patch_gdn_attention
        patch_gdn_attention()

        # Must explicitly set VLLM_USE_V1 to pass this check: https://github.com/vllm-project/vllm/pull/14972
        os.environ["VLLM_USE_V1"] = str(vllm_config.pop("VLLM_USE_V1", 1))
        self.sleep_level = vllm_config.pop("sleep_level", 1)

        vllm_version = Version(vllm.__version__)
        if vllm_config.get("enable_expert_parallel", False) and vllm_version.release[:2] < (0, 16):
            raise RuntimeError(
                "vLLM expert parallelism with the mp backend requires vLLM 0.16.x or later, "
                f"but found {vllm.__version__}. Upgrade vLLM or disable enable_expert_parallel."
            )

        tensor_parallel_size = vllm_config.get("tensor_parallel_size", 1)
        pipeline_parallel_size = vllm_config.get("pipeline_parallel_size", 1)
        placements = self.worker_config.resource_placement_groups
        data_parallel_size = vllm_config.get("data_parallel_size", 1)
        topology = resolve_vllm_mp_topology(
            worker_rank=self.worker.rank,
            worker_world_size=self.worker.world_size,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            resource_placements=placements,
        )
        node_groups = topology["node_groups"]
        deployment_id = topology["deployment_id"]
        data_parallel_rank = topology["data_parallel_rank"]
        nodes_per_engine = topology["nodes_per_engine"]
        logical_node_count = topology["nnodes"]
        supports_multi_node_mp = vllm_version >= Version("0.11.1")
        uses_legacy_ray_executor = nodes_per_engine > 1 and not supports_multi_node_mp
        if data_parallel_size > 1 and vllm_version < Version("0.11.0"):
            raise RuntimeError("vLLM external data parallelism requires vLLM >= 0.11.0.")
        if uses_legacy_ray_executor:
            vllm_config["distributed_executor_backend"] = "ray"
            logger.info(
                "vLLM %s uses the legacy Ray executor for a TP/PP engine spanning %s nodes.",
                vllm.__version__,
                nodes_per_engine,
            )
        endpoint_key = (
            f"vllm_dp_endpoint:{self.worker.cluster_name}:"
            f"{self.worker.master_port}:{deployment_id}"
        )

        if data_parallel_rank == 0:
            endpoint = {
                "address": self.worker.get_node_ip(),
                "rpc_port": self.worker.get_free_port() if data_parallel_size > 1 else None,
                "master_port": (
                    self.worker.get_free_port()
                    if logical_node_count > 1 and supports_multi_node_mp
                    else None
                ),
                "nodes_per_engine": nodes_per_engine,
            }
            if data_parallel_size > 1:
                await self.worker.shared_storage.put.remote(endpoint_key, endpoint)
        elif data_parallel_size > 1:
            endpoint = None
            timeout_at = asyncio.get_running_loop().time() + self.worker_config.backend_timeout * 60
            while endpoint is None:
                endpoint = await self.worker.shared_storage.get_if_exists.remote(endpoint_key)
                if endpoint is None and asyncio.get_running_loop().time() >= timeout_at:
                    raise TimeoutError(f"Timed out waiting for {endpoint_key}")
                if endpoint is None:
                    await asyncio.sleep(1)

        if endpoint["nodes_per_engine"] != nodes_per_engine:
            raise ValueError(
                "All vLLM external-DP ranks in a deployment must span the same "
                f"number of nodes; rank 0 uses {endpoint['nodes_per_engine']}, "
                f"rank {data_parallel_rank} uses {nodes_per_engine}."
            )

        if data_parallel_size > 1:
            vllm_config.update(
                {
                    "data_parallel_rank": data_parallel_rank,
                    "data_parallel_address": endpoint["address"],
                    "data_parallel_rpc_port": endpoint["rpc_port"],
                }
            )
            logger.info(
                f"VllmStrategy {self.worker.cluster_name} enable external data parallel "
                f"deployment_id={deployment_id}, data_parallel_size={data_parallel_size}, "
                f"data_parallel_rank={data_parallel_rank}"
            )

        # vLLM 0.11.0 supports external DP, but not MP logical-node args.
        if logical_node_count > 1 and supports_multi_node_mp:
            vllm_config.update(
                {
                    "nnodes": logical_node_count,
                    "node_rank": topology["node_rank"],
                    "master_addr": endpoint["address"],
                    "master_port": endpoint["master_port"],
                }
            )
            logger.info(
                "vLLM mp topology: deployment_id=%s, DP=%s, TP=%s, PP=%s, "
                "nodes_per_engine=%s, nnodes=%s, node_rank=%s",
                deployment_id,
                data_parallel_size,
                tensor_parallel_size,
                pipeline_parallel_size,
                nodes_per_engine,
                logical_node_count,
                vllm_config["node_rank"],
            )

        if vllm_config.get("enable_expert_parallel", False):
            logger.info(
                f"vLLM expert parallel enabled: TP={tensor_parallel_size}, "
                f"DP={data_parallel_size}, EP={tensor_parallel_size * data_parallel_size}."
            )

        if self.worker_config.model_args.dtype == "fp32":
            dtype = "float32"
        elif self.worker_config.model_args.dtype == "fp16":
            dtype = "float16"
        elif self.worker_config.model_args.dtype == "bf16":
            dtype = "bfloat16"
        else:
            dtype = "auto"

        default_compilation_config = {
            "pass_config": {"fuse_allreduce_rms": False},
        }
        vllm_config.update(
            {
                "model": self.worker_config.model_args.model_name_or_path,
                "dtype": dtype,
                "enforce_eager": vllm_config.get("enforce_eager", False),
                "trust_remote_code": True,
                "seed": self.worker.pipeline_config.seed,
                "disable_custom_all_reduce": vllm_config.get(
                    "disable_custom_all_reduce", True
                ),  # potentially hangs in tp>1
                "enable_prefix_caching": vllm_config.get("enable_prefix_caching", True),
                "load_format": vllm_config.get("load_format", "dummy"),  # use model update passed value
                "max_num_batched_tokens": vllm_config.get("max_num_batched_tokens", 8192), # use default value of LLM class usage context
                "compilation_config": vllm_config.get("compilation_config", default_compilation_config), # disable flashiner fuse_allreduce_rms
            }
        )

        self.is_lora = self.worker_config.model_args.lora_target is not None
        if self.is_lora:
            lora_kwargs = {
                "enable_lora": True,
                "max_loras": 1,
                "max_lora_rank": self.worker_config.model_args.lora_rank,
            }
            vllm_config.update(lora_kwargs)
            vllm_config["load_format"] = "auto"  # enables vLLM to load the base model for add_lora

        # Router replay (R3): mirror sglang_strategy.
        self.enable_rollout_routing_replay = self.worker_config.router_replay.mode != "disable"
        if self.enable_rollout_routing_replay:
            vllm_config["enable_return_routed_experts"] = True
            logger.info(
                f"{self.enable_rollout_routing_replay=} and {self.worker_config.router_replay.mode=}"
            )

        logger.info(f"vllm_config: {vllm_config}")
        assert not dist.is_initialized()

        # Can not set VLLM_PORT explicitly in DP. Each call of get_engine_client_zmq_addr in
        # DPCoordinator will return the same port, which will cause port conflict.
        # https://github.com/vllm-project/vllm/blob/releases/v0.10.0/vllm/v1/engine/coordinator.py#L72
        if not data_parallel_size > 1:
            # set VLLM_PORT to avoid port conflict applied by vllm
            vllm_port = self.get_free_port_for_rank()
            logger.info(f"Allocated vllm_port {vllm_port} for rank {self.worker.rank}")
            os.environ["VLLM_PORT"] = str(vllm_port)

        try:
            if not uses_legacy_ray_executor:
                await self._start_headless_workers(node_groups, vllm_config)
            self.model = await create_async_llm(
                resource_placement_groups=placements, **vllm_config
            )
        except Exception:
            self._shutdown_headless_workers()
            raise


        self.tokenizer = await call_maybe_await(self.model.get_tokenizer)

        assert self.worker.rank_info.dp_rank == self.worker.rank
        assert self.worker.rank_info.dp_size == self.worker.world_size

        self.is_model_in_gpu = True

        try:
            from vllm.v1.metrics.reader import get_metrics_snapshot
            self._metrics_task = asyncio.create_task(self._collect_metrics_snapshot())
        except Exception as e:
            logger.warning(f"Failed to create metrics collector task: {e}")

        self._log_stats_task = asyncio.create_task(self._log_stats_periodically())

    def op_compute_log_probs(self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        vllm实现compute log probs在这里实现即可
        """
        pass

    async def generate(self, batch: DataProto, generation_config) -> torch.Tensor:
        # Check if beam search is requested
        if self._should_use_beam_search(generation_config):
            return await self._generate_with_beam_search(batch, generation_config)
        else:
            return await self._generate_standard(batch, generation_config)

    def _should_use_beam_search(self, generation_config) -> bool:
        """Check if beam search should be used based on generation_config."""
        return generation_config.get("num_beams", 1) > 1 or generation_config.get("use_beam_search", False)

    async def _generate_standard(self, batch: DataProto, generation_config: Dict) -> torch.Tensor:
        """Standard generate method for non-beam search cases."""
        sampling_params = SamplingParams(**create_sampling_params_for_vllm(gen_kwargs=generation_config))

        input_ids = batch.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = batch.batch["attention_mask"]  # left-padded attention_mask

        if "multi_modal_data" in batch.non_tensor_batch:
            prompts = [build_tokens_prompt(data) for data in batch.non_tensor_batch["multi_modal_data"]]
        else:
            prompts = [TokensPrompt(prompt_token_ids=prompt)
                for prompt in gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)
            ]

        lora_request = None
        if self.is_lora:
            lora_int_ids = list(await self.model.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_request = LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="dummy_lora_path")

        async def _generate(prompt):
            request_id = random_uuid()
            result_generator = self.model.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
            )
            output: Optional[RequestOutput] = None
            async for result in result_generator:
                output = result
            return output

        vllm_outputs = await asyncio.gather(*[_generate(prompt) for prompt in prompts])

        # (bs * num_return_sequences, max_response_len)
        output_ids = gather_outputs_to_pad_tensor(
            request_outputs=vllm_outputs,
            pad_token_id=self.tokenizer.pad_token_id,
            device=input_ids.device,
        )

        # (bs * num_return_sequences, input_len + max_response_len)
        output = concatenate_input_and_output(
            input_ids=input_ids, output_ids=output_ids, num_return_sequences=sampling_params.n
        )

        return output

    async def _generate_with_beam_search(self, batch: DataProto, generation_config: Dict) -> torch.Tensor:
        """Generate using beam search method."""
        # Create beam search parameters
        beam_params = BeamSearchParams(
            beam_width=generation_config.get("num_beams", 1),
            max_tokens=generation_config.get("max_new_tokens", 50),
            temperature=generation_config.get("temperature", 0.0),
            ignore_eos=generation_config.get("ignore_eos", False),
            length_penalty=generation_config.get("length_penalty", 1.0),
            include_stop_str_in_output=generation_config.get("include_stop_str_in_output", False),
        )

        input_ids = batch.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = batch.batch["attention_mask"]  # left-padded attention_mask

        # Prepare prompts for beam_search
        if "multi_modal_data" in batch.non_tensor_batch:
            # For multimodal data, we need to handle it differently
            # This is a simplified approach - may need refinement based on actual multimodal format
            prompts = [build_tokens_prompt(data) for data in batch.non_tensor_batch["multi_modal_data"]]
        else:
            # Convert to token lists format expected by beam_search
            token_lists = gather_unpadded_input_ids(
                input_ids=input_ids, attention_mask=attention_mask
            )
            # Convert to TokensPrompt format expected by vLLM beam_search
            prompts = [{"prompt_token_ids": token_ids} for token_ids in token_lists]

        # Call beam_search method
        async def _beam_search(prompt):
            request_id = random_uuid()
            result_generator = self.model.beam_search(
                prompt=prompt,
                request_id=request_id,
                params=beam_params,
            )
            output: Optional[RequestOutput] = None
            async for result in result_generator:
                output = result
            return output

        beam_search_outputs = await asyncio.gather(*[_beam_search(prompt) for prompt in prompts])

        generated_token_ids = []
        for request_output in beam_search_outputs:
            for completion_output in request_output.outputs:
                generated_tokens = completion_output.token_ids
                generated_token_ids.append(torch.tensor(generated_tokens, device=input_ids.device))

        # Pad the sequences
        output_ids = pad_sequence(generated_token_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        # Concatenate input and output
        output = concatenate_input_and_output(
            input_ids=input_ids,
            output_ids=output_ids,
            num_return_sequences=beam_params.beam_width
        )

        return output

    async def generate_request(self, payload: Dict):
        if "multi_modal_data" in payload:
            prompt = build_tokens_prompt(payload["multi_modal_data"])
        else:
            prompt = TokensPrompt(prompt_token_ids=payload["input_ids"])

        lora_request = None
        if self.is_lora:
            lora_int_ids = list(await self.model.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_request = LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="dummy_lora_path")

        result_generator = self.model.generate(
            prompt=prompt,
            sampling_params=SamplingParams(**payload["sampling_params"]),
            request_id=payload["rid"],
            lora_request=lora_request,
        )
        output: Optional[RequestOutput] = None
        # vLLM support partial rollout in v1 from 0.10.1, and will return finished output
        # with finish_reason setted no matter what RequestOutputKind is.
        # For compatibility, the following except block are only for v0 and older version of v1.
        try:
            async for result in result_generator:
                output = result
        except asyncio.CancelledError:
            if output is None:
                return {"finish_reasons": ["abort"]}

        output_token_ids, finish_reasons, logprobs = [], [], []
        routed_experts = []
        if "multi_modal_data" in payload:
            # Use expanded_prompt_len (vision tokens expanded) for R3 slicing; prompt_token_ids are text-only.
            prompt_len = payload.get("expanded_prompt_len", len(payload["multi_modal_data"]["prompt_token_ids"]))
        else:
            prompt_len = len(payload["input_ids"])
        for completion_output in output.outputs:
            output_token_ids.append(completion_output.token_ids)
            # For compatibility, older version may return unfinished result, set finish_reason of those to 'abort'.
            finish_reason = "abort" if completion_output.finish_reason is None else completion_output.finish_reason
            finish_reasons.append(finish_reason)
            if completion_output.logprobs is not None:
                logprobs.append(
                    [
                        float(lps[token_id].logprob)
                        for token_id, lps in zip(completion_output.token_ids, completion_output.logprobs)
                    ]
                )

            # Router replay (R3): R3 training consumes the next-token-aligned prompt_len + gen_len - 1 rows.
            re = getattr(completion_output, "routed_experts", None)
            if re is not None:
                re = torch.as_tensor(re)
                expected_rows = prompt_len + len(completion_output.token_ids) - 1
                assert re.size(0) >= expected_rows, (
                    f"routed_experts rows {re.size(0)} < expected {expected_rows} "
                    f"(rid={payload.get('rid')}, prompt_len={prompt_len}, "
                    f"gen_len={len(completion_output.token_ids)}, finish_reason={finish_reason})"
                )
                re = re[:expected_rows]
            routed_experts.append(re)

        result = {
            "output_token_ids": output_token_ids,
            "finish_reasons": finish_reasons,
            "output_logprobs": logprobs,
        }

        # Mirroring sglang_strategy.
        if any(re is not None for re in routed_experts):
            result["routed_experts"] = routed_experts

        # Add speculative metrics if available
        spec_metrics = self.get_speculative_metrics()
        if spec_metrics:
            result["metrics"] = spec_metrics

        return result

    async def abort_requests(self, request_ids):
        for id in request_ids:
            await self.model.abort(request_id=id)

    # offload/reload 接口
    async def load_states(self, *args, **kwargs):
        await self.model.reset_prefix_cache()
        if not self.is_model_in_gpu:
            await self.model.load_states()
            self.is_model_in_gpu = True

    async def offload_states(self, include=None, non_blocking=False):
        await self.model.reset_prefix_cache()
        if include is None or OffloadStateType.model_params in include:
            if self.is_model_in_gpu and self.worker.pipeline_config.is_actor_infer_colocated:
                await self.model.offload_states(self.sleep_level)
                self.is_model_in_gpu = False
        clear_memory()
    
    async def process_weights_after_loading(self,*args, **kwargs):
        await self.model.process_weights_after_loading()

    # 参数同步相关接口
    async def setup_collective_group(self, master_address, master_port, rank_offset, world_size, group_name, backend=None):
        logger.info(f"setup_collective_group {group_name=}")
        backend = backend if backend is not None else current_platform.communication_backend
        await self.model.setup_collective_group(master_address, master_port, rank_offset, world_size, group_name, backend)

    async def broadcast_parameter(self, names, dtypes, shapes, group_name, is_lora=False):
        await self.model.broadcast_parameter(names, dtypes, shapes, group_name, is_lora)

    async def update_parameter_in_bucket(self, serialized_named_tensors, is_lora=False):
        await self.model.update_parameter_in_bucket(serialized_named_tensors, is_lora)

    async def add_lora(self, peft_config):
        target = self.worker_config.model_args.lora_target
        config = dict(peft_config)
        config["target_modules"] = target if isinstance(target, str) else list(target)
        acknowledgements = await self.model.add_lora(config)
        expected = self.worker_config.num_gpus_per_worker
        if (not isinstance(acknowledgements, (list, tuple))
                or len(acknowledgements) != expected
                or any(added is not True for added in acknowledgements)):
            raise RuntimeError(
                f"LoRA admission requires {expected} successful rank acknowledgements; "
                f"received {acknowledgements!r}"
            )
        return acknowledgements

    # Mapping from raw vLLM metric names to internal keys
    _VLLM_METRIC_MAP = {
        "vllm:kv_cache_usage_perc": "vllm/kv_cache_usage_perc_max",
        "vllm:num_requests_waiting": "vllm/num_requests_waiting_max",
        "vllm:num_preemptions": "vllm/num_preemptions_max",
        "vllm:spec_decode_num_drafts": "vllm/spec_decode_num_drafts",
        "vllm:spec_decode_num_draft_tokens": "vllm/spec_decode_num_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens": "vllm/spec_decode_num_accepted_tokens",
    }

    async def _collect_metrics_snapshot(self):
        """Collect metrics snapshots periodically in a background task."""
        from vllm.v1.metrics.reader import get_metrics_snapshot
        while True:
            raw_metrics = get_metrics_snapshot()
            snapshot = {key: [] for key in self._VLLM_METRIC_MAP.values()}
            for metric in raw_metrics:
                mapped_key = self._VLLM_METRIC_MAP.get(metric.name)
                if mapped_key:
                    snapshot[mapped_key].append(metric.value)
            self._metrics_snapshots.append(snapshot)
            await asyncio.sleep(self._metrics_snapshot_interval)

    async def _log_stats_periodically(self):
        while True:
            await asyncio.sleep(self._log_stats_interval)
            # Skip while the engine is offloaded/asleep, there is nothing to report then.
            if not self.is_model_in_gpu:
                continue
            try:
                await self.model.do_log_stats()
            except Exception as e:
                logger.warning(f"Failed to log engine stats: {e}")

    def get_metrics(self, metric_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get aggregated metrics for the time interval since last call.

        Args:
            metric_names: Optional list of specific metric names to filter

        Returns:
            Dictionary of metric names to aggregated values
        """
        if not self._metrics_snapshots:
            return {}
        metrics_snapshots = list_of_dict_to_dict_of_list(self._metrics_snapshots)
        self._metrics_snapshots.clear()
        return reduce_metrics(metrics_snapshots)

    def get_speculative_metrics(self) -> Dict[str, float]:
        """Get speculative decoding metrics in a format aligned with SGLang."""
        metrics = self.get_metrics()
        draft_tokens = metrics.get('vllm/spec_decode_num_draft_tokens', 0)
        accepted_tokens = metrics.get('vllm/spec_decode_num_accepted_tokens', 0)
        draft_count = metrics.get('vllm/spec_decode_num_drafts', 0)
        if draft_tokens == 0:
            return {}
        return {
            'spec_draft_token_num': draft_tokens,
            'spec_accept_token_num': accepted_tokens,
            'spec_accept_rate': accepted_tokens / draft_tokens if draft_tokens > 0 else 0.0,
            'spec_accept_length': 1 + (accepted_tokens / draft_count) if draft_count > 0 else 1.0,
        }


def gather_outputs_to_pad_tensor(request_outputs: List["RequestOutput"], pad_token_id, device=None) -> torch.Tensor:
    if device is None:
        device = current_platform.device_type
    token_ids_list_of_lists = [
        torch.tensor(completion_output.token_ids, device=device)
        for request_output in request_outputs
        for completion_output in request_output.outputs
    ]
    output_tensor = pad_sequence(token_ids_list_of_lists, batch_first=True, padding_value=pad_token_id)
    return output_tensor


def create_sampling_params_for_vllm(gen_kwargs, collect_unfinished=False):
    # TODO vLLM support partial rollout in v1 from 0.10.1, and do not need to set RequestOutputKind to CUMULATIVE
    output_kind = RequestOutputKind.CUMULATIVE if collect_unfinished else RequestOutputKind.FINAL_ONLY
    return dict(
        max_tokens=gen_kwargs["max_new_tokens"],
        temperature=gen_kwargs["temperature"],
        top_p=gen_kwargs["top_p"],
        top_k=gen_kwargs["top_k"],
        stop_token_ids=gen_kwargs["eos_token_id"],
        repetition_penalty=gen_kwargs["repetition_penalty"],
        n=gen_kwargs["num_return_sequences"],
        stop=gen_kwargs["stop_strings"],
        logprobs=gen_kwargs.get("logprobs", 0),
        output_kind=output_kind,
        include_stop_str_in_output=gen_kwargs.get("include_stop_str_in_output", True),
        seed=gen_kwargs.get("seed"),
    )
