import asyncio
import itertools
import math
import time
import uuid
import httpx
import weakref
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Set
from urllib.parse import quote

import ray

from roll.distributed.executor.cluster import Cluster
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto
from roll.configs.base_config import RouterArguments
from roll.models.model_providers import default_tokenizer_provider
from roll.utils.functionals import gather_unpadded_input_ids
from roll.utils.checkpoint_manager import download_model
from roll.utils.logging import get_logger


logger = get_logger()

def is_report_data_finished(data: DataProto) -> bool:
    finish_reasons = data.meta_info.get("finish_reasons", [])
    assert isinstance(finish_reasons, list), f"{finish_reasons}"
    assert all(isinstance(finish_reason, str) for finish_reason in finish_reasons), f"{finish_reasons}"
    return not any(finish_reason == "abort" for finish_reason in finish_reasons)

def raise_for_status(response: httpx.Response):
    if not response.is_success:
        try:
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(str(e))

async def wait_sglang_router_ready(router_process, url):
    async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
        for attempt in range(60):
            await asyncio.sleep(1)
            try:
                response = await client.get(url)
                if response.status_code in [200, 404]:
                    break
                else:
                    logger.info(f"Waiting for sglang router {url} to ready ({attempt=}) (status={response.status_code})...")
                    raise_for_status(response)
                assert router_process.is_alive()
            except httpx.ConnectError:
                logger.info(f"Waiting for sglang router {url} to start ({attempt=})...")

async def wait_sglang_router_workflow(router_url, expected):
    expected = set(expected)
    async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
        while True:
            await asyncio.sleep(3)
            response = await client.get(f"{router_url}/workers")
            raise_for_status(response)
            response = response.json()
            if {worker["url"] for worker in response["workers"]} == expected:
                break
            logger.info(f"Waiting for sglang router worker workflow {router_url} ready, "
                        f"{expected=}, current count={response['total']}, workers={response['workers']} ...")

class RouterManager:
    def __init__(self, actor_cluster: Cluster, router_args: RouterArguments, num_gpus_per_node: int):
        self.actor_cluster = actor_cluster
        self.workers = actor_cluster.workers

        self.strategy_name = actor_cluster.worker_config.strategy_args.strategy_name 
        self.model_path = download_model(actor_cluster.worker_config.model_args.model_name_or_path)
        self.tokenizer = default_tokenizer_provider(model_args=actor_cluster.worker_config.model_args)

        router_name = router_args.router_name
        if router_name == "PromptAffinityRouter":
            self.router_cls = PromptAffinityRouter
        elif router_name == "EnvAffinityRouter":
            self.router_cls = EnvAffinityRouter
        else:
            self.router_cls = SglangRouter
        assert self.router_cls is not SglangRouter or self.strategy_name == "sglang"
        assert (self.router_cls is SglangRouter) == (actor_cluster.worker_config.strategy_args.strategy_config.get("grpc_mode", None) is not None) # xnor
        logger.info(f"RouterManager use router {self.router_cls.__name__}")
        self.router: Router = self.router_cls(router_manager=self, workers=self.workers, model_path=self.model_path, router_args=router_args)

        self.inflight_requests = set()
        self.need_suspend = False
        self.need_shutdown = False
        self.suspend_notifier = asyncio.Event()
        self.empty_notifier = asyncio.Event()

        self.partial_gpu_manager = PartialGPUManager(actor_cluster=actor_cluster, router=self.router, num_gpus_per_node=num_gpus_per_node)

    async def initialize(self):
        await self.router.initialize()

    def router_meta(self):
        return {
            "strategy_name": self.strategy_name,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "sglang_router": self.router_cls is SglangRouter,
            "router_ip": self.router.router_ip if self.router_cls is SglangRouter else None,
            "router_port": self.router.router_port if self.router_cls is SglangRouter else None,
            "worker_urls": self.router.worker_urls if self.router_cls is SglangRouter else None,
        }

    @classmethod
    def create_client_sync(cls, self) -> "RouterClient":
        if isinstance(self, ray.actor.ActorHandle):
            meta = ray.get(self.router_meta.remote())
            proxy_cls = RayProxy
        elif isinstance(self, cls):
            meta = self.router_meta()
            proxy_cls = InprocProxy
        else:
            raise ValueError(f"self {self} is not a ray actor or RouterManager")

        proxy = proxy_cls(self)
        if meta["sglang_router"]:
            proxy = SglangProxy(proxy, meta)
        return RouterClient(proxy, meta)

    @classmethod
    async def create_client(cls, self) -> "RouterClient":
        """
        self may be a ray actor or normal class.
        """
        if isinstance(self, ray.actor.ActorHandle):
            meta = await self.router_meta.remote()
            proxy_cls = RayProxy
        elif isinstance(self, cls):
            meta = self.router_meta()
            proxy_cls = InprocProxy
        else:
            raise ValueError(f"self {self} is not a ray actor or RouterManager")

        proxy = proxy_cls(self)
        if meta["sglang_router"]:
            proxy = SglangProxy(proxy, meta)
        return RouterClient(proxy, meta)

    async def generate_request(self, payload, request_id, uid):
        return await self.router.generate_request(payload=payload, request_id=request_id, uid=uid)

    async def abort_requests(self, request_ids, uid):
        return await self.router.abort_requests(request_ids, uid)

    async def abort_all(self):
        logger.info(f"abort all requests, remaining requests: {len(self.inflight_requests)}")
        return await self.router.abort_all(list(self.inflight_requests))

    async def on_send_request(self, request_id) -> bool:
        while self.need_suspend:
            await self.suspend_notifier.wait()
        if self.need_shutdown:
            return False
        self.inflight_requests.add(request_id)
        return True

    async def on_request_routed(self, request_id):
        self.inflight_requests.remove(request_id)
        self.empty_notifier.set()

    def suspend(self):
        """
        Suspend all running requests.

        All following call of generate will be blocked until resume.
        """
        if self.need_suspend:
            return
        self.suspend_notifier.clear()
        self.need_suspend = True

    def resume(self):
        if not self.need_suspend:
            return
        self.need_suspend = False
        self.suspend_notifier.set()

    async def shutdown(self):
        self.need_shutdown = True
        await self.abort_all()
        self.resume()
        await self.wait_complete()

    async def wait_complete(self):
        """
        Wait until all running requests are finished (no matter whether suspended or not).
        """
        logger.info(f"RouterManager: wait all requests complete {self.inflight_requests=}")
        while len(self.inflight_requests) > 0:
            self.empty_notifier.clear()
            await self.empty_notifier.wait()
        logger.info(f"RouterManager: all requests completed")

    def size(self):
        return len(self.inflight_requests)

    async def shrink_workers(self, target_gpus: List[int]) -> Dict[str, Any]:
        logger.info(f"RouterManager shrink_workers {target_gpus=}")
        return await self.partial_gpu_manager.shrink_workers(target_gpus)

    async def expand_workers(self, target_gpus: List[int], skip_load: bool = False) -> Dict[str, Any]:
        logger.info(f"RouterManager expand_workers {target_gpus=}")
        return await self.partial_gpu_manager.expand_workers(target_gpus, skip_load)

class PartialGPUManager:
    def __init__(self, actor_cluster, router, num_gpus_per_node: int):
        self.infer_cluster = actor_cluster
        self.router = router
        self.num_gpus_per_node = num_gpus_per_node

    def _get_gpus_for_dp_rank(self, dp_rank: int) -> List[int]:
        """Map DP rank to GPU IDs using cluster's device info.

        Args:
            dp_rank: Data parallel rank index (0 to dp_size-1)

        Returns:
            List of GPU IDs used by this DP rank's workers

        Example:
            # Pure DP: rank == dp_rank
            # DP rank 0 uses GPUs [0], DP rank 1 uses GPUs [1], etc.
            gpus = self._get_gpus_for_dp_rank(dp_rank=0)
            # Returns: [0]
        """
        # In agentic pipeline (pure DP): rank == dp_rank, so directly access rank2devices
        devices_info = self.infer_cluster.rank2devices[dp_rank]

        # Extract GPU IDs: gpu_id = node_rank * num_gpus_per_node + gpu_rank
        gpu_ids = []
        for device in devices_info:
            gpu_id = device["node_rank"] * self.num_gpus_per_node + device["gpu_rank"]
            gpu_ids.append(gpu_id)

        return sorted(set(gpu_ids))  # Remove duplicates and sort

    def _validate_target_gpus(self, target_gpus: List[int], mode: str) -> None:
        """Validate target_gpus input for shrink/expand operations.

        Args:
            target_gpus: List of GPU IDs to free (shrink) or restore (expand)
            mode: Operation mode ("shrink" or "expand")

        Raises:
            ValueError: If target_gpus is empty, has duplicates, or mode is invalid

        Example:
            self._validate_target_gpus([4, 5, 6, 7], mode="shrink")
            # Validates successfully

            self._validate_target_gpus([], mode="shrink")
            # Raises: ValueError("[shrink] target_gpus cannot be empty")

            self._validate_target_gpus([4, 4, 5], mode="expand")
            # Raises: ValueError("[expand] target_gpus has duplicates: [4, 4, 5]")
        """
        # VAL: VAL_NON_EMPTY
        if not target_gpus:
            raise ValueError(f"[{mode}] target_gpus cannot be empty")

        # VAL: VAL_NO_DUPLICATES
        if len(target_gpus) != len(set(target_gpus)):
            raise ValueError(f"[{mode}] target_gpus has duplicates: {target_gpus}")

        if mode not in ("shrink", "expand"):
            raise ValueError(f"Invalid mode: {mode}")

    def _validate_calculated_ranks(self, ranks: List[int], mode: str) -> None:
        """Validate calculated DP ranks against current active_dp_ranks state.

        Args:
            ranks: List of DP ranks calculated from target_gpus
            mode: Operation mode ("shrink" or "expand")

        Raises:
            ValueError: If ranks is empty, contains out-of-range values,
                       or violates state consistency (shrink: must be active,
                       expand: must be inactive)

        Example:
            # Shrink validation
            self.active_dp_ranks = {0, 1, 2, 3}
            self._validate_calculated_ranks([2, 3], mode="shrink")
            # Validates successfully (ranks 2, 3 are active)

            self._validate_calculated_ranks([4], mode="shrink")
            # Raises: ValueError("[shrink] DP rank 4 not active")

            # Expand validation
            self.active_dp_ranks = {0, 1}
            self._validate_calculated_ranks([2, 3], mode="expand")
            # Validates successfully (ranks 2, 3 are inactive)

            self._validate_calculated_ranks([0], mode="expand")
            # Raises: ValueError("[expand] DP rank 0 already active")
        """
        # VAL: VAL_NON_EMPTY
        if not ranks:
            raise ValueError(f"[{mode}] Calculated ranks list is empty")

        # VAL: VAL_INT_RANGE
        for dp_rank in ranks:
            if not (0 <= dp_rank < self.infer_cluster.world_size):
                raise ValueError(f"[{mode}] DP rank {dp_rank} out of range [0, {self.infer_cluster.world_size})")

        # AST: State consistency

        # TODO: fix this validation and move to EnvAffinityRouter
        # for dp_rank in ranks:
        #     if dp_rank not in self.active_dp_ranks:
        #         raise ValueError(f"DP rank {dp_rank} not active {mode=}")

    async def shrink_workers(self, target_gpus: List[int]) -> Dict[str, Any]:
        """Complete atomic shrink operation: validate → rebalance → offload → update routing.

        Orchestrates the full worker shrink process:
        1. Validates target_gpus input
        2. Calculates DP ranks to offload based on GPU overlap
        3. Validates calculated ranks against active state
        4. Do shrink:
           - Rebalances routing (aborts requests on shrinking workers)
           - Offloads model states from shrinking workers
        5. Returns metrics for monitoring

        Args:
            target_gpus: GPU IDs to free (e.g., [4, 5, 6, 7] to free second half of 8 GPUs)

        Returns:
            Metrics dict containing:
                - "aborted": Number of requests aborted during rebalancing
                - "remapped": Number of src_ranks remapped (cleared from routing)
                - "shrink_duration_ms": Total operation time in milliseconds
                - "offload_ranks": List of DP ranks that were offloaded

        Raises:
            ValueError: If target_gpus invalid (empty, duplicates) or
                       calculated ranks invalid (not active, out of range)
            RuntimeError: If rebalance or offload operations fail

        Example:
            # Shrink to free GPUs [4, 5, 6, 7] (second half of 8-GPU setup)
            result = await scheduler.shrink_workers([4, 5, 6, 7])
            # Returns: {"aborted": 10, "remapped": 5, "shrink_duration_ms": 2340.5, "offload_ranks": [2, 3]}

        Side Effects:
            - Updates active_dp_ranks (removes offload_ranks)
            - Aborts in-flight requests on shrinking workers
            - Clears src_rank mappings for remapped environments
            - Offloads model states from shrinking workers to CPU
        """
        start_time = time.time()

        # VAL: VAL_NON_EMPTY, VAL_NO_DUPLICATES
        self._validate_target_gpus(target_gpus, mode="shrink")
        # Calculate DP ranks to offload
        target_gpus = set(target_gpus)
        offload_ranks = [dp for dp in range(self.infer_cluster.world_size)
                         if set(self._get_gpus_for_dp_rank(dp)).intersection(target_gpus)]

        # VAL: VAL_NON_EMPTY, state consistency check
        self._validate_calculated_ranks(offload_ranks, mode="shrink")

        result = await self.router.rebalance_on_shrink(offload_ranks)

        # release the lock before blocking offload so that active dp rank can work immediately
        # Offload states from target workers
        offload_refs = self.infer_cluster.offload_states_partial(offload_ranks, blocking=False)
        await asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in offload_refs])

        return {**result, "shrink_duration_ms": (time.time() - start_time) * 1000,
                "offload_ranks": offload_ranks}

    async def expand_workers(self, target_gpus: List[int], skip_load: bool = False) -> Dict[str, Any]:
        """Complete atomic expand operation: validate → load → rebalance → update routing.

        Orchestrates the full worker expand process:
        1. Validates target_gpus input
        2. Calculates DP ranks to restore based on GPU overlap
        3. Validates calculated ranks against active state (skip if skip_load=True)
        4. Do expand:
           - Loads model states on expanding workers (skip if skip_load=True)
           - Rebalances routing (proportionally redistributes requests)
        5. Returns metrics for monitoring

        Args:
            target_gpus: GPU IDs to restore (e.g., [4, 5, 6, 7] to restore second half of 8 GPUs)
            skip_load: If True, skip model loading and validation (use when model_update already loaded states).
                      This only updates active_dp_ranks to restore routing state without re-loading models.

        Returns:
            Metrics dict containing:
                - "aborted": Number of requests aborted during rebalancing (proportional redistribution)
                - "remapped": Number of src_ranks remapped (cleared from routing)
                - "expand_duration_ms": Total operation time in milliseconds
                - "load_ranks": List of DP ranks that were restored

        Raises:
            ValueError: If target_gpus invalid (empty, duplicates) or
                       calculated ranks invalid (already active, out of range)
            RuntimeError: If load or rebalance operations fail

        Example:
            # Expand to restore GPUs [4, 5, 6, 7] (second half of 8-GPU setup)
            result = await scheduler.expand_workers([4, 5, 6, 7])
            # Returns: {"aborted": 3, "remapped": 3, "expand_duration_ms": 1850.2, "load_ranks": [2, 3]}

            # After model_update already loaded states to all GPUs, just restore routing:
            result = await scheduler.expand_workers([4, 5, 6, 7], skip_load=True)

        Side Effects:
            - Updates active_dp_ranks (adds load_ranks)
            - Loads model states from CPU to expanding workers (unless skip_load=True)
            - Aborts some requests from old workers for proportional rebalancing
            - Clears src_rank mappings for rebalanced environments (will route to new workers)
        """
        start_time = time.time()

        # VAL: VAL_NON_EMPTY, VAL_NO_DUPLICATES
        self._validate_target_gpus(target_gpus, mode="expand")

        # Calculate DP ranks to restore
        target_gpus = set(target_gpus)
        load_ranks = [dp for dp in range(self.infer_cluster.world_size)
                      if set(self._get_gpus_for_dp_rank(dp)).issubset(target_gpus)]

        # VAL: VAL_NON_EMPTY, state consistency check
        # Skip validation when skip_load=True because ranks may already be "active" in cluster
        # (model states loaded by model_update) but not tracked in active_dp_ranks yet
        if not skip_load:
            self._validate_calculated_ranks(load_ranks, mode="expand")
            load_refs = self.infer_cluster.load_states_partial(load_ranks, blocking=False)
            await asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in load_refs])

        result = await self.router.rebalance_on_expand(load_ranks)

        return {**result, "expand_duration_ms": (time.time() - start_time) * 1000,
                "load_ranks": load_ranks}

class RouterProxy:
    """
    Proxy to RouterManager
    """
    @abstractmethod
    async def generate_request(self, payload, request_id, uid):
        pass

    @abstractmethod
    async def on_send_request(self, request_id):
        pass

    @abstractmethod
    async def on_request_routed(self, request_id):
        pass

    def generate_request_sync(self, payload, request_id, uid):
        raise NotImplementedError

    def on_send_request_sync(self, request_id):
        raise NotImplementedError

    def on_request_routed_sync(self, request_id):
        raise NotImplementedError

class InprocProxy(RouterProxy):
    def __init__(self, router_manager: RouterManager):
        self.router_manager = router_manager

    async def generate_request(self, payload, request_id, uid):
        return await self.router_manager.generate_request(payload=payload, request_id=request_id, uid=uid)

    async def on_send_request(self, request_id):
        return await self.router_manager.on_send_request(request_id)

    async def on_request_routed(self, request_id):
        return await self.router_manager.on_request_routed(request_id)

class RayProxy(RouterProxy):
    def __init__(self, router_manager: RouterManager):
        self.router_manager = router_manager

    async def generate_request(self, payload, request_id, uid):
        return await self.router_manager.generate_request.remote(payload=payload, request_id=request_id, uid=uid)

    async def on_send_request(self, request_id):
        return await self.router_manager.on_send_request.remote(request_id)

    async def on_request_routed(self, request_id):
        return await self.router_manager.on_request_routed.remote(request_id)

    def generate_request_sync(self, payload, request_id, uid):
        return ray.get(self.router_manager.generate_request.remote(payload=payload, request_id=request_id, uid=uid))

    def on_send_request_sync(self, request_id):
        return ray.get(self.router_manager.on_send_request.remote(request_id))

    def on_request_routed_sync(self, request_id):
        return ray.get(self.router_manager.on_request_routed.remote(request_id))

class SglangProxy(RouterProxy):
    def __init__(self, proxy: RouterProxy, router_meta):
        self.proxy = proxy
        self.router_ip = router_meta["router_ip"]
        self.router_port = router_meta["router_port"]
        self.worker_urls = router_meta["worker_urls"]
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(None))
        self.client_sync = httpx.Client(timeout=httpx.Timeout(None))

    async def generate_request(self, payload, request_id, uid):
        from roll.distributed.strategy.sglang_strategy import postprocess_generate
        assert "multi_modal_data" not in payload
        url = f"http://{self.router_ip}:{self.router_port}/generate"
        response = await self.client.post(url, json=payload)
        raise_for_status(response)
        response = response.json()
        response = response if isinstance(response, list) else [response]
        return postprocess_generate(response)

    async def on_send_request(self, request_id):
        return await self.proxy.on_send_request(request_id)

    async def on_request_routed(self, request_id):
        return await self.proxy.on_request_routed(request_id)

    def generate_request_sync(self, payload, request_id, uid):
        from roll.distributed.strategy.sglang_strategy import postprocess_generate
        assert "multi_modal_data" not in payload
        url = f"http://{self.router_ip}:{self.router_port}/generate"
        response = self.client_sync.post(url, json=payload)
        raise_for_status(response)
        response = response.json()
        response = response if isinstance(response, list) else [response]
        return postprocess_generate(response)

    def on_send_request_sync(self, request_id):
        return self.proxy.on_send_request_sync(request_id)

    def on_request_routed_sync(self, request_id):
        return self.proxy.on_request_routed_sync(request_id)

class RouterClient:
    def __init__(self, proxy, meta):
        self.proxy = proxy
        self.strategy_name = meta["strategy_name"]
        self.eos_token_id = meta["eos_token_id"]
        self.pad_token_id = meta["pad_token_id"]

    def _preprocess_generate(self, req: DataProto, request_id):
        if request_id is None:
            request_id = str(uuid.uuid4())
        payload = {"rid": str(request_id)}

        generation_config = req.meta_info.get("generation_config")
        collect_unfinished = req.meta_info.get("collect_unfinished", False)
        num_return_sequences = generation_config["num_return_sequences"]
        assert num_return_sequences == 1 or not collect_unfinished, "collect_unfinished is not supported in parallel sampling"

        max_new_tokens = req.meta_info.get("max_new_tokens", generation_config["max_new_tokens"])
        max_new_tokens = min(max_new_tokens, generation_config["max_new_tokens"])
        generation_config["max_new_tokens"] = max_new_tokens

        generation_config["eos_token_id"] = [self.eos_token_id, self.pad_token_id]
        generation_config["pad_token_id"] = self.pad_token_id

        if "multi_modal_data" in req.non_tensor_batch:
            multi_modal_data = req.non_tensor_batch["multi_modal_data"]
            assert len(multi_modal_data) == 1
            payload["multi_modal_data"] = multi_modal_data[0]
        else:
            input_ids = req.batch["input_ids"]
            assert not collect_unfinished or input_ids.size(0) == 1
            attention_mask = req.batch["attention_mask"]
            input_ids = gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)
            payload["input_ids"] = input_ids[0]

        match self.strategy_name:
            case "sglang":
                from roll.distributed.strategy.sglang_strategy import create_sampling_params_for_sglang
                sampling_params = create_sampling_params_for_sglang(gen_kwargs=generation_config)
                payload["sampling_params"] = sampling_params
                payload["return_logprob"] = generation_config.get("logprobs", 0) is not None
            case "vllm":
                from roll.distributed.strategy.vllm_strategy import create_sampling_params_for_vllm
                # vllm is hard coded to return logprob
                sampling_params = create_sampling_params_for_vllm(generation_config, collect_unfinished)
                payload["sampling_params"] = sampling_params
            case _:
                raise NotImplementedError(f"strategy {self.strategy_name} is not supported")
        return payload, request_id

    def _postprocess_generate(self, req, response):
        output_data = DataProto(meta_info=req.meta_info)
        output_data.meta_info["finish_reasons"] = response["finish_reasons"]
        output_data.meta_info["output_token_ids"] = response["output_token_ids"]
        output_data.meta_info["output_logprobs"] = response.get("output_logprobs", None)
        output_data.meta_info["eos_token_id"] = [self.eos_token_id, self.pad_token_id]
        output_data.meta_info["pad_token_id"] = self.pad_token_id
        return output_data

    async def generate_request(self, req: DataProto, request_id, uid):
        """
        Request format is adapted for sglang generate (specificly, use rid rather than request_id),
        which can be directly used by SglangRouter.
        Request is expected to be scalar (single request).

        Response format is adapted for ROLL DataProto.
        Response is expected to be vector (expanded for parallel sample).
        """
        payload, request_id = self._preprocess_generate(req, request_id)

        if not await self.proxy.on_send_request(request_id):
            return None # shutdown
        try:
            response = await self.proxy.generate_request(payload=payload, request_id=request_id, uid=uid)
        finally:
            await self.proxy.on_request_routed(request_id)

        return self._postprocess_generate(req, response)

    def generate_request_sync(self, req: DataProto, request_id, uid):
        payload, request_id = self._preprocess_generate(req, request_id)

        if not self.proxy.on_send_request_sync(request_id):
            return None # shutdown
        try:
            response = self.proxy.generate_request_sync(payload=payload, request_id=request_id, uid=uid)
        finally:
            self.proxy.on_request_routed_sync(request_id)

        return self._postprocess_generate(req, response)

class Router:
    def __init__(self, router_manager, workers, model_path, router_args: RouterArguments):
        self.router_manager_ref = weakref.ref(router_manager)
        self.workers = workers
        self.model_path = model_path
        self.router_args = router_args

    @abstractmethod
    async def initialize(self):
        pass

    @abstractmethod
    async def generate_request(self, payload, request_id, uid):
        pass

    @abstractmethod
    async def abort_requests(self, request_ids, uid):
        pass

    @abstractmethod
    async def abort_all(self, request_ids):
        pass

    async def rebalance_on_shrink(self, shrink_dp_ranks: List[int]) -> Dict[str, int]:
        raise NotImplementedError

    async def rebalance_on_expand(self, expand_dp_ranks: List[int]) -> Dict[str, int]:
        raise NotImplementedError

class SglangRouter(Router):
    """
    Wrap of https://docs.sglang.io/advanced_features/router.html#api-surface

    This is act as a client to sglang-router, can instantiate one SglangRouterClient for every env,
    """
    async def initialize(self):
        self.router_ip = Worker.get_node_ip()
        self.router_port = Worker.get_free_port()

        self.client = httpx.AsyncClient(timeout=httpx.Timeout(None))

        self.worker_urls = await asyncio.gather(
            *[
                worker.get_url.remote()
                for worker in self.workers
            ]
        )
        self.http_mode = False if self.worker_urls[0].startswith("grpc") else True
        assert self.http_mode

        import multiprocessing
        from sglang_router.launch_router import RouterArgs, launch_router

        multiprocessing.set_start_method("spawn")

        router_config = {
            "host": self.router_ip,
            "port": self.router_port,
            "prometheus_port": Worker.get_free_port(),
            "log_level": "warn",
            "policy": "cache_aware",
            "request_timeout_secs": 1800,
            "max_concurrent_requests": -1,
            "dp_aware": False,
            "worker_urls": self.worker_urls,
        }
        extra_router_config = self.router_args.router_config
        if router_config:
            router_config.update(extra_router_config)
        router_args = RouterArgs(**router_config)
        self.router_process = multiprocessing.Process(
            target=launch_router,
            args=(router_args,),
            daemon=True
        )
        self.router_process.start()
        logger.info(f"Launch sglang-router {router_args=}")
        await wait_sglang_router_ready(self.router_process, f"http://{self.router_ip}:{self.router_port}")
        await wait_sglang_router_workflow(f"http://{self.router_ip}:{self.router_port}", self.worker_urls)

    async def generate_request(self, payload, request_id, uid):
        raise RuntimeError("SglangRouter.generate_request is not expected to be called directly, use RouterClient.")

    async def abort_requests(self, request_ids, uid):
        async def abort_request(self, url, request_id):
            response = await self.client.post(f"{url}/abort_request", json={"rid": request_id})
            raise_for_status(response)
        await asyncio.gather(
            *[
                abort_request(self, url=url, request_id=request_id)
                for request_id in request_ids for url in self.worker_urls
            ]
        )

    async def abort_all(self, request_ids):
        # Cannot use abort_all of sglang, because actor_cluster may be shared between different Routers.
        await self.abort_requests(request_ids, uid=None)

    async def abort_all_worker(self, url):
        # Can only be used when router is not shared between two scheudlers.
        response = await self.client.post(f"{url}/abort_request", json={"abort_all": True})
        raise_for_status(response)

    async def post_workers(self, urls):
        responses = await asyncio.gather(
            *[
                self.client.post(
                    f"http://{self.router_ip}:{self.router_port}/workers",
                    json={"url": url},
                )
                for url in urls
            ]
        )
        for response in responses:
            raise_for_status(response)

    async def delete_workers(self, urls):
        encoded_urls = [quote(url, safe="") for url in urls]
        responses = await asyncio.gather(
            *[self.client.delete(f"http://{self.router_ip}:{self.router_port}/workers/{url}") for url in encoded_urls]
        )
        for response in responses:
            raise_for_status(response)

    async def get_worker_loads(self, url):
        response = await self.client.get(f"{url}/get_load")
        raise_for_status(response)
        return response.json()

    async def wait_worker_complete(self, url):
        while True:
            loads = await self.get_worker_loads(url)
            if all(load["num_reqs"] == 0 and load["num_waiting_reqs"] == 0 for load in loads):
                break
            await asyncio.sleep(1)

    async def rebalance_on_shrink(self, shrink_dp_ranks: List[int]) -> Dict[str, int]:
        shrink_urls = [self.worker_urls[dp_rank] for dp_rank in shrink_dp_ranks]

        router_manager: RouterManager = self.router_manager_ref()
        router_manager.suspend()

        await self.delete_workers(shrink_urls)
        logger.info(f"SglangRouter: delete workers on shrink {shrink_dp_ranks=} {shrink_urls=}")

        # FIXME: Do not abort and wait for all workers.
        # Because call wait_worker_complete of shrink workers may not be accurate. There may be
        # a client called on_request_routed but has not calling generate_request yet.
        # Instead, we use RouterManager.wait_complete to make sure no more requests to shrink workers.
        await asyncio.gather(*[self.abort_all_worker(url) for url in self.worker_urls])
        logger.info(f"SglangRouter: abort all requests on shrink {shrink_dp_ranks=} {shrink_urls=}")

        logger.info(f"SglangRouter: wait for running requests on shrink ")
        await router_manager.wait_complete()

        await wait_sglang_router_workflow(f"http://{self.router_ip}:{self.router_port}", {url for url in self.worker_urls if url not in shrink_urls})

        router_manager.resume()

        logger.info(f"SglangRouter: rebalance on shrink finish")

        return {"aborted": 0, "remapped": 0} # for compatibility

    async def rebalance_on_expand(self, expand_dp_ranks: List[int]) -> Dict[str, int]:
        expand_urls = [self.worker_urls[dp_rank] for dp_rank in expand_dp_ranks]

        await self.post_workers(expand_urls)
        logger.info(f"SglangRouter: post workers on expand {expand_dp_ranks=}")

        # simply abort all requests to let sglang-router to re-schedule
        await asyncio.gather(*[self.abort_all_worker(url) for url in self.worker_urls])
        logger.info(f"SglangRouter: aborted all requests on expand {expand_dp_ranks=}")

        # FIXME: assume expand all workers currently
        await wait_sglang_router_workflow(f"http://{self.router_ip}:{self.router_port}", self.worker_urls)

        return {"aborted": 0, "remapped": 0} # for compatibility

class PromptAffinityRouter(Router):
    """
    Schedule requests of the same prompt to the same worker. Choose worker using best fit
    strategy (using linear search for simplicity), blocking generate request if no worker available.

    Limit the number of running requests of each dp rank below max_running_requests.
    """
    async def initialize(self):
        self.max_running_requests = self.router_args.max_running_requests

        # key: dp_rank, value: num_inflight_requests
        self.worker_loads = {dp_rank: 0 for dp_rank in range(len(self.workers))}
        # cache-aware scheduling by uid
        self.id_to_dp_rank: Dict[int, int] = {}
        # dp_rank -> request_ids, used by abort_all
        self.dp_inflight_requests: List[int, Set[str]] = [set() for _ in self.workers]

        self.lock = asyncio.Lock()
        # used by acquire
        self.event = asyncio.Event()
        # used by reacquire
        self.worker_event = {dp_rank: asyncio.Event() for dp_rank in range(len(self.workers))}

    def __repr__(self):
        return f"worker loads: {self.worker_loads}"

    async def generate_request(self, payload, request_id, uid):
        credit = payload["sampling_params"]["n"]
        dp_rank = None
        if uid not in self.id_to_dp_rank:
            # To prevent multiple generate requests for the same prompt.
            # It is safe and no performance issue to acquire lock here.
            # Because acquire is guaranteed to return as long as there has
            # one worker whose running_requests < max_running_requests no matter
            # how large credit is.
            async with self.lock:
                if uid not in self.id_to_dp_rank:
                    dp_rank = await self.acquire(credit=credit)
                    self.id_to_dp_rank[uid] = dp_rank
        if dp_rank is None:
            assert uid in self.id_to_dp_rank
            dp_rank = self.id_to_dp_rank[uid]
            assert dp_rank is not None
            await self.reacquire(dp_rank=dp_rank, credit=credit)
        try:
            self.dp_inflight_requests[dp_rank].add(request_id)
            # InferWorker.generate_request only return data with finish_reason=="abort" on abort
            # but not raise asyncio.CancelledError. This try finally block may be not necessary.
            return await self.workers[dp_rank].generate_request.remote(payload)
            # TODO ray.cancel(ref) on asyncio.CancelledError
        finally:
            self.dp_inflight_requests[dp_rank].remove(request_id)
            self.release(dp_rank=dp_rank, credit=credit)

    async def abort_requests(self, request_ids, uid):
        assert uid is not None
        dp_rank = self.id_to_dp_rank[uid]
        await self.workers[dp_rank].abort_requests.remote(request_ids=request_ids)

    async def abort_all(self, request_ids):
        await asyncio.gather(
            *[
                self.workers[dp_rank].abort_requests.remote(list(request_ids))
                for dp_rank, request_ids in enumerate(self.dp_inflight_requests)
            ]
        )
        self.id_to_dp_rank.clear() # gc uid cache here

    async def acquire(self, credit: int) -> int:
        while True:
            # TODO add check of suspend here to stop early
            target = -1
            for dp_rank, running_requests in self.worker_loads.items():
                if running_requests >= self.max_running_requests:
                    continue
                if target == -1 or running_requests < self.worker_loads[target]:
                    target = dp_rank
            if target != -1:
                # may send more requests than max_running_requests,
                # i.e. worker_loads[target] + credit > max_running_requests
                self.worker_loads[target] += credit
                return target
            self.event.clear()
            await self.event.wait()

    async def reacquire(self, dp_rank: int, credit: int):
        assert dp_rank in self.worker_loads
        while True:
            # TODO add check of suspend here to stop early
            if self.worker_loads[dp_rank] < self.max_running_requests:
                self.worker_loads[dp_rank] += credit
                return
            self.worker_event[dp_rank].clear()
            await self.worker_event[dp_rank].wait()

    def release(self, dp_rank: int, credit: int):
        assert credit >= 0
        self.worker_loads[dp_rank] -= credit
        assert self.worker_loads[dp_rank] >= 0
        self.event.set()
        self.worker_event[dp_rank].set()

    def size(self):
        return sum(self.worker_loads.values())

    def full(self) -> bool:
        return all(running_requests >= self.max_running_requests for running_requests in self.worker_loads.values())

class EnvAffinityRouter(Router):
    """
    Schedule requests of the same (env) uid, to the same dp_rank.

    Choose dp_rank by RR for the first time.

    No rate limit now.

    Do not support partial rollout now.
    """
    async def initialize(self):
        self.src_rank2_dp_rank = {}
        self.request_id_2_src_rank: Dict[str, int] = {}  # Reverse lookup for abort
        self.running_requests: List[set[str]] = [set() for _ in range(len(self.workers))]
        self.worker_iter = itertools.cycle(range(len(self.workers)))

        # Active DP ranks for request routing
        self.active_dp_ranks: Set[int] = set(range(len(self.workers)))  # All ranks initially active
        self.routing_lock = asyncio.Lock()  # Protect routing updates

    async def generate_request(self, payload, request_id, uid):
        src_rank = uid
        # Atomic routing assignment under lock to prevent TOCTOU race with shrink/expand
        async with self.routing_lock:
            # Least-loaded dispatch
            if src_rank not in self.src_rank2_dp_rank:
                dp_rank = self._get_least_active_dp_rank()
                self.src_rank2_dp_rank[src_rank] = dp_rank
            dp_rank = self.src_rank2_dp_rank[src_rank]

        self.request_id_2_src_rank[request_id] = src_rank
        self.running_requests[dp_rank].add(request_id)

        try:
            return await self.workers[dp_rank].generate_request.remote(payload)
        finally:
            self.running_requests[dp_rank].remove(request_id)
            # Cleanup tracking (on both success and abort paths)
            self.request_id_2_src_rank.pop(request_id, None)

    async def abort_requests(self, request_ids, uid):
        raise NotImplementedError

    async def abort_all(self, request_ids):
        await asyncio.gather(*(
            self.workers[dp_rank].abort_requests.remote(list(self.running_requests[dp_rank]))
            for dp_rank in range(len(self.workers))
            if self.running_requests[dp_rank]
        ))

    def _get_least_active_dp_rank(self) -> int:
        """Find DP rank with fewest assigned src_ranks (environments).

        Returns:
            DP rank with minimum src_rank count from src_rank2_dp_rank

        Raises:
            RuntimeError: If no active ranks

        Note:
            Counts unique src_ranks (environments) per worker, not in-flight requests.
            With sticky mapping, one src_rank generates multiple sequential requests.
        """
        candidate_ranks = list(self.active_dp_ranks)
        if not candidate_ranks:
            raise RuntimeError("No active DP ranks")
        # todo optimization: (yangpeng) not efficient, better to use counter for this
        # Count src_ranks per dp_rank
        src_rank_count = defaultdict(int)
        for src_rank, dp_rank in self.src_rank2_dp_rank.items():
            if dp_rank in self.active_dp_ranks:
                src_rank_count[dp_rank] += 1

        # Return dp_rank with minimum src_rank count
        return min(candidate_ranks, key=lambda r: src_rank_count[r])

    def _clear_src_rank_mappings(self, src_ranks: Set[int]) -> None:
        """Clear sticky mappings to allow re-routing on retry."""
        for src_rank in src_ranks:
            self.src_rank2_dp_rank.pop(src_rank, None)

    async def rebalance_on_shrink(self, shrink_dp_ranks: List[int]) -> Dict[str, int]:
        # Atomic operation under routing_lock
        async with self.routing_lock:
            # Rebalance (abort + update active_dp_ranks)
            return await self.rebalance_on_shrink_impl(shrink_dp_ranks)

    async def rebalance_on_shrink_impl(self, shrink_dp_ranks: List[int]) -> Dict[str, int]:
        """Abort requests on shrinking workers, clear mappings for natural re-dispatch.

        Args:
            shrink_dp_ranks: DP ranks to remove from active set

        Returns:
            {"aborted": count, "remapped": count}

        Raises:
            ValueError: If shrink_dp_ranks empty/invalid/duplicates
            RuntimeError: If timeout or operation fails
        """
        # VAL: VAL_NON_EMPTY, VAL_TYPE_CHECK, VAL_INT_RANGE, VAL_NO_DUPLICATES
        if not shrink_dp_ranks:
            raise ValueError("shrink_dp_ranks cannot be empty")

        for rank in shrink_dp_ranks:
            if not isinstance(rank, int):
                raise TypeError(f"Expected int, got {type(rank)}")
            if not (0 <= rank < len(self.workers)):
                raise ValueError(f"rank {rank} out of range")

        if len(shrink_dp_ranks) != len(set(shrink_dp_ranks)):
            raise ValueError(f"Duplicates in shrink_dp_ranks")

        # P0: LOCK_TIMEOUT
        try:
            return await asyncio.wait_for(
                self._rebalance_on_shrink(shrink_dp_ranks),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            raise RuntimeError("rebalance_on_shrink timed out after 30s")

    async def _rebalance_on_shrink(self, shrink_dp_ranks: List[int]) -> Dict[str, int]:
        """Internal implementation of shrink rebalancing.

        PRE-CONDITION: routing_lock MUST be held by caller (shrink_workers).
        This method does NOT acquire the lock internally to avoid double-lock deadlock.

        Args:
            shrink_dp_ranks: DP ranks to remove from active set

        Returns:
            {"aborted": count, "remapped": count}

        Raises:
            RuntimeError: If shrink operation fails
        """
        keep_ranks = list(self.active_dp_ranks - set(shrink_dp_ranks))
        if not keep_ranks:
            raise ValueError("Cannot shrink to zero active ranks")

        old_active_ranks = self.active_dp_ranks.copy()
        self.active_dp_ranks = set(keep_ranks)

        try:
            total_aborted = 0
            abort_futures = []

            for dp_rank in shrink_dp_ranks:
                request_ids = list(self.running_requests[dp_rank])
                if not request_ids:
                    continue

                total_aborted += len(request_ids)

                abort_futures.append(
                    self.workers[dp_rank].abort_requests.remote(request_ids)
                )

            await asyncio.gather(*abort_futures)

            while True:
                remain = sum(len(self.running_requests[dp_rank]) for dp_rank in shrink_dp_ranks)
                if remain == 0:
                    break
                logger.info(f"Shrink: waiting for {len(shrink_dp_ranks)} workers {remain=} to finish abort")
                await asyncio.sleep(3)

            # Clear ALL mappings pointing to shrinking workers (not just in-flight)
            shrink_dp_ranks_set = set(shrink_dp_ranks)
            src_ranks_to_remap = set([
                src_rank for src_rank, dp_rank in self.src_rank2_dp_rank.items()
                if dp_rank in shrink_dp_ranks_set
            ])
            self._clear_src_rank_mappings(src_ranks_to_remap)

            logger.info(
                f"Shrink: aborted {total_aborted} requests, "
                f"cleared {len(src_ranks_to_remap)} mappings"
            )

            return {"aborted": total_aborted, "remapped": len(src_ranks_to_remap)}

        except Exception as e:
            self.active_dp_ranks = old_active_ranks
            raise RuntimeError(f"Shrink failed: {e}") from e

    async def rebalance_on_expand(self, expand_dp_ranks: List[int]) -> Dict[str, int]:
        # Atomic operation under routing_lock
        async with self.routing_lock:
            # Rebalance (update active_dp_ranks + conditional abort)
            return await self.rebalance_on_expand_impl(expand_dp_ranks)

    async def rebalance_on_expand_impl(self, expand_dp_ranks: List[int]) -> Dict[str, int]:
        """Add workers and rebalance via src_rank-level abort.

        Args:
            expand_dp_ranks: DP ranks to add to active set

        Returns:
            {"aborted": count, "remapped": count}

        Raises:
            ValueError: If expand_dp_ranks invalid
            RuntimeError: If timeout or operation fails
        """
        # VAL: VAL_NON_EMPTY, VAL_TYPE_CHECK, VAL_INT_RANGE, VAL_NO_DUPLICATES
        if not expand_dp_ranks:
            raise ValueError("expand_dp_ranks cannot be empty")
        for rank in expand_dp_ranks:
            if not isinstance(rank, int):
                raise TypeError(f"Expected int, got {type(rank)}")
            if not (0 <= rank < len(self.workers)):
                raise ValueError(f"rank {rank} out of range")
        if len(expand_dp_ranks) != len(set(expand_dp_ranks)):
            raise ValueError(f"Duplicates in expand_dp_ranks")

        # P0: LOCK_TIMEOUT
        try:
            return await asyncio.wait_for(
                self._rebalance_on_expand(expand_dp_ranks),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            raise RuntimeError("rebalance_on_expand timed out after 30s")

    async def _rebalance_on_expand(self, expand_dp_ranks: List[int]) -> Dict[str, int]:
        """Internal implementation of expand rebalancing.

        PRE-CONDITION: routing_lock MUST be held by caller (expand_workers).
        This method does NOT acquire the lock internally to avoid double-lock deadlock.

        Algorithm: Round-robin selection across old workers
        1. Calculate proportional src_ranks to abort: src_ranks_to_keep = ceil(total * old_count / new_count)
        2. Group existing src_ranks by dp_rank (only old workers)
        3. Round-robin iterate over old workers using cycle()
        4. Select one src_rank at a time until remaining_to_abort reaches 0
        5. Abort ALL requests from selected src_ranks
        6. Clear src_rank mappings for reallocation to new workers

        Implementation Notes:
        - Uses cycle() for infinite round-robin iteration over old workers
        - Check at line 1146 (if not dp_rank in old_active_dp_ranks) is redundant
          since dp_rank_to_src_ranks already contains only old workers, but kept as defensive guard
        - Loop terminates when remaining_to_abort <= 0 or all worker lists are exhausted
        - If all workers exhausted before reaching target, loop may cycle indefinitely
          (no explicit check for empty state, but pop(0) will eventually empty all lists)

        Args:
            expand_dp_ranks: DP ranks to add to active set (already validated)

        Returns:
            {"aborted": count, "remapped": count} - count of src_ranks aborted/remapped

        Preconditions:
            - routing_lock MUST be held by caller
            - expand_dp_ranks validated (non-empty, int, in range, no duplicates)

        Postconditions:
            - active_dp_ranks updated with expand_dp_ranks
            - Selected src_ranks aborted and removed from mappings
            - Requests from aborted src_ranks reported as is_abort=True
        """
        # Calculate counts before updating active_dp_ranks
        old_dp_count = len(self.active_dp_ranks)
        old_active_dp_ranks = self.active_dp_ranks.copy()

        self.active_dp_ranks.update(expand_dp_ranks)
        new_dp_count = len(self.active_dp_ranks)

        total_src_ranks = len(self.src_rank2_dp_rank)
        if total_src_ranks == 0:
            return {"aborted": 0, "remapped": 0}

        # Proportional calculation
        src_ranks_to_keep = math.ceil(int(total_src_ranks * old_dp_count / new_dp_count))
        src_ranks_to_abort = total_src_ranks - src_ranks_to_keep

        if src_ranks_to_abort <= 0:
            logger.info("Expand: no rebalancing needed (src_ranks_to_abort <= 0)")
            return {"aborted": 0, "remapped": 0}

        # Group src_ranks by dp_rank (old workers only)
        dp_rank_to_src_ranks = defaultdict(list)
        for src_rank, dp_rank in self.src_rank2_dp_rank.items():
            if dp_rank in old_active_dp_ranks:
                dp_rank_to_src_ranks[dp_rank].append(src_rank)

        # Round-robin selection: iterate over old workers and select one src_rank at a time
        # todo optimization:(yangpeng) take uneven dp load into consideration and do dynamic load balancing, not just RR
        selected_src_ranks = []
        remaining_to_abort = src_ranks_to_abort
        for dp_rank in itertools.cycle(dp_rank_to_src_ranks.keys()):
            if not dp_rank in old_active_dp_ranks:
                continue

            if remaining_to_abort <= 0:
                break

            src_ranks_on_worker = dp_rank_to_src_ranks.get(dp_rank, [])
            if not src_ranks_on_worker:
                continue
            selected_src_ranks.append(src_ranks_on_worker.pop(0))

            remaining_to_abort -= 1

        # Remove from mapping and group by dp_rank for abort
        abort_by_dp_rank = defaultdict(list)
        for src_rank in selected_src_ranks:
            dp_rank = self.src_rank2_dp_rank.pop(src_rank)

            # Find request_id(s) for this src_rank
            for request_id, sr in self.request_id_2_src_rank.items():
                if sr == src_rank:
                    abort_by_dp_rank[dp_rank].append(request_id)

        # Send batched ABORT commands
        abort_futures = []
        total_aborted = 0
        for dp_rank, request_ids in abort_by_dp_rank.items():
            if not request_ids:
                continue

            total_aborted += len(request_ids)
            abort_futures.append(
                self.workers[dp_rank].abort_requests.remote(request_ids)
            )


        await asyncio.gather(*abort_futures)

        logger.info(
            f"Expand: aborted {len(selected_src_ranks)} src_ranks, "
            f"cleared {len(selected_src_ranks)} mappings "
            f"(proportional: {old_dp_count}/{new_dp_count})"
        )

        return {"aborted": len(selected_src_ranks), "remapped": len(selected_src_ranks)}
