from typing import Any

from vllm import envs
from vllm.v1.executor.multiproc_executor import WorkerProc


def patch_multiproc_executor_init_order() -> None:
    if getattr(WorkerProc, "_roll_init_order_patched", False):
        return

    init_message_queues = WorkerProc._init_message_queues

    def defer_message_queue_init(self, input_shm_handle: Any, vllm_config: Any) -> None:
        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            init_message_queues(self, input_shm_handle, vllm_config)
            return

        load_model = self.worker.load_model

        def load_model_then_init_queues(*args: Any, **kwargs: Any) -> Any:
            del self.worker.load_model
            result = load_model(*args, **kwargs)
            init_message_queues(self, input_shm_handle, vllm_config)
            return result

        self.worker.load_model = load_model_then_init_queues

    WorkerProc._init_message_queues = defer_message_queue_init
    WorkerProc._roll_init_order_patched = True
