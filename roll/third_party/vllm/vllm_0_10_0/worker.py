import gc
from typing import Optional

import torch
from vllm.worker.worker import Worker

from roll.third_party.vllm.worker_helper import WorkerHelper
from roll.utils.logging import get_logger

logger = get_logger()


class Worker0100(WorkerHelper, Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
