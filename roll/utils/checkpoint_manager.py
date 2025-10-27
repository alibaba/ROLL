import contextlib
import copy
import hashlib
import os
import shutil
import tempfile
import traceback
from typing import Dict, Optional, Any

import ray
from filelock import FileLock
from huggingface_hub import snapshot_download

from roll.distributed.scheduler.storage import SharedStorage
from roll.utils.constants import STORAGE_NAME, RAY_NAMESPACE
from roll.utils.logging import get_logger
from roll.utils.network_utils import get_node_ip
from roll.utils.upload_utils import uploader_registry

logger = get_logger()

model_download_registry: Dict[str, Any] = {}
model_download_registry["HUGGINGFACE_HUB"] = snapshot_download
try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download

    model_download_registry["MODELSCOPE"] = ms_snapshot_download
except Exception as e:
    logger.error(e)


@contextlib.contextmanager
def file_lock_context(lock_path: str):
    temp_lock_path = os.path.join(tempfile.gettempdir(), f"{hashlib.md5(lock_path.encode()).hexdigest()}.lock")
    with FileLock(temp_lock_path):
        yield

shared_storage = None

def model_path_cache(func):
    node_ip = get_node_ip()
    def wrapper(model_name_or_path: str, local_dir: Optional[str] = None):
        global shared_storage
        if shared_storage is None:
            shared_storage = SharedStorage.options(
                name=STORAGE_NAME, get_if_exists=True, namespace=RAY_NAMESPACE
            ).remote()
        cached_path = ray.get(shared_storage.get.remote(key=f"{node_ip}:{model_name_or_path}"))
        if cached_path is None or not os.path.exists(cached_path):
            cached_path = func(model_name_or_path, local_dir)
            ray.get(shared_storage.put.remote(key=f"{node_ip}:{model_name_or_path}", data=cached_path))
        return cached_path
    return wrapper


@model_path_cache
def download_model(model_name_or_path: str, local_dir: Optional[str] = None):
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    model_download_type = os.getenv("MODEL_DOWNLOAD_TYPE", "MODELSCOPE")
    if model_download_type not in model_download_registry:
        raise ValueError(f"Unknown model_download_type: {model_download_type},"
                         f" total registered model download type: {model_download_registry.keys()}")
    model_download_func = model_download_registry[model_download_type]

    with file_lock_context(model_name_or_path):
        return model_download_func(model_name_or_path, local_dir=local_dir)


class CheckpointManager:
    """
    ray.Actor创建到每个node上，负责将本地output_dir的文件上传到远程存储(oss/hdfs)
    """

    def __init__(self, checkpoint_config=None):
        self.checkpoint_config: Dict = copy.deepcopy(checkpoint_config)
        self.uploader = None
        logger.info(f"checkpoint_config: {checkpoint_config}")
        if self.checkpoint_config:
            upload_type = self.checkpoint_config.pop("type", "file_system")
            if upload_type not in uploader_registry:
                raise ValueError(
                    f"Unknown tracker name: {upload_type}, total registered trackers: {uploader_registry.keys()}")
            uploader_cls = uploader_registry[upload_type]
            self.uploader = uploader_cls(**self.checkpoint_config)

    def upload(self, ckpt_id, local_state_path, keep_local_file=False):
        try:
            if not self.uploader:
                logger.warning(f"uploader is None, skip upload...")
                return

            self.uploader.upload(ckpt_id=ckpt_id, local_state_path=local_state_path)
            if not keep_local_file:
                if os.path.isdir(local_state_path):
                    shutil.rmtree(local_state_path, ignore_errors=True)
                else:
                    os.remove(local_state_path)
        except Exception as e:
            traceback.print_exc()
            logger.error(f"upload failed, {e}")
