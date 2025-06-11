import os
import shutil

from roll.utils.logging import get_logger


logger = get_logger()

uploader_registry = {}


class FileSystemUploader:
    """
    将本地的ckpt目录上传到文件系统, oss/cpfs在多Role的场景下，
    每个Role会把自己的ckpt dir的内容上传到OUTPUT_DIR/ckpt_id/下
    {
        "type": "file_system",
        "output_dir": /data/oss_bucket_0/llm/models
    }
    """

    def __init__(self, output_dir, *args, **kwargs):
        self.output_dir = output_dir
        logger.info(f"use FileSystemUploader to upload {output_dir}")

    def upload(self, ckpt_id: str, local_state_path: str, **kwargs):
        ckpt_id_output_dir = os.path.join(self.output_dir, ckpt_id)
        os.makedirs(ckpt_id_output_dir, exist_ok=True)
        logger.info(f"{local_state_path} save to {ckpt_id_output_dir}, wait...")
        shutil.copytree(local_state_path, ckpt_id_output_dir, dirs_exist_ok=True)
        logger.info(f"{local_state_path} save to {ckpt_id_output_dir}, done...")


uploader_registry['file_system'] = FileSystemUploader
