import errno
import os
import re
import shutil
import stat
import tempfile

from roll.utils.logging import get_logger


logger = get_logger()

uploader_registry = {}


class FileSystemUploader:
    """
    将本地的ckpt目录上传到文件系统, oss/cpfs在多Role的场景下，
    每个Role会把自己的ckpt dir的内容上传到OUTPUT_DIR/ckpt_id/下
    {
        "type": "file_system",
        "output_dir": /data/oss_bucket_0/llm/models,
        "use_hard_links": false
    }

    ``use_hard_links`` is an opt-in for immutable checkpoint files. It avoids
    duplicating payloads on the same filesystem; cross-filesystem uploads still
    copy. Linked source files may be removed but must never be modified in place.
    """

    def __init__(self, output_dir, *args, use_hard_links=False, **kwargs):
        self.output_dir = output_dir
        self.use_hard_links = use_hard_links
        logger.info(f"use FileSystemUploader to upload {output_dir}")
        self._re_checkpoint = re.compile(r"^" + "checkpoint" + r"\-(\d+)$")

    def upload(self, ckpt_id: str, local_state_path: str, **kwargs):
        ckpt_id_output_dir = os.path.join(self.output_dir, ckpt_id)
        os.makedirs(ckpt_id_output_dir, exist_ok=True)
        logger.info(f"{local_state_path} save to {ckpt_id_output_dir}, wait...")
        copy_function = self._link_or_copy if self.use_hard_links else shutil.copy2
        shutil.copytree(local_state_path, ckpt_id_output_dir, dirs_exist_ok=True, copy_function=copy_function)
        logger.info(f"{local_state_path} save to {ckpt_id_output_dir}, done...")

    @staticmethod
    def _link_or_copy(source, destination):
        # Never truncate an existing destination: it may itself be a hard link
        # to another rank's immutable local checkpoint. Publish each completed
        # file with atomic replacement, retaining the previous file on failure.
        with tempfile.TemporaryDirectory(prefix=".roll-checkpoint-", dir=os.path.dirname(destination)) as temporary:
            staged = os.path.join(temporary, "payload")
            if stat.S_ISREG(os.stat(source).st_mode):
                try:
                    os.link(os.path.realpath(source), staged)
                except OSError as error:
                    if error.errno not in {errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.ENOTSUP}:
                        raise
                    shutil.copy2(source, staged)
            else:
                shutil.copy2(source, staged)
            os.replace(staged, destination)
        return destination

    def register(self, ckpt_info):
        pass

    def get_latest_ckpt(self):
        content = os.listdir(self.output_dir)
        checkpoints = [
            path
            for path in content
            if self._re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(self.output_dir, path))
        ]
        if len(checkpoints) == 0:
            return None
        return os.path.join(self.output_dir, max(checkpoints, key=lambda x: int(self._re_checkpoint.search(x).groups()[0])))


uploader_registry['file_system'] = FileSystemUploader
