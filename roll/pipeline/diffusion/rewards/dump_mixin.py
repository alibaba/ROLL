"""Shared step-output dumping for diffusion reward workers.

Reward workers dump per-sample images plus a jsonl record so that runs can be
inspected offline (e.g. to tell genuine reward improvement from reward hacking).
The gating rules, directory layout, async IO and jsonl schema are identical
across rewards; only the payload fields differ. :class:`RewardDumpMixin` owns
everything generic and lets each worker declare its own extra columns.

Enabled by three pipeline config keys::

    dump_step_output_type: ["train"]        # phases to dump; empty disables all
    dump_step_output_idx: "list(range(0,100))"
    dump_step_output_path: /some/base/dir

Layout::

    <path>/<dump_run_name>/step<N>/<phase>/<step>-<sample_id>.png
    <path>/<dump_run_name>/step<N>/<phase>/output-<reward_name>-rank<R>.jsonl
"""

import json
import os
import queue
import shutil
import tempfile
import threading
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torchvision

from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto

# jsonl keys written for every reward; extra_columns must not shadow them.
RESERVED_DUMP_KEYS = frozenset(
    {"key", "reward_name", "sample_id", "img_path", "reward_score", "prompt", "rollout_seed"}
)


def resolve_dump_base_dir(pipeline_config) -> str:
    """Run-level dump directory, or "" when dumping has no path configured.

    Shared by pipelines and reward workers so every writer agrees on the layout
    (including the fallback used when ``dump_run_name`` is absent).
    """
    dump_path = getattr(pipeline_config, "dump_step_output_path", "") or ""
    if not dump_path:
        return ""
    run_name = getattr(pipeline_config, "dump_run_name", "") or ""
    if not run_name:
        # Fallback for configs that did not go through BaseConfig.__post_init__
        exp_name = getattr(pipeline_config, "exp_name", "experiment")
        run_name = f"{exp_name}_unknown"
    return os.path.join(dump_path, run_name)


def is_dump_enabled(pipeline_config, global_step: int, is_val: bool) -> bool:
    """Whether the given phase/step is selected for dumping."""
    dump_path = getattr(pipeline_config, "dump_step_output_path", "") or ""
    dump_types = getattr(pipeline_config, "dump_step_output_type", []) or []
    dump_steps = getattr(pipeline_config, "dump_step_output_idx", []) or []
    if not dump_path or not dump_types or not dump_steps:
        return False
    phase = "val" if is_val else "train"
    return phase in dump_types and global_step in dump_steps


def resolve_dump_step_dir(pipeline_config, global_step: int, is_val: bool) -> str:
    """Step/phase dump directory, or "" when dumping is disabled for it."""
    if not is_dump_enabled(pipeline_config, global_step, is_val):
        return ""
    base = resolve_dump_base_dir(pipeline_config)
    if not base:
        return ""
    phase = "val" if is_val else "train"
    return os.path.join(base, f"step{global_step}", phase)


def to_jsonable(value: Any) -> Any:
    """Recursively convert numpy/torch values into JSON-native Python objects.

    Reward workers produce scores from heterogeneous backends (torch tensors,
    numpy scalars from detectors, nested dicts of per-tag details), and
    ``json.dumps`` rejects numpy scalars outright. Sanitizing here keeps every
    reward's payload writable without each worker converting by hand.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    return str(value)


class RewardDumpMixin:
    """Async per-step image + jsonl dumping shared by diffusion reward workers.

    Mix into a :class:`~roll.distributed.executor.worker.Worker` subclass; the
    queue and writer thread are created on first use, so subclasses need no
    extra ``initialize`` wiring. ``flush_dump`` is registered as an RPC and is
    discovered through the MRO by ``Cluster._bind_worker_method``.
    """

    _dump_queue: Optional[queue.Queue] = None
    _dump_thread: Optional[threading.Thread] = None

    @property
    def reward_name(self) -> str:
        """Name of this reward, used to keep concurrent rewards' jsonl apart."""
        return getattr(self.worker_config, "name", None) or type(self).__name__

    def tensor_to_pil_image(self, image_tensor: torch.Tensor):
        """Convert a [C, H, W] tensor (either [0,1] or [-1,1]) to a PIL image."""
        image = image_tensor.detach().cpu().float()
        if image.ndim != 3:
            raise ValueError(f"Expected image tensor with shape [C, H, W], got {tuple(image.shape)}")
        if image.min().item() < 0:
            image = (image + 1.0) / 2.0
        image = image.clamp(0.0, 1.0)
        return torchvision.transforms.functional.to_pil_image(image)

    def _should_dump(self, global_step: int, is_val: bool) -> bool:
        """Whether the current phase/step is selected for dumping."""
        pipeline_config = getattr(self, "pipeline_config", None)
        if pipeline_config is None:
            return False
        return is_dump_enabled(pipeline_config, global_step, is_val)

    def _get_dump_base_dir(self) -> str:
        """Run-level dump directory shared by all workers of one pipeline run."""
        pipeline_config = getattr(self, "pipeline_config", None)
        if pipeline_config is None:
            return ""
        return resolve_dump_base_dir(pipeline_config)

    def _get_dump_step_dir(self, global_step: int, is_val: bool) -> str:
        pipeline_config = getattr(self, "pipeline_config", None)
        if pipeline_config is None:
            return ""
        return resolve_dump_step_dir(pipeline_config, global_step, is_val)

    def _ensure_dump_thread(self) -> queue.Queue:
        if self._dump_queue is None:
            self._dump_queue = queue.Queue()
            self._dump_thread = threading.Thread(target=self._dump_worker, daemon=True)
            self._dump_thread.start()
        return self._dump_queue

    def _dump_worker(self):
        """Background thread draining the dump queue."""
        while True:
            item = self._dump_queue.get()
            try:
                if item is None:
                    return
                self._write_dump(**item)
            except Exception as e:  # never let a dump failure kill the thread
                self.logger.exception("[ROLL-DUMP] async dump failed: %s", e)
            finally:
                self._dump_queue.task_done()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    def flush_dump(self) -> None:
        """Block until every queued dump task has been written."""
        if self._dump_queue is not None:
            self._dump_queue.join()

    def _write_dump(self, step_dir: str, global_step: int, images: list, records: List[Dict[str, Any]]):
        """Write images and jsonl lines (runs on the dump thread)."""
        os.makedirs(step_dir, exist_ok=True)
        jsonl_path = os.path.join(step_dir, f"output-{self.reward_name}-rank{self.rank}.jsonl")

        for image, record in zip(images, records):
            save_path = os.path.join(step_dir, record["img_path"])
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_f:
                    tmp_path = tmp_f.name
                image.save(tmp_path)
                shutil.copy2(tmp_path, save_path)
                os.unlink(tmp_path)
            except Exception as e:
                self.logger.exception("[ROLL-DUMP] failed to save image %s: %s", save_path, e)

        try:
            lines = [json.dumps(record, ensure_ascii=False) for record in records]
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp_f:
                tmp_path = tmp_f.name
                tmp_f.write("\n".join(lines) + "\n")
            with open(tmp_path, "r") as src:
                content = src.read()
            with open(jsonl_path, "a") as dst:
                dst.write(content)
            os.unlink(tmp_path)
        except Exception as e:
            self.logger.exception("[ROLL-DUMP] failed to write jsonl %s: %s", jsonl_path, e)

        self.logger.info(
            "[ROLL-DUMP] step %s dump complete: %d images + jsonl written to %s",
            global_step, len(images), step_dir,
        )

    def _build_dump_records(
        self,
        global_step: int,
        sample_ids: Sequence,
        reward_scores: Sequence,
        prompts: Sequence,
        rollout_seeds: Sequence,
        extra_columns: Dict[str, Sequence],
    ) -> List[Dict[str, Any]]:
        records = []
        for idx, sample_id in enumerate(sample_ids):
            sid = sample_id if sample_id is not None else "unk"
            seed = rollout_seeds[idx]
            record = {
                "key": f"{global_step}-{sid}",
                "reward_name": self.reward_name,
                "sample_id": to_jsonable(sid),
                "img_path": f"{global_step}-{sid}.png",
                "reward_score": to_jsonable(reward_scores[idx]),
                "prompt": to_jsonable(prompts[idx]) if prompts[idx] is not None else "",
                "rollout_seed": int(seed) if seed is not None else None,
            }
            for column, values in extra_columns.items():
                record[column] = to_jsonable(values[idx])
            records.append(record)
        return records

    @staticmethod
    def _as_list(values, length: int, default=None) -> list:
        if values is None:
            return [default] * length
        if hasattr(values, "tolist"):
            values = values.tolist()
        return list(values)

    def maybe_dump_step_output(
        self,
        data: DataProto,
        sample_ids: Sequence,
        images: list,
        reward_scores: Sequence,
        extra_columns: Optional[Dict[str, Sequence]] = None,
    ) -> None:
        """Enqueue an async dump of this step when dumping is configured.

        ``sample_ids`` should be unique per sample (e.g. ``"train-19356-0"``).
        ``reward_scores`` is the scalar actually used for training, written under
        the canonical ``reward_score`` key so records stay comparable across
        rewards. ``extra_columns`` maps a jsonl field name to a per-sample
        sequence; every sequence must align with ``images``.

        ``prompt``/``rollout_seed`` are pulled from ``data`` automatically.
        Images are copied so the writer thread owns its own references.
        """
        global_step = data.meta_info.get("global_step", None)
        if global_step is None:
            return
        is_val = data.meta_info.get("is_val", False)
        if not self._should_dump(global_step, is_val):
            return
        step_dir = self._get_dump_step_dir(global_step, is_val)
        if not step_dir:
            return

        extra_columns = dict(extra_columns or {})
        reserved = RESERVED_DUMP_KEYS & set(extra_columns)
        assert not reserved, f"extra_columns must not shadow reserved dump keys: {sorted(reserved)}"

        num_images = len(images)
        sample_ids = self._as_list(sample_ids, num_images)
        reward_scores = self._as_list(reward_scores, num_images)
        prompts = self._as_list(data.non_tensor_batch.get("prompt"), num_images)
        rollout_seeds = self._as_list(data.non_tensor_batch.get("rollout_seed"), num_images)

        # Fail loudly on misaligned columns: silently padding would hide field
        # shifts that make the whole dump untrustworthy.
        lengths = {
            "sample_ids": len(sample_ids),
            "reward_scores": len(reward_scores),
            "prompts": len(prompts),
            "rollout_seeds": len(rollout_seeds),
            **{name: len(self._as_list(values, num_images)) for name, values in extra_columns.items()},
        }
        mismatched = {name: n for name, n in lengths.items() if n != num_images}
        assert not mismatched, f"dump column length mismatch, expected {num_images}: {mismatched}"
        extra_columns = {name: self._as_list(values, num_images) for name, values in extra_columns.items()}

        records = self._build_dump_records(
            global_step, sample_ids, reward_scores, prompts, rollout_seeds, extra_columns
        )
        images_copy = [image.copy() for image in images]

        self._ensure_dump_thread().put(
            {
                "step_dir": step_dir,
                "global_step": global_step,
                "images": images_copy,
                "records": records,
            }
        )
        self.logger.info(
            "[ROLL-DUMP] step %s enqueued for async dump (%d images, reward=%s, val=%s)",
            global_step, num_images, self.reward_name, is_val,
        )
