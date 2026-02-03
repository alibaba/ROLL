import json
from functools import wraps
from typing import Optional, Dict, Any

import torch

from roll.utils.logging import get_logger

logger = get_logger()

tracker_registry: Dict[str, Any] = {}


def _strip_metric_tag(values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strip reducer tags from metric keys before logging.

    We may annotate metric keys with reducer tags for internal aggregation:
      "actor/pg_loss@sum", "actor/kl_loss@mean", ...
    Dashboards (TensorBoard/W&B/...) should log clean names, so we remove "@...":
      "actor/pg_loss@sum" -> "actor/pg_loss"

    - Only strips the last "@tag" part (rsplit("@", 1))
    - Recursively strips nested dict keys (e.g. add_scalars)
    - Returns a new dict (does not mutate the input)
    """
    def strip_key(k: str) -> str:
        return k.rsplit("@", 1)[0] if isinstance(k, str) and "@" in k else k

    out: Dict[str, Any] = {}
    for k, v in values.items():
        nk = strip_key(k)
        if isinstance(v, dict):
            v = _strip_metric_tag(v)
        out[nk] = v
    return out


def strip_at_tag_in_log(func):
    """
    Decorator for Tracker.log(...).

    Purpose:
      Remove "@tag" suffixes from metric keys right before sending them to the
      logging backend. This is name-cleaning only (no reduction happens here).
    """
    @wraps(func)
    def wrapper(self, values: dict, step: Optional[int] = None, **kwargs):
        if isinstance(values, dict):
            values = _strip_metric_tag(values)
        return func(self, values, step, **kwargs)
    return wrapper



class BaseTracker:

    def log(self, values: dict, step: Optional[int], **kwargs):
        pass

    def finish(self):
        pass


class TensorBoardTracker(BaseTracker):

    def __init__(self, config: dict, **kwargs):
        log_dir = kwargs.pop("log_dir")
        from torch.utils import tensorboard

        kwargs["max_queue"] = 1000
        kwargs["flush_secs"] = 10
        self.writer = tensorboard.SummaryWriter(log_dir=log_dir, **kwargs)
        self.config = config
        for k in list(self.config.keys())[:]:
            if not isinstance(self.config[k], (int, float, str, bool, torch.Tensor)):
                self.config[k] = str(self.config[k])
        self.writer.add_hparams(hparam_dict=self.config, metric_dict={})
        self.writer.flush()

    @strip_at_tag_in_log
    def log(self, values: dict, step: Optional[int], **kwargs):
        for k, v in values.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, global_step=step, **kwargs)
            elif isinstance(v, str):
                self.writer.add_text(k, v, global_step=step, **kwargs)
            elif isinstance(v, dict):
                self.writer.add_scalars(k, v, global_step=step, **kwargs)
        self.writer.flush()

    def finish(self):
        self.writer.close()


class WandbTracker(BaseTracker):

    def __init__(self, config: dict, **kwargs):
        self.config = config
        project = kwargs.pop("project", None)
        tags = kwargs.pop("tags", None)
        name = kwargs.pop("name", None)
        notes = kwargs.pop("notes", None)
        log_dir = kwargs.pop("log_dir", None)
        api_key = kwargs.pop("api_key", None)
        mode = kwargs.pop("mode", None)
        settings = kwargs.pop("settings", {"console": "off"})
        import wandb
        if api_key:
            wandb.login(key=api_key)
        self.run = wandb.init(project=project, tags=tags, name=name, notes=notes, dir=log_dir, mode=mode, settings=settings)

        self.run.config.update(config, allow_val_change=True)

    @strip_at_tag_in_log
    def log(self, values: dict, step: Optional[int], **kwargs):
        self.run.log(values, step=step, **kwargs)

    def finish(self):
        self.run.finish()


class SwanlabTracker(BaseTracker):

    def __init__(self, config: dict, **kwargs):
        self.config = config
        project = kwargs.pop("project", None)
        workspace = kwargs.pop("workspace", None)
        experiment_name = kwargs.pop("experiment_name", None)
        description = kwargs.pop("description", None)
        tags = kwargs.pop("tags", None)
        logdir = kwargs.pop("logdir", None)
        login_kwargs = kwargs.pop("login_kwargs", None)
        import swanlab
        if login_kwargs:
            swanlab.login(**login_kwargs)
        self.run = swanlab.init(project=project, workspace=workspace, experiment_name=experiment_name, description=description,
                                tags=tags, logdir=logdir, **kwargs)

    @strip_at_tag_in_log
    def log(self, values: dict, step: Optional[int], **kwargs):
        self.run.log(values, step=step, **kwargs)

    def finish(self):
        self.run.finish()


class StdoutTracker(BaseTracker):

    def __init__(self, config: dict, **kwargs):
        self.config = config

    @strip_at_tag_in_log
    def log(self, values: dict, step: Optional[int], **kwargs):
        logger.info(f"metrics_tag: {json.dumps({'step': step, 'metrics': values})}")

    def finish(self):
        pass


def create_tracker(tracker_name: str, config: dict, **kwargs) -> BaseTracker:
    if not tracker_name:
        return BaseTracker()
    logger.info(f"create tracker {tracker_name}, kwargs: {kwargs}")

    if tracker_name not in tracker_registry:
        raise ValueError(f"Unknown tracker name: {tracker_name}, total registered trackers: {tracker_registry.keys()}")
    tracker_cls = tracker_registry[tracker_name]
    return tracker_cls(config, **kwargs)

tracker_registry["tensorboard"] = TensorBoardTracker
tracker_registry["wandb"] = WandbTracker
tracker_registry["stdout"] = StdoutTracker
tracker_registry["swanlab"] = SwanlabTracker
