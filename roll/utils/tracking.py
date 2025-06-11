import json
from typing import Optional, Dict, Any

import torch

from roll.utils.logging import get_logger

logger = get_logger()

tracker_registry: Dict[str, Any] = {}


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
        settings = kwargs.pop("settings", {"console": "off"})
        import wandb
        if api_key:
            wandb.login(key=api_key)
        self.run = wandb.init(project=project, tags=tags, name=name, notes=notes, dir=log_dir, settings=settings)

        self.run.config.update(config, allow_val_change=True)

    def log(self, values: dict, step: Optional[int], **kwargs):
        self.run.log(values, step=step, **kwargs)

    def finish(self):
        self.run.finish()


class StdoutTracker(BaseTracker):

    def __init__(self, config: dict, **kwargs):
        self.config = config

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
