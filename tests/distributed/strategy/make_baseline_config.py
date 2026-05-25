from dacite import from_dict
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from roll.pipeline.rlvr.rlvr_config import RLVRConfig


def make_baseline_config(config_path, config_name):

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name=config_name)
    ppo_config = from_dict(data_class=RLVRConfig, data=OmegaConf.to_container(cfg, resolve=True))

    return ppo_config
