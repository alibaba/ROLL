from dacite import from_dict
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

def make_pipeline_config(config_path, config_name, data_class):

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name=config_name)
    pipeline_config = from_dict(data_class=data_class, data=OmegaConf.to_container(cfg, resolve=True))

    return pipeline_config
