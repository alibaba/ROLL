from dacite import from_dict
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

def make_pipeline_config(config_path, config_name, data_class):

    initialize(config_path=config_path)
    cfg = compose(config_name=config_name)
    pipeline_config = from_dict(data_class=data_class, data=OmegaConf.to_container(cfg, resolve=True))

    return pipeline_config
