import argparse

from dacite import from_dict
try:
    from hydra import compose, initialize
except ImportError:
    from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.diffusion.diffusion_config import DiffusionConfig
from roll.pipeline.diffusion.diffusion_pipeline import DiffusionPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="The path of the main configuration file", default="config")
    parser.add_argument("--config_name", help="The name of the main configuration file.", default="flow_grpo_fsdp2")
    args = parser.parse_args()

    initialize(config_path=args.config_path, job_name="app")
    cfg = compose(config_name=args.config_name)
    print(OmegaConf.to_yaml(cfg, resolve=True))

    pipeline_config = from_dict(data_class=DiffusionConfig, data=OmegaConf.to_container(cfg, resolve=True))
    init()
    DiffusionPipeline(pipeline_config=pipeline_config).run()


if __name__ == "__main__":
    main()
