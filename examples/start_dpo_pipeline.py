import argparse

from dacite import from_dict
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.dpo.dpo_config import DPOConfig
from roll.pipeline.dpo.dpo_pipeline import DPOPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="The path of the main configuration file", default="config")
    parser.add_argument(
        "--config_name", help="The name of the main configuration file (without extension).", default="sppo_config"
    )
    args = parser.parse_args()

    initialize(config_path=args.config_path, job_name="app")
    cfg = compose(config_name=args.config_name)

    print(OmegaConf.to_yaml(cfg, resolve=True))

    dpo_config = from_dict(data_class=DPOConfig, data=OmegaConf.to_container(cfg, resolve=True))

    init()

    pipeline = DPOPipeline(pipeline_config=dpo_config)

    pipeline.run()


if __name__ == "__main__":
    main()
