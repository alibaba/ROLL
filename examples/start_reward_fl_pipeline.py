import argparse

from dacite import from_dict
from hydra import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.diffusion.reward_fl.reward_fl_config import RewardFLConfig

from roll.pipeline.diffusion.reward_fl.reward_fl_pipeline import RewardFLPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="The path of the main configuration file", default="config")
    parser.add_argument(
        "--config_name", help="The name of the main configuration file (without extension).", default="reward_fl_config"
    )
    args = parser.parse_args()

    initialize(config_path=args.config_path, job_name="app")
    cfg = compose(config_name=args.config_name)

    print(OmegaConf.to_yaml(cfg, resolve=True))

    reward_fl_config = from_dict(data_class=RewardFLConfig, data=OmegaConf.to_container(cfg, resolve=True))

    init()

    pipeline = RewardFLPipeline(pipeline_config=reward_fl_config)

    pipeline.run()


if __name__ == "__main__":
    main()
