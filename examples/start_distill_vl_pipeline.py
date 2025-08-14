import argparse

from dacite import from_dict, Config
from hydra import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.distill.distill_config import DistillConfig

from roll.pipeline.distill.distill_vlm_pipeline import DistillVLMPipeline


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

    distill_config = from_dict(data_class=DistillConfig, data=OmegaConf.to_container(cfg, resolve=True))

    init()

    pipeline = DistillVLMPipeline(pipeline_config=distill_config)

    pipeline.run()


if __name__ == "__main__":
    main()
