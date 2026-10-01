import argparse

from dacite import from_dict
from hydra import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.diffusion.diffusion_config import DiffusionConfig
from roll.pipeline.diffusion.diffusion_rollout_pipeline import DiffusionRolloutPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="qwen-image-diffusion")
    parser.add_argument("--config_name", default="qwen_image_rollout")
    args = parser.parse_args()

    initialize(config_path=args.config_path, job_name="app")
    cfg = compose(config_name=args.config_name)
    print(OmegaConf.to_yaml(cfg, resolve=True))

    pipeline_config = from_dict(
        data_class=DiffusionConfig,
        data=OmegaConf.to_container(cfg, resolve=True),
    )
    init()
    DiffusionRolloutPipeline(pipeline_config=pipeline_config).run()


if __name__ == "__main__":
    main()
