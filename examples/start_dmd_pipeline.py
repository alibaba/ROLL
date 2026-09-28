import argparse

from dacite import from_dict
try:
    from hydra import compose, initialize
except ImportError:
    from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.diffusion.dmd.dmd_config import DMDConfig
from roll.pipeline.diffusion.dmd.dmd_pipeline import DMDPipeline


def main() -> None:
    """Start the DMD pipeline from a Hydra config."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="The path of the main configuration file", default="config")
    parser.add_argument(
        "--config_name",
        help="The name of the main configuration file without extension.",
        default="dmd_config",
    )
    args, overrides = parser.parse_known_args()

    initialize(config_path=args.config_path, job_name="dmd_train")
    cfg = compose(config_name=args.config_name, overrides=overrides)

    print(OmegaConf.to_yaml(cfg, resolve=True))

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    dmd_config: DMDConfig = from_dict(data_class=DMDConfig, data=config_dict)

    init()
    pipeline = DMDPipeline(pipeline_config=dmd_config)
    pipeline.run()


if __name__ == "__main__":
    main()
