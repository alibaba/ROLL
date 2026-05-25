import argparse
import os

import pytest
from dacite import from_dict
from hydra import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.rlvr.rlvr_config import RLVRConfig


DEFAULT_CONFIG_NAME = "rlvr_megatron_config"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PPO Configuration")
    parser.add_argument("--config_name", type=str, default=DEFAULT_CONFIG_NAME, help="Name of the PPO configuration.")
    return parser.parse_args(argv)


def make_ppo_config(config_name=DEFAULT_CONFIG_NAME):
    config_path = "."

    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name=config_name)
    ppo_config = from_dict(data_class=RLVRConfig, data=OmegaConf.to_container(cfg, resolve=True))

    return ppo_config


def test_make_ppo_config():
    ppo_config = make_ppo_config()
    print(ppo_config)


@pytest.mark.skipif(
    os.environ.get("RUN_PIPELINE_INTEGRATION") != "1",
    reason="Full pipeline integration run is disabled by default.",
)
def test_ppo_pipeline(config_name=DEFAULT_CONFIG_NAME):

    ppo_config = make_ppo_config(config_name)

    init()

    from roll.pipeline.rlvr.rlvr_pipeline import RLVRPipeline
    pipeline = RLVRPipeline(pipeline_config=ppo_config)

    pipeline.run()


if __name__ == "__main__":
    cli_args = parse_args()
    test_ppo_pipeline(cli_args.config_name)
