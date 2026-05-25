import argparse
import json
from dataclasses import asdict

from dacite import from_dict
from hydra import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.agentic.agentic_config import AgenticConfig


DEFAULT_CONFIG_NAME = "agentic_pipeline_config"


def _align_resource_shape_with_config(ppo_config, configured_num_gpus_per_node):
    total_devices = []
    for value in vars(ppo_config).values():
        device_mapping = getattr(value, "device_mapping", None)
        if device_mapping is not None:
            total_devices.extend(device_mapping)

    if not total_devices or configured_num_gpus_per_node is None:
        return

    max_gpu_num = max(total_devices) + 1
    ppo_config.num_gpus_per_node = max(
        ppo_config.num_gpus_per_node,
        configured_num_gpus_per_node,
    )
    ppo_config.num_nodes = (max_gpu_num + ppo_config.num_gpus_per_node - 1) // ppo_config.num_gpus_per_node


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PPO Configuration")
    parser.add_argument("--config_name", type=str, default=DEFAULT_CONFIG_NAME, help="Name of the PPO configuration.")
    return parser.parse_args(argv)


def make_ppo_config(config_name=DEFAULT_CONFIG_NAME):
    config_path = "."

    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name=config_name)
    print(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    ppo_config = from_dict(data_class=AgenticConfig, data=cfg_dict)
    _align_resource_shape_with_config(ppo_config, cfg_dict.get("num_gpus_per_node"))
    return ppo_config


def test_make_ppo_config():
    ppo_config = make_ppo_config()
    print(ppo_config)


def test_ppo_pipeline(config_name=DEFAULT_CONFIG_NAME):
    from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline

    ppo_config = make_ppo_config(config_name)

    init()

    pipeline = AgenticPipeline(pipeline_config=ppo_config)

    pipeline.run()

    output_file = "ppo_pipeline.json"
    with open(output_file, "w") as f:
        json.dump(asdict(pipeline.state), f, ensure_ascii=False)


if __name__ == "__main__":
    cli_args = parse_args()
    test_ppo_pipeline(cli_args.config_name)
