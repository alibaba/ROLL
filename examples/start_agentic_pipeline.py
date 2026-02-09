import argparse

from dacite import from_dict
from hydra import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.utils.import_utils import safe_import_class
from roll.utils.str_utils import print_pipeline_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="The path of the main configuration file", default="config")
    parser.add_argument(
        "--config_name", help="The name of the main configuration file (without extension).", default="sppo_config"
    )
    args = parser.parse_args()

    initialize(config_path=args.config_path, job_name="app")
    cfg = compose(config_name=args.config_name)

    ppo_config = from_dict(data_class=AgenticConfig, data=OmegaConf.to_container(cfg, resolve=True))

    init()

    print_pipeline_config(ppo_config)

    pipeline_cls = getattr(cfg, "pipeline_cls", "roll.pipeline.agentic.agentic_pipeline.AgenticPipeline")
    if isinstance(pipeline_cls, str):
        pipeline_cls = safe_import_class(pipeline_cls)

    pipeline = pipeline_cls(pipeline_config=ppo_config)

    pipeline.run()


if __name__ == "__main__":
    main()
