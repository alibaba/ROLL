"""CLI entry point for causal ODE trajectory distillation."""

import argparse

from dacite import from_dict
try:
    from hydra import compose, initialize
except ImportError:
    from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.diffusion.ode_distillation.ode_distillation_config import ODEDistillationConfig
from roll.pipeline.diffusion.ode_distillation.ode_distillation_pipeline import ODEDistillationPipeline


def main() -> None:
    """Start the ODE distillation pipeline from a Hydra config."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="config")
    parser.add_argument("--config_name", default="ode_distillation_config")
    args, overrides = parser.parse_known_args()

    initialize(config_path=args.config_path, job_name="ode_distillation_train")
    cfg = compose(config_name=args.config_name, overrides=overrides)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    pipeline_config = from_dict(
        data_class=ODEDistillationConfig,
        data=OmegaConf.to_container(cfg, resolve=True),
    )

    init()
    ODEDistillationPipeline(pipeline_config=pipeline_config).run()


if __name__ == "__main__":
    main()
