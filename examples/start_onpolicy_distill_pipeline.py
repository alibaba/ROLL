"""
On-Policy Distill Pipeline Launcher

Supports both RLVR and Agentic pipelines based on `pure_opd_pipeline_type` config field:
- 'rlvr' (default): Uses RLVRConfig + RLVRPipeline
- 'agentic': Uses AgenticConfig + AgenticPipeline
"""

import argparse

from dacite import from_dict, Config
from hydra import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.pipeline.rlvr.rlvr_pipeline import RLVRPipeline
from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline

def main():
    parser = argparse.ArgumentParser(description="On-Policy Distill Pipeline")
    parser.add_argument(
        "--config_path",
        type=str,
        default="examples/qwen3-8B-onpolicy-distill-megatron",
        help="Directory path where the config file is located"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="onpolicy_distill_config",
        help="Name of the config file (without extension)"
    )

    args = parser.parse_args()

    # Initialize Hydra
    initialize(config_path=args.config_path, job_name="onpolicy_distill")
    cfg = compose(config_name=args.config_name)

    # Print configuration
    print("=" * 80)
    print("On-Policy Distill Pipeline Config:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("=" * 80)

    # Convert to dict
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Force set is_pure_opd=True (this launcher is for pure OPD mode only)
    config_dict["is_pure_opd"] = True

    # Determine pipeline type from config
    pure_opd_pipeline_type = config_dict.get("pure_opd_pipeline_type", "rlvr")

    # Configure dacite to allow internal fields (prefixed with _)
    dacite_config = Config(check_types=False)

    if pure_opd_pipeline_type == "agentic":
        print("OPD pipeline type: agentic")
        pipeline_config = from_dict(data_class=AgenticConfig, data=config_dict, config=dacite_config)
        pipeline_cls = AgenticPipeline
    else:
        print("OPD pipeline type: rlvr")
        pipeline_config = from_dict(data_class=RLVRConfig, data=config_dict, config=dacite_config)
        pipeline_cls = RLVRPipeline

    # Initialize Ray
    init()

    # Create and run pipeline
    pipeline = pipeline_cls(pipeline_config=pipeline_config)
    pipeline.run()


if __name__ == "__main__":
    main()