import json
import os

from roll.configs.worker_config import StrategyArguments
from roll.distributed.scheduler.initialize import init
from roll.utils.logging import get_logger
from tests.distributed.strategy.make_baseline_config import \
    make_baseline_config
from tests.distributed.strategy.model_update.model_update_pipeline import \
    ModelUpdatePipeline

logger = get_logger()


def vllm_model_update_baseline():
    os.environ["RAY_PROFILING"] = "1"

    init()

    ppo_config = make_baseline_config(config_path="./model_update", config_name="model_update_baseline_config")
    # Enable stat logging for vLLM to allow metrics collection
    if (
        hasattr(ppo_config.actor_infer, "strategy_args")
        and ppo_config.actor_infer.strategy_args.strategy_name == "vllm"
    ):
        if "disable_log_stats" not in ppo_config.actor_infer.strategy_args.strategy_config:
            ppo_config.actor_infer.strategy_args.strategy_config["disable_log_stats"] = False

    pipeline = ModelUpdatePipeline(pipeline_config=ppo_config)

    metric_list = pipeline.run()
    generate_times = [metric["time/model_update"] for metric in metric_list[:-2]]
    total_time = sum(generate_times)

    logger.info(f"{json.dumps({'total_time': total_time, 'time_list': generate_times})}")

    output_file = "model_update_baseline.json"
    with open(output_file, "w") as f:
        json.dump(metric_list, f, ensure_ascii=False)


def ds_2_hf_model_update_baseline():
    os.environ["RAY_PROFILING"] = "1"

    init()

    ppo_config = make_baseline_config(config_path="./model_update", config_name="model_update_baseline_config")

    pipeline = ModelUpdatePipeline(pipeline_config=ppo_config)

    metric_list = pipeline.run()
    generate_times = [metric["time/model_update"] for metric in metric_list]
    total_time = sum(generate_times)

    logger.info(f"{json.dumps({'total_time': total_time, 'time_list': generate_times})}")


def fsdp2_train_model_update():
    os.environ["RAY_PROFILING"] = "1"

    init()

    ppo_config = make_baseline_config(config_path="./model_update", config_name="model_update_fsdp")
    # Enable stat logging for vLLM to allow metrics collection
    if (
        hasattr(ppo_config.actor_infer, "strategy_args")
        and ppo_config.actor_infer.strategy_args.strategy_name == "vllm"
    ):
        if "disable_log_stats" not in ppo_config.actor_infer.strategy_args.strategy_config:
            ppo_config.actor_infer.strategy_args.strategy_config["disable_log_stats"] = False

    pipeline = ModelUpdatePipeline(pipeline_config=ppo_config)

    metric_list = pipeline.run()
    generate_times = [metric["time/model_update"] for metric in metric_list[:-2]]
    total_time = sum(generate_times)

    logger.info(f"{json.dumps({'total_time': total_time, 'time_list': generate_times})}")

    output_file = "model_update_fsdp.json"
    with open(output_file, "w") as f:
        json.dump(metric_list, f, ensure_ascii=False)


if __name__ == "__main__":
    # vllm_model_update_baseline()
    # ds_2_hf_model_update_baseline()
    fsdp2_train_model_update()
