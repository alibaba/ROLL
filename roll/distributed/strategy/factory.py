from typing import Union

from roll.distributed.executor.worker import Worker
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy


def create_strategy(worker: Worker) -> Union[InferenceStrategy, TrainStrategy]:
    strategy_name = worker.worker_config.strategy_args.strategy_name

    # Lazy import strategy to avoid cuda initialized
    if strategy_name == "deepspeed_infer":
        from roll.distributed.strategy.deepspeed_strategy import DeepSpeedInferStrategy as strategy_cls
    elif strategy_name == "deepspeed_train":
        from roll.distributed.strategy.deepspeed_strategy import DeepSpeedTrainStrategy as strategy_cls
    elif strategy_name == "diffusion_deepspeed_train":
        from roll.distributed.strategy.diffusion_strategy import DeepSpeedTrainStrategy as strategy_cls    
    elif strategy_name == "hf_infer":
        from roll.distributed.strategy.hf_strategy import HfInferStrategy as strategy_cls
    elif strategy_name == "vllm":
        from roll.distributed.strategy.vllm_strategy import VllmStrategy as strategy_cls
    elif strategy_name == "sglang":
        from roll.distributed.strategy.sglang_strategy import SgLangStrategy as strategy_cls
    elif strategy_name == "megatron_infer":
        from roll.distributed.strategy.megatron_strategy import MegatronInferStrategy as strategy_cls
    elif strategy_name == "megatron_train":
        from roll.distributed.strategy.megatron_strategy import MegatronTrainStrategy as strategy_cls
    elif strategy_name == "mock_infer":
        from roll.distributed.strategy.mock_strategy import MockInferStrategy as strategy_cls
    else:
        raise ValueError(f"Unknown strategy name: {strategy_name}")

    return strategy_cls(worker)
