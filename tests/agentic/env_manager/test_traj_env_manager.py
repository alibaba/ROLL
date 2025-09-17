"""
usage:

conda create -n python310_torch260_em  python=3.10

pip3 install torch torchvision torchaudio py-cpuinfo
pip install -r requirements_em_local_debug.txt

python tests/agentic/env_manager/test_traj_env_manager.py
"""
import threading

import ray

from roll.distributed.scheduler.rollout_scheduler import GroupQueueManager
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider, default_processor_provider
from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.pipeline.agentic.env_manager.step_env_manager import StepEnvManager
from roll.pipeline.agentic.env_manager.traj_env_manager import TrajEnvManager
from roll.pipeline.agentic.env_manager.vl_traj_env_manager import VLTrajEnvManager
from tests.agentic.env_manager.config_load_utils import make_pipeline_config


def test_debug_traj_env_manager():
    config_path = ""
    config_name = "traj_env_manager_debug"

    pipeline_config: AgenticConfig = make_pipeline_config(config_path, config_name, AgenticConfig)

    pipeline_config.model_download_type = "MODELSCOPE"
    pipeline_config.async_generation_ratio = 2

    worker_config = pipeline_config.train_env_manager
    tokenizer = default_tokenizer_provider(model_args=worker_config.model_args)
    generate_scheduler = None

    ray.init(log_to_driver=True)

    output_queue = GroupQueueManager.remote(config=pipeline_config, env_manager_config=worker_config, mode="train")

    env_manager = TrajEnvManager(worker_config=worker_config,
                                 pipeline_config=pipeline_config,
                                 env_config=worker_config.env_configs[0][0],
                                 tokenizer=tokenizer,
                                 generate_scheduler=generate_scheduler,
                                 output_queue=output_queue,
                                 thread_lock=threading.Lock(),
                                 mode="train")

    data = DataProto(meta_info={"current_step": 0, "seed": 0})
    env_manager.run_rollout_loop(data=data)

    batch = ray.get(output_queue.get_batch.remote(batch_size=pipeline_config.rollout_batch_size))
    print(batch)


def test_debug_vl_traj_env_manager():
    config_path = ""
    config_name = "vl_traj_env_manager_debug"

    pipeline_config: AgenticConfig = make_pipeline_config(config_path, config_name, AgenticConfig)
    pipeline_config.model_download_type = "MODELSCOPE"
    pipeline_config.async_generation_ratio = 2
    worker_config = pipeline_config.train_env_manager
    tokenizer = default_tokenizer_provider(model_args=worker_config.model_args)
    processor = default_processor_provider(model_args=worker_config.model_args)
    generate_scheduler = None

    ray.init(log_to_driver=True)

    output_queue = GroupQueueManager.remote(config=pipeline_config, env_manager_config=worker_config, mode="train")

    env_manager = VLTrajEnvManager(worker_config=worker_config,
                                     pipeline_config=pipeline_config,
                                     env_config=worker_config.env_configs[0][0],
                                     tokenizer=tokenizer,
                                     processor=processor,
                                     generate_scheduler=generate_scheduler,
                                     output_queue=output_queue,
                                     thread_lock=threading.Lock(),
                                     mode="train")

    data = DataProto(meta_info={"current_step": 0, "seed": 0})
    env_manager.run_rollout_loop(data=data)

    batch = ray.get(output_queue.get_batch.remote(batch_size=pipeline_config.rollout_batch_size))
    print(batch)


def test_debug_step_env_manager():
    config_path = ""
    config_name = "step_env_manager_debug"

    pipeline_config: AgenticConfig = make_pipeline_config(config_path, config_name, AgenticConfig)

    pipeline_config.model_download_type = "MODELSCOPE"
    pipeline_config.async_generation_ratio = 2

    worker_config = pipeline_config.train_env_manager
    tokenizer = default_tokenizer_provider(model_args=worker_config.model_args)
    generate_scheduler = None

    ray.init(log_to_driver=True)

    output_queue = GroupQueueManager.remote(config=pipeline_config, env_manager_config=worker_config, mode="train")

    env_manager = StepEnvManager(worker_config=worker_config,
                                 pipeline_config=pipeline_config,
                                 env_config=worker_config.env_configs[0][0],
                                 tokenizer=tokenizer,
                                 generate_scheduler=generate_scheduler,
                                 output_queue=output_queue,
                                 thread_lock=threading.Lock(),
                                 mode="train")

    data = DataProto(meta_info={"current_step": 0, "seed": 0})
    env_manager.run_rollout_loop(data=data)

    batch = ray.get(output_queue.get_batch.remote(batch_size=pipeline_config.rollout_batch_size))
    print(batch)



if __name__ == '__main__':
    test_debug_traj_env_manager()