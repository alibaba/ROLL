import os

import ray


@ray.remote
def get_visible_gpus(device_control_env_var: str):
    # Ray may export a different visibility variable (e.g. ONEAPI_DEVICE_SELECTOR for XPU); use Ray's IDs.
    if device_control_env_var not in os.environ:
        return [str(gpu_id) for gpu_id in ray.get_gpu_ids()]
    return os.environ.get(device_control_env_var, "").split(",")


@ray.remote
def get_node_rank():
    return int(os.environ.get("NODE_RANK", "0"))
