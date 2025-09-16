from vllm.v1.executor.ray_distributed_executor import RayDistributedExecutor

from roll.third_party.vllm.vllm_0_10_0.ray_distributed_executor import ( 
    CustomRayDistributedExecutor as CustomRayDistributedExecutorV0)

# Force RayDistributedExecutor to come before CustomRayDistributedExecutorV0
# to ensure correct method resolution order (MRO) and override behavior.
class CustomRayDistributedExecutor(RayDistributedExecutor, CustomRayDistributedExecutorV0):
    pass
