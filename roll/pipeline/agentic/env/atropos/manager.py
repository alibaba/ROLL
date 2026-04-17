import importlib
import logging
from typing import Any, Dict, Optional, Type

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, ServerBaseline

logger = logging.getLogger(__name__)

def load_atropos_env_class(env_path: str) -> Type[BaseEnv]:
    """
    Dynamically load an Atropos environment class from a string.
    Format: 'module_path:ClassName'
    Example: 'atropos.environments.gsm8k_server:GSM8kEnv'
    """
    try:
        module_path, class_name = env_path.split(":")
        module = importlib.import_module(module_path)
        env_class = getattr(module, class_name)
        if not issubclass(env_class, BaseEnv):
            raise TypeError(f"{class_name} is not a subclass of BaseEnv")
        return env_class
    except Exception as e:
        logger.error(f"Failed to load Atropos environment from {env_path}: {e}")
        raise

def create_atropos_instance(
    env_class: Type[BaseEnv],
    env_config_dict: Dict[str, Any],
    server_configs: Optional[Any] = None
) -> BaseEnv:
    """
    Creates an instance of an Atropos environment with the provided config.
    
    This factory ensures the environment is initialized for controlled 
    rollout execution within the ROLL framework.
    """
    # Initialize default configs if not provided
    # Initialize default configs if not provided
    base_config, base_servers = env_class.config_init()

    # Merge provided config into the base config class
    # base_config is an instance of the environment-specific config class
    config_cls = type(base_config)
    env_config = config_cls(**{**base_config.model_dump(), **env_config_dict})
    
    # If no server configs provided, use the defaults from config_init
    if server_configs is None:
        server_configs = base_servers
        
    return env_class(
        config=env_config,
        server_configs=server_configs,
        slurm=False,
        testing=True # Default to testing mode for ROLL integration (avoids slurm/gpu check)
    )

async def safe_get_next_item(env: BaseEnv) -> Dict[str, Any]:
    """
    Safely get the next item from the environment, with fallback logic.
    """
    if hasattr(env, "get_next_item") and callable(env.get_next_item):
        try:
            return await env.get_next_item()
        except Exception as e:
            logger.warning(f"env.get_next_item() failed: {e}. Falling back.")
            
    # Fallback: Check if there's a dataset we can iterate
    if hasattr(env, "train") and isinstance(env.train, (list, tuple)):
        it = getattr(env, "_fallback_iter", 0)
        item = env.train[it % len(env.train)]
        env._fallback_iter = it + 1
        return item
    
    # Final fallback: Empty task (not ideal but avoids crash)
    logger.error("No valid data source found in Atropos environment.")
    return {"question": "No task provided.", "problem_statement": "No task provided."}
