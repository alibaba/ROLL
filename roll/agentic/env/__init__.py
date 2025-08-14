"""
base agentic codes reference: https://github.com/RAGEN-AI/RAGEN
"""
import gem

from roll.utils.logging import get_logger
logger = get_logger()

gem.register(id="sokoban", entry_point="roll.agentic.env.sokoban:SokobanEnv")
gem.register(id="frozen_lake", entry_point="roll.agentic.env.frozen_lake:FrozenLakeEnv")


try:
    # add webshop-minimal to PYTHONPATH
    import os
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = "../../../third_party/webshop-minimal"
    module_path = os.path.join(current_dir, relative_path)
    sys.path.append(module_path)

    from .webshop.env import WebShopEnv
    # gem.register(id="webshop", entry_point="roll.agentic.env.frozen_lake:FrozenLakeEnv")
    logger.warning(f"webshop interface is not ready yet, please wait")
except Exception as e:
    logger.info(f"Failed to import webshop: {e}")
