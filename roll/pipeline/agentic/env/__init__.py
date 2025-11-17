"""
base agentic codes reference: https://github.com/RAGEN-AI/RAGEN
"""
import gem

from roll.utils.logging import get_logger
logger = get_logger()

gem.register("sokoban", entry_point="roll.pipeline.agentic.env.sokoban:SokobanEnv")
gem.register("frozen_lake", entry_point="roll.pipeline.agentic.env.frozen_lake:FrozenLakeEnv")
gem.register("roll_math", entry_point="roll.pipeline.agentic.env.gem.math_env:MathEnv")
gem.register("roll_code", entry_point="roll.pipeline.agentic.env.gem.code_env:CodeEnv")
gem.register("roll_qa", entry_point="roll.pipeline.agentic.env.gem.qa_env:QaEnv")
gem.register("sokoban_sandbox", entry_point="roll.pipeline.agentic.env.sandbox:SokobanSandboxEnv")


try:
    # add webshop-minimal to PYTHONPATH
    import os
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = "../../../../third_party/webshop-minimal"
    module_path = os.path.join(current_dir, relative_path)
    sys.path.append(module_path)
    gem.register("webshop", entry_point="roll.pipeline.agentic.env.webshop.env:WebShopEnv")

except Exception as e:
    logger.info(f"Failed to import webshop: {e}")
