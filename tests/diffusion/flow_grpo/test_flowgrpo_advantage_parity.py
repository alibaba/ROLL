import ast
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from roll.pipeline.diffusion.utils import compute_grpo_outcome_advantage as roll_flow_grpo


def _load_verl_flow_grpo():
    repo_root = Path(__file__).resolve().parents[4]
    source_path = repo_root / "verl" / "verl" / "trainer" / "ppo" / "core_algos.py"
    module_ast = ast.parse(source_path.read_text(), filename=str(source_path))

    for node in module_ast.body:
        if isinstance(node, ast.FunctionDef) and node.name == "compute_flow_grpo_outcome_advantage":
            function_ast = deepcopy(node)
            function_ast.decorator_list = []
            isolated_module = ast.Module(body=[function_ast], type_ignores=[])
            ast.fix_missing_locations(isolated_module)
            namespace = {
                "DictConfig": object,
                "Optional": Optional,
                "defaultdict": defaultdict,
                "np": np,
                "torch": torch,
            }
            exec(compile(isolated_module, str(source_path), "exec"), namespace)
            return namespace["compute_flow_grpo_outcome_advantage"]

    raise AssertionError("Failed to load verl Flow-GRPO advantage function from source.")


def test_roll_flow_grpo_matches_verl_for_random_input():
    verl_flow_grpo = _load_verl_flow_grpo()

    torch.manual_seed(7)
    batch_size = 12
    num_steps = 5
    scores = torch.randn(batch_size, 1, dtype=torch.float32)
    group_ids = np.array([idx // 3 for idx in range(batch_size)])
    response_mask = torch.ones((batch_size, num_steps), dtype=torch.float32)

    roll_advantages, roll_returns = roll_flow_grpo(
        scores=scores.squeeze(-1),
        group_ids=group_ids.tolist(),
        num_steps=num_steps,
    )
    verl_advantages, verl_returns = verl_flow_grpo(
        token_level_rewards=scores,
        response_mask=response_mask,
        index=group_ids,
    )

    torch.testing.assert_close(roll_advantages, verl_advantages, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(roll_returns, verl_returns, rtol=0.0, atol=1e-6)
