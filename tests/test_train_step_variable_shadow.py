"""Test that loop variables in train_step don't shadow the outer `data` variable.

Bug: In CriticWorker.train_step and diffusion ActorWorker.train_step, the for-loop
used `data` as both the outer batch variable and the loop iteration variable:

    for batch_idx, data in tqdm(enumerate(dataloader), ...):
        ...
    data.to("cpu")  # Only releases last mini-batch, not the full batch

This causes `data.to("cpu")` after the loop to move only the last mini-batch to CPU,
leaking the full batch on GPU memory.

Fix: Rename loop variable to `mini_batch`.
"""

import ast
import textwrap


def _get_for_loop_targets(source: str) -> list:
    """Extract all for-loop target variable names from source code."""
    tree = ast.parse(textwrap.dedent(source))
    targets = []
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            # Handle `for batch_idx, var in ...`
            target = node.target
            if isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        targets.append(elt.id)
            elif isinstance(target, ast.Name):
                targets.append(target.id)
    return targets


class TestCriticWorkerTrainStepNoShadow:
    """Verify CriticWorker.train_step loop variable doesn't shadow outer `data`."""

    def test_base_worker_critic_train_step_no_data_shadow(self):
        """The for-loop in CriticWorker.train_step must not use `data` as loop variable."""
        import roll.pipeline.base_worker as mod
        import inspect

        source = inspect.getsource(mod.CriticWorker.train_step)
        targets = _get_for_loop_targets(source)
        assert "data" not in targets, (
            "CriticWorker.train_step for-loop uses `data` as loop variable, "
            "which shadows the outer `data` and causes only the last mini-batch "
            "to be moved to CPU instead of the full batch (GPU memory leak)."
        )

    def test_diffusion_actor_worker_train_step_no_data_shadow(self):
        """The for-loop in diffusion ActorWorker.train_step must not use `data` as loop variable."""
        import roll.pipeline.diffusion.reward_fl.actor_worker as mod
        import inspect

        source = inspect.getsource(mod.ActorWorker.train_step)
        targets = _get_for_loop_targets(source)
        assert "data" not in targets, (
            "diffusion ActorWorker.train_step for-loop uses `data` as loop variable, "
            "which shadows the outer `data` and causes only the last mini-batch "
            "to be moved to CPU instead of the full batch (GPU memory leak)."
        )

    def test_loop_variable_is_mini_batch(self):
        """The loop variable should be renamed to `mini_batch`."""
        import roll.pipeline.base_worker as mod
        import roll.pipeline.diffusion.reward_fl.actor_worker as diffusion_mod
        import inspect

        for cls, name in [
            (mod.CriticWorker, "CriticWorker"),
            (diffusion_mod.ActorWorker, "diffusion ActorWorker"),
        ]:
            source = inspect.getsource(cls.train_step)
            targets = _get_for_loop_targets(source)
            assert "mini_batch" in targets, (
                f"{name}.train_step for-loop should use `mini_batch` as loop variable, "
                f"got targets: {targets}"
            )
