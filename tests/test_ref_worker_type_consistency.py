"""Test that reference log prob computation uses Cluster (not WorkerConfig) for dp_size.

Bug: In RLVRPipeline._train, when use_ref_model=False:

    worker_config = self.pipeline_config.reference if self.use_ref_model else self.pipeline_config.actor_train
    worker = self.reference if self.use_ref_model else self.pipeline_config.actor_train  # BUG

The `worker` variable is set to `self.pipeline_config.actor_train` (a WorkerConfig),
but it should be `self.actor_train` (a Cluster). WorkerConfig has no `dp_size` attribute,
so `worker.dp_size` on line 548 raises AttributeError.

Fix: Change `self.pipeline_config.actor_train` to `self.actor_train` on that line.
"""

import ast
import inspect
import textwrap


def test_ref_worker_uses_cluster_not_config():
    """When use_ref_model=False, log-prob computation must use `self.actor_train` (Cluster), not WorkerConfig."""
    import roll.pipeline.rlvr.rlvr_pipeline as mod

    source = inspect.getsource(mod.RLVRPipeline)
    tree = ast.parse(textwrap.dedent(source))

    def is_self_attr(node: ast.AST, *attrs: str) -> bool:
        expected = ("self", *attrs)
        current = node
        actual = []
        while isinstance(current, ast.Attribute):
            actual.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            actual.append(current.id)
        return tuple(reversed(actual)) == expected

    def is_not_self_use_ref_model(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.Not)
            and is_self_attr(node.operand, "use_ref_model")
        )

    lora_ref_branch = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and is_not_self_use_ref_model(node.test):
            lora_ref_branch = node
            break

    assert lora_ref_branch is not None, (
        "Could not find the `if not self.use_ref_model` branch in RLVRPipeline. "
        "The reference log-prob code structure may have changed."
    )

    found_actor_train_dp_size = False
    found_actor_train_compute_log_probs = False
    for node in ast.walk(lora_ref_branch):
        if isinstance(node, ast.Attribute) and node.attr == "dp_size":
            assert is_self_attr(node.value, "actor_train"), (
                "`dp_size` in the LoRA reference branch must come from "
                "`self.actor_train` (Cluster), not WorkerConfig or another alias."
            )
            found_actor_train_dp_size = True

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "compute_log_probs":
                assert is_self_attr(node.func.value, "actor_train"), (
                    "`compute_log_probs` in the LoRA reference branch must be called on "
                    "`self.actor_train` (Cluster), not WorkerConfig."
                )
                found_actor_train_compute_log_probs = True

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "worker":
                    assert "pipeline_config" not in ast.dump(node.value), (
                        "Bug: `worker` is assigned from `self.pipeline_config.actor_train` "
                        "(WorkerConfig) instead of `self.actor_train` (Cluster)."
                    )

    assert found_actor_train_dp_size, "LoRA reference branch should use `self.actor_train.dp_size`."
    assert found_actor_train_compute_log_probs, (
        "LoRA reference branch should call `self.actor_train.compute_log_probs`."
    )


def test_worker_config_has_no_dp_size():
    """WorkerConfig should NOT have dp_size - it's only on Cluster."""
    from roll.configs.worker_config import WorkerConfig

    assert not hasattr(WorkerConfig, "dp_size"), (
        "WorkerConfig should not have dp_size attribute; "
        "dp_size is a property of Cluster, not WorkerConfig."
    )
