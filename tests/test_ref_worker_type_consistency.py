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
    """When use_ref_model=False, `worker` must be `self.actor_train` (Cluster), not `self.pipeline_config.actor_train` (WorkerConfig)."""
    import roll.pipeline.rlvr.rlvr_pipeline as mod

    source = inspect.getsource(mod.RLVRPipeline)

    # The buggy pattern: `self.pipeline_config.actor_train` used where `self.actor_train` is needed
    # The fix ensures `worker = ... else self.actor_train` (without pipeline_config prefix)
    #
    # We check: in the line that assigns `worker = ...`, the else-branch must NOT
    # reference `self.pipeline_config.actor_train`
    tree = ast.parse(textwrap.dedent(source))

    found_worker_assign = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        # Look for: worker = <ternary>
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "worker":
                if isinstance(node.value, ast.IfExp):
                    found_worker_assign = True
                    # Check the orelse (else branch) of the ternary
                    orelse = node.value.orelse
                    # It should be self.actor_train, NOT self.pipeline_config.actor_train
                    source_segment = ast.dump(orelse)
                    assert "pipeline_config" not in source_segment, (
                        "Bug: `worker` assignment else-branch references "
                        "`self.pipeline_config.actor_train` (WorkerConfig) instead of "
                        "`self.actor_train` (Cluster). WorkerConfig has no `dp_size` property."
                    )

    assert found_worker_assign, (
        "Could not find `worker = ... if ... else ...` ternary assignment in RLVRPipeline. "
        "The code structure may have changed."
    )


def test_worker_config_has_no_dp_size():
    """WorkerConfig should NOT have dp_size - it's only on Cluster."""
    from roll.configs.worker_config import WorkerConfig

    assert not hasattr(WorkerConfig, "dp_size"), (
        "WorkerConfig should not have dp_size attribute; "
        "dp_size is a property of Cluster, not WorkerConfig."
    )
