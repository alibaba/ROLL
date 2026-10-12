"""The rewards package must import without every worker's optional dependencies.

``base_pipeline.create_clusters_parallel`` pre-imports each configured worker
class by path, which imports ``roll.pipeline.rlvr.rewards``. When that package
eagerly re-exports a worker whose third-party scoring dependency is absent, the
import fails and every RLVR run aborts with "Failed to pre-import worker class",
no matter which reward worker the run configured.
"""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest

PACKAGE = "roll.pipeline.rlvr.rewards"
LAZY_DEPENDENCY = "mathruler"
LAZY_WORKER = "Geo3kRewardWorker"


@pytest.fixture
def without_lazy_dependency(monkeypatch: pytest.MonkeyPatch):
    """Reload the rewards package with the optional dependency unimportable."""
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == LAZY_DEPENDENCY or name.startswith(f"{LAZY_DEPENDENCY}."):
            raise ModuleNotFoundError(f"No module named '{LAZY_DEPENDENCY}'", name=name)
        return real_import(name, globals, locals, fromlist, level)

    for name in [n for n in sys.modules if n == PACKAGE or n.startswith(f"{PACKAGE}.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    for name in [n for n in sys.modules if n == LAZY_DEPENDENCY or n.startswith(f"{LAZY_DEPENDENCY}.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    return importlib.import_module(PACKAGE)


def test_package_imports_without_the_optional_dependency(without_lazy_dependency) -> None:
    assert without_lazy_dependency.__name__ == PACKAGE


def test_eagerly_exported_workers_resolve_without_the_optional_dependency(without_lazy_dependency) -> None:
    for name in ("MathRuleRewardWorker", "GSM8KMathRewardWorker", "RemoteRewardSystemWorker"):
        assert isinstance(getattr(without_lazy_dependency, name), type)


def test_worker_pre_import_path_succeeds_without_the_optional_dependency(without_lazy_dependency) -> None:
    """safe_import_class is what base_pipeline uses to resolve a cluster's worker."""
    from roll.utils.import_utils import safe_import_class

    resolved = safe_import_class(f"{PACKAGE}.multiple_choice_boxed_rule_reward_worker.MultipleChoiceBoxedRuleRewardWorker")
    assert isinstance(resolved, type)


def test_lazy_worker_still_raises_its_own_missing_dependency(without_lazy_dependency) -> None:
    with pytest.raises(ModuleNotFoundError, match=LAZY_DEPENDENCY):
        getattr(without_lazy_dependency, LAZY_WORKER)


def test_lazy_worker_is_advertised_and_unknown_names_still_fail(without_lazy_dependency) -> None:
    assert LAZY_WORKER in without_lazy_dependency.__all__
    assert LAZY_WORKER in dir(without_lazy_dependency)
    with pytest.raises(AttributeError):
        without_lazy_dependency.NoSuchRewardWorker


def test_lazy_worker_resolves_when_its_dependency_is_available() -> None:
    """With mathruler installed the export behaves exactly as the eager one did."""
    pytest.importorskip(LAZY_DEPENDENCY)
    package = importlib.import_module(PACKAGE)
    from roll.pipeline.rlvr.rewards.geo3k_reward_worker import Geo3kRewardWorker as direct

    assert getattr(package, LAZY_WORKER) is direct
