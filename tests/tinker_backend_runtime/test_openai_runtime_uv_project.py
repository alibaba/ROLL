from __future__ import annotations

import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UV_PROJECT = PROJECT_ROOT / "roll" / "pipeline" / "tinker_backend_runtime" / "openai_runtime" / "pyproject.toml"


def test_openai_runtime_has_dedicated_uv_project() -> None:
    assert UV_PROJECT.exists()

    project = tomllib.loads(UV_PROJECT.read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]

    assert project["project"]["name"] == "roll-tinker-openai-runtime"
    assert any(dependency.startswith("openai==") for dependency in dependencies)
    assert any(dependency.startswith("PyYAML==") for dependency in dependencies)
    assert any(dependency.startswith("sglang==") for dependency in dependencies)
    assert not any(
        dependency.split("==", 1)[0].lower() in {"torch", "vllm", "ray"}
        for dependency in dependencies
    )


def test_openai_runtime_uv_project_contains_runtime_entrypoint() -> None:
    runtime_entrypoint = UV_PROJECT.with_name("runtime_pipeline_openai.py")

    assert runtime_entrypoint.exists()
    assert runtime_entrypoint.parent.name == "openai_runtime"
    assert not list(UV_PROJECT.parent.glob("run_*pipeline_openai.py"))
