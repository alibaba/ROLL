from __future__ import annotations

from pathlib import Path

import pytest

from roll.pipeline.tinker_backend_runtime import model_paths


def test_resolve_model_path_keeps_existing_local_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_paths,
        "_download_modelscope_snapshot",
        lambda _model_id: pytest.fail("existing paths must not be downloaded"),
    )

    assert model_paths.resolve_model_path(str(tmp_path), use_modelscope=True) == str(tmp_path.resolve())


def test_resolve_model_path_rejects_missing_absolute_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing-model"

    with pytest.raises(FileNotFoundError, match="model path does not exist"):
        model_paths.resolve_model_path(str(missing), use_modelscope=True)


def test_resolve_model_path_downloads_modelscope_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    calls: list[str] = []

    def fake_download(model_id: str) -> str:
        calls.append(model_id)
        return str(model_dir)

    monkeypatch.setattr(model_paths, "_download_modelscope_snapshot", fake_download)

    assert model_paths.resolve_model_path("Qwen/example", use_modelscope=True) == str(model_dir.resolve())
    assert calls == ["Qwen/example"]


@pytest.mark.parametrize("download_type", ["OPENLM_HUB", "openlm_hub"])
def test_resolve_model_path_rejects_internal_download(download_type: str) -> None:
    with pytest.raises(ValueError, match="public.*Hugging Face"):
        model_paths.resolve_model_path("Qwen/example", download_type=download_type)


def test_resolve_model_path_rejects_internal_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_DOWNLOAD_TYPE", "OPENLM_HUB")
    with pytest.raises(ValueError, match="OPENLM_HUB is unsupported"):
        model_paths.resolve_model_path("Qwen/example", use_modelscope=True)


def test_local_path_never_requires_internal_downloader(tmp_path: Path) -> None:
    assert model_paths.resolve_model_path(str(tmp_path), download_type="OPENLM_HUB") == str(tmp_path.resolve())


def test_resolve_model_path_leaves_repo_id_for_huggingface() -> None:
    assert model_paths.resolve_model_path("Qwen/example", use_modelscope=False) == "Qwen/example"
