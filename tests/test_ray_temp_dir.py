from types import SimpleNamespace

import pytest

from roll.utils.ray_temp_dir import resolve_ray_temp_dir


def test_unset_ray_temp_dir_preserves_default(monkeypatch):
    monkeypatch.delenv("ROLL_RAY_TEMP_DIR", raising=False)
    assert resolve_ray_temp_dir() is None


def test_configured_ray_temp_dir_is_created_and_returned(tmp_path, monkeypatch):
    target = tmp_path / "ray-tmp"
    monkeypatch.setenv("ROLL_RAY_TEMP_DIR", str(target))
    monkeypatch.setenv("ROLL_RAY_TEMP_MIN_FREE_BYTES", "1")
    assert resolve_ray_temp_dir() == target.resolve()
    assert target.is_dir()


def test_configured_ray_temp_dir_rejects_low_free_space(tmp_path, monkeypatch):
    target = tmp_path / "ray-tmp"
    monkeypatch.setenv("ROLL_RAY_TEMP_DIR", str(target))
    monkeypatch.setenv("ROLL_RAY_TEMP_MIN_FREE_BYTES", "100")

    import roll.utils.ray_temp_dir as module

    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=99))
    with pytest.raises(RuntimeError, match="bytes free"):
        resolve_ray_temp_dir()
