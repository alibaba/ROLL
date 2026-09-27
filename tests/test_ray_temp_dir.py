from types import SimpleNamespace

import pytest

from roll.utils.ray_temp_dir import (
    owned_ray_process_ids,
    resolve_ray_session_dir,
    resolve_ray_temp_dir,
    should_force_new_cluster,
)


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


def test_dedicated_temp_dir_forces_an_isolated_cluster(monkeypatch, tmp_path):
    monkeypatch.delenv("ROLL_RAY_FORCE_NEW_CLUSTER", raising=False)
    assert should_force_new_cluster(tmp_path / "ray") is True


def test_explicit_force_without_temp_dir_is_supported(monkeypatch):
    monkeypatch.setenv("ROLL_RAY_FORCE_NEW_CLUSTER", "true")
    assert should_force_new_cluster(None) is True


def test_session_dir_must_resolve_inside_dedicated_temp_dir(tmp_path):
    temp_dir = tmp_path / "ray"
    temp_dir.mkdir()
    session = temp_dir / "session_123"
    session.mkdir()
    (temp_dir / "session_latest").symlink_to(session, target_is_directory=True)
    assert resolve_ray_session_dir(temp_dir) == session.resolve()

    (temp_dir / "session_latest").unlink()
    outside = tmp_path / "outside"
    outside.mkdir()
    (temp_dir / "session_latest").symlink_to(outside, target_is_directory=True)
    assert resolve_ray_session_dir(temp_dir) is None


def test_owned_process_filter_requires_exact_session_marker():
    table = """
      101 raylet --session-dir=/data/ray/session_123
      102 gcs_server --log_dir=/data/ray/session_123/logs
      103 raylet --session-dir=/data/ray/session_1234
      104 python train.py
    """
    assert owned_ray_process_ids("/data/ray/session_123", table) == [101, 102]
