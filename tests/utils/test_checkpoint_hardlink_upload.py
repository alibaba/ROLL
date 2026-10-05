"""An immutable same-filesystem checkpoint need not be copied a second time."""
import errno
import os
import shutil

import pytest

from roll.utils.upload_utils import FileSystemUploader


def source(tmp_path, name, payload=b"checkpoint weights"):
    path = tmp_path / name
    path.mkdir()
    (path / "weights").write_bytes(payload)
    return path


def test_default_upload_still_copies_mutable_source(tmp_path):
    local = source(tmp_path, "local")
    destination = tmp_path / "archive"
    FileSystemUploader(destination).upload("checkpoint-3", local)
    archived = destination / "checkpoint-3/weights"
    assert not os.path.samefile(local / "weights", archived)
    (local / "weights").write_bytes(b"subsequent write")
    assert archived.read_bytes() == b"checkpoint weights"


def test_opt_in_links_payload_and_survives_local_cleanup(tmp_path):
    local = source(tmp_path, "local")
    inode = (local / "weights").stat().st_ino
    destination = tmp_path / "archive"
    FileSystemUploader(destination, use_hard_links=True).upload("checkpoint-3", local)
    shutil.rmtree(local)
    archived = destination / "checkpoint-3/weights"
    assert archived.stat().st_ino == inode
    assert archived.read_bytes() == b"checkpoint weights"
    assert not local.exists()


def test_replacing_existing_destination_does_not_mutate_its_original_source(tmp_path):
    first = source(tmp_path, "first", b"old payload")
    second = source(tmp_path, "second", b"new payload")
    destination = tmp_path / "archive"
    uploader = FileSystemUploader(destination, use_hard_links=True)
    uploader.upload("checkpoint-3", first)
    uploader.upload("checkpoint-3", second)
    archived = destination / "checkpoint-3/weights"
    assert archived.read_bytes() == b"new payload"
    assert (first / "weights").read_bytes() == b"old payload"
    assert os.path.samefile(second / "weights", archived)


def test_cross_filesystem_link_falls_back_to_independent_copy(tmp_path, monkeypatch):
    local = source(tmp_path, "local")
    destination = tmp_path / "archive"

    def cross_device(*args, **kwargs):
        raise OSError(errno.EXDEV, "injected cross-device link")

    monkeypatch.setattr(os, "link", cross_device)
    FileSystemUploader(destination, use_hard_links=True).upload("checkpoint-3", local)
    archived = destination / "checkpoint-3/weights"
    assert archived.read_bytes() == b"checkpoint weights"
    assert not os.path.samefile(local / "weights", archived)


def test_link_failure_propagates_and_keeps_previous_archive_and_local_state(tmp_path, monkeypatch):
    local = source(tmp_path, "local", b"new payload")
    destination = tmp_path / "archive"
    checkpoint = destination / "checkpoint-3"
    checkpoint.mkdir(parents=True)
    (checkpoint / "weights").write_bytes(b"previous payload")

    def no_space(*args, **kwargs):
        raise OSError(errno.ENOSPC, "injected full filesystem")

    monkeypatch.setattr(os, "link", no_space)
    uploader = FileSystemUploader(destination, use_hard_links=True)
    with pytest.raises(Exception, match="full filesystem"):
        uploader.upload("checkpoint-3", local)
    assert (local / "weights").read_bytes() == b"new payload"
    assert (checkpoint / "weights").read_bytes() == b"previous payload"
    assert [p.name for p in checkpoint.iterdir()] == ["weights"]


def test_rank_directories_merge_without_modifying_other_rank_source(tmp_path):
    first = source(tmp_path, "rank0", b"rank zero metadata")
    second = source(tmp_path, "rank1", b"rank one metadata")
    (first / "rank0.distcp").write_bytes(b"rank zero state")
    (second / "rank1.distcp").write_bytes(b"rank one state")
    destination = tmp_path / "archive"
    uploader = FileSystemUploader(destination, use_hard_links=True)
    uploader.upload("checkpoint-3", first)
    uploader.upload("checkpoint-3", second)
    assert (first / "weights").read_bytes() == b"rank zero metadata"
    for local, filename in [(first, "rank0.distcp"), (second, "rank1.distcp")]:
        assert os.path.samefile(local / filename, destination / "checkpoint-3" / filename)


def test_failed_fallback_copy_does_not_publish_partial_payload(tmp_path, monkeypatch):
    local = source(tmp_path, "local", b"new payload")
    destination = tmp_path / "archive"
    checkpoint = destination / "checkpoint-3"
    checkpoint.mkdir(parents=True)
    (checkpoint / "weights").write_bytes(b"previous payload")

    def cross_device(*args, **kwargs):
        raise OSError(errno.EXDEV, "injected cross-device link")

    def partial_copy(src, dst):
        with open(dst, "wb") as stream:
            stream.write(b"partial")
        raise OSError(errno.EIO, "injected copy failure")

    monkeypatch.setattr(os, "link", cross_device)
    monkeypatch.setattr(shutil, "copy2", partial_copy)
    with pytest.raises(Exception, match="copy failure"):
        FileSystemUploader(destination, use_hard_links=True).upload("checkpoint-3", local)
    assert (checkpoint / "weights").read_bytes() == b"previous payload"
    assert (local / "weights").read_bytes() == b"new payload"
    assert [p.name for p in checkpoint.iterdir()] == ["weights"]


def test_symlink_source_publishes_its_payload_as_a_regular_file(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    payload = tmp_path / "payload"
    payload.write_bytes(b"checkpoint weights")
    (local / "weights").symlink_to(payload)
    destination = tmp_path / "archive"
    FileSystemUploader(destination, use_hard_links=True).upload("checkpoint-3", local)
    archived = destination / "checkpoint-3/weights"
    assert not archived.is_symlink()
    assert os.path.samefile(payload, archived)
    assert archived.read_bytes() == b"checkpoint weights"


def test_checkpoint_manager_forwards_link_option_and_cleans_up(tmp_path):
    pytest.importorskip("ray")
    from roll.utils.checkpoint_manager import CheckpointManager
    local = source(tmp_path, "local")
    inode = (local / "weights").stat().st_ino
    destination = tmp_path / "archive"
    manager = CheckpointManager(dict(type="file_system", output_dir=str(destination), use_hard_links=True))
    manager.upload("checkpoint-3", str(local))
    assert not local.exists()
    assert (destination / "checkpoint-3/weights").stat().st_ino == inode
