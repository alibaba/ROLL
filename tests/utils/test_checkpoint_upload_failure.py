"""Synchronous checkpoint upload errors must reach the caller and preserve state."""
from pathlib import Path
import pytest
from roll.utils.checkpoint_manager import CheckpointManager


def test_failed_filesystem_upload_raises_and_retains_local_checkpoint(tmp_path):
    local=tmp_path/'local';local.mkdir();(local/'weights').write_bytes(b'checkpoint payload')
    target=tmp_path/'target';target.mkdir()
    (target/'checkpoint-3').write_text('not a directory')
    manager=CheckpointManager(dict(type='file_system',output_dir=str(target)))
    with pytest.raises(FileExistsError):manager.upload('checkpoint-3',str(local))
    assert (local/'weights').read_bytes()==b'checkpoint payload'
    assert (target/'checkpoint-3').read_text()=='not a directory'


@pytest.mark.parametrize('keep_local', [False,True])
def test_successful_upload_preserves_payload_and_requested_local_retention(tmp_path,keep_local):
    local=tmp_path/'local';local.mkdir();(local/'weights').write_bytes(b'checkpoint payload')
    target=tmp_path/'target'
    manager=CheckpointManager(dict(type='file_system',output_dir=str(target)))
    manager.upload('checkpoint-3',str(local),keep_local_file=keep_local)
    assert (target/'checkpoint-3/weights').read_bytes()==b'checkpoint payload'
    assert local.exists() is keep_local
