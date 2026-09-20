from types import SimpleNamespace

import pytest
import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.sft.sft_worker import SFTWorker


class _TinySFTStrategy:
    def __init__(self):
        self.model = torch.nn.Linear(3, 1)

    @staticmethod
    def get_data_input(data):
        return data

    def forward_step(self, batch, forward_func):
        _, metrics = forward_func(batch, self.model(batch.batch["features"]))
        return metrics

    def train_step(self, batch, loss_func):
        loss, metrics = loss_func(batch, self.model(batch.batch["features"]))
        loss.backward()
        return metrics

    @staticmethod
    def op_compute_language_loss(output_tensor, labels, batch_num_tokens):
        loss = torch.nn.functional.mse_loss(output_tensor, labels, reduction="sum") / batch_num_tokens
        return loss, {"sft_train/loss@sum": loss.detach().item()}


def test_val_step_disables_autograd_without_disabling_later_training():
    torch.manual_seed(17)
    worker = SFTWorker.__new__(SFTWorker)
    worker.worker_config = SimpleNamespace(infer_batch_size=2)
    worker.strategy = _TinySFTStrategy()
    features = torch.tensor([[1.0, -2.0, 0.5], [0.25, 3.0, -1.0]])
    labels = torch.tensor([[0.75], [-1.25]])
    data = DataProto.from_dict(
        tensors={"features": features, "labels": labels},
        meta_info={"batch_num_tokens": {"labels": 2}},
    )

    with torch.no_grad():
        expected_loss = torch.nn.functional.mse_loss(
            worker.strategy.model(features), labels, reduction="sum"
        ).item() / 2

    saved_tensors = []

    def save_tensor(tensor):
        saved_tensors.append(tensor)
        return tensor

    with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(save_tensor, lambda tensor: tensor):
        assert torch.is_grad_enabled()
        output = worker.val_step(data)

    assert saved_tensors == []
    assert output.meta_info["metrics"]["sft_train/loss@sum"] == pytest.approx(expected_loss)
    assert all(parameter.grad is None for parameter in worker.strategy.model.parameters())

    worker.train_step(data)

    assert all(parameter.grad is not None for parameter in worker.strategy.model.parameters())
    assert any(torch.count_nonzero(parameter.grad).item() for parameter in worker.strategy.model.parameters())


class _CheckpointSFTStrategy:
    """Small real Adam checkpoint to isolate the worker initialization boundary."""

    def initialize(self, model_provider):
        torch.manual_seed(734)
        self.model = torch.nn.Linear(3, 1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)

    def update(self):
        self.optimizer.zero_grad()
        loss = self.model(torch.tensor([[1., -2., 3.]])).square().sum()
        loss.backward()
        self.optimizer.step()

    def load_checkpoint(self, load_dir, tag):
        from pathlib import Path

        state = torch.load(Path(load_dir) / "state.pt", weights_only=True)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])


@pytest.mark.parametrize("resume", [False, True, "explicit"])
def test_sft_initialize_restores_next_optimizer_update(tmp_path, monkeypatch, resume):
    from roll.pipeline.sft import sft_worker

    reference = _CheckpointSFTStrategy()
    reference.initialize(None)
    reference.update()
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    torch.save({"model": reference.model.state_dict(), "optimizer": reference.optimizer.state_dict()},
               checkpoint / "state.pt")
    if resume:
        reference.update()

    worker = SFTWorker.__new__(SFTWorker)
    worker.worker_name = "sft_resume_test"
    worker.worker_config = SimpleNamespace(model_args=SimpleNamespace(model_name_or_path=None))
    monkeypatch.setattr(sft_worker, "create_strategy", lambda worker: _CheckpointSFTStrategy())
    # Download caching uses a Ray actor. The fixture is already a local checkpoint;
    # keep real latest-directory selection and actual tensor/Adam deserialization.
    monkeypatch.setattr(sft_worker, "download_model", lambda path: path, raising=False)
    config = SimpleNamespace(
        resume_from_checkpoint=str(checkpoint) if resume == "explicit" else resume,
        checkpoint_config={"type": "file_system", "output_dir": str(tmp_path)},
    )
    worker.initialize(config)
    worker.strategy.update()
    torch.testing.assert_close(worker.strategy.model.state_dict(), reference.model.state_dict(), atol=0, rtol=0)
    torch.testing.assert_close(worker.strategy.optimizer.state_dict(), reference.optimizer.state_dict(), atol=0, rtol=0)
