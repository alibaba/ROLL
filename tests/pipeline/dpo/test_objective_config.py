"""Exercise objective configuration through the DPO driver and worker loss on CPU."""

import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.dpo.actor_worker import ActorWorker
from roll.pipeline.dpo.dpo_pipeline import DPOPipeline


class _LocalActor:
    """Replace remote/model execution while retaining the real DPO worker loss."""

    def __init__(self):
        self.theta = torch.tensor(1.0, requires_grad=True)
        self.token_logps = torch.tensor([[-0.5, -0.5], [-1.0, -1.0]])
        self.worker = SimpleNamespace(strategy=SimpleNamespace(op_compute_log_probs=lambda **kw: kw["logits"]))
        self.loss = None
        self.gradient = None

    def compute_log_probs(self, data, blocking):
        return DataProto.from_dict(tensors={"log_probs": self.theta.detach() * self.token_logps})

    def train_step(self, data, blocking):
        # The driver runs under no_grad; real training happens in another worker.
        with torch.enable_grad():
            loss, metrics = ActorWorker.loss_func(self.worker, data, self.theta * self.token_logps)
            loss.backward()
        self.loss = loss.item()
        self.gradient = self.theta.grad.item()
        return DataProto(meta_info={"metrics": metrics})


@pytest.fixture
def pipeline(monkeypatch):
    pipeline = DPOPipeline.__new__(DPOPipeline)
    pipeline.actor_train = _LocalActor()
    pipeline.reference = SimpleNamespace(
        compute_log_probs=lambda data, blocking: DataProto.from_dict(tensors={"log_probs": torch.zeros(2, 2)})
    )
    batch = {
        "input_ids": torch.zeros((2, 3), dtype=torch.long),
        "attention_mask": torch.ones((2, 3), dtype=torch.long),
        "prompt_id_lens": torch.ones(2, dtype=torch.long),
    }
    pipeline.dataloader = [batch]
    pipeline.val_dataloader = [batch]
    pipeline.val_dataset = True
    pipeline.state = SimpleNamespace(step=-1, log_history=[])
    pipeline.tracker = Mock()
    pipeline.do_checkpoint = Mock()
    # Local cluster methods return materialized DataProto objects, not Ray refs.
    monkeypatch.setattr(DataProto, "materialize_concat", staticmethod(lambda data_refs: data_refs))
    monkeypatch.delenv("RAY_PROFILING", raising=False)
    return pipeline


@pytest.mark.parametrize(
    "ipo,beta,label_smoothing",
    [
        pytest.param(False, 0.1, 0.0, id="default-dpo"),
        pytest.param(False, 0.7, 0.0, id="nondefault-beta"),
        pytest.param(False, 0.1, 0.2, id="label-smoothing"),
        pytest.param(True, 0.1, 0.0, id="ipo"),
        pytest.param(True, 0.7, 0.0, id="ipo-nondefault-beta"),
    ],
)
def test_objective_config_controls_training_and_validation(pipeline, ipo, beta, label_smoothing):
    pipeline.pipeline_config = SimpleNamespace(
        ipo=ipo,
        beta=beta,
        label_smoothing=label_smoothing,
        actor_train=SimpleNamespace(training_args=SimpleNamespace(num_train_epochs=1)),
        is_offload_states=False,
        is_offload_optimizer_states_in_train_step=False,
        eval_steps=1,
        logging_steps=1,
        max_steps=1,
    )

    # One pair has policy log-ratio theta=1 and reference log-ratio zero.
    # Compute the oracle independently of the production loss function.
    if ipo:
        expected_loss = (1 - 1 / (2 * beta)) ** 2
        expected_gradient = 2 * (1 - 1 / (2 * beta))
    else:
        expected_loss = (1 - label_smoothing) * math.log1p(math.exp(-beta)) + label_smoothing * math.log1p(
            math.exp(beta)
        )
        expected_gradient = beta * (1 / (1 + math.exp(-beta)) - 1 + label_smoothing)

    # run() also calls the real val() before the training step.
    pipeline.run()

    assert pipeline.actor_train.loss == pytest.approx(expected_loss)
    assert pipeline.actor_train.gradient == pytest.approx(expected_gradient)
    pipeline.tracker.log.assert_called_once()
    metrics = pipeline.tracker.log.call_args.kwargs["values"]
    assert metrics["actor/loss"] == pytest.approx(expected_loss)
    assert metrics["val/actor/loss"] == pytest.approx(expected_loss)
    for prefix in ("actor", "val/actor"):
        assert metrics[f"{prefix}/chosen_reward"] == pytest.approx(-beta)
        assert metrics[f"{prefix}/reject_reward"] == pytest.approx(-2 * beta)
        assert metrics[f"{prefix}/acc"] == 1.0
