import math

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto
from roll.utils.functionals import agg_loss
from roll.utils.metrics.metrics_manager import MetricsManager


def make_batch(mask: torch.Tensor, domains: list[str], agg_entropy: torch.Tensor | None = None) -> DataProto:
    batch_size, response_length = mask.shape[0], mask.shape[1] - 1
    tensors = {
        "scores": torch.arange(batch_size, dtype=torch.float32) % 2,
        "token_level_rewards": torch.zeros(batch_size, response_length),
        "advantages": torch.ones(batch_size, response_length),
        "raw_advantages": torch.ones(batch_size, response_length),
        "returns": torch.ones(batch_size, response_length),
        "prompt_mask": torch.ones(batch_size, 1),
        "response_mask": mask,
        "final_response_mask": mask[:, 1:].clone(),
        "responses": torch.zeros(batch_size, response_length),
    }
    meta_info = {} if agg_entropy is None else {"agg_entropy": agg_entropy}
    return DataProto(
        batch=TensorDict(tensors, batch_size=[batch_size]),
        non_tensor_batch={"domain": np.array(domains, dtype=object)},
        meta_info=meta_info,
    )


def test_domain_entropy_is_token_weighted_after_reordering() -> None:
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]])
    entropy = torch.tensor([[0.1, 0.1, 0.0], [1.0, 1.0, 1.0], [0.4, 0.0, 0.0], [2.0, 0.0, 0.0]])
    global_entropy = agg_loss(entropy, mask[:, 1:], "token-mean")
    batch = make_batch(mask, ["A", "B", "A", "B"], global_entropy)
    manager = MetricsManager()

    batch.batch["final_response_mask"][2] = 0
    manager.record_domain_entropy(batch, entropy)
    batch.reorder(torch.tensor([2, 1, 0, 3]))
    first_groups = batch.group_by("domain")
    batch = DataProto.concat([first_groups["B"], first_groups["A"]])
    manager.add_all_metrics(0, batch)
    manager.add_domain_all_metrics(0, batch.group_by("domain"))
    metrics = manager.get_metrics()

    assert metrics["critic/entropy/mean"] == pytest.approx(0.8)
    assert metrics["A/critic/entropy/mean"] == pytest.approx(0.2)
    assert metrics["B/critic/entropy/mean"] == pytest.approx(1.25)


def test_domain_entropy_masks_padding_and_reports_empty_domain_as_nan() -> None:
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1]])
    entropy = torch.tensor([[0.5, 100.0], [9.0, 9.0], [1.0, 2.0]])
    batch = make_batch(mask, ["A", "empty", "B"], torch.tensor(1.0))
    manager = MetricsManager()

    manager.record_domain_entropy(batch, entropy)
    manager.add_domain_all_metrics(0, batch.group_by("domain"))
    metrics = manager.get_metrics()

    assert metrics["A/critic/entropy/mean"] == pytest.approx(0.5)
    assert metrics["B/critic/entropy/mean"] == pytest.approx(1.5)
    assert math.isnan(metrics["empty/critic/entropy/mean"])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_domain_entropy_accumulates_low_precision_inputs_in_float32(dtype: torch.dtype) -> None:
    mask = torch.ones(2, 8193, dtype=torch.long)
    entropy = torch.full((2, 8192), 10.0, dtype=dtype)
    batch = make_batch(mask, ["A", "B"])
    manager = MetricsManager()

    manager.record_domain_entropy(batch, entropy)
    manager.add_domain_all_metrics(0, batch.group_by("domain"))
    metrics = manager.get_metrics()

    assert metrics["A/critic/entropy/mean"] == pytest.approx(10.0)
    assert metrics["B/critic/entropy/mean"] == pytest.approx(10.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_domain_entropy_on_cuda() -> None:
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]])
    entropy = torch.tensor([[0.1, 0.1, 0.0], [1.0, 1.0, 1.0], [0.4, 0.0, 0.0], [2.0, 0.0, 0.0]])
    batch = make_batch(mask, ["A", "B", "A", "B"]).to("cuda")
    entropy = entropy.to("cuda")
    manager = MetricsManager()

    manager.record_domain_entropy(batch, entropy)
    manager.add_domain_all_metrics(0, batch.group_by("domain"))
    metrics = manager.get_metrics()

    assert entropy.device.type == "cuda"
    assert metrics["A/critic/entropy/mean"] == pytest.approx(0.2)
    assert metrics["B/critic/entropy/mean"] == pytest.approx(1.25)


def test_no_recomputed_entropy_and_clear_preserve_existing_default() -> None:
    mask = torch.ones(2, 3, dtype=torch.long)
    batch = make_batch(mask, ["A", "B"])
    manager = MetricsManager()
    manager.record_domain_entropy(batch, torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
    manager.clear_metrics()

    manager.add_all_metrics(0, batch)
    manager.add_domain_all_metrics(0, batch.group_by("domain"))
    metrics = manager.get_metrics()

    assert metrics["critic/entropy/mean"] == 0.0
    assert metrics["A/critic/entropy/mean"] == 0.0
    assert metrics["B/critic/entropy/mean"] == 0.0
