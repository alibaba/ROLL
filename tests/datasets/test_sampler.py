from collections import Counter

import datasets
import pytest
from torch.utils.data import Dataset

from roll.datasets.sampler import BatchStratifiedSampler


class MockDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


@pytest.fixture
def sample_dataset():
    data = []
    for i in range(10):
        data.append({"domain": "a", "id": f"a_{i}"})
    for i in range(6):
        data.append({"domain": "b", "id": f"b_{i}"})
    for i in range(4):
        data.append({"domain": "c", "id": f"c_{i}"})
    return MockDataset(data)


def test_basic_function(sample_dataset):
    domain_ratios = {"a": 5, "b": 3, "c": 2}
    sampler = BatchStratifiedSampler(sample_dataset, domain_ratios=domain_ratios, batch_size=10, drop_last=True)

    assert sampler.domain_ratios == {"a": 0.5, "b": 0.3, "c": 0.2}
    assert sampler.domain_batch_num == {"a": 5, "b": 3, "c": 2}

    assert len(sampler) == 2

    all_samples = []
    for batch in sampler:
        assert len(batch) == 10
        domains = [sample_dataset[i]["domain"] for i in batch]
        counts = Counter(domains)
        assert counts["a"] == 5
        assert counts["b"] == 3
        assert counts["c"] == 2
        all_samples.extend(batch)

    assert len(all_samples) == 2 * 10


def test_index_expansion():
    data = [{"domain": "c", "id": i} for i in range(3)]
    dataset = MockDataset(data)

    sampler = BatchStratifiedSampler(dataset, domain_ratios={"c": 1}, batch_size=2, drop_last=False)

    indices = list(sampler.__iter__())
    assert len(indices) == 2
    flat_indices = [idx for batch in indices for idx in batch]

    assert set(flat_indices) == set(range(0, 3))


def test_ratio_calculation():
    dataset = MockDataset([{"domain": "a"}, {"domain": "b"}])
    sampler = BatchStratifiedSampler(dataset, domain_ratios={"a": 9, "b": 1}, batch_size=10)
    assert sampler.domain_batch_num == {"a": 9, "b": 1}


def test_randomness(sample_dataset, monkeypatch):
    from roll.datasets import sampler as sampler_module

    def reverse_shuffle(values):
        values[:] = values[::-1]

    monkeypatch.setattr(sampler_module.np.random, "shuffle", reverse_shuffle)
    monkeypatch.setattr(sampler_module.random, "shuffle", reverse_shuffle)

    sampler = BatchStratifiedSampler(sample_dataset, domain_ratios={"a": 5, "b": 3, "c": 2}, batch_size=10)
    batches = list(sampler.__iter__())

    for batch in batches:
        domains = [sample_dataset[i]["domain"] for i in batch]
        counts = Counter(domains)
        assert counts["a"] == 5
        assert counts["b"] == 3
        assert counts["c"] == 2
        assert domains != ["a"] * 5 + ["b"] * 3 + ["c"] * 2


def test_edge_cases():
    with pytest.raises(AssertionError):
        BatchStratifiedSampler(MockDataset([]), domain_ratios={"a": 1}, batch_size=10)

    dataset = MockDataset([{"domain": "a"}])
    sampler = BatchStratifiedSampler(dataset, domain_ratios={"a": 1}, batch_size=1, drop_last=False)
    assert len(sampler) == 1


def test_data_integrity(sample_dataset):
    sampler = BatchStratifiedSampler(
        sample_dataset, domain_ratios={"a": 5, "b": 3, "c": 2}, batch_size=10, drop_last=False
    )

    all_samples = []
    for batch in sampler:
        all_samples.extend(batch)

    original_ids = set(range(len(sample_dataset)))
    used_ids = set(all_samples)
    assert used_ids.issuperset(original_ids)
