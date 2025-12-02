import random
from typing import Dict, Optional, Any, Callable

import ray
from datasets import Dataset

from roll.datasets.dataset import create_local_dataset
from roll.utils.random_utils import all_seed
from roll.utils.logging import get_logger

logger = get_logger()

@ray.remote
class GlobalDataset:
    def __init__(self, dataset_name, split: str = "train", mode="sample", dataset_kwargs: Dict = None):
        self.mode = mode
        self.dataset_name = dataset_name
        self.split = split
        self.dataset_kwargs = dataset_kwargs
        self.dataset: Optional[Dataset] = create_local_dataset(dataset_name=self.dataset_name,
                                                               split=split,
                                                               dataset_kwargs=dataset_kwargs)
        self.filter_names = set()
        logger.info(f"dataset_name: {self.dataset_name} len: {len(self.dataset)}")
        self.epoch = 0
        self.idx = 0
        self.seed_to_idx = {}

    async def get_data_item(self, seed: int, **kwargs) -> Dict:
        if self.mode == "traversal":
            data = None
            if seed not in self.seed_to_idx:
                self.seed_to_idx[seed] = self.idx
                if self.idx < len(self.dataset):
                    data = self.dataset[self.idx]
                    self.idx += 1
            else:
                stored_idx = self.seed_to_idx[seed]
                if stored_idx < len(self.dataset):
                    data = self.dataset[stored_idx]
            return data

        with all_seed(seed):
            if seed is not None:
                self.idx = random.randint(0, len(self.dataset) - 1)
            else:
                if self.idx == len(self.dataset):
                    self.epoch += 1
                    self.dataset = self.dataset.shuffle(seed=self.epoch)
                    self.idx = 0
            data = self.dataset[self.idx]
        return data

    async def size(self):
        return len(self.dataset)

    async def reset(self):
        self.idx = 0
        self.seed_to_idx.clear()

    async def filter(self, filter_name: str, function: Optional[Callable] = None, **kwargs):
        if filter_name in self.filter_names:
            return
        logger.info(f"---- before filter-- {filter_name}, dataset_name: {self.dataset_name} len: {len(self.dataset)}")
        self.dataset = self.dataset.filter(function, **kwargs)
        self.filter_names.add(filter_name)
        logger.info(f"---- after filter-- {filter_name}, dataset_name: {self.dataset_name} len: {len(self.dataset)}")


@ray.remote
class GlobalDatasetManager:
    def __init__(self):
        self.global_dataset_dict: Dict[str, Any] = {}

    async def register(self, dataset_name, dataset_ref):
        self.global_dataset_dict[dataset_name] = dataset_ref

    async def reset(self):
        refs = []
        for dataset_name, dataset_ref in self.global_dataset_dict.items():
            refs.append(dataset_ref.reset.remote())
        ray.get(refs)
