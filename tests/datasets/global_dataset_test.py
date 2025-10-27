import ray

from roll.datasets.global_dataset import GlobalDataset
from roll.utils.constants import RAY_NAMESPACE


def read_global_dataset():
    ray.init()

    dataset = GlobalDataset.remote(dataset_name="data/math_benchmarks.jsonl")

    item = ray.get(dataset.get_data_item.remote(seed=0))

    print(item)


@ray.remote
class ReadActor:
    def __init__(self, dataset_name="data/math_benchmarks.jsonl", mode="sample"):
        self.dataset = GlobalDataset.options(name=dataset_name,
                                             get_if_exists=True,
                                             namespace=RAY_NAMESPACE).remote(dataset_name=dataset_name, mode=mode)

    def get_data_item(self, seed=0):
        return ray.get(self.dataset.get_data_item.remote(seed=seed))

def read_global_dataset_parallel():
    ray.init()
    dataset_name = "data/math_benchmarks.jsonl"
    dataset = GlobalDataset.options(name=dataset_name,
                                    get_if_exists=True,
                                    namespace=RAY_NAMESPACE).remote(dataset_name=dataset_name)

    actor_num = 10
    actor_list = [ReadActor.remote(dataset_name=dataset_name, mode="traversal") for _ in range(actor_num)]

    data_list = ray.get([actor.get_data_item.remote(seed=i) for i, actor in enumerate(actor_list)])

    print(data_list)

    data_list2 = ray.get([actor.get_data_item.remote(seed=i) for i, actor in enumerate(actor_list)])
    print(data_list2)

def filter_dataset():
    dataset_name = "data/math_benchmarks.jsonl"
    dataset = GlobalDataset.options(name=dataset_name,
                                    get_if_exists=True,
                                    namespace=RAY_NAMESPACE).remote(dataset_name=dataset_name)
    print(f"len(dataset): {ray.get(dataset.size.remote())}")
    ray.get(dataset.filter.remote(filter_name="test", function=lambda x: int(x["id"]) in list(range(10))))
    print(f"len(dataset) after filter: {ray.get(dataset.size.remote())}")


if __name__ == '__main__':
    # read_global_dataset()
    read_global_dataset_parallel()
    # filter_dataset()

