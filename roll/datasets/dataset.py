import os
from typing import Callable, Dict, Union

from datasets import Dataset, IterableDataset, load_dataset

from roll.configs.data_args import DataArguments
from roll.utils.logging import get_logger


logger = get_logger()

REGISTERED_DATASETS: Dict[str, Callable[[DataArguments], Union[Dataset, IterableDataset]]] = {}


def register_dataset(key: str):
    def decorator(func: Callable[[DataArguments], Union[Dataset, IterableDataset]]):
        if key in REGISTERED_DATASETS:
            raise ValueError(f"Dataset type '{key}' already exists!")
        REGISTERED_DATASETS[key] = func
        return func

    return decorator


def get_dataset(data_args: "DataArguments"):
    data_path = None
    data_name = data_args.file_name
    data_files = []
    dataset_dir = getattr(data_args, "dataset_dir", ".")
    dataset_type = getattr(data_args, "dataset_type", "default")
    FILEEXT2TYPE = {
        "arrow": "arrow",
        "csv": "csv",
        "json": "json",
        "jsonl": "json",
        "parquet": "parquet",
        "txt": "text",
    }
    if isinstance(data_name, list):
        local_path = ""
    else:
        local_path: str = os.path.join(dataset_dir, data_name)

    if os.path.isdir(local_path):
        for file_name in os.listdir(local_path):
            data_files.append(os.path.join(local_path, file_name))
            if data_path is None:
                data_path = FILEEXT2TYPE.get(file_name.split(".")[-1], None)
            elif data_path != FILEEXT2TYPE.get(file_name.split(".")[-1], None):
                raise ValueError("File types should be identical.")
    elif os.path.isfile(local_path):  # is file
        data_files.append(local_path)
        data_path = FILEEXT2TYPE.get(local_path.split(".")[-1], None)
    else:
        assert local_path == ""
        for file_name in data_name:
            data_files.append(os.path.join(dataset_dir, file_name))
            if data_path is None:
                data_path = FILEEXT2TYPE.get(file_name.split(".")[-1], None)
            elif data_path != FILEEXT2TYPE.get(file_name.split(".")[-1], None):
                raise ValueError("File types should be identical.")

    if data_path not in REGISTERED_DATASETS:
        raise ValueError(
            f"Dataset type '{data_path}' is not found! Available datasets: {list(REGISTERED_DATASETS.keys())}"
        )

    logger.info(f"load_data_files: {chr(10)} {chr(10).join(data_files)}")
    logger.info(f"prompt column: {data_args.prompt}  label column: {data_args.response}")

    return REGISTERED_DATASETS[data_path](data_files, data_args)


@register_dataset("default")
@register_dataset("json")
def default_json_dataset(
    data_files: "DataPaths",
    data_args: "DataArguments",
) -> Union["Dataset", "IterableDataset"]:
    return load_dataset("json", data_files=data_files)["train"]


@register_dataset("arrow")
def default_arrow_dataset(
    data_files: "DataPaths",
    data_args: "DataArguments",
) -> Union["Dataset", "IterableDataset"]:
    return load_dataset("arrow", data_files=data_files)["train"]


@register_dataset("csv")
def default_csv_dataset(
    data_files: "DataPaths",
    data_args: "DataArguments",
) -> Union["Dataset", "IterableDataset"]:
    return load_dataset("csv", data_files=data_files)["train"]


@register_dataset("parquet")
def default_parquet_dataset(
    data_files: "DataPaths",
    data_args: "DataArguments",
) -> Union["Dataset", "IterableDataset"]:
    return load_dataset("parquet", data_files=data_files)["train"]


@register_dataset("text")
def default_text_dataset(
    data_files: "DataPaths",
    data_args: "DataArguments",
) -> Union["Dataset", "IterableDataset"]:
    return load_dataset("text", data_files=data_files)["train"]
