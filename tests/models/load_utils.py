from torch.utils.data import DistributedSampler, DataLoader
from transformers import DataCollatorWithPadding

from roll.configs import ModelArguments, DataArguments
from roll.configs.training_args import TrainingArguments
from roll.datasets.loader import get_dataset
from roll.models.model_providers import default_tokenizer_provider


def get_model_input_device(model):
    if hasattr(model, "get_input_embeddings"):
        input_embeddings = model.get_input_embeddings()
        if input_embeddings is not None:
            return input_embeddings.weight.device
    return next(model.parameters()).device


def get_generation_eos_token_ids(tokenizer):
    additional_token_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
    if additional_token_ids is None:
        additional_tokens = getattr(tokenizer, "additional_special_tokens", [])
        additional_token_ids = tokenizer.convert_tokens_to_ids(additional_tokens)
    if isinstance(additional_token_ids, int):
        additional_token_ids = [additional_token_ids]

    token_ids = [tokenizer.eos_token_id]
    token_ids.extend(token_id for token_id in additional_token_ids if token_id is not None)
    return token_ids


def get_mock_dataloader(model_args: ModelArguments, data_args: DataArguments, batch_size: int = 4):

    tokenizer = default_tokenizer_provider(model_args=model_args)

    dataset = get_dataset(
        tokenizer=tokenizer,
        data_args=data_args,
    )
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in ("input_ids", "attention_mask")]
    )
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
    sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=42,
        drop_last=True,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return dataloader, tokenizer
