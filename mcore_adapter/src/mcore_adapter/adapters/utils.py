import re
from typing import Callable

from megatron.core.extensions.transformer_engine import TEGroupedLinear, TELayerNormColumnParallelLinear, TELinear
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.transformer.moe.router import TopKRouter
from transformers import PreTrainedModel


def set_linear_is_expert(model):
    for n, module in model.named_modules():
        if (
            ".experts." in n
            and isinstance(module, (TELinear, TELayerNormColumnParallelLinear))
            or isinstance(module, TEGroupedLinear)
        ):
            module.is_expert = True


def find_layers(model: "PreTrainedModel", cond: Callable):
    inner_nodes = set()
    for name, module in model.named_modules():
        name = re.sub(r"\d+\.", "{}.", name)
        if not cond(module):
            inner_nodes.add(name)
    target_module_names = set()
    for name, module in model.named_modules():
        if cond(module):
            module_name_list = name.split(".")
            module_name = module_name_list.pop()
            for inner_node in inner_nodes:
                processed_module_name = re.sub(r"\d+\.", "{}.", module_name)
                while module_name_list and inner_node.endswith(processed_module_name):
                    module_name = f"{module_name_list.pop()}.{module_name}"
            target_module_names.add(module_name)
    return list(target_module_names)


def find_all_linear_modules(model):
    return find_layers(
        model, lambda module: isinstance(module, (TELinear, TEGroupedLinear, TELayerNormColumnParallelLinear))
    )


def find_all_embedding_modules(model):
    return find_layers(model, lambda module: isinstance(module, LanguageModelEmbedding))


def find_all_router_modules(model):
    return find_layers(model, lambda module: isinstance(module, TopKRouter))
