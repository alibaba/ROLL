from functools import partial
from pickle import FALSE
from typing import TYPE_CHECKING, Callable, Dict

from roll.utils.prompt import BASE_CHAT_FORMAT, LONGCOT_QWEN_2_5_SYSTEM


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


chat_templates: Dict[str, Callable[..., str]] = {}


def add_default_system(func, tokenizer: "PreTrainedTokenizer", conversation, default_system: str = None, **kwargs):
    if default_system is not None and conversation[0]["role"] != "system":
        conversation = [{"role": "system", "content": default_system}] + conversation
    return func(tokenizer, conversation, **kwargs)


def register_chat_template(key, **kwargs):
    def decorator(func):
        if key in chat_templates:
            raise ValueError(f"chat template {key} already exists")
        chat_templates[key] = partial(add_default_system, func, **kwargs)
        return func

    return decorator


def get_chat_template(key, tokenizer: "PreTrainedTokenizer"):
    if key not in chat_templates:
        raise ValueError(f"chat template {key} not found")
    return partial(chat_templates[key], tokenizer)


@register_chat_template("native")
@register_chat_template("qwen2_5")
def native_chat_template(tokenizer: "PreTrainedTokenizer", conversation, tools=None, documents=None, **kwargs):
    kwargs["tokenize"] = False
    kwargs["add_generation_prompt"] = kwargs.get("add_generation_prompt", True)
    return tokenizer.apply_chat_template(conversation, tools, documents, **kwargs)

@register_chat_template("qwen3")
def qwen3_chat_template(tokenizer: "PreTrainedTokenizer", conversation, tools=None, documents=None, **kwargs):
    kwargs["tokenize"] = False
    kwargs["add_generation_prompt"] = kwargs.get("add_generation_prompt", True)
    kwargs["enable_thinking"] = True
    return tokenizer.apply_chat_template(conversation, tools, documents, **kwargs)

@register_chat_template("qwen2_5_dpo")
def dpo_chat_template(tokenizer: "PreTrainedTokenizer", conversation, tools=None, documents=None, **kwargs):
    kwargs["tokenize"] = False

    # Disable generation prompt ('<|assistant|>') to avoid redundant tokens in DPO training
    kwargs["add_generation_prompt"] = kwargs.get("add_generation_prompt", False)

    return tokenizer.apply_chat_template(conversation, tools, documents, **kwargs)

# TODO: change template name ?
@register_chat_template("chatml")
def chatml_chat_template(tokenizer: "PreTrainedTokenizer", conversation, tools=None, documents=None, **kwargs):
    chat_template = (
        "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' "
        "+ '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    kwargs["tokenize"] = False
    kwargs["add_generation_prompt"] = kwargs.get("add_generation_prompt", True)
    return tokenizer.apply_chat_template(conversation, tools, documents, chat_template=chat_template, **kwargs)


@register_chat_template("base", base_format=BASE_CHAT_FORMAT)
@register_chat_template("empty", base_format="<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n")
def base_chat_template(tokenizer: "PreTrainedTokenizer", conversation, base_format: str, **kwargs):
    assert len(conversation) == 1 and conversation[0]["role"] == "user"
    return base_format.replace("{{content}}", conversation[0]["content"])


@register_chat_template("longCOT_qwen2.5", default_system=LONGCOT_QWEN_2_5_SYSTEM)
def longcot_qwen2_5_chat_template(
    tokenizer: "PreTrainedTokenizer", conversation, tools=None, documents=None, **kwargs
):
    for i in range(len(conversation)):
        if conversation[i]["role"] == "user":
            conversation[i]["content"] = "Return your final response within \\boxed{}. " + conversation[i]["content"]
    kwargs["tokenize"] = False
    kwargs["add_generation_prompt"] = kwargs.get("add_generation_prompt", True)
    return tokenizer.apply_chat_template(conversation, tools, documents, **kwargs)


@register_chat_template("longcot_V3")
def longcot_think_chat_template(tokenizer: "PreTrainedTokenizer", conversation, tools=None, documents=None, **kwargs):
    kwargs["tokenize"] = False
    kwargs["add_generation_prompt"] = kwargs.get("add_generation_prompt", True)
    return tokenizer.apply_chat_template(conversation, tools, documents, **kwargs) + "<think>\n"
