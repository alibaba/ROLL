import json
from typing import List, Dict
from functools import lru_cache
from transformers import PreTrainedTokenizer

from roll.datasets.collator import DataCollatorWithPaddingForMM
from roll.utils.logging import get_logger
logger = get_logger()


@lru_cache(maxsize=10)
def compute_conversation_end_token_id(tokenizer: PreTrainedTokenizer) -> List[int]:
    """
    find '<|im_end|>' token id
    """
    assistant_mock = [{"role": "user", "content": ""}]
    assistant_token_ids_mock: List[int] = tokenizer.apply_chat_template(assistant_mock, tokenize=True)
    for token_id in reversed(assistant_token_ids_mock):
        if token_id in tokenizer.all_special_ids:
            return [token_id]
    return []

def custom_apply_chat_template(messages: List[Dict], tokenizer: PreTrainedTokenizer, add_generation_prompt=True, enable_thinking=False) -> List:
    if len(messages) == 0:
        return []
    if messages[0]["role"] == "system":
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking)
        return token_ids
    else:
        system_mock = [{"role": "system", "content": ""}]
        system_token_ids_mock = tokenizer.apply_chat_template(system_mock, tokenize=True)
        token_ids = tokenizer.apply_chat_template(system_mock + messages, tokenize=True, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking)
        return token_ids[len(system_token_ids_mock):]

def custom_vl_apply_chat_template(messages: List[Dict], collator: DataCollatorWithPaddingForMM, add_generation_prompt=True) -> Dict:
    if len(messages) == 0:
        return {}

    images = []
    for message in messages:
        if message["role"] == "user":
            content: List[Dict] = message["content"]
            images.extend([content[i].pop("image_PIL") for i in range(len(content)) if content[i]["type"] == "image"])

    if messages[0]["role"] == "system":
        messages_text = collator.processor.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)
        features = [{
            collator.prompt_key: messages_text,
            collator.image_key: images,
            collator.image_flag_key: True
        }]
        inputs = collator(features)
        inputs.pop("position_ids", None)
        return inputs
    else:
        system_mock = [{"role": "system", "content": ""}]
        system_token_ids_mock = collator.processor.apply_chat_template(system_mock, tokenize=True)
        messages_text = collator.processor.apply_chat_template(system_mock + messages)
        features = [{
            collator.prompt_key: messages_text,
            collator.image_key: images,
            collator.image_flag_key: True
        }]
        inputs = collator(features)
        inputs.pop("position_ids", None)
        inputs["input_ids"] = inputs["input_ids"][:, len(system_token_ids_mock):]
        inputs["attention_mask"] = inputs["attention_mask"][:, len(system_token_ids_mock):]
        return inputs

def messages_to_tokens_and_masks(messages: List[Dict], tokenizer: PreTrainedTokenizer, add_generation_prompt=False):
    """
    Convert a list of chat messages into token IDs and response masks.

    Args:
        messages: List of dictionaries representing chat messages, each with 'role' and 'content' keys.
        tokenizer: PreTrainedTokenizer instance to tokenize the messages.
        add_generation_prompt: Whether to add a generation prompt for the last message (default: False).

    Returns:
        Tuple[List[List[int]], List[List[int]]]: A tuple containing:
            - List of token ID lists for each message.
            - List of response mask lists (1 for tokens to be predicted, 0 otherwise).

    case:
        messages = [
            {"role": "system",
             "content": "You're a helpful assistant. "},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris"},
            {"role": "assistant", "content": "Landon"}
        ]
        <|im_start|>system
        You're a helpful assistant. <|im_end|>
        <|im_start|>user
        What is the capital of France?<|im_end|>
        <|im_start|>assistant
        Paris<|im_end|>
        <|im_start|>assistant
        Landon<|im_end|>

        [[151644, 8948, 198, 2610, 2299, 264, 10950, 17847, 13, 220, 151645, 198], [151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198], [151644, 77091, 198, 59604, 151645, 198], [151644, 77091, 198, 43, 11037, 151645]]
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1]]
        [('<|im_start|>', 0, 151644), ('system', 0, 8948), ('\n', 0, 198), ('You', 0, 2610), ("'re", 0, 2299), (' a', 0, 264), (' helpful', 0, 10950), (' assistant', 0, 17847), ('.', 0, 13), (' ', 0, 220), ('<|im_end|>', 0, 151645), ('\n', 0, 198), ('<|im_start|>', 0, 151644), ('user', 0, 872), ('\n', 0, 198), ('What', 0, 3838), (' is', 0, 374), (' the', 0, 279), (' capital', 0, 6722), (' of', 0, 315), (' France', 0, 9625), ('?', 0, 30), ('<|im_end|>', 0, 151645), ('\n', 0, 198), ('<|im_start|>', 0, 151644), ('assistant', 0, 77091), ('\n', 0, 198), ('Paris', 1, 59604), ('<|im_end|>', 1, 151645), ('\n', 0, 198), ('<|im_start|>', 0, 151644), ('assistant', 0, 77091), ('\n', 0, 198), ('L', 1, 43), ('andon', 1, 11037), ('<|im_end|>', 1, 151645)]

    """
    token_ids_list = []
    response_masks_list = []
    system_mock = {"role": "system", "content": ""}
    system_token_ids_mock = tokenizer.apply_chat_template([system_mock], tokenize=True)
    for i, message in enumerate(messages):
        if message["role"].lower() == "system":
            token_ids = tokenizer.apply_chat_template([message], tokenize=True)
            token_ids_list.append(token_ids)
            response_masks_list.append([0] * len(token_ids))
        if message["role"].lower() in ["user", "assistant"]:
            token_ids = tokenizer.apply_chat_template([system_mock, message], tokenize=True,
                                                      add_generation_prompt=add_generation_prompt and i == len(messages) - 1)
            token_ids = token_ids[len(system_token_ids_mock):]
            if message["role"].lower() == "user":
                token_ids_list.append(token_ids)
                response_masks_list.append([0] * len(token_ids))
            elif message["role"].lower() == "assistant":
                assistant_token_ids = []
                response_masks = []
                token_id_without_format = tokenizer.encode(message["content"])
                first_assistant_idx = token_ids.index(token_id_without_format[0])
                assistant_token_ids.extend(token_ids[: first_assistant_idx])
                response_masks.extend([0] * first_assistant_idx)
                after_eos_token_id = False
                for idx in range(first_assistant_idx, len(token_ids)):
                    if i == len(messages) - 1 and after_eos_token_id and not add_generation_prompt:
                        break
                    assistant_token_ids.append(token_ids[idx])
                    if after_eos_token_id:
                        response_masks.append(0)
                    else:
                        response_masks.append(1)
                    if token_ids[idx] == tokenizer.eos_token_id:
                        after_eos_token_id = True
                token_ids_list.append(assistant_token_ids)
                response_masks_list.append(response_masks)

    return token_ids_list, response_masks_list


def token_ids_to_assistant_mask(messages: List[Dict], input_ids_list: List[List], tokenizer: PreTrainedTokenizer):
    """
    Reconstruct per-message assistant masks from already-tokenized inputs.

    Args:
        messages: Original chat messages; only the `"assistant"` ones are inspected.
        input_ids_list: Already-tokenized version of `messages`, aligned index-by-index.
        tokenizer: Hugging Face tokenizer whose `.eos_token_id` marks the end of an assistant turn.

    Returns:
        List[List[int]] where each inner list contains 1 for tokens belonging to the assistant’s
        actual response (excluding format / control tokens) and 0 elsewhere.
    """
    token_ids_list = []
    assistant_mask_list = []
    for i, (message, token_ids) in enumerate(zip(messages, input_ids_list)):
        if message["role"].lower() in ["system", "user"]:
            token_ids_list.append(token_ids)
            assistant_mask_list.append([0] * len(token_ids))
        if message["role"].lower() in ["assistant"]:
            assistant_token_ids = []
            assistant_mask = []
            token_id_without_format = tokenizer.encode(message["content"])
            first_assistant_idx = len(token_ids)
            if len(token_id_without_format) > 0:
                first_assistant_idx = token_ids.index(token_id_without_format[0])
            assistant_token_ids.extend(token_ids[: first_assistant_idx])
            assistant_mask.extend([0] * first_assistant_idx)
            after_eos_token_id = False
            for idx in range(first_assistant_idx, len(token_ids)):
                assistant_token_ids.append(token_ids[idx])
                if after_eos_token_id:
                    assistant_mask.append(0)
                else:
                    assistant_mask.append(1)
                if token_ids[idx] == tokenizer.eos_token_id:
                    after_eos_token_id = True
            token_ids_list.append(assistant_token_ids)
            assistant_mask_list.append(assistant_mask)

    return assistant_mask_list


def split_by_token(input_ids: list, token: int, messages: List[Dict], tokenizer: PreTrainedTokenizer) -> list[list]:
    """
    Split the input_ids list by the given token and return a list of lists.
    Each sub-list starts with that token.

    Args:
        input_ids: Sequence of input IDs to be split (list[int]).
        token: Token ID used for splitting and marking the start of each sub-list.

    Returns:
        A list containing multiple sub-lists (List[List[int]]).

    Example:
      >>> split_by_token(
      ...     input_ids=[151644, 8948, 198, 151644, 872, 198],
      ...     token=151644
      ... )
      [[151644, 8948, 198], [151644, 872, 198]]
    """

    if not input_ids:
        return []

    result = []
    current_segment = []

    for item in input_ids:
        if item == token:
            if current_segment:
                result.append(current_segment)
            current_segment = [item]
        else:
            current_segment.append(item)

    if current_segment:
        result.append(current_segment)

    if len(result) == len(messages):
        return result
    input_ids_list = result[:]
    result = []
    # spliting by start token is vulnerable since the format of responses cannot be guaranteed
    # input_ids_list has large length than messages when format error, which is caused by
    # responses includeing more than one start token
    # adjustment according to messages
    segment_mismatch = True
    ids_next_idx = 0  # index in input_ids_list for the next message
    bos_token_id = input_ids_list[0][0]
    for i, message in enumerate(messages):
        segment_mismatch = len(input_ids_list) - ids_next_idx != len(messages) - i
        if segment_mismatch:
            # str or list of dict
            content = (
                "".join([item["text"] for item in message["content"] if item["type"] == "text"])
                if not isinstance(message["content"], str)
                else message["content"]
            )
            token_id_without_format = tokenizer.encode(content)
            bos_num = token_id_without_format.count(bos_token_id) + 1  # generated + chat_format
            current_segment = sum(input_ids_list[ids_next_idx : ids_next_idx + bos_num], [])
            ids_next_idx += bos_num
        else:
            current_segment = input_ids_list[ids_next_idx]
            ids_next_idx += 1
        result.append(current_segment)
    return result




def convert_list_content_str(messages: List[Dict], parse_tool_call_parameter_to_dict=False) -> List[Dict]:
    """
    Convert state0.json format to tokenizer-compatible format.

    The state0.json may have content as either:
    1. A string (already compatible)
    2. A list of dictionaries with 'type' and 'text' keys

    This function ensures all content is converted to strings by concatenating
    text from list objects when needed.

    Args:
        messages: List of message dictionaries from iflow_state0.json
        parse_tool_call_parameter_to_dict: Whether to convert tool call arguments to dict, https://github.com/QwenLM/Qwen3-Coder/issues/444

    Returns:
        List of message dictionaries with string content suitable for tokenizer
    """
    converted_messages = []

    for message in messages:
        converted_message = message.copy()

        # Handle content field
        content = message.get('content')
        if isinstance(content, list):
            # Concatenate all text elements from the list
            text_parts = []
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    text_parts.append(item['text'])
                elif isinstance(item, str):
                    text_parts.append(item)
            converted_message['content'] = ''.join(text_parts)
        elif isinstance(content, str):
            # Already in correct format
            converted_message['content'] = content
        else:
            # Handle other cases (convert to string)
            converted_message['content'] = str(content)

        if parse_tool_call_parameter_to_dict:
            if message['role'] == 'assistant':
                if "tool_calls" in message:
                    tool_calls: List[Dict] = message['tool_calls']
                    try:
                        for tool_call in tool_calls:
                            if "arguments" in tool_call["function"] and isinstance(tool_call['function']['arguments'], str):
                                tool_call['function']["arguments"] = json.loads(tool_call['function']['arguments'])
                    except Exception as e:
                        # NOTE: check 兜底逻辑是否合理
                        #       现在更倾向于把assistant部分的内容替换为 content=response_text
                        #       not isinstance(tool_call['function']['arguments'], str)的情况在model_update的时候会出现，abort的request会被convert两次
                        content = converted_message.get('content', '')
                        tool_calls = message.pop("tool_calls")
                        converted_message['content'] = f"{content}{tool_calls}"
                        logger.error(f"Error parsing tool call arguments: {e}, src arguments: {json.dumps(tool_calls)}, parsing drawback to {converted_message['content']}")
        converted_messages.append(converted_message)

    return converted_messages