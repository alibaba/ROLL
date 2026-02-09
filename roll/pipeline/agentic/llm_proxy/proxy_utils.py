"""
Utility functions for LLM proxy operations.
"""

from typing import List, Dict, Any, Optional, Union

import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizer

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.llm_proxy.base_llm_proxy import BaseLLMProxy
from roll.utils.logging import get_logger

logger = get_logger()


def generate_by_proxy(
    messages: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    proxy: BaseLLMProxy,
    enable_thinking: bool = False,
    generation_config: Optional[Dict[str, Any]] = None,
    collator: Optional[Any] = None,
    mm_data: Optional[Dict[str, Any]] = None,
    src_rank: Optional[int] = None,
) -> Optional[str]:
    """
    Generate text through proxy with support for multimodal inputs.

    This function formats messages using chat template, creates a DataProto
    with tokenized input (and optional multimodal data), calls proxy.generate(),
    and returns the decoded text response.

    For text-only generation, it uses tokenizer directly. For multimodal generation,
    it uses collator to process images/videos along with text.

    Args:
        messages: List of message dictionaries for the prompt.
                 For text: [{"role": "user", "content": "..."}]
                 For multimodal: [{"role": "user", "content": [{"type": "text", "text": "..."},
                                                                {"type": "image", "image": PIL.Image}]}]
        tokenizer: Tokenizer for the inference model
        proxy: LLM proxy for model inference
        enable_thinking: Whether to enable thinking tags in chat template (text-only mode)
        generation_config: Optional generation config to override defaults
                          (temperature, max_new_tokens, etc.)
        collator: Optional DataCollatorWithPaddingForMM for multimodal processing.
                 If provided, multimodal mode is used.
        mm_data: Optional multimodal data dict with "image" and/or "video" keys.
                Only used when collator is provided.
        src_rank: Optional source rank for request routing in scheduler.
                 If not provided, defaults to 0.

    Returns:
        Decoded text response from the LLM, or None if the request fails

    Examples:
        Text-only generation:
        >>> messages = [{"role": "user", "content": "Judge this response..."}]
        >>> response_text = generate_by_proxy(
        ...     messages=messages,
        ...     tokenizer=tokenizer,
        ...     proxy=proxy,
        ...     enable_thinking=True,
        ...     generation_config={"temperature": 0.2, "max_new_tokens": 2048}
        ... )

        Multimodal generation:
        >>> messages = [{"role": "user", "content": "Describe this image"}]
        >>> mm_data = {"image": [pil_image]}
        >>> response_text = generate_by_proxy(
        ...     messages=messages,
        ...     tokenizer=tokenizer,
        ...     proxy=proxy,
        ...     collator=collator,
        ...     mm_data=mm_data
        ... )
    """
    # Multimodal mode: use collator to process features
    if collator is not None:
        # Get text from chat template without tokenization
        lm_input_texts = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        # Build feature dict
        feature = {
            collator.prompt_key: lm_input_texts,
        }

        # Add multimodal data if provided
        if mm_data:
            if "image" in mm_data:
                feature[collator.image_key] = mm_data["image"]
            if "video" in mm_data:
                feature[collator.video_key] = mm_data["video"]

        # Process through collator
        inputs = collator([feature])
        lm_input: DataProto = DataProto.from_single_dict(inputs)

    # Text-only mode: tokenize directly
    else:
        # Format messages using chat template with optional thinking tags
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

        # Create DataProto with tokenized input
        input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor([1] * input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        position_ids = attention_mask.cumsum(dim=-1)

        lm_input = DataProto()
        lm_input.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }, batch_size=input_ids.shape[0])

    # Set generation config with defaults if not provided
    if generation_config is None:
        generation_config = {
            "temperature": 0.2,  # Lower temperature for stable judgments
            "max_new_tokens": 2048,
            "top_p": 0.95,
        }

    # Set src_rank for request routing in scheduler
    lm_input.meta_info["src_rank"] = src_rank if src_rank is not None else 0

    # Call proxy.generate() for inference
    lm_output: Optional[DataProto] = proxy.generate(
        messages=messages,
        lm_input=lm_input,
        generation_config=generation_config
    )

    # Handle failure cases
    if lm_output is None:
        logger.warning("LLM generation failed (returned None)")
        return None

    # Extract response token IDs and decode to text
    if "responses" not in lm_output.batch.keys():
        logger.error("LLM output missing 'responses' key")
        return None

    response_ids = lm_output.batch['responses'][0]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    return response_text
