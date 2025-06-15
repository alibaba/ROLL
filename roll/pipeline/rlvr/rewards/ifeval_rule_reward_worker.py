# Import necessary libraries and modules
from functools import partial
from typing import Optional, Union, Iterator
import json
import re
import inspect
import os

import ray
import torch
from codetiming import Timer
from tqdm import tqdm
import signal
import multiprocessing
import itertools, collections
from collections import defaultdict

# Import from existing WorkerConfig, Worker, Dispatch and other modules
from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
import string
from difflib import SequenceMatcher
import nltk

nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))
os.environ["NLTK_DATA"] = os.path.join(os.path.dirname(__file__), "nltk_data")
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk

# Assume tokenizer still comes from default_tokenizer_provider
from roll.models.model_providers import default_reward_model_provider, default_tokenizer_provider

# Import dictionary mapping of ifeval validation functions
# IF_FUNCTIONS_MAP is the function mapping included in the complete implementation given above
from typing import Union, Dict, List

from roll.utils.logging import get_logger

logger = get_logger()  # Get logger instance


def first_boxed(text: str) -> str | None:
    """
    Extract content of the first \boxed{...}, supporting nested {} inside boxed.
    """
    marker = r"\boxed{"
    start = text.find(marker)
    if start == -1:
        return ""  # No \boxed{ found

    i = start + len(marker)  # Skip '\boxed{'
    depth = 1  # Already entered 1 level of {
    buf = []

    while i < len(text) and depth:  # Scan until balanced
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:  # Return to 0 means boxed is complete
                break
        if depth:  # Only record characters when brackets are not balanced
            buf.append(ch)
        i += 1

    return "".join(buf) if depth == 0 else ""


class timeout:
    """
    Timeout context similar to MathRewardWorker example, for demonstration,
    can be omitted directly if timeout is not needed.
    """

    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


# Contains keywords: Your answer should include keywords {keyword1}, {keyword2}
def verify_keywords(text, keyword_list):
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    # Convert response text to lowercase for case-insensitive matching
    response_lower = text.lower()

    # Check if response contains all keywords (each keyword also converted to lowercase for matching)
    return all(keyword.lower() in response_lower for keyword in keyword_list)


# Keyword frequency: In your answer, word {word} should appear {N} times
def verify_keyword_frequency(text, word, N):
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    # Convert text to lowercase for case-insensitive search
    text = text.lower()
    word = word.lower()

    # Use regex to match word boundaries and split text into word list
    words = re.findall(r"\b\w+\b", text)

    # Count actual occurrences (exact match keywords)
    actual_count = sum(1 for word in words if word == word)

    # Check if actual occurrence count equals expected N times
    constraint_met = actual_count == N

    return constraint_met


# Forbidden words: Answer should not contain keywords {forbidden words}
def validate_forbidden_words(text, forbidden_words):
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()

    # Check if each forbidden word appears in the text
    found_words = [word for word in forbidden_words if word.lower() in text_lower]

    # If no forbidden words found, return True; otherwise return False
    return len(found_words) == 0


# Letter frequency: In your answer, letter {letter} should appear exactly {N} times
def verify_letter_frequency(text: str, letter: str, N: int) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    if len(letter) != 1:
        raise ValueError("Letter parameter must be a single character")

    actual_count = text.count(letter)
    return actual_count == N


# Response language constraint: Your entire answer should use {language}, no other language content allowed
def validate_response_language(text, language):
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    from langdetect import detect

    # Detect text language
    detected_language = detect(text)
    # Check if detected language matches expected language
    return detected_language == language


# Paragraph count: Answer should contain {N} paragraphs, separated by markdown separator "* * *"
def verify_paragraph_count(text: str, N: int) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """

    def clean_text(text: str) -> str:
        """Remove excess whitespace and normalize line breaks"""
        return "\n".join(line.strip() for line in text.splitlines()).strip()

    # Clean input text
    text = clean_text(text)

    # Split text by markdown separator, each separator creates n+1 paragraphs
    paragraphs = text.split("* * *")
    actual_count = len(paragraphs)

    # Verify each split result contains non-empty content
    valid_paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if len(valid_paragraphs) != actual_count:
        return False

    return actual_count == N


# Word count constraint: Answer word count should be at least/around/at most {N}
def validate_word_constraint(text: str, N: int, quantifier: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    # Remove excess whitespace and split text into word list
    words = text.strip().split()
    actual_count = len(words)

    # Define tolerance range for "around" constraint (±10% of target word count, at least 1 word)
    tolerance = max(round(N * 0.1), 1)

    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "at most":
        return actual_count <= N
    elif quantifier == "around":
        return abs(actual_count - N) <= tolerance
    else:
        return False


# Sentence count constraint: Answer should contain at least/around/at most {N} sentences
def verify_sentence_constraint(text: str, N: int, quantifier: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    # Use regex to split text into sentence list based on periods or question marks followed by spaces
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)

    # Count actual sentence count
    actual_count = len(sentences)

    # Compare based on different quantifiers
    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "around":
        return abs(actual_count - N) <= 1
    elif quantifier == "at most":
        return actual_count <= N
    else:
        return False


# Paragraph count and specific paragraph first word constraint: Answer should contain {N} paragraphs separated only by two newlines, paragraph {i} must start with {first word}
def validate_paragraphs(text, N, first_word, i):
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    # Split text into paragraphs by two newlines
    paragraphs = text.split("\n\n")

    # Check if total paragraph count meets requirements
    if len(paragraphs) != N:
        return False

    # Check if paragraph i starts with specified word
    if paragraphs[i - 1].strip().startswith(first_word):
        return True
    return False


# Postscript validation: Please clearly add a postscript starting with {postscript marker} at the end of your answer
def verify_postscript(text, postscript_marker):
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    # Check if text contains postscript marker
    if postscript_marker in text:
        # Get marker index position
        marker_index = text.find(postscript_marker)
        # Check if there's other content near the marker
        remaining_text = text[marker_index:].strip()
        # Verify postscript is not just the marker itself
        return len(remaining_text) > len(postscript_marker)
    return False


# Placeholder validation: Answer should contain at least {N} placeholders in square brackets, e.g. [address]
def validate_placeholders(text: str, N: int) -> tuple[bool, List[str]]:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    # Use regex to find all content within square brackets
    pattern = r"\[(.*?)\]"
    placeholders = re.findall(pattern, text)

    # Check if at least N placeholders were found
    has_enough = len(placeholders) >= N

    return has_enough


# Bullet point validation: Answer must contain exactly {N} bullet points. Use markdown format bullet points, e.g.: * This is a point.
def verify_bullet_points(text: str, N: int) -> tuple[bool, str]:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    # Split text by lines and count lines starting with * or -
    lines = text.split("\n")
    bullet_points = [line.strip() for line in lines if line.strip().startswith(("*", "-"))]
    actual_count = len(bullet_points)

    if actual_count == N:
        return True
    else:
        return False


# Title validation: Answer must contain a title wrapped in double angle brackets, e.g. <<poem of joy>>
def validate_title(text: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    pattern = r"<<(.*?)>>"
    matches = re.findall(pattern, text)

    if len(matches) > 0:
        return True
    else:
        return False


# Multiple choice validation: Answer content must be one of the following options: {options}
def validate_choice(text: str, options: list) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    for option in options:
        if text in option:
            return True
    return False


# Highlighted section count validation: Answer must highlight at least {N} sections using markdown format, e.g. *highlighted section*
def validate_highlighted_sections(text: str, N: int) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    pattern = r"\*(.*?)\*"
    matches = re.findall(pattern, text)

    if len(matches) >= N:
        return True
    else:
        return False


# Multi-section validation: Answer must contain {N} sections, each section should start with {section splitter}
def validate_sections(text: str, N: int, section_splitter: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    sections = text.split(section_splitter)
    # First section may not start with splitter, so adjustment needed
    if sections[0] == "":
        sections.pop(0)
    if len(sections) == N:
        return True
    else:
        return False


# JSON format validation: Entire output must be wrapped in JSON format
def validate_json_format(text: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


# Repeat prompt validation: First repeat user's request content unchanged, then give your answer (repeated content should not contain other additional information)
def validate_repeat_prompt(text: str, original_prompt: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    if text.startswith(original_prompt):
        return True
    else:
        return False


# Two responses validation: Provide two different answers separated only by six asterisks "******"
def validate_two_responses(text: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    if text.count("******") == 1:
        response_list = text.split("******")
        first_response = response_list[0].strip()
        second_response = response_list[1].strip()
        if first_response != second_response:
            return True
    return False


# All uppercase: Entire answer must use English uppercase letters
def validate_uppercase(text: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    if text == text.upper():
        return True
    else:
        return False


# All lowercase: Entire answer must use English lowercase letters, no uppercase allowed
def validate_lowercase(text: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    if text == text.lower():
        return True
    else:
        return False


# All-caps word frequency validation: In the answer, all-caps words should appear at least/around/at most {N} times
def validate_frequency_capital_words(text: str, N: int, quantifier: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    words = re.findall(r"\b[A-Z]+\b", text)
    if quantifier == "at least":
        return len(words) >= N
    elif quantifier == "around":
        return len(words) == N
    elif quantifier == "at most":
        return len(words) <= N
    else:
        return False


# End phrase validation: Answer must end with exact phrase {end phrase}, with no other content after it
def validate_end(text: str, end_phrase: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    if text.endswith(end_phrase):
        return True
    else:
        return False


# Quotation wrapping validation: Entire answer must be wrapped in double quotes
def validate_quotation(text: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    if text.startswith('"') and text.endswith('"'):
        return True
    else:
        return False


# Comma ban: No commas allowed in entire answer
def validate_no_commas(text: str) -> bool:
    """
    Reference implementation from: https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
    """
    if "," not in text:
        return True
    else:
        return False


def call_ifeval_function(func, text: str, constraint_dict: dict):
    """
    1) Get function signature of func
    2) Only keep parameters that match signature and are not None
    3) Call func(text, **filtered_args)
    """
    # 1) Get function signature
    sig = inspect.signature(func)
    valid_params = set(sig.parameters.keys())  # Set of function parameter names

    # 2) Filter out irrelevant fields and None values in constraint_dict
    #    (If a function parameter is None value but still valid, keep it; otherwise you can add extra judgment)
    filtered_args = {}
    for k, v in constraint_dict.items():
        if k in valid_params:  # Parameter list actually has this field
            # If you want to completely discard None, you can add: if v is not None:
            filtered_args[k] = v

    # 3) Call function
    return func(text, **filtered_args)


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py
    """
    # If max_penalty is positive, throw error directly, indicating negative values should be used for penalty
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    # Internal function zipngram for splitting text into ngrams
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    # repetition_penalty_reward function calculates n-gram repetition degree in given response
    def repetition_penalty_reward(response, **kwargs) -> float:
        """
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py
        """
        # If response is empty or insufficient ngram size, return 0 directly
        if response == "" or len(response.split()) < ngram_size:
            return 0.0

        # Iterate all ngrams, count unique ngram and total ngram quantities
        ngrams = set()
        total = 0
        for ng in zipngram(response, ngram_size):
            ngrams.add(ng)
            total += 1

        # scaling = 1 - (non-repeated ngrams / total ngram count)
        # The fewer non-repeated (more repeated), the larger scaling
        scaling = 1 - len(ngrams) / total
        # reward is scaling multiplied by max_penalty
        reward = scaling * max_penalty
        return reward

    return repetition_penalty_reward


def extract_after_last_think(input_string, end_think="</think>"):
    """
    Extract content after the last '</think>' tag in the input string,
    and remove all newlines at the beginning of the result string.

    Args:
    input_string: Original string.

    Returns:
    Extracted and processed string. Returns empty string if '</think>' tag not found.
    """
    # Find starting position of last end_think
    last_index = input_string.rfind(end_think)

    # If end_think not found
    if last_index == -1:
        # return ""
        return input_string  # Or return None or original string as needed

    # Calculate position after end_think ends
    start_pos = last_index + len(end_think)

    # Extract part after end_think
    extracted_part = input_string[start_pos:]

    # Remove all leading newlines '\n'
    cleaned_part = extracted_part.lstrip("\n")

    return cleaned_part


class GeneralRuleRewardWorker(Worker):
    """
    Example Reward Worker for executing ifeval validation and putting each func result into output.tensors.
    In this example, ground_truths string
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None
        self.repetition_penalty_reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-0.5)
        # nltk.download('wordnet')
        # nltk.download('omw-1.4')
        # use os to export a nltk_data

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        pass

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def compute_rewards(self, data: DataProto):
        """
        Only call "func_name" in data.non_tensor_batch['ground_truth'],
        and return its result as single response-level reward.
        """

        # 1) Decode response text
        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=False)
        batch_size = len(response_text_list)

        # 2) Read ground_truth (which is a JSON string containing func_name and other parameters)
        prompts = data.non_tensor_batch["prompt"]
        ground_truths = data.non_tensor_batch["ground_truth"]
        tags = data.non_tensor_batch["tag"]

        # 3) Prepare a list to store validation results
        results = [0.0] * batch_size
        repetition_penalty_rewards = []
        response_length_rewards = []

        for i, (resp_tokens, ground_truth, tag, prompt) in enumerate(
            zip(data.batch["responses"], ground_truths, tags, prompts)
        ):
            # Decode current entry
            resp_text = self.tokenizer.decode(resp_tokens, skip_special_tokens=False)
            resp_text1 = resp_text.replace("<|endoftext|>", "").replace("<pad>", "").replace("<|im_end|>", "")
            resp_text = extract_after_last_think(resp_text1)
            # logger.info(f"extract_after_last_think(resp_text): {resp_text}")

            if tag == "ifeval":
                # Parse ground_truth (JSON) to get constraint information
                if isinstance(ground_truth, str):
                    constraint_dict = json.loads(ground_truth)
                else:
                    constraint_dict = ground_truth  # If already dict, use directly

                # Extract func_name from constraints
                func_name = constraint_dict.get("func_name", None)
                if not func_name or func_name not in IF_FUNCTIONS_MAP:
                    self.logger.warning("constraint missing func_name")
                    # If no func_name or corresponding function not found
                    # Then we record result as 0.0 (can do other processing)
                    results[i] = 0.0
                    continue

                # Remove func_name, pass other parameters to function
                constraint_dict.pop("func_name")
                func = IF_FUNCTIONS_MAP[func_name]
                # print(f"Running function {func_name} with Response text: {resp_text}")
                # print(f"Response text: {resp_text}")

                # Call function for validation
                try:
                    result = call_ifeval_function(func, resp_text, constraint_dict)
                except Exception as e:
                    self.logger.error(f"Error in function {func_name}: {e}")
                    result = False
            else:
                self.logger.warning(f"Unknown tag: {tag}")

            # Convert result to float: bool -> (1.0/0.0), numeric -> float(...), other structures -> bool(...)
            if isinstance(result, bool):
                val = 1.0 if result else 0.0
            elif isinstance(result, (int, float)):
                val = float(result)
            else:
                val = 1.0 if result else 0.0

            # Store to results
            results[i] = val
            repetition_penalty_rewards.append(self.repetition_penalty_reward_fn(resp_text1))

        # 4) Prepare output tensors:
        #   - token_level_rewards: same shape as responses, all 0
        #   - response_level_rewards: i.e. results
        #   - scores: can be same as response_level_rewards (for statistics/logging)
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        scores = torch.tensor(results, dtype=torch.float16)
        repetition_penalty_rewards = torch.tensor(repetition_penalty_rewards, dtype=torch.float16)
        response_level_rewards = scores + repetition_penalty_rewards

        # 5) Pack these tensors into the same dictionary
        output_tensors = {
            "token_level_rewards": token_level_rewards,
            "response_level_rewards": response_level_rewards,
            "scores": scores
        }

        # 6) Use DataProto.from_dict(...) to construct return value
        output = DataProto.from_dict(tensors=output_tensors)
        return output


IF_FUNCTIONS_MAP = {
    "verify_keywords": verify_keywords,
    "verify_keyword_frequency": verify_keyword_frequency,
    "validate_forbidden_words": validate_forbidden_words,
    "verify_letter_frequency": verify_letter_frequency,
    "validate_response_language": validate_response_language,
    "verify_paragraph_count": verify_paragraph_count,
    "validate_word_constraint": validate_word_constraint,
    "verify_sentence_constraint": verify_sentence_constraint,
    "validate_paragraphs": validate_paragraphs,
    "verify_postscript": verify_postscript,
    "validate_placeholders": validate_placeholders,
    "verify_bullet_points": verify_bullet_points,
    "validate_title": validate_title,
    "validate_choice": validate_choice,
    "validate_highlighted_sections": validate_highlighted_sections,
    "validate_sections": validate_sections,
    "validate_json_format": validate_json_format,
    "validate_repeat_prompt": validate_repeat_prompt,
    "validate_two_responses": validate_two_responses,
    "validate_uppercase": validate_uppercase,
    "validate_lowercase": validate_lowercase,
    "validate_frequency_capital_words": validate_frequency_capital_words,
    "validate_end": validate_end,
    "validate_quotation": validate_quotation,
    "validate_no_commas": validate_no_commas,
}

ALL_FUNCS = list(IF_FUNCTIONS_MAP.keys())
