# Copyright (c) 2025, ALIBABA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import string
from typing import Dict, Any, Optional, List, Tuple, Callable


############################Tool Fuctions############################
def normalize_text(text: str) -> str:
    """Normalize text by removing whitespace, punctuation, and converting to lowercase."""
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def extract_answer_from_text(text: str) -> str:
    """Extract answer from text with various patterns."""
    patterns = [
        r"The answer is:?\s*(.*?)(?:\n|$)",
        r"Answer:?\s*(.*?)(?:\n|$)",
        r"Final answer:?\s*(.*?)(?:\n|$)",
        r"Therefore,\s*(.*?)(?:\n|$)",
        r"Thus,\s*(.*?)(?:\n|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # If no pattern matches, return the last line as a fallback
    lines = text.strip().split("\n")
    return lines[-1].strip()


# ====== Dataset Processors ======


def process_metamathqa(item: Dict[str, Any]) -> Tuple[str, str]:
    """Process MetaMathQA dataset item."""
    question = item["query"]
    answer = extract_answer_from_text(item["response"])
    return question, answer


def process_gsm8k(item: Dict[str, Any]) -> Tuple[str, str]:
    """Process GSM8K dataset item."""
    question = item["question"]
    answer = item["answer"]
    answer = answer.split("####")[1].strip().lower()
    return question, answer


def process_theoremqa(item: Dict[str, Any]) -> Tuple[str, str]:
    """Process TheoremQA dataset item."""
    question = item["Question"]
    answer = str(item["Answer"])
    return question, answer


def process_mmlu(item: Dict[str, Any]) -> Tuple[str, str]:
    """Process MMLU dataset with multiple choice format."""
    question = item["question"]
    choices = [item["choices"][i] for i in range(len(item["choices"]))]
    formatted_question = question + "\n" + "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    answer = chr(65 + item["answer"])  # Convert to A, B, C, D format
    return formatted_question, answer


def process_gpqa(item: Dict[str, Any]) -> Tuple[str, str]:
    """Process GPQA dataset item."""
    question = item["Question"]
    answer = extract_answer_from_text(item["Correct Answer"])
    return question, answer


# ====== Scoring Functions ======


def compute_score_exact_match(prediction: str, label: str) -> Dict[str, Any]:
    """Basic exact match after normalization."""
    norm_pred = normalize_text(prediction)
    norm_label = normalize_text(label)

    is_correct = norm_pred == norm_label
    is_valid = len(norm_pred) > 0  # Simple validity check

    return {
        "is_correct": is_correct,
        "is_valid": is_valid,
        "normalized_prediction": norm_pred,
        "normalized_label": norm_label,
    }


def compute_score_numeric(prediction: str, label: str) -> Dict[str, Any]:
    """Extract numeric values and compare them."""
    # Extract the first numeric value from both prediction and label
    pred_match = re.search(r"(\d+(?:\.\d+)?)", prediction)
    label_match = re.search(r"(\d+(?:\.\d+)?)", label)

    is_valid = pred_match is not None

    if pred_match and label_match:
        pred_answer = pred_match.group(0)
        label_answer = label_match.group(0)

        try:
            is_correct = float(pred_answer) == float(label_answer)
        except ValueError:
            is_correct = False
    else:
        is_correct = False

    # Also try text match as fallback
    text_match = normalize_text(prediction) == normalize_text(label)
    is_correct = is_correct or text_match

    return {
        "is_correct": is_correct,
        "is_valid": is_valid,
        "numeric_match": is_correct and not text_match,
        "text_match": text_match,
    }


def compute_score_multiple_choice(prediction: str, label: str) -> Dict[str, Any]:
    """Score multiple choice answers (A, B, C, D)."""
    pred_match = re.search(r"([A-D])", prediction.upper())
    label_match = re.search(r"([A-D])", label.upper())

    is_valid = pred_match is not None

    if pred_match and label_match:
        pred_choice = pred_match.group(0)
        label_choice = label_match.group(0)
        is_correct = pred_choice == label_choice
    else:
        # Fallback to text comparison
        is_correct = normalize_text(prediction) == normalize_text(label)

    return {
        "is_correct": is_correct,
        "is_valid": is_valid,
        "extracted_prediction": pred_match.group(0) if pred_match else None,
        "extracted_label": label_match.group(0) if label_match else None,
    }


##########################registration###########################
REGISTERD_STATIC_ENV = {
    "metamathqa": {
        "config": {
            "path": "meta-math/MetaMathQA",
        },
        "processor": process_metamathqa,
        "compute_score": compute_score_exact_match,
    },
    "gsm8k": {
        "config": {"path": "openai/gsm8k", "name": "main"},
        "processor": process_gsm8k,
        "compute_score": compute_score_numeric,
    },
    # "theoremqa": {
    #     "config": {
    #         "path": "TIGER-Lab/TheoremQA",
    #     },
    #     "processor": process_theoremqa,
    #     "compute_score": compute_score_numeric
    # },
    "mmlu": {
        "config": {
            "path": "cais/mmlu",
            "name": "abstract_algebra",
        },
        "processor": process_mmlu,
        "compute_score": compute_score_multiple_choice,
    },
    # "gpqa":{
    #     "config": {
    #         "path": "Idavidrein/gpqa",
    #         "name": "gpqa_main",
    #     },
    #     "processor": process_gpqa,
    #     "compute_score": compute_score_exact_match
    # }
}
