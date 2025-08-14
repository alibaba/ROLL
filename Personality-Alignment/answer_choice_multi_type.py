import json
import torch
import random
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from tqdm import tqdm
import argparse
import re
import os

# ========================= 1. Prompt 模板 ========================= #
GENERATION_PROMPT = """
[System]
You are an answer agent for multiple‑choice questions.
• Your task: Given a multiple-choice QUESTION, output exactly one letter 'A' or 'B' as the correct answer you thought.
• The output should be only 'A' if you think the first option 'A.' is correct, or 'B' if you think the second option 'B.'is correct.
• Do NOT explain, do NOT copy the option, produce only one letter.

[User]
QUESTION: {question}

[Assistant] 
"""

GENERATION_MESSAGE = [
    {
        "role": "system",
        "content": "You are an answer agent for multiple-choice questions. Your task is to output one letter 'A' or 'B'. Just output one letter. Do not explain, do not copy the option, do not output anything about conversation. Just output the letter.",
    },
    {
        "role": "user",
        "content": "Choose 'A' or 'B' for the following question: {question}\n{reference_data}\nYour output should be a single letter 'A' or 'B'.\nNow, your output is:",
    },
]


def extract_profile_conv_from_prompt(prompt):
    """Extract profile and conversation from the prompt"""
    profile = ""
    conversation_history = ""

    if "[Profile Begin]" in prompt and "[Profile End]" in prompt:
        start = prompt.find("[Profile Begin]") + len("[Profile Begin]")
        end = prompt.find("[Profile End]")
        profile = prompt[start:end].strip()

    # Extract conversation history
    if "[Conversation History Begin]" in prompt and "[Conversation History End]" in prompt:
        start = prompt.find("[Conversation History Begin]") + len("[Conversation History Begin]")
        end = prompt.find("[Conversation History End]")
        conversation_history = prompt[start:end].strip()

    return profile, conversation_history


def load_dataset(file_path):
    """Load dataset from jsonl file"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    return data


def create_binary_choice_question(correct_answer, distractor, profile, conversation):
    """Create a binary choice question with correct answer and distractor"""
    # Randomly decide whether correct answer is A or B
    if random.choice([True, False]):
        option_a = correct_answer
        option_b = distractor
        correct_choice = "A"
    else:
        option_a = distractor
        option_b = correct_answer
        correct_choice = "B"

    # Create the question
    question = f"[Profile Begin]{profile}[Profile End]\n"
    question += f"[Conversation History Begin]{conversation}[Conversation History End]\n\n"
    question += "Which response is most appropriate for this person in this context?\n\n"
    question += f"A. {option_a}\n"
    question += f"B. {option_b}\n"

    return question, correct_choice


def prepare_prompts_for_distractor_type(data, distractor_type, tokenizer):
    """Prepare prompts for a specific distractor type"""
    prompts = []
    correct_choices = []
    valid_items = []

    for item in data:
        # Check if this distractor type exists for this item
        distractor_key = f"{distractor_type}_distractor"
        if distractor_key not in item or item[distractor_key].startswith("Failed_"):
            continue

        profile, conversation = extract_profile_conv_from_prompt(item["prompt"])
        correct_answer = item["output"]
        distractor = item[distractor_key]

        # Create binary choice question
        question, correct_choice = create_binary_choice_question(correct_answer, distractor, profile, conversation)

        # Apply chat template
        messages = [
            GENERATION_MESSAGE[0],  # system message
            {"role": "user", "content": GENERATION_MESSAGE[1]["content"].format(question=question, reference_data="")},
        ]

        prompt = tokenizer.apply_chat_template(
            conversation=messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )

        prompts.append(prompt)
        correct_choices.append(correct_choice)
        valid_items.append(item)

    return prompts, correct_choices, valid_items


def batch_inference(model, tokenizer, prompts, batch_size=16):
    """Perform batch inference"""
    results = []
    valid_indices = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        batch_prompts = prompts[i : i + batch_size]

        # Tokenize batch
        input_tokens = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        ).to(model.device)

        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                **input_tokens, max_new_tokens=2, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )

        # Decode responses
        for j, output in enumerate(outputs):
            response = tokenizer.decode(output[input_tokens["input_ids"].shape[1] :], skip_special_tokens=True)
            response = response.strip().upper()

            # print(f"Response: {response}")

            # Extract A or B from response
            if "A" in response and "B" not in response:
                results.append("A")
                valid_indices.append(i + j)
            elif "B" in response and "A" not in response:
                results.append("B")
                valid_indices.append(i + j)
            else:
                # Invalid response, skip
                continue

    return results, valid_indices


def evaluate_distractor_type(model, tokenizer, data, distractor_type, batch_size=16):
    """Evaluate accuracy for a specific distractor type"""
    print(f"\n{'='*60}")
    print(f"Evaluating distractor type: {distractor_type}")
    print(f"{'='*60}")

    # Prepare prompts for this distractor type
    prompts, correct_choices, valid_items = prepare_prompts_for_distractor_type(data, distractor_type, tokenizer)

    if len(prompts) == 0:
        print(f"No valid data found for distractor type: {distractor_type}")
        return None

    print(f"Valid samples for {distractor_type}: {len(prompts)}")

    # Perform inference
    predictions, valid_indices = batch_inference(model, tokenizer, prompts, batch_size)

    if len(predictions) == 0:
        print(f"No valid predictions for distractor type: {distractor_type}")
        return None

    # Filter correct choices to match valid predictions
    valid_correct_choices = [correct_choices[i] for i in valid_indices]

    # Calculate accuracy
    correct = sum(1 for pred, truth in zip(predictions, valid_correct_choices) if pred == truth)
    total = len(predictions)
    accuracy = correct / total * 100 if total > 0 else 0

    print(f"Total samples: {len(prompts)}")
    print(f"Valid predictions: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

    return {
        "distractor_type": distractor_type,
        "total_samples": len(prompts),
        "valid_predictions": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
    }


def main():
    # File path
    file_path = "/project/hdtaccuracy/Personality-Alignment/choice_ver/raw_choice_data_v7_hard.jsonl"

    # Load model and tokenizer
    print("Loading Qwen3-8B model...")
    model_name = "/project/hdtaccuracy/models/base/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load dataset
    print("Loading dataset...")
    data = load_dataset(file_path)  # Load full dataset

    distractor_types = [
        "style_violation",
        "topic_violation",
        "richness_violation",
        "free_violation",
        "profile_violation_w",
        "conversation_violation_w",
        "both_violation_w",
        "profile_violation_w/o",
        "conversation_violation_w/o",
        "both_violation_w/o",
    ]

    # Store results for all distractor types
    all_results = []

    # Test each distractor type
    for distractor_type in distractor_types:
        result = evaluate_distractor_type(model, tokenizer, data, distractor_type, batch_size=32)
        if result:
            all_results.append(result)

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL DISTRACTOR TYPES")
    print(f"{'='*80}")
    print(f"{'Distractor Type':<25} {'Samples':<8} {'Valid':<7} {'Correct':<8} {'Accuracy':<10}")
    print("-" * 80)

    for result in all_results:
        print(
            f"{result['distractor_type']:<25} "
            f"{result['total_samples']:<8} "
            f"{result['valid_predictions']:<7} "
            f"{result['correct_predictions']:<8} "
            f"{result['accuracy']:<10.2f}%"
        )

    # Calculate average accuracy
    if all_results:
        avg_accuracy = sum(r["accuracy"] for r in all_results) / len(all_results)
        print("-" * 80)
        print(f"{'AVERAGE':<25} {'':<8} {'':<7} {'':<8} {avg_accuracy:<10.2f}%")

    # Save results to file
    output_file = file_path.replace(".jsonl", "_evaluation_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
