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


def extract_profile_conv_choices_from_prompt(prompt):
    """Extract profile conversation choices from the prompt"""
    if "[Profile Begin]" in prompt and "[Profile End]" in prompt:
        start = prompt.find("[Profile Begin]") + len("[Profile Begin]")
        end = prompt.find("[Profile End]")
        profile = prompt[start:end].strip()

    # Extract conversation history
    if "[Conversation History Begin]" in prompt and "[Conversation History End]" in prompt:
        start = prompt.find("[Conversation History Begin]") + len("[Conversation History Begin]")
        end = prompt.find("[Conversation History End]")
        conversation_history = prompt[start:end].strip()

    if "Possible outputs of the person:" in prompt and "Your choice:" in prompt:
        start = prompt.find("Possible outputs of the person:") + len("Possible outputs of the person:")
        end = prompt.find("Your choice:")
        choices = prompt[start:end].strip()

    return profile, conversation_history, choices


def load_dataset(file_path):
    """Load dataset from jsonl file"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    return data


def prepare_prompts(data):
    """Prepare prompts using the GENERATION_MESSAGE template"""
    prompts = []
    for item in data:
        question = item["prompt"]
        # formatted_prompt = GENERATION_MESSAGE[1]["content"].format(question=question)
        prompts.append(question)
    return prompts


def prepare_prompts_beta(data, tokenizer):
    """Prepare prompts using the GENERATION_MESSAGE template with tokenizer"""
    prompts = []
    for item in data:
        question = item["prompt"]
        profile, conversation_history, choices = extract_profile_conv_choices_from_prompt(question)
        formatted_prompt = GENERATION_MESSAGE[1]["content"].format(
            question=choices,
            reference_data=f"You are a person with the following profile: {profile}\n"
            f"Conversation history: {conversation_history}\n",
            # f"Possible outputs of the person: {choices}\n",
        )
        # Apply chat template for each item
        messages = [
            GENERATION_MESSAGE[0],  # system message
            {"role": "user", "content": formatted_prompt},
        ]
        # Get the string representation of the messages
        prompt = tokenizer.apply_chat_template(
            conversation=messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        prompts.append(prompt)
    return prompts


def batch_inference(model, tokenizer, data):
    """Perform batch inference using the message format"""
    batch_size = 32
    results = []
    valid_indices = []
    device = next(model.parameters()).device

    for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        batch_data = data[i : i + batch_size]

        # Prepare messages for each item in batch
        batch_messages = []
        # for item in batch_data:
        #     messages = [
        #         GENERATION_MESSAGE[0],  # system message
        #         {"role": "user", "content": GENERATION_MESSAGE[1]["content"].format(question=item["prompt"])},
        #     ]
        #     batch_messages.append(messages)

        # # Apply chat template for batch (get strings, not tokenized)
        # batch_prompts = [
        #     tokenizer.apply_chat_template(
        #         conversation=messages,
        #         add_generation_prompt=True,
        #         tokenize=False,
        #         enable_thinking=False,
        #     )
        #     for messages in batch_messages
        # ]
        batch_prompts = prepare_prompts_beta(batch_data, tokenizer)
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

        # Decode responses and track valid predictions
        for j, output in enumerate(outputs):
            # 提取问句部分
            response = tokenizer.decode(output[input_tokens["input_ids"].shape[1] :], skip_special_tokens=True)
            # Extract A or B from response
            response = response.strip().upper()
            # print(f"Input {i+j}: {batch_prompts[j]}\n")  # Debugging output
            # print(f"Response {i+j}: {response}\n")  # Debugging output
            if "</think>\n\n" in response:
                response = response.replace("</think>\n\n", "")
            if "A" in response and "B" not in response:
                results.append("A")
                valid_indices.append(i + j)
            elif "B" in response and "A" not in response:
                results.append("B")
                valid_indices.append(i + j)
            # Skip responses that don't contain exactly A or B
    return results, valid_indices


def main():
    file_path = "/home/szhangfa/ROLL/Personality-Alignment/changed_dialogue_dataset_v6_weak_gai_conver_little.jsonl"

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
    data = load_dataset(file_path)[:1000]

    # Extract ground truth
    ground_truth = [item["output"].strip().upper() for item in data]

    # Perform batch inference
    print("Starting inference...")
    predictions, valid_indices = batch_inference(model, tokenizer, data)

    # Filter ground truth to only include valid predictions
    valid_ground_truth = [ground_truth[i] for i in valid_indices]

    # Calculate accuracy
    correct = sum(1 for pred, truth in zip(predictions, valid_ground_truth) if pred == truth)
    total = len(valid_indices)
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\nResults:")
    print(f"Total samples processed: {len(data)}")
    print(f"Valid predictions: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
