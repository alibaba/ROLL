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
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from traceback import print_exc

try:
    # OpenAI Python SDK v1.x
    from openai import OpenAI
except Exception:
    OpenAI = None

# ========================= 1. Prompt 模板 ========================= #
GENERATION_MESSAGE = [
    {
        "role": "system",
        "content": "You are an answer agent for multiple-choice questions. Your task is to output one letter 'A', 'B', 'C', or 'D'. Do not explain, do not copy the option, do not output anything about conversation. Just output the letter.",
    },
    {
        "role": "user",
        "content": "Choose 'A', 'B', 'C', or 'D' for the following question: {question}\n{reference_data}\nYour output should be a single letter 'A', 'B', 'C', or 'D', Do not explain, do not copy the option, do not output anything about conversation.\nNow, your output is:",
    },
]


def load_questions(file_path):
    """Load four-choice questions from JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompts(file_path):
    """Load prompts from JSONL file"""
    prompts_dict = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            prompts_dict[data["qid"]] = data["prompt"]
    return prompts_dict


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


def create_four_choice_question_text(question_data, profile, conversation_history):
    """Create question text from question data with profile and conversation"""
    # Build the question text with profile and conversation
    question_text = ""

    if profile:
        question_text += f"[Profile Begin]\n{profile}\n[Profile End]\n\n"

    if conversation_history:
        question_text += f"[Conversation History Begin]\n{conversation_history}\n[Conversation History End]\n\n"

    question_text += "Which response is most appropriate for this person in this context?\n\n"

    # Add choices
    for choice in sorted(question_data["choices"], key=lambda x: x["label"]):
        question_text += f"{choice['label']}. {choice['text']}\n"

    return question_text


def prepare_prompts_for_evaluation(questions, prompts_dict, tokenizer=None, backend="hf"):
    """Prepare prompts for evaluation"""
    prompts = []
    correct_choices = []
    question_ids = []

    for question in questions:
        qid = question["qid"]

        # Get corresponding prompt data
        if qid not in prompts_dict:
            print(f"Warning: No prompt found for qid {qid}, skipping...")
            continue

        prompt_text = prompts_dict[qid]
        profile, conversation_history = extract_profile_conv_from_prompt(prompt_text)

        # Create question text
        question_text = create_four_choice_question_text(question, profile, conversation_history)

        # Build messages (system + user)
        messages = [
            GENERATION_MESSAGE[0],  # system message
            {
                "role": "user",
                "content": GENERATION_MESSAGE[1]["content"].format(question=question_text, reference_data=""),
            },
        ]

        if backend == "hf":
            if tokenizer is None:
                raise ValueError("Tokenizer is required for backend='hf'")
            prompt = tokenizer.apply_chat_template(
                conversation=messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
            prompts.append(prompt)
        else:
            # For GPT backend, keep raw messages
            prompts.append(messages)

        correct_choices.append(question["correct_answer"])
        question_ids.append(qid)

    return prompts, correct_choices, question_ids


def _get_openai_creds():
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("openai_base_url")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY (or openai_api_key).")
    return base_url, api_key


def batch_inference_gpt(prompts, model_name="gpt-4o-mini", batch_size=16, workers=8, request_timeout=30):
    """Perform batch inference using OpenAI GPT with multithreading per batch"""
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    base_url, api_key = _get_openai_creds()
    client = OpenAI(base_url=base_url, api_key=api_key) if base_url else OpenAI(api_key=api_key)
    results = []
    valid_indices = []

    def request_one(messages):
        # simple retry with backoff
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0,
                    max_tokens=1,
                    timeout=request_timeout,
                )
                response = (resp.choices[0].message.content or "").strip()
                tqdm.write(f"Response: {response}", end="\n")
                # 只解析首个明确字母
                m = re.match(r'^\s*(?:ANSWER|OPTION|CHOICE)?\s*[:\-]?\s*["\'`(]*\b([ABCD])\b', response.upper())
                if not m:
                    m = re.match(r'^\s*["\'`(]*([ABCD])(?:[\).\s]|$)', response.upper())
                return m.group(1) if m else None
            except Exception as e:
                tqdm.write(f"[GPT error attempt {attempt+1}] {e}", end="\n")
                time.sleep(1.5**attempt + random.random() * 0.2)
        return None

    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches (GPT)"):
        batch = prompts[i : i + batch_size]
        max_workers = min(workers, len(batch))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(request_one, messages): i + j for j, messages in enumerate(batch)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    choice = future.result()
                    if choice is not None:
                        results.append(choice)
                        valid_indices.append(idx)
                except Exception as e:
                    tqdm.write(f"[Future error] idx={idx} {e}", end="\n")

    return results, valid_indices


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

            # Extract A, B, C, or D from response
            valid_choices = ["A", "B", "C", "D"]
            found_choices = [choice for choice in valid_choices if choice in response]

            if len(found_choices) == 1:
                results.append(found_choices[0])
                valid_indices.append(i + j)
            else:
                # Invalid response or multiple choices, skip
                continue

    return results, valid_indices


def calculate_scores(predictions, correct_choices, questions, valid_indices):
    """Calculate accuracy and scores"""
    correct_count = 0
    total_score = 0
    # score_breakdown = {"score_1": 0, "score_2": 0, "score_3": 0, "score_4": 0, "score_5": 0}
    score_breakdown = {f"score_{i}": 0 for i in range(0, 6)}

    for i, (pred, correct) in enumerate(zip(predictions, correct_choices)):
        question_idx = valid_indices[i]
        question = questions[question_idx]

        if pred == correct:
            correct_count += 1
            total_score += 5  # Correct answer always gets 5 points
            score_breakdown["score_5"] += 1
        else:
            # Find the score for the predicted choice
            for choice in question["choices"]:
                if choice["label"] == pred:
                    score = choice["score"]
                    total_score += score
                    score_breakdown[f"score_{score}"] += 1
                    break

    total_questions = len(predictions)
    accuracy = correct_count / total_questions * 100 if total_questions > 0 else 0
    avg_score = total_score / total_questions if total_questions > 0 else 0

    return {
        "total_questions": total_questions,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "total_score": total_score,
        "avg_score": avg_score,
        "score_breakdown": score_breakdown,
    }


def analyze_by_difficulty(predictions, correct_choices, questions, valid_indices):
    """Analyze performance by question difficulty"""
    difficulty_bins = {"easy": [], "medium": [], "hard": []}

    for i, (pred, correct) in enumerate(zip(predictions, correct_choices)):
        question_idx = valid_indices[i]
        question = questions[question_idx]

        # Determine difficulty based on max similarity of distractors
        max_similarity = question["difficulty_metrics"]["max_similarity"]

        if max_similarity < 0.4:
            difficulty = "easy"
        elif max_similarity < 0.7:
            difficulty = "medium"
        else:
            difficulty = "hard"

        is_correct = pred == correct
        difficulty_bins[difficulty].append(is_correct)

    difficulty_stats = {}
    for difficulty, results in difficulty_bins.items():
        if results:
            accuracy = sum(results) / len(results) * 100
            difficulty_stats[difficulty] = {"count": len(results), "accuracy": accuracy}
        else:
            difficulty_stats[difficulty] = {"count": 0, "accuracy": 0}

    return difficulty_stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on four-choice questions")
    parser.add_argument("--question_file", required=True, help="Four-choice questions JSON file")
    parser.add_argument("--prompt_file", required=True, help="Prompts JSONL file")
    parser.add_argument(
        "--model_path", default="/project/hdtaccuracy/models/base/Qwen3-8B", help="HF model path (backend=hf)"
    )
    parser.add_argument("--backend", choices=["hf", "gpt"], default="hf", help="Inference backend: hf or gpt")
    parser.add_argument("--gpt_model", default="gpt-4o-mini", help="OpenAI model name (backend=gpt)")
    parser.add_argument("--gpt_workers", type=int, default=8, help="Threads per batch for GPT backend")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--sample_size", type=int, help="Number of questions to sample (default: all)")
    parser.add_argument("--output_file", help="Output file for detailed results")

    args = parser.parse_args()

    # Load model and tokenizer
    if args.backend == "hf":
        print("Loading HF model...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    else:
        print("Using OpenAI GPT backend...")
        tokenizer = None
        model = None
        if OpenAI is None:
            print("openai package not installed. Run: pip install openai")
            return

    # Load questions and prompts
    print("Loading questions...")
    questions = load_questions(args.question_file)

    print("Loading prompts...")
    prompts_dict = load_prompts(args.prompt_file)

    # Sample questions if requested
    if args.sample_size and args.sample_size < len(questions):
        questions = random.sample(questions, args.sample_size)
        print(f"Sampled {len(questions)} questions")

    print(f"Total questions: {len(questions)}")
    print(f"Total prompts available: {len(prompts_dict)}")

    # Prepare prompts
    print("Preparing prompts...")
    prompts, correct_choices, question_ids = prepare_prompts_for_evaluation(
        questions, prompts_dict, tokenizer=tokenizer, backend=args.backend
    )
    print(f"Successfully prepared {len(prompts)} prompts")

    # Perform inference
    print("Performing inference...")
    if args.backend == "hf":
        predictions, valid_indices = batch_inference(model, tokenizer, prompts, args.batch_size)
    else:
        predictions, valid_indices = batch_inference_gpt(prompts, args.gpt_model, args.batch_size, args.gpt_workers)

    if len(predictions) == 0:
        print("No valid predictions generated!")
        return

    # Filter correct choices and questions to match valid predictions
    valid_correct_choices = [correct_choices[i] for i in valid_indices]
    valid_questions = [questions[i] for i in valid_indices]

    # Calculate scores
    results = calculate_scores(predictions, valid_correct_choices, valid_questions, list(range(len(predictions))))

    # Analyze by difficulty
    difficulty_stats = analyze_by_difficulty(
        predictions, valid_correct_choices, valid_questions, list(range(len(predictions)))
    )

    # Print results
    print(f"\n{'='*60}")
    print("QWEN3-8B FOUR-CHOICE EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total questions processed: {len(questions)}")
    print(f"Valid predictions: {results['total_questions']}")
    print(f"Correct answers: {results['correct_count']}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"Total score: {results['total_score']}")
    print(f"Average score: {results['avg_score']:.2f}")

    print(f"\nScore distribution:")
    for score_type, count in results["score_breakdown"].items():
        score_num = score_type.split("_")[1]
        print(f"  Score {score_num}: {count} questions")

    print(f"\nPerformance by difficulty:")
    for difficulty, stats in difficulty_stats.items():
        print(f"  {difficulty.capitalize()}: {stats['count']} questions, {stats['accuracy']:.2f}% accuracy")

    # Save detailed results if requested
    if args.output_file:
        detailed_results = {
            "summary": results,
            "difficulty_analysis": difficulty_stats,
            "predictions": [
                {
                    "qid": question_ids[valid_indices[i]],
                    "prediction": pred,
                    "correct_answer": correct,
                    "is_correct": pred == correct,
                }
                for i, (pred, correct) in enumerate(zip(predictions, valid_correct_choices))
            ],
        }

        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)

        print(f"\nDetailed results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
