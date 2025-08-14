import json
import random
import numpy as np
from typing import List, Dict, Tuple
import argparse

# Threshold for "high similarity" distractors
HI_SIM_THRESHOLD = 0.8


def load_similarity_data(file_path: str) -> List[Dict]:
    """Load similarity data from JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_distractors(item: Dict) -> List[Tuple[str, float, str]]:
    """
    Extract all valid distractors with their similarities

    Returns:
        List of tuples: (distractor_type, similarity_score, distractor_text)
    """
    distractors = []

    # Define all distractor types
    distractor_types = [
        "style_violation_distractor",
        "topic_violation_distractor",
        "richness_violation_distractor",
        "free_violation_distractor",
        "profile_violation_w_distractor",
        "conversation_violation_w_distractor",
        "both_violation_w_distractor",
        "profile_violation_w/o_distractor",
        "conversation_violation_w/o_distractor",
        "both_violation_w/o_distractor",
    ]

    for distractor_type in distractor_types:
        similarity_key = f"{distractor_type}_similarity"

        if (
            distractor_type in item
            and similarity_key in item
            and item[distractor_type] is not None
            and item[similarity_key] is not None
        ):

            distractor_text = item[distractor_type]
            similarity_score = item[similarity_key]

            # Filter out failed distractors
            if not distractor_text.startswith("Failed_"):
                distractors.append((distractor_type, similarity_score, distractor_text))

    return distractors


def select_optimal_distractors(
    distractors: List[Tuple[str, float, str]], num_choices: int = 3
) -> List[Tuple[str, float, str]]:
    """
    Select optimal distractors for a balanced difficulty

    Strategy:
    1. Include at least one high similarity distractor (challenging)
    2. Include at least one low similarity distractor (easy to eliminate)
    3. Fill remaining slots with medium similarity distractors
    """
    if len(distractors) < num_choices:
        return distractors

    # Sort distractors by similarity score
    sorted_distractors = sorted(distractors, key=lambda x: x[1], reverse=True)

    selected = []

    # Strategy 1: Ensure diversity in difficulty
    high_sim = [d for d in sorted_distractors if d[1] > HI_SIM_THRESHOLD]  # Very similar (hard)
    medium_sim = [d for d in sorted_distractors if 0.3 <= d[1] <= HI_SIM_THRESHOLD]  # Medium
    low_sim = [d for d in sorted_distractors if d[1] < 0.3]  # Very different (easy)

    # Select one from each category if available
    if high_sim:
        selected.append(random.choice(high_sim))
    if low_sim:
        selected.append(random.choice(low_sim))

    # Fill remaining slots with medium similarity or remaining options
    remaining_slots = num_choices - len(selected)

    # Remove already selected items
    available = [d for d in sorted_distractors if d not in selected]

    if remaining_slots > 0:
        if medium_sim:
            # Prefer medium similarity for remaining slots
            medium_available = [d for d in medium_sim if d not in selected]
            selected.extend(random.sample(medium_available, min(remaining_slots, len(medium_available))))
            remaining_slots = num_choices - len(selected)

        # Fill any remaining slots
        if remaining_slots > 0:
            other_available = [d for d in available if d not in selected]
            selected.extend(random.sample(other_available, min(remaining_slots, len(other_available))))

    return selected


def calculate_choice_score(similarity: float) -> int:
    """
    Calculate score for each choice based on similarity
    Higher similarity = Higher score (more points for choosing similar option)
    """
    if similarity >= 0.8:
        return 3  # Very high similarity
    elif similarity >= 0.6:
        return 2  # High similarity
    elif similarity >= 0.3:
        return 1  # Medium similarity
    else:
        return 0  # Low similarity


def build_four_choice_question(item: Dict) -> Dict:
    """Build a four-choice question from similarity data"""
    qid = item["qid"]
    correct_answer = item["output"]

    # Extract valid distractors
    distractors = extract_distractors(item)

    if len(distractors) < 3:
        print(f"Warning: Question {qid} has only {len(distractors)} valid distractors")
        return None

    # Select 3 optimal distractors
    selected_distractors = select_optimal_distractors(distractors, num_choices=3)

    # Build choices list
    choices = []

    # Add correct answer (always gets score 5)
    choices.append({"text": correct_answer, "label": "A", "is_correct": True, "score": 5, "type": "correct_output"})

    # Add distractors
    labels = ["B", "C", "D"]
    for i, (distractor_type, similarity, distractor_text) in enumerate(selected_distractors):
        score = calculate_choice_score(similarity)
        choices.append(
            {
                "text": distractor_text,
                "label": labels[i],
                "is_correct": False,
                "score": score,
                "type": distractor_type,
                "similarity": similarity,
            }
        )

    # Shuffle choices while keeping track of correct answer
    random.shuffle(choices)

    # Update labels after shuffling
    for i, choice in enumerate(choices):
        choice["label"] = chr(65 + i)  # A, B, C, D
        if choice["is_correct"]:
            correct_label = choice["label"]

    # Calculate question difficulty metrics
    similarities = [c["similarity"] for c in choices if "similarity" in c]
    difficulty_metrics = {
        "avg_similarity": np.mean(similarities) if similarities else 0,
        "max_similarity": max(similarities) if similarities else 0,
        "min_similarity": min(similarities) if similarities else 0,
        "similarity_std": np.std(similarities) if similarities else 0,
    }

    question = {
        "qid": qid,
        "question_text": f"Choose the most appropriate response:",
        "choices": choices,
        "correct_answer": correct_label,
        "difficulty_metrics": difficulty_metrics,
        "original_prompt": item.get("prompt", ""),
        "metadata": {
            "total_available_distractors": len(distractors),
            "selected_distractors": len(selected_distractors),
        },
    }

    return question


def analyze_question_quality(questions: List[Dict]) -> Dict:
    """Analyze the quality and distribution of generated questions"""
    if not questions:
        return {}

    total_questions = len(questions)

    # Difficulty analysis
    difficulties = [q["difficulty_metrics"]["avg_similarity"] for q in questions]
    max_similarities = [q["difficulty_metrics"]["max_similarity"] for q in questions]

    # Score distribution analysis (scores are limited to 0,1,2,3)
    score_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for q in questions:
        for choice in q["choices"]:
            if not choice["is_correct"]:
                s = int(choice.get("score", 0))
                if s in score_counts:
                    score_counts[s] += 1

    # Distractor type distribution
    distractor_types = {}
    for q in questions:
        for choice in q["choices"]:
            if not choice["is_correct"]:
                dtype = choice["type"]
                distractor_types[dtype] = distractor_types.get(dtype, 0) + 1

    analysis = {
        "total_questions": total_questions,
        "difficulty_stats": {
            "mean_avg_similarity": np.mean(difficulties),
            "std_avg_similarity": np.std(difficulties),
            "mean_max_similarity": np.mean(max_similarities),
            "questions_with_high_similarity": len([d for d in max_similarities if d >= HI_SIM_THRESHOLD]),
            "questions_with_low_similarity": len([d for d in difficulties if d < 0.3]),
        },
        "score_distribution": {
            "score_0": score_counts[0],
            "score_1": score_counts[1],
            "score_2": score_counts[2],
            "score_3": score_counts[3],
        },
        "distractor_type_usage": distractor_types,
    }

    return analysis


def classify_question_by_difficulty(question: Dict) -> str:
    """
    Classify a question into 'hard' | 'medium' | 'easy' | '' based on:
    - Hard:   >=2 high-sim choices and avg_similarity > 0.75
    - Medium: ==1 high-sim choice and 0.5 < avg_similarity < 0.75
    - Easy:   ==0 high-sim choices and avg_similarity < 0.5
    A "high-sim" choice means a distractor with similarity >= HI_SIM_THRESHOLD.
    """
    avg_sim = question.get("difficulty_metrics", {}).get("avg_similarity", 0.0)
    high_sim_count = sum(
        1
        for c in question.get("choices", [])
        if (not c.get("is_correct")) and (c.get("similarity", 0.0) >= HI_SIM_THRESHOLD)
    )

    if high_sim_count >= 2 and avg_sim > 0.75:
        return "hard"
    if high_sim_count == 1 and 0.5 < avg_sim < 0.75:
        return "medium"
    if high_sim_count == 0 and avg_sim < 0.5:
        return "easy"
    return ""


def main():
    parser = argparse.ArgumentParser(description="Build four-choice questions from similarity data")
    parser.add_argument("input_file", help="Input JSON file with similarity data")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file for questions")
    parser.add_argument("--analysis", "-a", help="Output file for quality analysis")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--sample-size", type=int, help="Number of questions to generate (default: all)")
    # Difficulty bucket outputs
    parser.add_argument("--hard-output", help="Output JSON for hard-mode questions")
    parser.add_argument("--medium-output", help="Output JSON for medium-mode questions")
    parser.add_argument("--easy-output", help="Output JSON for easy-mode questions")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    print(f"Loading similarity data from: {args.input_file}")
    data = load_similarity_data(args.input_file)
    print(f"Loaded {len(data)} items")

    # Sample data if requested
    if args.sample_size and args.sample_size < len(data):
        data = random.sample(data, args.sample_size)
        print(f"Sampled {len(data)} items")

    # Build questions
    print("Building four-choice questions...")
    questions = []
    failed_count = 0

    for item in data:
        question = build_four_choice_question(item)
        if question:
            questions.append(question)
        else:
            failed_count += 1

    print(f"Successfully built {len(questions)} questions")
    if failed_count > 0:
        print(f"Failed to build {failed_count} questions (insufficient distractors)")

    # Save questions
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"Questions saved to: {args.output}")

    # Analyze question quality
    analysis = analyze_question_quality(questions)

    print("\n=== Question Quality Analysis ===")
    print(f"Total questions generated: {analysis['total_questions']}")
    print(f"Mean difficulty (avg similarity): {analysis['difficulty_stats']['mean_avg_similarity']:.4f}")
    print(
        f"Questions with challenging distractors (max sim >= {HI_SIM_THRESHOLD}): {analysis['difficulty_stats']['questions_with_high_similarity']}"
    )
    print(
        f"Questions with easy elimination (avg sim < 0.3): {analysis['difficulty_stats']['questions_with_low_similarity']}"
    )

    print("\nScore distribution:")
    for score, count in analysis["score_distribution"].items():
        print(f"  {score}: {count}")

    print("\nDistractor type usage:")
    for dtype, count in sorted(analysis["distractor_type_usage"].items()):
        print(f"  {dtype}: {count}")

    # Classify into difficulty datasets
    hard_questions, medium_questions, easy_questions = [], [], []
    for q in questions:
        bucket = classify_question_by_difficulty(q)
        if bucket == "hard":
            hard_questions.append(q)
        elif bucket == "medium":
            medium_questions.append(q)
        elif bucket == "easy":
            easy_questions.append(q)

    print("\n=== Difficulty buckets ===")
    print(f"Hard   (>=2 high-sim, avg>0.75): {len(hard_questions)}")
    print(f"Medium (1 high-sim,  0.5<avg<0.75): {len(medium_questions)}")
    print(f"Easy   (0 high-sim,  avg<0.5): {len(easy_questions)}")

    # Save per-bucket outputs if requested
    if args.hard_output:
        with open(args.hard_output, "w", encoding="utf-8") as f:
            json.dump(hard_questions, f, indent=2, ensure_ascii=False)
        print(f"Hard questions saved to: {args.hard_output}")

    if args.medium_output:
        with open(args.medium_output, "w", encoding="utf-8") as f:
            json.dump(medium_questions, f, indent=2, ensure_ascii=False)
        print(f"Medium questions saved to: {args.medium_output}")

    if args.easy_output:
        with open(args.easy_output, "w", encoding="utf-8") as f:
            json.dump(easy_questions, f, indent=2, ensure_ascii=False)
        print(f"Easy questions saved to: {args.easy_output}")

    # Save analysis if requested
    if args.analysis:
        with open(args.analysis, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\nAnalysis saved to: {args.analysis}")

    print("Done!")


if __name__ == "__main__":
    main()
