#!/usr/bin/env python3
"""
Comprehensive evaluation script that combines automated metrics (BLEU/ROUGE)
and LLM-based evaluation for personality alignment dialogue models.
"""

import json
import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_automated_evaluation(ground_truth_file: str, model_results_file: str, output_file: str) -> Dict[str, Any]:
    """
    Run automated evaluation using BLEU/ROUGE metrics

    Args:
        ground_truth_file: Path to ground truth JSONL file
        model_results_file: Path to model results JSONL file
        output_file: Output file for automated evaluation results

    Returns:
        Dictionary with automated evaluation statistics
    """
    logger.info("Running automated evaluation (BLEU/ROUGE)...")

    cmd = [
        sys.executable,
        "automated_eval.py",
        "--ground-truth",
        ground_truth_file,
        "--model-results",
        model_results_file,
        "--output",
        output_file,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Automated evaluation completed successfully")

        # Parse the output to extract statistics
        stats = {}
        for line in result.stdout.split("\n"):
            if "BLEU Score - Mean:" in line:
                parts = line.split()
                stats["bleu_mean"] = float(parts[4])
                stats["bleu_std"] = float(parts[6])
                stats["bleu_median"] = float(parts[9])
            elif "ROUGE-L Score - Mean:" in line:
                parts = line.split()
                stats["rouge_l_mean"] = float(parts[4])
                stats["rouge_l_std"] = float(parts[6])
                stats["rouge_l_median"] = float(parts[9])
            elif "Entries evaluated:" in line:
                stats["num_evaluated"] = int(line.split()[-1])

        return stats

    except subprocess.CalledProcessError as e:
        logger.error(f"Automated evaluation failed: {e.stderr}")
        return {}


def run_llm_evaluation(
    ground_truth_file: str, model_results_file: str, output_file: str, evaluator_type: str = "gpt", **kwargs
) -> Dict[str, Any]:
    """
    Run LLM-based evaluation

    Args:
        ground_truth_file: Path to ground truth JSONL file
        model_results_file: Path to model results JSONL file
        output_file: Output file for LLM evaluation results
        evaluator_type: Type of evaluator ('gpt' or 'qwen')
        **kwargs: Additional arguments for the LLM evaluator

    Returns:
        Dictionary with LLM evaluation statistics
    """
    logger.info(f"Running LLM evaluation with {evaluator_type}...")

    # First, prepare evaluation data in the format expected by llm_eval.py
    eval_data_file = prepare_llm_eval_data(ground_truth_file, model_results_file)

    cmd = [
        sys.executable,
        "llm_eval.py",
        "--input",
        eval_data_file,
        "--output",
        output_file,
        "--evaluator-type",
        evaluator_type,
    ]

    # Add additional arguments based on evaluator type
    if evaluator_type == "gpt" and kwargs.get("api_key"):
        cmd.extend(["--api-key", kwargs["api_key"]])
    if kwargs.get("parallel"):
        cmd.append("--parallel")
    if kwargs.get("qwen_model_path"):
        cmd.extend(["--qwen-model-path", kwargs["qwen_model_path"]])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("LLM evaluation completed successfully")

        # Parse the output to extract statistics
        stats = {}
        for line in result.stdout.split("\n"):
            if "num_evaluated:" in line:
                stats["num_evaluated"] = int(line.split()[-1])
            elif "topic_alignment_mean:" in line:
                stats["topic_alignment_mean"] = float(line.split()[-1])
            elif "persona_consistency_mean:" in line:
                stats["persona_consistency_mean"] = float(line.split()[-1])
            elif "preference_consistency_mean:" in line:
                stats["preference_consistency_mean"] = float(line.split()[-1])
            elif "history_consistency_mean:" in line:
                stats["history_consistency_mean"] = float(line.split()[-1])

        return stats

    except subprocess.CalledProcessError as e:
        logger.error(f"LLM evaluation failed: {e.stderr}")
        return {}


def prepare_llm_eval_data(ground_truth_file: str, model_results_file: str) -> str:
    """
    Prepare evaluation data in the format expected by llm_eval.py

    Args:
        ground_truth_file: Path to ground truth JSONL file
        model_results_file: Path to model results JSONL file

    Returns:
        Path to prepared evaluation data file
    """
    logger.info("Preparing data for LLM evaluation...")

    # Load ground truth data
    ground_truth = {}
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            qid = data.get("qid")
            if qid:
                ground_truth[qid] = data

    # Load model results
    model_results = {}
    with open(model_results_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            qid = data.get("qid")
            if qid:
                model_results[qid] = data.get("response", "")

    # Prepare evaluation data
    eval_data = []
    for qid in ground_truth:
        if qid in model_results:
            gt_data = ground_truth[qid]

            # Extract profile and conversation history from prompt
            prompt = gt_data.get("prompt", "")
            profile, conversation_history = extract_profile_and_history(prompt)

            eval_entry = {
                "qid": qid,
                "profile": profile,
                "conversation_history": conversation_history,
                "ground_truth": gt_data.get("output", ""),
                "model_output": model_results[qid],
            }
            eval_data.append(eval_entry)

    # Save prepared data
    eval_data_file = "prepared_eval_data.jsonl"
    with open(eval_data_file, "w", encoding="utf-8") as f:
        for entry in eval_data:
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Prepared {len(eval_data)} entries for LLM evaluation")
    return eval_data_file


def extract_profile_and_history(prompt: str) -> tuple:
    """
    Extract profile and conversation history from the prompt

    Args:
        prompt: The full prompt text

    Returns:
        Tuple of (profile, conversation_history)
    """
    profile = ""
    conversation_history = ""

    # Extract profile
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


def generate_combined_report(
    automated_stats: Dict[str, Any], llm_stats: Dict[str, Any], output_file: str, model_name: str
):
    """
    Generate a combined evaluation report

    Args:
        automated_stats: Statistics from automated evaluation
        llm_stats: Statistics from LLM evaluation
        output_file: Output file path
        model_name: Name of the evaluated model
    """
    logger.info("Generating combined evaluation report...")

    report = {
        "model_name": model_name,
        "evaluation_summary": {"automated_metrics": automated_stats, "llm_evaluation": llm_stats},
        "overall_assessment": {},
    }

    # Calculate overall scores
    if automated_stats and llm_stats:
        # Normalize BLEU and ROUGE to 0-10 scale to match LLM evaluation
        normalized_bleu = automated_stats.get("bleu_mean", 0) * 5
        normalized_rouge = automated_stats.get("rouge_l_mean", 0) * 5

        # Calculate weighted overall score (all metrics now on 1-5 scale)
        automated_score = (normalized_bleu + normalized_rouge) / 2
        llm_score = (
            llm_stats.get("topic_alignment_mean", 1)
            + llm_stats.get("persona_consistency_mean", 1)
            + llm_stats.get("preference_consistency_mean", 1)
            + llm_stats.get("history_consistency_mean", 1)
        ) / 4

        overall_score = automated_score * 0.3 + llm_score * 0.7  # Weight LLM evaluation higher

        report["overall_assessment"] = {
            "automated_score": automated_score,
            "llm_score": llm_score,
            "overall_score": overall_score,
            "grade": get_grade(overall_score),
        }

    # Save report
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 80)
    print(f"EVALUATION REPORT FOR {model_name.upper()}")
    print("=" * 80)

    if automated_stats:
        print(f"\nAUTOMATED METRICS:")
        print(f"  BLEU Score:   {automated_stats.get('bleu_mean', 0):.4f} ± {automated_stats.get('bleu_std', 0):.4f}")
        print(
            f"  ROUGE-L:      {automated_stats.get('rouge_l_mean', 0):.4f} ± {automated_stats.get('rouge_l_std', 0):.4f}"
        )
        print(f"  Evaluated:    {automated_stats.get('num_evaluated', 0)} entries")

    if llm_stats:
        print(f"\nLLM EVALUATION:")
        print(f"  Topic Alignment:      {llm_stats.get('topic_alignment_mean', 1):.2f}/5")
        print(f"  Persona Consistency:  {llm_stats.get('persona_consistency_mean', 1):.2f}/5")
        print(f"  Preference Consistency: {llm_stats.get('preference_consistency_mean', 1):.2f}/5")
        print(f"  History Consistency:  {llm_stats.get('history_consistency_mean', 1):.2f}/5")
        print(f"  Evaluated:            {llm_stats.get('num_evaluated', 0)} entries")

    if "overall_assessment" in report and report["overall_assessment"]:
        assessment = report["overall_assessment"]
        print(f"\nOVERALL ASSESSMENT:")
        print(f"  Overall Score: {assessment['overall_score']:.2f}/5 ({assessment['grade']})")

    print(f"\nDetailed report saved to: {output_file}")
    print("=" * 80)


def get_grade(score: float) -> str:
    """Convert numerical score to letter grade (1-5 scale)"""
    if score >= 4.5:
        return "A+"
    elif score >= 4.2:
        return "A"
    elif score >= 4.0:
        return "A-"
    elif score >= 3.7:
        return "B+"
    elif score >= 3.5:
        return "B"
    elif score >= 3.2:
        return "B-"
    elif score >= 3.0:
        return "C+"
    elif score >= 2.7:
        return "C"
    elif score >= 2.5:
        return "C-"
    elif score >= 2.0:
        return "D"
    else:
        return "F"


def main():
    parser = argparse.ArgumentParser(description="Comprehensive evaluation of personality alignment dialogue models")

    # Required arguments
    parser.add_argument("--ground-truth", "-g", required=True, help="Path to ground truth JSONL file")
    parser.add_argument("--model-results", "-m", required=True, help="Path to model results JSONL file")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for evaluation results")
    parser.add_argument("--model-name", "-n", required=True, help="Name of the model being evaluated")

    # Evaluation options
    parser.add_argument("--skip-automated", action="store_true", help="Skip automated evaluation (BLEU/ROUGE)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM evaluation")

    # LLM evaluation options
    parser.add_argument(
        "--evaluator-type", choices=["gpt", "qwen"], default="gpt", help="Type of LLM evaluator to use"
    )
    parser.add_argument("--api-key", help="OpenAI API key (for GPT evaluator)")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing (for GPT evaluator)")
    parser.add_argument("--qwen-model-path", default="Qwen/Qwen2.5-7B-Instruct", help="Path to Qwen model")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Define output files
    automated_output = output_dir / f"{args.model_name}_automated_eval.txt"
    llm_output = output_dir / f"{args.model_name}_llm_eval.jsonl"
    combined_report = output_dir / f"{args.model_name}_evaluation_report.json"

    automated_stats = {}
    llm_stats = {}

    try:
        # Run automated evaluation
        if not args.skip_automated:
            automated_stats = run_automated_evaluation(args.ground_truth, args.model_results, str(automated_output))

        # Run LLM evaluation
        if not args.skip_llm:
            llm_kwargs = {"api_key": args.api_key, "parallel": args.parallel, "qwen_model_path": args.qwen_model_path}
            llm_stats = run_llm_evaluation(
                args.ground_truth, args.model_results, str(llm_output), args.evaluator_type, **llm_kwargs
            )

        # Generate combined report
        generate_combined_report(automated_stats, llm_stats, str(combined_report), args.model_name)

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
