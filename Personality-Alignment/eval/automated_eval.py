import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk

# Try to import rouge, with fallback
try:
    from rouge import Rouge

    ROUGE_AVAILABLE = True
except ImportError:
    try:
        from rouge_score import rouge_scorer

        ROUGE_AVAILABLE = "rouge_score"
    except ImportError:
        ROUGE_AVAILABLE = False

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_bleu_rouge(reference: str, simulation: str) -> tuple:
    """
    Calculate BLEU and ROUGE-L scores between reference and simulation texts

    Args:
        reference: Ground truth text
        simulation: Generated text

    Returns:
        Tuple of (bleu_score, rouge_l_score)
    """
    try:
        # Tokenize and lowercase
        reference_tokens = word_tokenize(reference.lower())
        simulation_tokens = word_tokenize(simulation.lower())

        # Calculate BLEU score
        bleu = sentence_bleu([reference_tokens], simulation_tokens)

        # Calculate ROUGE-L score
        rouge_l = 0.0
        if ROUGE_AVAILABLE == True:
            rouge = Rouge()
            rouge_scores = rouge.get_scores(simulation, reference)
            rouge_l = rouge_scores[0]["rouge-l"]["f"]
        elif ROUGE_AVAILABLE == "rouge_score":
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            scores = scorer.score(reference, simulation)
            rouge_l = scores["rougeL"].fmeasure
        else:
            # Simple fallback ROUGE-L implementation
            rouge_l = simple_rouge_l(reference_tokens, simulation_tokens)

        return bleu, rouge_l
    except Exception as e:
        logger.warning(f"Error calculating metrics: {e}")
        return 0.0, 0.0


def simple_rouge_l(reference_tokens: list, simulation_tokens: list) -> float:
    """
    Simple ROUGE-L implementation as fallback
    """
    if not reference_tokens or not simulation_tokens:
        return 0.0

    # Find LCS (Longest Common Subsequence)
    def lcs_length(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    lcs_len = lcs_length(reference_tokens, simulation_tokens)
    if lcs_len == 0:
        return 0.0

    # ROUGE-L F-measure
    recall = lcs_len / len(reference_tokens)
    precision = lcs_len / len(simulation_tokens)

    if precision + recall == 0:
        return 0.0

    f_measure = 2 * precision * recall / (precision + recall)
    return f_measure


def load_ground_truth_data(file_path: str) -> Dict[str, str]:
    """
    Load ground truth data from JSONL file

    Args:
        file_path: Path to ground truth JSONL file

    Returns:
        Dictionary mapping qid to output text
    """
    ground_truth = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                qid = data.get("qid")
                output = data.get("output", "")
                if qid and output:
                    ground_truth[qid] = output
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing ground truth line: {e}")

    logger.info(f"Loaded {len(ground_truth)} ground truth entries")
    return ground_truth


def load_model_results(file_path: str) -> Dict[str, str]:
    """
    Load model results from JSONL file

    Args:
        file_path: Path to model results JSONL file

    Returns:
        Dictionary mapping qid to response text
    """
    results = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                qid = data.get("qid")
                response = data.get("response", "")
                if qid and response:
                    results[qid] = response
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing model results line: {e}")

    logger.info(f"Loaded {len(results)} model result entries")
    return results


def evaluate_automated_metrics(ground_truth: Dict[str, str], model_results: Dict[str, str]) -> Dict[str, Any]:
    """
    Evaluate automated metrics (BLEU and ROUGE) for all matching entries

    Args:
        ground_truth: Dictionary of ground truth responses
        model_results: Dictionary of model responses

    Returns:
        Dictionary containing evaluation results and statistics
    """
    bleu_scores = []
    rouge_scores = []
    detailed_results = []

    # Find common qids
    common_qids = set(ground_truth.keys()) & set(model_results.keys())
    logger.info(f"Found {len(common_qids)} common entries for evaluation")

    for qid in common_qids:
        reference = ground_truth[qid]
        simulation = model_results[qid]

        bleu, rouge_l = calculate_bleu_rouge(reference, simulation)

        bleu_scores.append(bleu)
        rouge_scores.append(rouge_l)

        detailed_results.append(
            {"qid": qid, "bleu": bleu, "rouge_l": rouge_l, "reference": reference, "simulation": simulation}
        )

    # Calculate statistics
    results = {
        "num_evaluated": len(bleu_scores),
        "bleu_mean": np.mean(bleu_scores) if bleu_scores else 0.0,
        "bleu_std": np.std(bleu_scores) if bleu_scores else 0.0,
        "bleu_median": np.median(bleu_scores) if bleu_scores else 0.0,
        "rouge_l_mean": np.mean(rouge_scores) if rouge_scores else 0.0,
        "rouge_l_std": np.std(rouge_scores) if rouge_scores else 0.0,
        "rouge_l_median": np.median(rouge_scores) if rouge_scores else 0.0,
        "detailed_results": detailed_results,
    }

    return results


def save_results(results: Dict[str, Any], output_file: str):
    """
    Save evaluation results to file

    Args:
        results: Evaluation results
        output_file: Output file path
    """
    with open(output_file, "w", encoding="utf-8") as f:
        # Save summary statistics
        summary = {k: v for k, v in results.items() if k != "detailed_results"}
        f.write("=== AUTOMATED EVALUATION SUMMARY ===\n")
        f.write(json.dumps(summary, indent=2) + "\n\n")

        # Save detailed results
        f.write("=== DETAILED RESULTS ===\n")
        for result in results["detailed_results"]:
            f.write(json.dumps(result) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model outputs using automated metrics (BLEU/ROUGE)")
    parser.add_argument("--ground-truth", "-g", required=True, help="Path to ground truth JSONL file")
    parser.add_argument("--model-results", "-m", required=True, help="Path to model results JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output file for evaluation results")

    args = parser.parse_args()

    try:
        # Load data
        logger.info("Loading ground truth data...")
        ground_truth = load_ground_truth_data(args.ground_truth)

        logger.info("Loading model results...")
        model_results = load_model_results(args.model_results)

        # Evaluate
        logger.info("Evaluating automated metrics...")
        results = evaluate_automated_metrics(ground_truth, model_results)

        # Save results
        save_results(results, args.output)

        # Print summary
        logger.info("=== EVALUATION SUMMARY ===")
        logger.info(f"Entries evaluated: {results['num_evaluated']}")
        logger.info(
            f"BLEU Score - Mean: {results['bleu_mean']:.4f} ± {results['bleu_std']:.4f}, Median: {results['bleu_median']:.4f}"
        )
        logger.info(
            f"ROUGE-L Score - Mean: {results['rouge_l_mean']:.4f} ± {results['rouge_l_std']:.4f}, Median: {results['rouge_l_median']:.4f}"
        )
        logger.info(f"Results saved to: {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
