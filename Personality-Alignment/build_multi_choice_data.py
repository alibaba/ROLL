import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm
import argparse


def load_jsonl(file_path):
    """Load JSONL file and return list of dictionaries"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def compute_similarities(data, model_name="all-MiniLM-L6-v2", output_file=None, batch_size=32):
    """
    Compute semantic similarities between distractors and correct output using batch inference

    Args:
        data: List of dictionaries from JSONL file
        model_name: SentenceTransformer model name
        output_file: Optional output file path for results
        batch_size: Batch size for inference

    Returns:
        List of dictionaries with similarity scores
    """
    # Load embedding model
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Define distractor types
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

    # Collect all texts for batch encoding
    all_outputs = []
    all_distractors = {dt: [] for dt in distractor_types}
    item_indices = []

    print("Preparing texts for batch encoding...")
    for i, item in enumerate(data):
        all_outputs.append(item["output"])
        item_indices.append(i)

        for distractor_type in distractor_types:
            if distractor_type in item and item[distractor_type] is not None:
                all_distractors[distractor_type].append(item[distractor_type])
            else:
                all_distractors[distractor_type].append(None)

    # Batch encode all outputs
    print(f"Encoding {len(all_outputs)} outputs in batches of {batch_size}...")
    output_embeddings = model.encode(all_outputs, batch_size=batch_size, show_progress_bar=True)

    # Batch encode each distractor type
    distractor_embeddings = {}
    for distractor_type in distractor_types:
        valid_distractors = [d for d in all_distractors[distractor_type] if d is not None]
        if valid_distractors:
            print(f"Encoding {len(valid_distractors)} {distractor_type} distractors...")
            embeddings = model.encode(valid_distractors, batch_size=batch_size, show_progress_bar=True)

            # Create mapping back to original indices
            embedding_map = {}
            embedding_idx = 0
            for i, distractor in enumerate(all_distractors[distractor_type]):
                if distractor is not None:
                    embedding_map[i] = embeddings[embedding_idx]
                    embedding_idx += 1
                else:
                    embedding_map[i] = None

            distractor_embeddings[distractor_type] = embedding_map
        else:
            distractor_embeddings[distractor_type] = {i: None for i in range(len(data))}

    # Compute similarities
    results = []
    print("Computing similarities...")

    for i, item in enumerate(tqdm(data)):
        qid = item["qid"]
        output = item["output"]
        output_embedding = output_embeddings[i]

        similarity_scores = {"qid": qid, "output": output}

        # Compute similarity for each distractor type
        for distractor_type in distractor_types:
            if distractor_type in item and item[distractor_type] is not None:
                distractor_text = item[distractor_type]
                distractor_embedding = distractor_embeddings[distractor_type][i]

                if distractor_embedding is not None:
                    # Compute cosine similarity
                    similarity = cosine_similarity(
                        output_embedding.reshape(1, -1), distractor_embedding.reshape(1, -1)
                    )[0][0]

                    similarity_scores[f"{distractor_type}_similarity"] = float(similarity)
                    similarity_scores[distractor_type] = distractor_text
                else:
                    similarity_scores[f"{distractor_type}_similarity"] = None
                    similarity_scores[distractor_type] = None
            else:
                similarity_scores[f"{distractor_type}_similarity"] = None
                similarity_scores[distractor_type] = None

        results.append(similarity_scores)

    # Save results if output file specified
    if output_file:
        if output_file.endswith(".json"):
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        elif output_file.endswith(".jsonl"):
            with open(output_file, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
        elif output_file.endswith(".csv"):
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
        else:
            # Default to JSONL
            with open(output_file, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Results saved to: {output_file}")

    return results


def analyze_similarities(results, output_file=None):
    """Analyze similarity scores and provide detailed statistics"""
    distractor_types = [col for col in results[0].keys() if col.endswith("_similarity")]

    analysis_output = []
    analysis_output.append("=== Detailed Similarity Analysis ===")

    # Overall statistics
    all_similarities = []
    for distractor_type in distractor_types:
        similarities = [r[distractor_type] for r in results if r[distractor_type] is not None]
        all_similarities.extend(similarities)

    if all_similarities:
        analysis_output.append("\nOverall Statistics (all distractors):")
        analysis_output.append(f"  Total samples: {len(all_similarities)}")
        analysis_output.append(f"  Mean: {np.mean(all_similarities):.4f}")
        analysis_output.append(f"  Median: {np.median(all_similarities):.4f}")
        analysis_output.append(f"  Std: {np.std(all_similarities):.4f}")
        analysis_output.append(f"  25th percentile: {np.percentile(all_similarities, 25):.4f}")
        analysis_output.append(f"  75th percentile: {np.percentile(all_similarities, 75):.4f}")

    # Per distractor type analysis - create DataFrame for better visualization
    distractor_stats = []

    for distractor_type in distractor_types:
        similarities = [r[distractor_type] for r in results if r[distractor_type] is not None]
        if similarities:
            high_sim_count = len([s for s in similarities if s > 0.8])
            low_sim_count = len([s for s in similarities if s < 0.3])

            stats = {
                "Distractor Type": distractor_type.replace("_similarity", ""),
                "Count": len(similarities),
                "Mean": round(np.mean(similarities), 4),
                "Median": round(np.median(similarities), 4),
                "Std": round(np.std(similarities), 4),
                "Min": round(np.min(similarities), 4),
                "Max": round(np.max(similarities), 4),
                "Q25": round(np.percentile(similarities, 25), 4),
                "Q75": round(np.percentile(similarities, 75), 4),
                "High Sim (>0.8)": f"{high_sim_count} ({high_sim_count/len(similarities)*100:.1f}%)",
                "Low Sim (<0.3)": f"{low_sim_count} ({low_sim_count/len(similarities)*100:.1f}%)",
            }
            distractor_stats.append(stats)

    # Create DataFrame and display as markdown
    df_stats = pd.DataFrame(distractor_stats)
    analysis_output.append("\n## Per Distractor Type Analysis:")
    analysis_output.append(df_stats.to_markdown(index=False))

    # Effectiveness ranking
    effectiveness_df = df_stats[["Distractor Type", "Mean"]].sort_values("Mean").reset_index(drop=True)
    effectiveness_df["Rank"] = range(1, len(effectiveness_df) + 1)
    effectiveness_df = effectiveness_df[["Rank", "Distractor Type", "Mean"]]

    analysis_output.append("\n## Distractor Effectiveness Ranking (by mean similarity, lower is better):")
    analysis_output.append(effectiveness_df.to_markdown(index=False))

    # Consistency ranking
    consistency_df = df_stats[["Distractor Type", "Std"]].sort_values("Std").reset_index(drop=True)
    consistency_df["Rank"] = range(1, len(consistency_df) + 1)
    consistency_df = consistency_df[["Rank", "Distractor Type", "Std"]]

    analysis_output.append("\n## Distractor Consistency Ranking (by std deviation, lower is more consistent):")
    analysis_output.append(consistency_df.to_markdown(index=False))

    # Coverage analysis
    coverage_data = []
    total_items = len(results)
    for distractor_type in distractor_types:
        count = len([r for r in results if r[distractor_type] is not None])
        coverage = count / total_items * 100
        coverage_data.append(
            {
                "Distractor Type": distractor_type.replace("_similarity", ""),
                "Available": count,
                "Total": total_items,
                "Coverage (%)": round(coverage, 1),
            }
        )

    coverage_df = pd.DataFrame(coverage_data)
    analysis_output.append("\n## Coverage Analysis:")
    analysis_output.append(coverage_df.to_markdown(index=False))

    # Print to console
    for line in analysis_output:
        print(line)

    # Save to file if specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            for line in analysis_output:
                f.write(line + "\n")
        print(f"\nAnalysis results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute semantic similarities between distractors and correct outputs"
    )
    parser.add_argument("input_file", help="Input JSONL file path")
    parser.add_argument("--output", "-o", help="Output file path (supports .json, .jsonl, .csv)")
    parser.add_argument(
        "--model", "-m", default="all-MiniLM-L6-v2", help="SentenceTransformer model name (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument("--analyze", "-a", action="store_true", help="Show similarity analysis statistics")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="Batch size for inference (default: 32)")

    args = parser.parse_args()

    # Load data
    print(f"Loading data from: {args.input_file}")
    data = load_jsonl(args.input_file)
    print(f"Loaded {len(data)} items")

    # Compute similarities
    results = compute_similarities(data, model_name=args.model, output_file=args.output, batch_size=args.batch_size)

    # Show analysis if requested
    if args.analyze:
        analyze_file = args.output.replace(".json", "_analysis.json") if args.output else "similarity_analysis.txt"
        analyze_similarities(results, analyze_file)

    print("Done!")


if __name__ == "__main__":
    main()
