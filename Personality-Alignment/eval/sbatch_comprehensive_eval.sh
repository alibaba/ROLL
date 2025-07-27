# [-h] --ground-truth GROUND_TRUTH --model-results MODEL_RESULTS --output-dir OUTPUT_DIR --model-name MODEL_NAME [--skip-automated]
# [--skip-llm] [--evaluator-type {gpt,qwen}] [--api-key API_KEY] [--parallel] [--qwen-model-path QWEN_MODEL_PATH]

python comprehensive_eval.py \
    --ground-truth dialogue_dataset_all_v5_summarized.jsonl \
    --model-results qwen_multicard_results.jsonl \
    --output-dir /project/hdtaccuracy/results/base/Qwen3-8B-Convo \
    --model-name Qwen3-8B \
    --evaluator-type qwen \
    --qwen-model-path /project/hdtaccuracy/models/base/Qwen3-8B
