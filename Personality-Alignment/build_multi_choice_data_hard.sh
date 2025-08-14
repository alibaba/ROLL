python build_multi_choice_data.py /project/hdtaccuracy/Personality-Alignment/choice_ver/raw_choice_data_v7_hard.jsonl \
    --model /project/hdtaccuracy/models/base/Qwen3-8B-Embedding \
    --output /project/hdtaccuracy/Personality-Alignment/choice_ver/multi_choice_similarity_v7_hard.json \
    --batch-size 32 \
    --analyze