python build_multi_choice_data.py /project/hdtaccuracy/Personality-Alignment/choice_ver/raw_choice_data_v6.jsonl \
    --model /project/hdtaccuracy/models/base/Qwen3-8B-Embedding \
    --output /project/hdtaccuracy/Personality-Alignment/choice_ver/multi_choice_similarity_v7.json \
    --batch-size 32 \
    --analyze