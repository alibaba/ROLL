export openai_base_url=https://jeniya.cn/v1
export openai_api_key=sk-Nz81G8iMtQjlj605yti5KLI4CKcNTxySJj6fmwD5xFed0nLP

python answer_choice_four_type.py \
    --question_file /project/hdtaccuracy/Personality-Alignment/choice_ver/four_choice_questions.json \
    --prompt_file /project/hdtaccuracy/Personality-Alignment/choice_ver/raw_choice_data_v6.jsonl \
    --sample_size 16 \
    --backend gpt \
    --gpt_model gpt-4o-mini \
    --gpt_workers 8 \
    --output_file /project/hdtaccuracy/Personality-Alignment/choice_ver/four_results/gpt-4o-mini_results.json