python answer_choice_four_type.py \
    --question_file /project/hdtaccuracy/Personality-Alignment/choice_ver/four_choices_question_v7_hard/all_questions.json \
    --prompt_file /project/hdtaccuracy/Personality-Alignment/choice_ver/raw_choice_data_v7_hard.jsonl \
    --model_path /project/hdtaccuracy/models/base/Qwen3-8B \
    --batch_size 48 \
    --output_file /project/hdtaccuracy/Personality-Alignment/choice_ver/four_results/results_v7_hard.json