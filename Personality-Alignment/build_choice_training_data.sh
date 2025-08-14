python build_choice_training_data.py \
    --questions /project/hdtaccuracy/Personality-Alignment/choice_ver/four_choices_question_v7/all_questions.json \
    --prompts /project/hdtaccuracy/Personality-Alignment/choice_ver/raw_choice_data_v6.jsonl \
    --out /project/hdtaccuracy/Personality-Alignment/choice_ver/four_choices_question_v7/v7.json \
    --split_mode user \
    --test_ratio 0.2 \
    --skip_missing_prompt