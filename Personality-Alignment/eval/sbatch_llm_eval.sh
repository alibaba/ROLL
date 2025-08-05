#!/bin/bash
#SBATCH --job-name=eval_mul
#SBATCH --nodes=1
#SBATCH --gpus=4            # 请求8张GPU
#SBATCH --ntasks=1          # 只起 1 个进程
#SBATCH --cpus-per-task=32  # 增加CPU数量以支持多卡
#SBATCH --time=4:00:00
#SBATCH --partition=normal
#SBATCH --account=hdtaccuracy
#SBATCH --output=eval_llm_judge.out

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=$(( 20000 + RANDOM % 10000 ))

srun --export=ALL \
    --container-image=/home/szhangfa/containers/roll.img \
    --container-mounts=/home/szhangfa:/home/szhangfa \
    --container-workdir=/home/szhangfa/ROLL/Personality-Alignment/eval \
    --container-writable \
    bash -c "
cd /home/szhangfa/ROLL/Personality-Alignment/eval
python llm_eval.py \
    --prompt-input /project/hdtaccuracy/Personality-Alignment/dialogue_dataset_all_v5_summarized.jsonl \
    --input /home/szhangfa/ROLL/Personality-Alignment/eval/qwen_base_results.jsonl \
    --output /project/hdtaccuracy/results/base/Qwen3-8B-Convo/Qwen3-8B_llm_eval.jsonl \
    --evaluator-type qwen-multi-gpu \
    --num-gpus 4\
    --qwen-model-path /project/hdtaccuracy/models/base/Qwen3-8B \
    --inference-batch-size 8 \
"



