#!/bin/bash
#SBATCH --job-name=eval_mul
#SBATCH --nodes=1
#SBATCH --gpus=4            # 请求8张GPU
#SBATCH --ntasks=1          # 只起 1 个进程
#SBATCH --cpus-per-task=32  # 增加CPU数量以支持多卡
#SBATCH --time=4:00:00
#SBATCH --partition=normal
#SBATCH --account=hdtaccuracy
#SBATCH --output=eval_multicard_data.out

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=$(( 20000 + RANDOM % 10000 ))

# 多卡推理配置
export NUM_GPUS=4
export PER_GPU_BATCH_SIZE=32  # 每张卡的批处理大小，可以根据显存调整
export USE_ASYNC=true         # 使用异步模式提高吞吐量

srun --export=ALL \
    --container-image=/home/szhangfa/containers/roll.img \
    --container-mounts=/home/szhangfa:/home/szhangfa \
    --container-workdir=/home/szhangfa/ROLL/Personality-Alignment/eval \
    --container-writable \
    bash -c "
cd /home/szhangfa/ROLL/Personality-Alignment/eval
bash run_llm_gen_multicard.sh
"
