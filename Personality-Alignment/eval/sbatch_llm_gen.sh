#!/bin/bash
#SBATCH --job-name=bconvo
#SBATCH --nodes=1
#SBATCH --gpus=2            # 使用2张GPU
#SBATCH --ntasks=1          # 只起 1 个进程
#SBATCH --cpus-per-task=16  # 视节点 CPU 数而定
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --account=msccsit2024
#SBATCH --output=eval_data.out

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=$(( 20000 + RANDOM % 10000 ))

# 配置推理参数以更好利用多卡
export BATCH_SIZE=32  # 降低批处理大小避免OOM

srun --export=ALL \
    --container-image=/home/szhangfa/containers/roll.img \
    --container-mounts=/home/szhangfa:/home/szhangfa \
    --container-workdir=/home/szhangfa/ROLL/Personality-Alignment/eval \
    --container-writable \
    bash -c "
cd /home/szhangfa/ROLL/Personality-Alignment/eval
bash run_llm_gen.sh
"
