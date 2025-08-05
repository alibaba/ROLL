#!/bin/bash
#SBATCH --job-name=hdtppo
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=dgx-34
#SBATCH --time=24:00:00
#SBATCH --account=hdtaccuracy
#SBATCH --partition=normal
##SBATCH --container-writable
##SBATCH --container-image /home/szhangfa/containers/roll.img
##SBATCH --container-save /home/szhangfa/containers/roll.img

# cd /home/szhangfa/LLaMA-Factory
export WANDB_API_KEY=dce12064d30900b2cc538f73e82997de5aafbb96

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=$(( 20000 + RANDOM % 10000 ))

srun --export=ALL \
    --container-image=/home/szhangfa/containers/roll.img \
    --container-mounts=/home/szhangfa:/home/szhangfa \
    --container-workdir=/home/szhangfa/ROLL \
    --container-writable \
    bash -c "
cd /home/szhangfa/ROLL
bash examples/qwen3-8B-rlvr_sglang/run_rlvr_pipeline.sh
"