#!/bin/bash
#SBATCH --job-name=bconvo
#SBATCH --nodes=1
#SBATCH --gpus=1            # 一行就够
#SBATCH --ntasks=1          # 只起 1 个进程
#SBATCH --cpus-per-task=16  # 视节点 CPU 数而定
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --account=msccsit2024
#SBATCH --output=build_convo_data.out

# ------------ Pyxis/Enroot ---------------
#SBATCH --container-image=/home/szhangfa/containers/roll.img
#SBATCH --container-mounts=/home/szhangfa:/home/szhangfa,/project:/project     
#SBATCH --container-workdir=/home/szhangfa/ROLL/Personality-Alignment
#SBATCH --container-writable
# ----------------------------------------
# pip install accelerate

python build_convo_data.py
