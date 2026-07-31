#!/bin/bash
#SBATCH --job-name=coseg_test
#SBATCH --account=mst115208
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH --output=test_gpu_%j.out
#SBATCH --error=test_gpu_%j.err

module load miniconda3/26.1.1

# ===== 這是關鍵的魔法指令 =====
# 它會強制讓接下來所有的 python 指令，都自動使用 coseg 環境裡的 Python！
export PATH="/home/u9444861/.conda/envs/coseg/bin:$PATH"

echo "===== CONDA ENV ====="
conda env list

echo "===== PYTHON PATH ====="
which python

echo "===== GPU TEST ====="
python -u run_pipeline.py