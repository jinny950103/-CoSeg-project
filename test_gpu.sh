#!/bin/bash
#SBATCH --job-name=coseg_test
#SBATCH --account=mst115208
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=test_gpu_%j.out
#SBATCH --error=test_gpu_%j.err

module load miniconda3/26.1.1

echo "===== CONDA ENV ====="
conda env list

echo "===== PYTHON PATH ====="
conda run -n coseg which python

echo "===== GPU TEST ====="
conda run -n coseg python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
