#!/bin/bash
#SBATCH --job-name=nnunet504_pred
#SBATCH --account=MST115208
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --exclude=25a-hgpn001
#SBATCH --output=logs/pred504_%j.out
#SBATCH --error=logs/pred504_%j.err
#
# 測試集推論:11 位醫院 + 18 位公開 = 29 位,全程沒參與訓練。
#   sbatch scripts/nano4/13_predict_504.sh
module load miniconda3
source activate coseg
set -euo pipefail

ROOT=/home/u9444861/-CoSeg-project
DNAME=Dataset504_MixedCanalClean
FOLD=${FOLD:-0}
TRAINER=${TRAINER:-nnUNetTrainer_500epochs}
IN=$ROOT/data/nnUNet_raw/$DNAME/imagesTs
OUT=$ROOT/data/predictions/${DNAME}_${TRAINER}_fold${FOLD}

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'GPU 不可用')"
mkdir -p "$OUT"
nnUNetv2_predict -i "$IN" -o "$OUT" -d 504 -c 3d_fullres -tr "$TRAINER" -f "$FOLD"

echo "預測檔數: $(ls "$OUT"/*.nii.gz | wc -l)  (應該是 29)"
echo "下一步:python scripts/evaluate_predictions.py --pred $OUT --gt $ROOT/data/nnUNet_raw/$DNAME/labelsTs_eval"
