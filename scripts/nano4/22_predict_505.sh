#!/bin/bash
#SBATCH --job-name=nnunet505_pred
#SBATCH --account=MST115208
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --exclude=25a-hgpn001
#SBATCH --output=logs/pred505_%j.out
#SBATCH --error=logs/pred505_%j.err
#
# 用微調後的模型預測 504 的測試集(11 位醫院 + 18 位公開),並直接算分數。
# 公開資料的分數會下降是預期中的:這個模型已經改學醫院的畫法。
#   sbatch scripts/nano4/22_predict_505.sh
module load miniconda3
source activate coseg
export PATH="$CONDA_PREFIX/bin:$PATH"   # sbatch 裡的 python 不一定是 coseg 的
set -euo pipefail

ROOT=/home/u9444861/-CoSeg-project
FOLD=${FOLD:-0}
IN=$ROOT/data/nnUNet_raw/Dataset504_MixedCanalClean/imagesTs
GT=$ROOT/data/nnUNet_raw/Dataset504_MixedCanalClean/labelsTs_eval
OUT=$ROOT/data/predictions/Dataset505_HospFT_fold${FOLD}

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'GPU 不可用')"
mkdir -p "$OUT"
nnUNetv2_predict -i "$IN" -o "$OUT" -d 505 -c 3d_fullres -tr nnUNetTrainer_HospFT -f "$FOLD"
echo "預測檔數: $(ls "$OUT"/*.nii.gz | wc -l)  (應該是 29)"

echo
python "$ROOT/scripts/evaluate_predictions.py" --pred "$OUT" --gt "$GT"
echo
python "$ROOT/scripts/centerline_analysis.py" --pred "$OUT" --gt "$GT" \
  --images "$ROOT/data/nnUNet_raw/Dataset504_MixedCanalClean/imagesTs"
