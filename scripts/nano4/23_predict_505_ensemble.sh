#!/bin/bash
#SBATCH --job-name=nnunet505_ens
#SBATCH --account=MST115208
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --exclude=25a-hgpn001
#SBATCH --output=logs/ens505_%j.out
#SBATCH --error=logs/ens505_%j.err
#
# A1:5 個醫院微調模型(505 fold 0-4)集成,預測 504 測試集並算分數。
# 要先跑完 21_train_505_hospft.sh 的 fold 1-4。
#   sbatch scripts/nano4/23_predict_505_ensemble.sh
module load miniconda3
source activate coseg
export PATH="$CONDA_PREFIX/bin:$PATH"   # sbatch 裡的 python 不一定是 coseg 的
set -euo pipefail

ROOT=/home/u9444861/-CoSeg-project
IN=$ROOT/data/nnUNet_raw/Dataset504_MixedCanalClean/imagesTs
GT=$ROOT/data/nnUNet_raw/Dataset504_MixedCanalClean/labelsTs_eval
OUT=$ROOT/data/predictions/Dataset505_HospFT_ens5

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'GPU 不可用')"
for f in 0 1 2 3 4; do
  [ -f "$nnUNet_results/Dataset505_HospFT/nnUNetTrainer_HospFT__nnUNetPlans__3d_fullres/fold_$f/checkpoint_final.pth" ]     || { echo "fold $f 還沒訓練完"; exit 1; }
done
mkdir -p "$OUT"
nnUNetv2_predict -i "$IN" -o "$OUT" -d 505 -c 3d_fullres -tr nnUNetTrainer_HospFT -f 0 1 2 3 4
echo "預測檔數: $(ls "$OUT"/*.nii.gz | wc -l)  (應該是 29)"

echo
python "$ROOT/scripts/evaluate_predictions.py" --pred "$OUT" --gt "$GT"
echo
python "$ROOT/scripts/centerline_analysis.py" --pred "$OUT" --gt "$GT" \
  --images "$ROOT/data/nnUNet_raw/Dataset504_MixedCanalClean/imagesTs"
echo
python "$ROOT/scripts/three_lines.py" --pred "$OUT" --gt "$GT"   --images "$ROOT/data/nnUNet_raw/Dataset504_MixedCanalClean/imagesTs"
