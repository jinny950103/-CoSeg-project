#!/bin/bash
#SBATCH --job-name=nnunet503_pred
#SBATCH --account=MST115208
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12      # dev 的上限是 GPU 張數 x 12
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/pred_%j.out
#SBATCH --error=logs/pred_%j.err
#
# 對測試集推論:6 位醫院 + 18 位公開,共 24 位。
#   sbatch scripts/nano4/04_predict.sh
#
# 這 24 位從頭到尾沒參與訓練,也不在任何 fold 的驗證集裡,所以是真正的考試。
module load miniconda3
source activate coseg
set -euo pipefail

ROOT=/home/u9444861/-CoSeg-project
DNAME=Dataset503_MixedCanal
DID=503
CONFIG=3d_fullres
FOLD=${FOLD:-0}
TRAINER=${TRAINER:-nnUNetTrainer_500epochs}

IN=$ROOT/data/nnUNet_raw/$DNAME/imagesTs
OUT=$ROOT/data/predictions/${DNAME}_${TRAINER}_fold${FOLD}

echo "=== 開始 $(date) ==="
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'GPU 不可用')"

echo "輸入: $IN  ($(ls "$IN" | wc -l) 個檔案)"
echo "輸出: $OUT"
mkdir -p "$OUT"

echo
echo "=== 推論 ==="
# -f 只用訓練好的那一折。之後若跑完 5 折,改成 -f 0 1 2 3 4 做集成,通常再多 1~2%。
nnUNetv2_predict \
  -i "$IN" \
  -o "$OUT" \
  -d "$DID" \
  -c "$CONFIG" \
  -tr "$TRAINER" \
  -f "$FOLD"

echo
echo "=== 產出 ==="
echo "預測檔數: $(ls "$OUT"/*.nii.gz 2>/dev/null | wc -l)  (應該是 24)"
ls "$OUT" | head -5 | sed 's/^/  /'

echo
echo "=== 完成 $(date) ==="
echo "下一步:python scripts/evaluate_predictions.py --pred $OUT"
