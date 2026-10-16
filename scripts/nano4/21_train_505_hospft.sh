#!/bin/bash
#SBATCH --job-name=nnunet505_ft
#SBATCH --account=MST115208
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:30:00         # 100 epochs x ~17.5s ≈ 30 分鐘
#SBATCH --exclude=25a-hgpn001
#SBATCH --output=logs/train505_%j.out
#SBATCH --error=logs/train505_%j.err
#
# 從 504 的模型出發,只用醫院資料微調。
#   sbatch scripts/nano4/21_train_505_hospft.sh
# 集成用(504 只有 fold 0,所以 1-4 折都從 504 fold 0 出發):
#   for f in 1 2 3 4; do sbatch --export=ALL,FOLD=$f,PRE_FOLD=0 scripts/nano4/21_train_505_hospft.sh; done
module load miniconda3
source activate coseg
export PATH="$CONDA_PREFIX/bin:$PATH"   # sbatch 裡的 python 不一定是 coseg 的
set -euo pipefail

FOLD=${FOLD:-0}
PRE_FOLD=${PRE_FOLD:-$FOLD}
PRE=$nnUNet_results/Dataset504_MixedCanalClean/nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres/fold_${PRE_FOLD}/checkpoint_final.pth
[ -f "$PRE" ] || { echo "找不到 504 的模型: $PRE"; exit 1; }

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'GPU 不可用')"
echo "=== 開始 $(date) on $(hostname) ==="
echo "起點: $PRE"

nnUNetv2_train 505 3d_fullres "$FOLD" -tr nnUNetTrainer_HospFT -pretrained_weights "$PRE"

RES=$nnUNet_results/Dataset505_HospFT/nnUNetTrainer_HospFT__nnUNetPlans__3d_fullres/fold_${FOLD}
echo
echo "=== fold $FOLD 醫院驗證分數 ==="
[ "$FOLD" != 0 ] && echo "  (注意:起點 504 fold 0 訓練時看過這些病人,這個驗證分數會偏高;看測試集才準)"
python - <<PY
import json, os
d = json.load(open("$RES/validation/summary.json"))
rows = sorted((os.path.basename(c["prediction_file"]), c["metrics"]["1"]["Dice"]) for c in d["metric_per_case"])
for n, v in rows:
    print(f"    {n:28s} {v:.4f}")
print(f"  微調後 平均 = {sum(v for _, v in rows)/len(rows):.4f}   (fold 0 的 504 微調前 = 0.6512)")
PY
echo "=== 完成 $(date) ==="
echo "下一步(中心線分析):"
echo "  python scripts/centerline_analysis.py --pred $RES/validation --gt \$nnUNet_raw/Dataset505_HospFT/labelsTr --images \$nnUNet_raw/Dataset505_HospFT/imagesTr"
echo "測試集:sbatch scripts/nano4/22_predict_505.sh"
