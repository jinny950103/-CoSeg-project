#!/bin/bash
#SBATCH --job-name=nnunet506_train
#SBATCH --account=MST115208
#SBATCH --partition=dev         # 只用 1 張卡
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12      # dev 上限 = GPU 張數 x 12
#SBATCH --mem=64G
#SBATCH --time=03:55:00         # 500 epochs x ~17.5s ≈ 2.5h
#SBATCH --exclude=25a-hgpn001   # 這台的 GPU 綁定壞掉
#SBATCH --output=logs/train506_%j.out
#SBATCH --error=logs/train506_%j.err
#
# Dataset506(稀疏標註,ignore 不算分)訓練一折。5 折都要跑,每位病人才會有一次
# 「沒看過這個人」的預測,拿來檢查模型在看不清楚的地方畫了什麼。
#   for f in 0 1 2 3 4; do FOLD=$f sbatch scripts/nano4/31_train_506.sh; done
#   被時限砍掉時續跑:FOLD=0 CONTINUE=1 sbatch scripts/nano4/31_train_506.sh
module load miniconda3
source activate coseg
export PATH="$CONDA_PREFIX/bin:$PATH"
set -euo pipefail

DID=506
DNAME=Dataset506_CanalSparse
CONFIG=3d_fullres
FOLD=${FOLD:-0}
TRAINER=${TRAINER:-nnUNetTrainer_Sparse}   # 預設 lr 1e-2 的 500epochs 在 fold 0 第 59 輪崩成全背景

echo "=== 開始 $(date) on $(hostname)  fold $FOLD ==="
ROOT=/home/u9444861/-CoSeg-project
PKG=$(python -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))")
cp "$ROOT/scripts/nnunet_trainers/nnUNetTrainer_Sparse.py" "$PKG/training/nnUNetTrainer/variants/"
python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("GPU 不可用")
print("  GPU:", torch.cuda.get_device_name(0))
PY

python - <<PY
import json, os
s = json.load(open(os.path.join(os.environ["nnUNet_preprocessed"], "$DNAME", "splits_final.json")))
f = s[$FOLD]
print(f"  fold $FOLD: train={len(f['train'])} val={len(f['val'])}")
assert len(f["train"]) + len(f["val"]) == 34, "splits 被 nnU-Net 自己的版本蓋掉了"
PY

if [ "${CONTINUE:-0}" = "1" ]; then
  nnUNetv2_train "$DID" "$CONFIG" "$FOLD" -tr "$TRAINER" --c
else
  nnUNetv2_train "$DID" "$CONFIG" "$FOLD" -tr "$TRAINER"
fi

RES=$nnUNet_results/$DNAME/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}
echo
echo "=== 驗證分數(只算看得到的地方,ignore 不算)==="
python - <<PY
import json, os
p = "$RES/validation/summary.json"
if not os.path.exists(p):
    raise SystemExit(f"  找不到 {p}")
d = json.load(open(p))
print("  val Dice:", round(d["foreground_mean"]["Dice"], 4))
for c in sorted(d["metric_per_case"], key=lambda c: c["prediction_file"]):
    print(f"      {os.path.basename(c['prediction_file'])[-11:-7]}  {c['metrics']['1']['Dice']:.4f}")
PY
echo "=== 完成 $(date) ===  5 折都完成後:sbatch scripts/nano4/32_predict_506.sh"
