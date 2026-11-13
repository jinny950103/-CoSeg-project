#!/bin/bash
#SBATCH --job-name=nnunet508_10f
#SBATCH --account=MST115208
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:55:00         # 和 51 一樣 500 輪,約 2.5h
#SBATCH --exclude=25a-hgpn001
#SBATCH --output=logs/train508_10f_%j.out
#SBATCH --error=logs/train508_10f_%j.err
#
# Dataset508 的 10 折版(教授 2026-11-12 要求)。前處理沿用 508,不用重做。
# 10 折切法:scripts/make_508_10fold.py,每個 5 折驗證組再切兩半(第 k 折在 5 折的第 k//2 折裡)。
# trainer 用 nnUNetTrainer_Sparse10:讀 splits_10fold.json,結果放在自己的資料夾,不會蓋掉 5 折。
#
# 先把本機 outputs/upload508/Dataset508_MCSTUEdge/splits_10fold.json 上傳到 $ROOT/,再:
#   for f in 0 1 2 3 4 5 6 7 8 9; do FOLD=$f sbatch scripts/nano4/53_train_508_10fold.sh; done
#   續跑:FOLD=3 CONTINUE=1 sbatch scripts/nano4/53_train_508_10fold.sh
module load miniconda3
source activate coseg
export PATH="$CONDA_PREFIX/bin:$PATH"
set -euo pipefail

DID=508
DNAME=Dataset508_MCSTUEdge
CONFIG=3d_fullres
FOLD=${FOLD:-0}
TRAINER=nnUNetTrainer_Sparse10
ROOT=/home/u9444861/-CoSeg-project
PREP=$nnUNet_preprocessed/$DNAME

echo "=== 開始 $(date) on $(hostname)  10-fold 第 $FOLD 折 ==="
PKG=$(python -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))")
cp "$ROOT/scripts/nnunet_trainers/nnUNetTrainer_Sparse.py" "$PKG/training/nnUNetTrainer/variants/"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'GPU 不可用')"

[ -f "$PREP/splits_10fold.json" ] || cp "$ROOT/splits_10fold.json" "$PREP/splits_10fold.json"
python - <<PY
import json
s = json.load(open("$PREP/splits_10fold.json"))
assert len(s) == 10, f"應該 10 折,現在 {len(s)}"
assert sorted(c for f in s for c in f["val"]) == sorted(set(s[0]["train"]) | set(s[0]["val"])), "每例要剛好在一折驗證"
f = s[$FOLD]
assert len(f["train"]) + len(f["val"]) == 218 and not any(c.startswith("HOSP_") for c in f["train"] + f["val"])
print(f"  fold $FOLD: train={len(f['train'])} val={len(f['val'])}")
PY

if [ "${CONTINUE:-0}" = "1" ]; then
  nnUNetv2_train "$DID" "$CONFIG" "$FOLD" -tr "$TRAINER" --c
else
  nnUNetv2_train "$DID" "$CONFIG" "$FOLD" -tr "$TRAINER"
fi

RES=$nnUNet_results/$DNAME/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}
python - <<PY
import json, os
d = json.load(open("$RES/validation/summary.json"))
ds = sorted((c["metrics"]["1"]["Dice"], os.path.basename(c["prediction_file"])) for c in d["metric_per_case"])
print("  val Dice 平均:", round(d["foreground_mean"]["Dice"], 4), f"(n={len(ds)},ignore 不算)")
print("  最低 3 例:", ", ".join(f"{n[:-7]} {v:.3f}" for v, n in ds[:3]))
PY
echo "=== 完成 $(date) ===  10 折都完成後:sbatch scripts/nano4/54_predict_508_10fold.sh"
