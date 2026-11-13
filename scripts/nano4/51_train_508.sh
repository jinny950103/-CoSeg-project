#!/bin/bash
#SBATCH --job-name=nnunet508_train
#SBATCH --account=MST115208
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:55:00         # 500 epochs x ~17.5s ≈ 2.5h(每輪 250 iteration,和病例數無關)
#SBATCH --exclude=25a-hgpn001
#SBATCH --output=logs/train508_%j.out
#SBATCH --error=logs/train508_%j.err
#
# Dataset508(MCSTU 218 例 edge 標註,ignore 不算分)訓練一折。
#   for f in 0 1 2 3 4; do FOLD=$f sbatch scripts/nano4/51_train_508.sh; done
#   被時限砍掉時續跑:FOLD=0 CONTINUE=1 sbatch scripts/nano4/51_train_508.sh
# trainer 用 nnUNetTrainer_Sparse(lr 1e-3、500 輪、從零):506 用預設 lr 1e-2 時 3 折崩成全背景。
module load miniconda3
source activate coseg
export PATH="$CONDA_PREFIX/bin:$PATH"
set -euo pipefail

DID=508
DNAME=Dataset508_MCSTUEdge
CONFIG=3d_fullres
FOLD=${FOLD:-0}
TRAINER=${TRAINER:-nnUNetTrainer_Sparse}

echo "=== 開始 $(date) on $(hostname)  fold $FOLD ==="
ROOT=/home/u9444861/-CoSeg-project
PKG=$(python -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))")
cp "$ROOT/scripts/nnunet_trainers/nnUNetTrainer_Sparse.py" "$PKG/training/nnUNetTrainer/variants/"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'GPU 不可用'); " && python -c "import torch; print('  GPU:', torch.cuda.get_device_name(0))"

python - <<PY
import json, os
s = json.load(open(os.path.join(os.environ["nnUNet_preprocessed"], "$DNAME", "splits_final.json")))
f = s[$FOLD]
print(f"  fold $FOLD: train={len(f['train'])} val={len(f['val'])}")
assert len(f["train"]) + len(f["val"]) == 218, "splits 被 nnU-Net 自己的版本蓋掉了"
assert not any(c.startswith("HOSP_") for c in f["train"] + f["val"]), "不應該有醫院病人"
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
ds = sorted((c["metrics"]["1"]["Dice"], os.path.basename(c["prediction_file"])) for c in d["metric_per_case"])
print("  val Dice 平均:", round(d["foreground_mean"]["Dice"], 4), f"(n={len(ds)})")
print("  最低 5 例:", ", ".join(f"{n[:-7]} {v:.3f}" for v, n in ds[:5]))
PY
echo "=== 完成 $(date) ===  5 折都完成後:sbatch scripts/nano4/52_predict_508.sh"
