#!/bin/bash
#SBATCH --job-name=nnunet504_train
#SBATCH --account=MST115208
#SBATCH --partition=dev         # 只用 1 張卡;8gpus 會把閒置的 7 張也算錢
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12      # dev 上限 = GPU 張數 x 12
#SBATCH --mem=64G
#SBATCH --time=03:55:00         # 500 epochs x ~17.5s ≈ 2.5h
#SBATCH --exclude=25a-hgpn001   # 這台的 GPU 綁定壞掉
#SBATCH --output=logs/train504_%j.out
#SBATCH --error=logs/train504_%j.err
#
# Dataset504(只用乾淨醫院資料)fold 0,條件跟 503 的 job 495049 相同,方便直接比較。
#   sbatch scripts/nano4/12_train_504.sh
#   被時限砍掉時續跑:CONTINUE=1 sbatch scripts/nano4/12_train_504.sh
module load miniconda3
source activate coseg
set -euo pipefail

DID=504
DNAME=Dataset504_MixedCanalClean
CONFIG=3d_fullres
FOLD=${FOLD:-0}
TRAINER=${TRAINER:-nnUNetTrainer_500epochs}

echo "=== 開始 $(date) on $(hostname) ==="
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
nh = sum(1 for c in f["val"] if c.startswith("HOSP_"))
print(f"  fold $FOLD: train={len(f['train'])} val={len(f['val'])} 醫院 val={nh}")
PY

if [ "${CONTINUE:-0}" = "1" ]; then
  nnUNetv2_train "$DID" "$CONFIG" "$FOLD" -tr "$TRAINER" --c
else
  nnUNetv2_train "$DID" "$CONFIG" "$FOLD" -tr "$TRAINER"
fi

RES=$nnUNet_results/$DNAME/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}
echo
echo "=== 驗證分數 ==="
python - <<PY
import json, os
p = "$RES/validation/summary.json"
if not os.path.exists(p):
    raise SystemExit(f"  找不到 {p}")
d = json.load(open(p))
print("  整體 val Dice:", round(d["foreground_mean"]["Dice"], 4))
for tag, pref in [("醫院", "HOSP_"), ("公開", "MCSTU_")]:
    rows = [(os.path.basename(c["prediction_file"]), c["metrics"]["1"]["Dice"])
            for c in d["metric_per_case"] if os.path.basename(c["prediction_file"]).startswith(pref)]
    if rows:
        ds = [r[1] for r in rows]
        print(f"  {tag}: n={len(ds)} 平均={sum(ds)/len(ds):.4f} 最低={min(ds):.4f} 最高={max(ds):.4f}")
    if pref == "HOSP_":
        for n, v in sorted(rows):
            print(f"      {n:28s} {v:.4f}")
PY
echo "=== 完成 $(date) ===  下一步:sbatch scripts/nano4/13_predict_504.sh"
