#!/bin/bash
#SBATCH --job-name=nnunet506_pred
#SBATCH --account=MST115208
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --exclude=25a-hgpn001
#SBATCH --output=logs/pred506_%j.out
#SBATCH --error=logs/pred506_%j.err
#
# 5 折都訓練完後:
#   1. 5 個模型集成,預測 504 測試集裡的 11 位醫院病人(沒有稀疏標註,只拿來看圖)
#   2. 收集 5 折的驗證預測(34 位,每位都是「沒看過的人」),和 1 一起打包帶回本機
#   sbatch scripts/nano4/32_predict_506.sh
module load miniconda3
source activate coseg
export PATH="$CONDA_PREFIX/bin:$PATH"
set -euo pipefail

ROOT=/home/u9444861/-CoSeg-project
TRAINER=${TRAINER:-nnUNetTrainer_Sparse}
RES=$nnUNet_results/Dataset506_CanalSparse/${TRAINER}__nnUNetPlans__3d_fullres
TS=$ROOT/data/nnUNet_raw/Dataset504_MixedCanalClean/imagesTs
IN=$ROOT/data/predictions/Dataset506_input_hosp_ts
OUT=$ROOT/data/predictions/Dataset506_CanalSparse_ens5
VAL=$ROOT/data/predictions/Dataset506_CanalSparse_cv

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'GPU 不可用')"
for f in 0 1 2 3 4; do
  [ -f "$RES/fold_$f/checkpoint_final.pth" ] || { echo "fold $f 還沒訓練完"; exit 1; }
done

echo "=== 1. 測試集醫院病人(集成 5 折)==="
mkdir -p "$IN" "$OUT"
for f in "$TS"/HOSP_*; do ln -sf "$f" "$IN/"; done
echo "  輸入 $(ls "$IN" | wc -l) 個(應該是 11)"
nnUNetv2_predict -i "$IN" -o "$OUT" -d 506 -c 3d_fullres -tr "$TRAINER" -f 0 1 2 3 4
echo "  預測 $(ls "$OUT"/*.nii.gz | wc -l) 個"

echo
echo "=== 2. 收集 5 折驗證預測 ==="
mkdir -p "$VAL"
for f in 0 1 2 3 4; do
  cp "$RES/fold_$f/validation/"*.nii.gz "$VAL/"
  cp "$RES/fold_$f/validation/summary.json" "$VAL/summary_fold$f.json"
done
echo "  共 $(ls "$VAL"/*.nii.gz | wc -l) 個(應該是 34)"
python - <<PY
import json, glob, os
ds = []
for p in sorted(glob.glob("$RES/fold_*/validation/summary.json")):
    d = json.load(open(p)); ds += [c["metrics"]["1"]["Dice"] for c in d["metric_per_case"]]
print(f"  5 折驗證 Dice(只算看得到的地方):n={len(ds)} 平均={sum(ds)/len(ds):.4f} 最低={min(ds):.4f} 最高={max(ds):.4f}")
PY

echo
echo "=== 3. 打包帶回本機 ==="
cd "$ROOT/data/predictions"
tar cf "$ROOT/download506.tar" Dataset506_CanalSparse_ens5 Dataset506_CanalSparse_cv
ls -lh "$ROOT/download506.tar"
echo "=== 完成 $(date) ===  把 $ROOT/download506.tar 下載回本機的 coseg-latest/"
