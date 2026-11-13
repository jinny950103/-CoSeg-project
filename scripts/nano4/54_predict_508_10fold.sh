#!/bin/bash
#SBATCH --job-name=nnunet508_10fp
#SBATCH --account=MST115208
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --exclude=25a-hgpn001
#SBATCH --output=logs/pred508_10f_%j.out
#SBATCH --error=logs/pred508_10f_%j.err
#
# 10 折都訓練完後:
#   1. 10 個模型集成,預測 504 的測試集 29 例(MCSTU 獨立測試 18 + 醫院 11;醫院只拿來看,沒訓練過)
#   2. 收集 10 折驗證預測(218 例)
#   3. 打包 download508_10fold.tar 帶回本機,在本機和 labelsTs_eval 算分
#   sbatch scripts/nano4/52_predict_508.sh
module load miniconda3
source activate coseg
export PATH="$CONDA_PREFIX/bin:$PATH"
set -euo pipefail

ROOT=/home/u9444861/-CoSeg-project
TRAINER=nnUNetTrainer_Sparse10
RES=$nnUNet_results/Dataset508_MCSTUEdge/${TRAINER}__nnUNetPlans__3d_fullres
TS=$ROOT/data/nnUNet_raw/Dataset504_MixedCanalClean/imagesTs
OUT=$ROOT/data/predictions/Dataset508_MCSTUEdge_ens10
VAL=$ROOT/data/predictions/Dataset508_MCSTUEdge_cv10

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'GPU 不可用')"
for f in 0 1 2 3 4 5 6 7 8 9; do
  [ -f "$RES/fold_$f/checkpoint_final.pth" ] || { echo "fold $f 還沒訓練完"; exit 1; }
done

echo "=== 1. 測試集 29 例(集成 10 折)==="
mkdir -p "$OUT"
echo "  輸入 $(ls "$TS" | wc -l) 個(應該是 29)"
nnUNetv2_predict -i "$TS" -o "$OUT" -d 508 -c 3d_fullres -tr "$TRAINER" -f 0 1 2 3 4 5 6 7 8 9
echo "  預測 $(ls "$OUT"/*.nii.gz | wc -l) 個"

echo
echo "=== 2. 收集 10 折驗證預測 ==="
mkdir -p "$VAL"
for f in 0 1 2 3 4 5 6 7 8 9; do
  cp "$RES/fold_$f/validation/"*.nii.gz "$VAL/"
  cp "$RES/fold_$f/validation/summary.json" "$VAL/summary_fold$f.json"
done
echo "  共 $(ls "$VAL"/*.nii.gz | wc -l) 個(應該是 218)"
python - <<PY
import json, glob
ds = []
for p in sorted(glob.glob("$RES/fold_*/validation/summary.json")):
    d = json.load(open(p)); ds += [c["metrics"]["1"]["Dice"] for c in d["metric_per_case"]]
print(f"  10 折驗證 Dice(只算看得到的地方):n={len(ds)} 平均={sum(ds)/len(ds):.4f} 最低={min(ds):.4f} 最高={max(ds):.4f}")
PY

echo
echo "=== 3. 打包帶回本機 ==="
cd "$ROOT/data/predictions"
tar cf "$ROOT/download508_10fold.tar" Dataset508_MCSTUEdge_ens10 Dataset508_MCSTUEdge_cv10
ls -lh "$ROOT/download508_10fold.tar"
echo "=== 完成 $(date) ===  把 $ROOT/download508_10fold.tar 下載回本機的 coseg-latest/"
