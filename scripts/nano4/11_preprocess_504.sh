#!/bin/bash
#SBATCH --job-name=nnunet504_prep
#SBATCH --account=MST115208
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1      # nano4 的 partition 全是 GPU 節點,必須指定
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00        # 503 實測 2 分鐘
#SBATCH --exclude=25a-hgpn001
#SBATCH --output=logs/prep504_%j.out
#SBATCH --error=logs/prep504_%j.err
#
# Dataset504 的 planning + 前處理,只做 3d_fullres(2d 會多佔約 26GB 又用不到)。
#   sbatch scripts/nano4/11_preprocess_504.sh
module load miniconda3
source activate coseg
set -euo pipefail

ROOT=/home/u9444861/-CoSeg-project
DNAME=Dataset504_MixedCanalClean
PREP=$ROOT/data/nnUNet_preprocessed/$DNAME

echo "=== 開始 $(date) ==="
df -h "$ROOT" | tail -1

nnUNetv2_plan_and_preprocess -d 504 -c 3d_fullres --verify_dataset_integrity -np 8

echo
echo "=== 放入自訂的 splits_final.json ==="
# nnU-Net 只認 nnUNet_preprocessed 底下這份;不覆蓋就會變成它自己隨機切的 5 折。
SRC=$ROOT/data/nnUNet_raw/$DNAME/splits_final.json
DST=$PREP/splits_final.json
if [ -f "$DST" ] && ! cmp -s "$SRC" "$DST"; then
  cp "$DST" "$DST.nnunet_generated.bak"
fi
cp "$SRC" "$DST"
python - <<PY
import json
s = json.load(open("$DST"))
for i, f in enumerate(s):
    nh = sum(1 for c in f["val"] if c.startswith("HOSP_"))
    print(f"  fold {i}: train={len(f['train'])} val={len(f['val'])} (醫院 {nh})")
assert all(sum(1 for c in f["val"] if c.startswith("HOSP_")) > 0 for f in s)
PY

echo
echo "=== plans ==="
python - <<PY
import json
d = json.load(open("$PREP/nnUNetPlans.json"))
c = d["configurations"]["3d_fullres"]
print("  spacing    =", [round(x, 4) for x in c["spacing"]], "(應該是 0.4)")
print("  patch_size =", c["patch_size"])
PY
df -h "$ROOT" | tail -1
echo "=== 完成 $(date) ===  下一步:sbatch scripts/nano4/12_train_504.sh"
