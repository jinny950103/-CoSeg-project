#!/bin/bash
#SBATCH --job-name=nnunet503_prep3d
#SBATCH --account=MST115208
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1      # nano4 的 partition 全是 GPU 節點,必須指定
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:50:00
#SBATCH --output=logs/prep3d_%j.out
#SBATCH --error=logs/prep3d_%j.err
#
# 只前處理 3d_fullres。
#
# 為什麼不用 nnUNetv2_plan_and_preprocess:它預設會把 2d / 3d_fullres / 3d_lowres
# 三種設定各存一份前處理資料。3d_lowres 已被 planner 判定不需要而放棄,但 2d 仍會
# 產生約 26GB 的副本 —— 加上 3d_fullres 的 26GB 就超過 /home 的 45GB 可用空間。
# 2D 對下頜管這種立體管狀結構也沒有用(那是舊 CoSeg++ 的做法),所以直接跳過。
#
# plans 已由前一次的 planning 階段算好並存在 nnUNetPlans.json,這裡只做前處理。
#   sbatch scripts/nano4/01b_preprocess_3dfullres.sh
module load miniconda3
source activate coseg
set -euo pipefail

ROOT=/home/u9444861/-CoSeg-project
DNAME=Dataset503_MixedCanal
PREP=$ROOT/data/nnUNet_preprocessed/$DNAME

echo "=== 開始 $(date) ==="
df -h "$ROOT" | tail -1

echo
echo "=== 只前處理 3d_fullres ==="
nnUNetv2_preprocess -d 503 -c 3d_fullres -np 8

echo
echo "--- 前處理後的磁碟 ---"
df -h "$ROOT" | tail -1
echo "preprocessed 各設定的大小:"
du -sh "$PREP"/* 2>/dev/null | sed 's/^/  /'

echo
echo "=== 放入自訂的 splits_final.json ==="
# 關鍵:nnU-Net 只認 nnUNet_preprocessed 底下的這份。不覆蓋的話它會用自己隨機切的
# 5-fold,「每折都要有醫院病人」和「同受試者不跨折」的設計會被無聲蓋掉。
SRC=$ROOT/data/nnUNet_raw/$DNAME/splits_final.json
DST=$PREP/splits_final.json
if [ -f "$DST" ] && ! cmp -s "$SRC" "$DST"; then
  cp "$DST" "$DST.nnunet_generated.bak"
  echo "  nnU-Net 產生的版本已備份為 splits_final.json.nnunet_generated.bak"
fi
cp "$SRC" "$DST"
python - <<PY
import json
s = json.load(open("$DST"))
print(f"  {len(s)} 折:")
for i, f in enumerate(s):
    nh = sum(1 for c in f["val"] if c.startswith("HOSP_"))
    print(f"    fold {i}: train={len(f['train'])} val={len(f['val'])} (其中醫院 {nh})")
assert all(sum(1 for c in f["val"] if c.startswith("HOSP_")) > 0 for f in s), "有折次沒有醫院病人"
PY

echo
echo "=== 完成 $(date) ==="
echo "下一步:sbatch scripts/nano4/02_smoke_train.sh"
