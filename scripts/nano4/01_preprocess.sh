#!/bin/bash
#SBATCH --job-name=nnunet503_prep
#SBATCH --account=MST115208
#SBATCH --partition=dev                # 4 小時上限,專給測試用,排隊最快
#SBATCH --gpus-per-node=1      # nano4 的 partition 全是 GPU 節點,必須指定
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:50:00
#SBATCH --output=logs/prep_%j.out
#SBATCH --error=logs/prep_%j.err
#
# nnU-Net 前處理。純 CPU,不需要 GPU。
#   mkdir -p logs && sbatch scripts/nano4/01_preprocess.sh
#
# 記憶體開 96G 是因為醫院影像是 501x501x438(110M 體素),前處理會把整個
# volume 以 float32 讀進來(約 442MB)並同時處理多個 case;開太小會在跑到
# 醫院那 17 位時 OOM,而且 OOM 發生在 218 位公開資料跑完之後,等於浪費幾小時。
# conda activate 會引用未定義變數,所以要在 set -u 之前完成環境啟用。
module load miniconda3
source activate coseg
set -euo pipefail

ROOT=/home/u9444861/-CoSeg-project
DID=503
DNAME=Dataset503_MixedCanal

echo "=== 開始 $(date) ==="
echo "nnUNet_raw          = ${nnUNet_raw:-未設定}"
echo "nnUNet_preprocessed = ${nnUNet_preprocessed:-未設定}"
echo
echo "--- 前處理前的磁碟 ---"
df -h "$ROOT" | tail -1

echo
echo "=== nnUNetv2_plan_and_preprocess ==="
# --verify_dataset_integrity 會檢查檔名配對、label 值、幾何一致性。
# 第一次跑一定要加:它抓到的問題在這裡是幾分鐘,訓練到一半才爆是幾小時。
nnUNetv2_plan_and_preprocess \
  -d "$DID" \
  --verify_dataset_integrity \
  -np 8

echo
echo "--- 前處理後的磁碟 ---"
df -h "$ROOT" | tail -1
echo "preprocessed 大小: $(du -sh "$ROOT/data/nnUNet_preprocessed/$DNAME" | cut -f1)"

echo
echo "=== 放入自訂的 splits_final.json ==="
# 關鍵步驟:nnU-Net 只認 nnUNet_preprocessed 底下的 splits_final.json。
# 如果不覆蓋,它會自己產生一份隨機 5-fold,我們「每折都要有醫院病人」
# 和「同一受試者不跨折」的設計就會被無聲蓋掉。
SRC=$ROOT/data/nnUNet_raw/$DNAME/splits_final.json
DST=$ROOT/data/nnUNet_preprocessed/$DNAME/splits_final.json
if [ -f "$DST" ]; then
  cp "$DST" "$DST.nnunet_generated.bak"
  echo "  nnU-Net 自己產生的版本已備份為 splits_final.json.nnunet_generated.bak"
fi
cp "$SRC" "$DST"
python - <<PY
import json
s = json.load(open("$DST"))
print(f"  {len(s)} 折:")
for i, f in enumerate(s):
    nh = sum(1 for c in f["val"] if c.startswith("HOSP_"))
    print(f"    fold {i}: train={len(f['train'])} val={len(f['val'])} (其中醫院 {nh})")
PY

echo
echo "=== 產生的 plans(確認 target spacing 與 patch size)==="
python - <<PY
import json, os
p = os.path.join("$ROOT", "data/nnUNet_preprocessed/$DNAME/nnUNetPlans.json")
d = json.load(open(p))
print("  原始中位 spacing :", [round(x, 4) for x in d.get("original_median_spacing_after_transp", [])])
print("  原始中位 shape   :", d.get("original_median_shape_after_transp"))
print("  可用的 configuration:", list(d.get("configurations", {}).keys()))
for name, c in d.get("configurations", {}).items():
    print(f"  [{name}]")
    print(f"     spacing    = {[round(x,4) for x in c.get('spacing', [])]}")
    print(f"     patch_size = {c.get('patch_size')}")
    print(f"     batch_size = {c.get('batch_size')}")
PY

echo
echo "=== 完成 $(date) ==="
echo "下一步:sbatch scripts/nano4/02_smoke_train.sh"
