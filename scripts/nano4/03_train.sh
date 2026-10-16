#!/bin/bash
#SBATCH --job-name=nnunet503_train
#SBATCH --account=MST115208
#SBATCH --partition=8gpus       # dev 只有 4h,跑不完 1000 epochs 的 ~5h
#SBATCH --gpus-per-node=8       # 這個 partition 必須整節點 8 張一起要:只要 1 張時
                                # nvidia-smi 看得到卡,但 torch.cuda.is_available() 是 False
#SBATCH --cpus-per-task=16      # 多給一點給資料增強的 worker
#SBATCH --mem=64G
#SBATCH --time=12:00:00         # 實測 17.5s/epoch -> 1000 epochs 約 5h,留足餘裕給最後的驗證
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#
# 正式訓練:fold 0,預設 trainer(1000 epochs)。
#   sbatch scripts/nano4/03_train.sh
#
# 續跑:如果被時限砍掉,用同一支腳本加 --c 從最後的 checkpoint 接著跑
#   CONTINUE=1 sbatch scripts/nano4/03_train.sh
module load miniconda3
source activate coseg
set -euo pipefail

DID=503
CONFIG=3d_fullres
FOLD=${FOLD:-0}

echo "=== 開始 $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# nvidia-smi 看得到卡 != PyTorch 用得到卡。GPU 張數申請錯時兩者會不一致,
# 而 nnU-Net 要到建好模型呼叫 .to(device) 才報錯,白等好幾分鐘。先擋在這裡。
echo "--- PyTorch 是否真的抓到 GPU ---"
python - <<'PY'
import os, sys, torch
print("  CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("  torch.cuda.is_available() =", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("  GPU 不可用 -> 這個 partition 的 --gpus-per-node 張數要求不對")
print("  使用:", torch.cuda.get_device_name(0))
PY

echo
echo "--- 確認自訂 splits 生效(最便宜的檢查)---"
# 這行印出的 train/val 數字如果不是 183/52,表示 splits_final.json 被蓋掉了,
# 「每折都有醫院病人」「同受試者不跨折」的設計就失效了。
python - <<'PY'
import json, os
p = os.path.join(os.environ["nnUNet_preprocessed"], "Dataset503_MixedCanal", "splits_final.json")
s = json.load(open(p))
for i, f in enumerate(s):
    nh = sum(1 for c in f["val"] if c.startswith("HOSP_"))
    print(f"  fold {i}: train={len(f['train'])} val={len(f['val'])} 醫院={nh}")
assert all(sum(1 for c in f["val"] if c.startswith("HOSP_")) > 0 for f in s), \
    "有折次的 val 沒有醫院病人 -> splits 被蓋掉了"
PY

echo
echo "=== 訓練 fold $FOLD ==="
if [ "${CONTINUE:-0}" = "1" ]; then
  echo "(從最後的 checkpoint 續跑)"
  nnUNetv2_train "$DID" "$CONFIG" "$FOLD" --c
else
  nnUNetv2_train "$DID" "$CONFIG" "$FOLD"
fi

echo
echo "=== 完成 $(date) ==="
RES=$nnUNet_results/Dataset503_MixedCanal/nnUNetTrainer__nnUNetPlans__${CONFIG}/fold_${FOLD}
ls -lh "$RES" | sed 's/^/  /'
echo
echo "--- 最終驗證分數 ---"
# summary.json 是 nnU-Net 對 val 全部 52 位做完整推論後算的 Dice,
# 比訓練中的 pseudo dice 準。
python - <<PY
import json, os
p = "$RES/validation/summary.json"
if os.path.exists(p):
    d = json.load(open(p))
    fg = d.get("foreground_mean", {})
    print("  整體 val Dice:", round(fg.get("Dice", float("nan")), 4))
    print()
    print("  分 cohort(我們真正要看的):")
    per = d.get("metric_per_case", [])
    for tag, pref in [("醫院", "HOSP_"), ("公開", "MCSTU_")]:
        ds = [c["metrics"]["1"]["Dice"] for c in per
              if os.path.basename(c["prediction_file"]).startswith(pref)]
        if ds:
            print(f"    {tag}: n={len(ds)}  平均 Dice={sum(ds)/len(ds):.4f}  "
                  f"最低={min(ds):.4f}  最高={max(ds):.4f}")
else:
    print("  找不到 summary.json")
PY
echo
echo "下一步:對測試集推論 -> nnUNetv2_predict"
