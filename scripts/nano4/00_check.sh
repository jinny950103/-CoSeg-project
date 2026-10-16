#!/bin/bash
# 上傳後的第一步:確認資料完整、環境正常、空間夠。
# 這支不需要 GPU,直接在登入節點跑。
#   bash scripts/nano4/00_check.sh
#
# 注意:這裡刻意不用 set -u。conda 的 activate 腳本會引用未定義變數,
# set -u 會讓環境啟用失敗,後續的 pip/python 全部失效(看起來像套件沒裝)。

ROOT=/home/u9444861/-CoSeg-project
DS=$ROOT/data/nnUNet_raw/Dataset503_MixedCanal

echo "===== 1. 磁碟空間 ====="
df -h "$ROOT" | tail -2
echo
echo "(前處理預估需要 20~25GB,如果可用空間少於 30GB 先清理)"
echo
echo "舊的 Dataset501 是否還在(該刪掉以釋放 7.7GB):"
if [ -d "$ROOT/data/nnUNet_raw/Dataset501_MandibularCanal" ]; then
  echo "  還在 -> rm -rf $ROOT/data/nnUNet_raw/Dataset501_MandibularCanal"
else
  echo "  已刪除"
fi

echo
echo "===== 2. 環境變數 ====="
if [ "${CONDA_DEFAULT_ENV:-}" != "coseg" ]; then
  module load miniconda3
  source activate coseg
fi
for v in nnUNet_raw nnUNet_preprocessed nnUNet_results; do
  printf '  %-22s = %s\n' "$v" "${!v:-<未設定>}"
done
echo
echo "nnU-Net 版本(應該是 nnunetv2):"
pip show nnunetv2 2>&1 | head -3 | sed 's/^/  /' || echo "  找不到 nnunetv2"
echo
echo "可用的 trainer(找短 epoch 的版本,給試訓練用):"
python - <<'PY'
import os, re, nnunetv2
# 短 epoch 的 trainer 是「同一個檔案裡的多個 class」(nnUNetTrainer_Xepochs.py),
# 不是一個檔案一個 trainer,所以要掃 class 定義而不是檔名。
base = os.path.join(os.path.dirname(nnunetv2.__file__), "training", "nnUNetTrainer")
classes = set()
for root, _, files in os.walk(base):
    for f in files:
        if not f.endswith(".py"):
            continue
        txt = open(os.path.join(root, f), encoding="utf-8", errors="ignore").read()
        classes |= set(re.findall(r"^class\s+(nnUNetTrainer\w*)", txt, re.M))
short = sorted(c for c in classes if re.search(r"_\d+epoch", c))
print("  短 epoch 可用:", ", ".join(short) if short else "(無)")
print("  trainer 總數:", len(classes))
PY

echo
echo "===== 3. 資料完整性 ====="
if [ ! -d "$DS" ]; then
  echo "  !!! 找不到 $DS"
  exit 1
fi
for d in imagesTr labelsTr imagesTs labelsTs_eval; do
  printf '  %-15s %s\n' "$d:" "$(ls "$DS/$d" 2>/dev/null | wc -l)"
done
echo "  (應該是 235 / 235 / 24 / 24)"
echo
echo "  JSON 檔:"
ls -1 "$DS"/*.json | sed 's|.*/|    |'
echo
echo "  總大小: $(du -sh "$DS" | cut -f1)   (應該約 7.9G)"

echo
echo "===== 4. gzip 完整性(抓傳輸截斷) ====="
FAIL=0
for d in imagesTr labelsTr imagesTs labelsTs_eval; do
  if ! gzip -t "$DS/$d"/*.nii.gz 2>/tmp/gziperr_$$; then
    echo "  !!! $d 有損壞檔案:"
    sed 's/^/      /' /tmp/gziperr_$$
    FAIL=1
  fi
done
rm -f /tmp/gziperr_$$
[ $FAIL -eq 0 ] && echo "  全部 518 個 .nii.gz 完整"

echo
echo "===== 5. SLURM 設定(寫 sbatch 腳本需要) ====="
echo "可用的 partition:"
sinfo -s -o "%20P %10a %10l %10D %N" 2>/dev/null | head -15 || echo "  sinfo 無法執行"
echo
echo "帳號可用的 partition/QoS:"
sacctmgr -n show assoc user="$USER" format=Account,Partition,QOS 2>/dev/null | head -10 || echo "  sacctmgr 無法執行"

echo
if [ $FAIL -eq 0 ]; then
  echo "檢查完成。把上面「SLURM 設定」那段貼回來,我就能把 partition 填進訓練腳本。"
else
  echo "有損壞檔案,請重新上傳那幾個檔案後再跑一次。"
  exit 1
fi
