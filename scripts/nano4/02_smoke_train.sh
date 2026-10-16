#!/bin/bash
#SBATCH --job-name=nnunet503_smoke
#SBATCH --account=MST115208
#SBATCH --partition=dev                # 4 小時上限,夠跑短試訓練
#SBATCH --gpus-per-node=1      # nano4 的 partition 全是 GPU 節點,必須指定
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
#
# 短時間試訓練。目的不是看分數,是確認:
#   GPU 抓得到 / 記憶體夠 / 資料讀得進去 / checkpoint 存得出來 / 驗證流程會動
# 跑幾個 epoch 就好,一小時內會結束。
#   mkdir -p logs && sbatch scripts/nano4/02_smoke_train.sh
#
# TRAINER 預設用短 epoch 的版本。00_check.sh 會列出安裝版本實際有哪些;
# 如果 nnUNetTrainer_5epochs 不存在,從那份清單挑一個換掉即可。
# conda activate 會引用未定義變數,所以要在 set -u 之前完成環境啟用。
module load miniconda3
source activate coseg
set -euo pipefail

DID=503
CONFIG=3d_fullres
FOLD=0
TRAINER=${TRAINER:-nnUNetTrainer_5epochs}

echo "=== 開始 $(date) ==="
echo "trainer=$TRAINER  config=$CONFIG  fold=$FOLD"
echo
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, '| cuda可用', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo
echo "--- 確認 splits 是我們的版本(每折都該有醫院病人)---"
python - <<'PY'
import json, os
p = os.path.join(os.environ["nnUNet_preprocessed"], "Dataset503_MixedCanal", "splits_final.json")
s = json.load(open(p))
for i, f in enumerate(s):
    nh = sum(1 for c in f["val"] if c.startswith("HOSP_"))
    print(f"  fold {i}: val={len(f['val'])} 醫院={nh}")
if any(sum(1 for c in f["val"] if c.startswith("HOSP_")) == 0 for f in s):
    raise SystemExit("!!! 有折次沒有醫院病人 -> splits_final.json 被 nnU-Net 蓋掉了,回去重跑 01_preprocess.sh 的複製步驟")
PY

echo
echo "=== 試訓練 ==="
if ! nnUNetv2_train "$DID" "$CONFIG" "$FOLD" -tr "$TRAINER"; then
  echo
  echo "訓練失敗。如果錯誤是找不到 trainer,用下面這行看有哪些可選:"
  echo "  python -c \"import nnunetv2,os;b=os.path.join(os.path.dirname(nnunetv2.__file__),'training','nnUNetTrainer');print('\n'.join(sorted(f[:-3] for r,_,fs in os.walk(b) for f in fs if f.startswith('nnUNetTrainer') and f.endswith('.py'))))\""
  echo "然後用 TRAINER=<名稱> sbatch scripts/nano4/02_smoke_train.sh 重跑"
  exit 1
fi

echo
echo "=== 產出檢查 ==="
RES=$nnUNet_results/Dataset503_MixedCanal/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}
ls -lh "$RES" 2>/dev/null | sed 's/^/  /'
echo
if [ -f "$RES/checkpoint_final.pth" ]; then
  echo "  checkpoint_final.pth 存在 -> 訓練流程完整走完"
else
  echo "  !!! 沒有 checkpoint_final.pth"
fi
echo
echo "--- 訓練記錄最後幾行(看 loss 有沒有在動)---"
tail -20 "$RES"/training_log_*.txt 2>/dev/null | sed 's/^/  /' || echo "  找不到 training log"

echo
echo "=== 完成 $(date) ==="
echo
echo "這支只是確認流程能動,分數沒有意義(只跑幾個 epoch)。"
echo "確認 checkpoint 有產生、loss 在下降之後,再跑正式訓練:"
echo "  nnUNetv2_train $DID $CONFIG 0    # 預設 1000 epochs,約 1~2 天"
