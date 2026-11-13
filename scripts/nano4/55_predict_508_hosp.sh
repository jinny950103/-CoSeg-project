#!/bin/bash
#SBATCH --job-name=nnunet508_hosp
#SBATCH --account=MST115208
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=25a-hgpn001
#SBATCH --output=logs/pred508_hosp_%j.out
#SBATCH --error=logs/pred508_hosp_%j.err
#
# 待辦 A2(2026-11-13):用 508(只用 MCSTU 訓練,5 折)預測醫院訓練組 34 例
# (33 位 + 標註錯位的 1 位,本機比對時會跳過)。508 沒看過任何醫院影像,所以 5 折集成都可以用。
# 本機拿預測結果,在超過上限的長缺口裡和 bridge/縱向斷面比對,一致的才保留。
#   sbatch scripts/nano4/55_predict_508_hosp.sh
#   完成後下載 $ROOT/download508_hosp.tar 到本機 coseg-latest/
module load miniconda3
source activate coseg
export PATH="$CONDA_PREFIX/bin:$PATH"
set -euo pipefail

ROOT=/home/u9444861/-CoSeg-project
TRAINER=${TRAINER:-nnUNetTrainer_Sparse}
RES=$nnUNet_results/Dataset508_MCSTUEdge/${TRAINER}__nnUNetPlans__3d_fullres
TR=$ROOT/data/nnUNet_raw/Dataset504_MixedCanalClean/imagesTr
IN=$ROOT/data/predictions/hosp34_input
OUT=$ROOT/data/predictions/Dataset508_MCSTUEdge_hosp34

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'GPU 不可用')"
for f in 0 1 2 3 4; do
  [ -f "$RES/fold_$f/checkpoint_final.pth" ] || { echo "fold $f 找不到 checkpoint_final.pth"; exit 1; }
done

echo "=== 1. 準備輸入(醫院影像連結,不複製)==="
mkdir -p "$IN" "$OUT"
for f in "$TR"/HOSP_*_0000.nii.gz; do ln -sf "$f" "$IN/"; done
N=$(ls "$IN" | wc -l)
echo "  輸入 $N 個(應該是 34)"
[ "$N" -eq 34 ] || { echo "數量不對"; exit 1; }

echo "=== 2. 預測(集成 5 折)==="
nnUNetv2_predict -i "$IN" -o "$OUT" -d 508 -c 3d_fullres -tr "$TRAINER" -f 0 1 2 3 4
echo "  預測 $(ls "$OUT"/*.nii.gz | wc -l) 個"

echo "=== 3. 打包帶回本機 ==="
cd "$ROOT/data/predictions"
tar cf "$ROOT/download508_hosp.tar" Dataset508_MCSTUEdge_hosp34
ls -lh "$ROOT/download508_hosp.tar"
echo "=== 完成 $(date) ===  把 $ROOT/download508_hosp.tar 下載回本機的 coseg-latest/"
