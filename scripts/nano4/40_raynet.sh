#!/bin/bash
#SBATCH --job-name=raynet
#SBATCH --account=MST115208
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:55:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn062
#SBATCH --output=logs/raynet_%j.out
#SBATCH --error=logs/raynet_%j.err
#
# 射線網路:從截面影像預測 72 個方向的管壁半徑(只用醫院影像和我們自己的 edge detection 半徑)。
# 先把 upload_raynet.tar 在 $ROOT 解開,再:
#   sbatch scripts/nano4/40_raynet.sh            # 切截面資料 + 5 折(同 504 的病人分折)
# 完成後把 $ROOT/download_raynet.tar 下載回本機的 coseg-latest/
module load miniconda3
source activate coseg
export PATH="$CONDA_PREFIX/bin:$PATH"
set -euo pipefail

ROOT=/home/u9444861/-CoSeg-project
cd "$ROOT"
OUT=outputs/raynet
mkdir -p "$OUT"
echo "=== 開始 $(date) on $(hostname) ==="
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'GPU 不可用'); " 
python -c "import torch; print('  GPU:', torch.cuda.get_device_name(0))"

if [ ! -f "$OUT/data.npz" ]; then
  echo "=== 1. 切截面(影像用 504 的 imagesTr)==="
  python scripts/canal_raynet_data.py --rays outputs/sparse_label_v6_rays/chunk_0*/*_rays.npz --out "$OUT/data.npz" \
    --img "data/nnUNet_raw/Dataset504_MixedCanalClean/imagesTr/HOSP_{}_0000.nii.gz" \
    --splits data/nnUNet_raw/Dataset504_MixedCanalClean/splits_final.json
fi

echo "=== 2. 5 折訓練 ==="
for f in 0 1 2 3 4; do
  python scripts/canal_raynet.py --data "$OUT/data.npz" --out "$OUT/cv" --folds $f --epochs ${EPOCHS:-40}
done

echo "=== 3. 打包 ==="
tar cf "$ROOT/download_raynet.tar" "$OUT"/cv/cv_pred_f*.npz "$OUT"/cv/net_fold*.pt "$OUT"/cv/args.json
ls -lh "$ROOT/download_raynet.tar"
echo "=== 完成 $(date) ===  把 $ROOT/download_raynet.tar 下載回本機的 coseg-latest/"
