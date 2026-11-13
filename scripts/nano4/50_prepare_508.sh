#!/bin/bash
#SBATCH --job-name=nnunet508_prep
#SBATCH --account=MST115208
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --exclude=25a-hgpn001
#SBATCH --output=logs/prep508_%j.out
#SBATCH --error=logs/prep508_%j.err
#
# 準備 Dataset508_MCSTUEdge:MCSTU 開發集 218 例 + edge detection 做的標註
# (0=背景 1=管(量到+推測+外插) 2=ignore)。先只用公開資料訓練基底模型,醫院資料等醫生確認後再混。
#   1. 影像從 504 的 imagesTr hard link 過來(上傳包只有標註),md5 逐檔比對
#   2. 檢查標註只有 0/1/2,和影像形狀、座標一致
#   3. 沿用 504 的 plans(0.4mm),前處理 3d_fullres
#   4. 5 折:沿用 504 分法裡的 MCSTU 部分
#
# 先把 upload508.tar 在 $ROOT 解開,再:
#   sbatch scripts/nano4/50_prepare_508.sh
module load miniconda3
source activate coseg
export PATH="$CONDA_PREFIX/bin:$PATH"
set -euo pipefail

ROOT=/home/u9444861/-CoSeg-project
SRC=$ROOT/data/nnUNet_raw/Dataset504_MixedCanalClean
DST=$ROOT/data/nnUNet_raw/Dataset508_MCSTUEdge
N=218

echo "=== 0. 空間(前處理約需 15-20GB)==="
FREE=$(df -BG --output=avail /home | tail -1 | tr -dc 0-9)
echo "  /home 剩 ${FREE}GB"
[ "$FREE" -ge 22 ] || { echo "  !!! 空間不夠,先刪舊的 nnUNet_preprocessed(例如 506/507)"; exit 1; }
python -c "from importlib.metadata import version; v=version('nnunetv2'); print('  nnunetv2', v); assert tuple(map(int, v.split('.')[:2])) >= (2, 2)"

echo
echo "=== 1. 影像從 504 連結過來 ==="
[ -f "$DST/images_md5.txt" ] || { echo "找不到 $DST/images_md5.txt -> upload508.tar 還沒在 $ROOT 解開"; exit 1; }
mkdir -p "$DST/imagesTr"
for lab in "$DST"/labelsTr/*.nii.gz; do
  b=$(basename "$lab" .nii.gz)
  f="$SRC/imagesTr/${b}_0000.nii.gz"
  [ -f "$f" ] || { echo "  !!! 504 裡找不到 $(basename "$f")"; exit 1; }
  ln -f "$f" "$DST/imagesTr/" 2>/dev/null || cp "$f" "$DST/imagesTr/"
done
( cd "$DST/imagesTr" && md5sum -c --quiet ../images_md5.txt ) && echo "  影像 $(wc -l < "$DST/images_md5.txt") 個,md5 全部一致"

echo
echo "=== 2. 標註檢查 ==="
python - <<PY
import os, json
import nibabel as nib, numpy as np
d = "$DST"
labs = sorted(f for f in os.listdir(d + "/labelsTr") if f.endswith(".nii.gz"))
imgs = sorted(os.listdir(d + "/imagesTr"))
assert len(labs) == len(imgs) == $N, f"應該各 $N 個,現在 labels={len(labs)} images={len(imgs)}"
tot = np.zeros(3)
for f in labs:
    l = nib.load(f"{d}/labelsTr/{f}"); i = nib.load(f"{d}/imagesTr/{f[:-7]}_0000.nii.gz")
    a = np.asanyarray(l.dataobj)
    u = set(np.unique(a).tolist())
    assert u <= {0, 1, 2}, f"{f} 有 0/1/2 以外的值 {u}"
    assert l.shape == i.shape and np.allclose(l.affine, i.affine, atol=1e-4), f"{f} 和影像的形狀或座標不同"
    tot += np.bincount(a.ravel(), minlength=3)[:3]
print(f"  $N 例都只有 0/1/2,形狀和座標都和影像一致;管 {int(tot[1])} 體素,ignore {int(tot[2])} 體素")
js = json.load(open(d + "/dataset.json"))
assert js["labels"] == {"background": 0, "canal": 1, "ignore": 2}, js["labels"]
s = json.load(open(d + "/splits_final.json"))
assert sorted(c for f in s for c in f["val"]) == [f[:-7] for f in labs], "每例要剛好在一折的驗證集"
for k, f in enumerate(s):
    print(f"  fold {k}: train={len(f['train'])} val={len(f['val'])}")
PY

echo
echo "=== 3. fingerprint + 沿用 504 的 plans + 前處理 ==="
nnUNetv2_extract_fingerprint -d 508 -np 4
nnUNetv2_move_plans_between_datasets -s 504 -t 508 -sp nnUNetPlans -tp nnUNetPlans
PREP=$ROOT/data/nnUNet_preprocessed/Dataset508_MCSTUEdge
cp "$DST/dataset.json" "$PREP/"        # move_plans 不會放 dataset.json
nnUNetv2_preprocess -d 508 -c 3d_fullres -np 3   # -np 8 在 506 用到 100GB 被砍
cp "$DST/splits_final.json" "$PREP/splits_final.json"
python - <<PY
import json
b = json.load(open("$PREP/nnUNetPlans.json"))["configurations"]["3d_fullres"]
print("  spacing    =", [round(x, 4) for x in b["spacing"]], "(應該是 0.4)")
print("  patch_size =", b["patch_size"])
PY
df -h /home | tail -1

echo
echo "=== 完成 $(date) ===  下一步:5 折各送一個 job"
echo "  for f in 0 1 2 3 4; do FOLD=\$f sbatch scripts/nano4/51_train_508.sh; done"
