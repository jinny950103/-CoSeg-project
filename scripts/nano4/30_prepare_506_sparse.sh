#!/bin/bash
#SBATCH --job-name=nnunet506_prep
#SBATCH --account=MST115208
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --exclude=25a-hgpn001
#SBATCH --output=logs/prep506_%j.out
#SBATCH --error=logs/prep506_%j.err
#
# 準備 Dataset506_CanalSparse:34 位醫院訓練病人 + 自己做的稀疏標註
# (scripts/canal_sparse_label.py 的 agree 版,0=背景 1=管 2=ignore)。
#   1. 影像從 504 的 imagesTr hard link 過來(上傳包只有標註),md5 逐檔比對
#   2. 檢查標註只有 0/1/2,和影像形狀一致
#   3. nnunetv2 要 >= 2.2 才支援 ignore 標籤
#   4. 沿用 504 的 plans(和 505 一樣),前處理 3d_fullres
#   5. 5 折:每折的醫院驗證病人和 504/505 相同
#
# 先把 upload506.tar 在 $ROOT 解開,再:
#   sbatch scripts/nano4/30_prepare_506_sparse.sh
module load miniconda3
source activate coseg
export PATH="$CONDA_PREFIX/bin:$PATH"   # sbatch 裡的 python 不一定是 coseg 的
set -euo pipefail

ROOT=/home/u9444861/-CoSeg-project
SRC=$ROOT/data/nnUNet_raw/Dataset504_MixedCanalClean
DST=$ROOT/data/nnUNet_raw/Dataset506_CanalSparse

echo "=== 0. nnunetv2 版本(ignore 標籤需要 >= 2.2)==="
python - <<'PY'
from importlib.metadata import version
v = version("nnunetv2"); print("  nnunetv2", v)
major, minor = (int(x) for x in v.split(".")[:2])
assert (major, minor) >= (2, 2), "nnunetv2 太舊,不支援 ignore 標籤 -> pip install -U nnunetv2"
PY

echo
echo "=== 1. 影像從 504 連結過來 ==="
[ -f "$DST/images_md5.txt" ] || { echo "找不到 $DST/images_md5.txt -> upload506.tar 還沒在 $ROOT 解開"; exit 1; }
mkdir -p "$DST/imagesTr"
for lab in "$DST"/labelsTr/HOSP_*.nii.gz; do
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
assert len(labs) == len(imgs) == 34, f"應該各 34 個,現在 labels={len(labs)} images={len(imgs)}"
tot = np.zeros(3)
for f in labs:
    l = nib.load(f"{d}/labelsTr/{f}"); i = nib.load(f"{d}/imagesTr/{f[:-7]}_0000.nii.gz")
    a = np.asanyarray(l.dataobj)
    u = set(np.unique(a).tolist())
    assert u <= {0, 1, 2}, f"{f} 有 0/1/2 以外的值 {u}"
    assert l.shape == i.shape and np.allclose(l.affine, i.affine, atol=1e-4), f"{f} 和影像的形狀或座標不同"
    tot += np.bincount(a.ravel(), minlength=3)[:3]
print(f"  34 例都只有 0/1/2,形狀和座標都和影像一致")
print(f"  管 {int(tot[1])} 體素,ignore {int(tot[2])} 體素")
js = json.load(open(d + "/dataset.json"))
assert js["labels"] == {"background": 0, "canal": 1, "ignore": 2}, js["labels"]
print("  dataset.json labels:", js["labels"])
PY

echo
echo "=== 3. 5 折(醫院驗證病人和 504/505 相同)==="
python - <<PY
import json, os
src, dst = "$SRC", "$DST"
have = {f[:-7] for f in os.listdir(dst + "/labelsTr") if f.endswith(".nii.gz")}
s = json.load(open(src + "/splits_final.json"))
out = []
for f in s:
    val = sorted(c for c in f["val"] if c in have)
    out.append({"train": sorted(have - set(val)), "val": val})
assert sorted(c for f in out for c in f["val"]) == sorted(have), "每位病人要剛好在一折的驗證集"
json.dump(out, open(dst + "/splits_final.json", "w"), indent=2)
for i, f in enumerate(out):
    print(f"  fold {i}: train={len(f['train'])} val={len(f['val'])}")
PY

echo
echo "=== 4. fingerprint + 沿用 504 的 plans + 前處理 ==="
nnUNetv2_extract_fingerprint -d 506 -np 8
nnUNetv2_move_plans_between_datasets -s 504 -t 506 -sp nnUNetPlans -tp nnUNetPlans
PREP=$ROOT/data/nnUNet_preprocessed/Dataset506_CanalSparse
cp "$DST/dataset.json" "$PREP/"        # move_plans 不會放 dataset.json,前處理少了它不能跑
nnUNetv2_preprocess -d 506 -c 3d_fullres -np 3   # -np 8 用到 100GB,超過 96G 被砍(job 603273)
cp "$DST/splits_final.json" "$PREP/splits_final.json"
python - <<PY
import json
b = json.load(open("$PREP/nnUNetPlans.json"))["configurations"]["3d_fullres"]
print("  spacing    =", [round(x, 4) for x in b["spacing"]], "(應該是 0.4)")
print("  patch_size =", b["patch_size"])
PY

echo
echo "=== 完成 $(date) ===  下一步:5 折各送一個 job"
echo "  for f in 0 1 2 3 4; do FOLD=\$f sbatch scripts/nano4/31_train_506.sh; done"
