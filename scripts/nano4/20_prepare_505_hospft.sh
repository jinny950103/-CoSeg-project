#!/bin/bash
#SBATCH --job-name=nnunet505_prep
#SBATCH --account=MST115208
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --exclude=25a-hgpn001
#SBATCH --output=logs/prep505_%j.out
#SBATCH --error=logs/prep505_%j.err
#
# 準備「只用醫院資料微調」的 Dataset505_HospFT:
#   1. 用 504 的 34 位醫院訓練病人組出 505(hard link,不多佔空間)
#   2. 沿用 504 的 plans(網路架構必須完全一樣,才能載入 504 的權重)
#   3. 前處理 3d_fullres
#   4. 5 折與 504 一致:fold 0 的 7 位醫院驗證病人和 504 fold 0 完全相同,分數可以直接比
#   5. 把 nnUNetTrainer_HospFT 安裝進 nnunetv2
#
#   sbatch scripts/nano4/20_prepare_505_hospft.sh
module load miniconda3
source activate coseg
export PATH="$CONDA_PREFIX/bin:$PATH"   # sbatch 裡的 python 不一定是 coseg 的
set -euo pipefail

ROOT=/home/u9444861/-CoSeg-project
SRC=$ROOT/data/nnUNet_raw/Dataset504_MixedCanalClean
DST=$ROOT/data/nnUNet_raw/Dataset505_HospFT

echo "=== 1. 組出 Dataset505 ==="
mkdir -p "$DST/imagesTr" "$DST/labelsTr"
for f in "$SRC"/imagesTr/HOSP_*; do ln -f "$f" "$DST/imagesTr/" 2>/dev/null || cp "$f" "$DST/imagesTr/"; done
for f in "$SRC"/labelsTr/HOSP_*; do ln -f "$f" "$DST/labelsTr/" 2>/dev/null || cp "$f" "$DST/labelsTr/"; done
python - <<PY
import json
src, dst = "$SRC", "$DST"
n = len([1 for _ in __import__("os").listdir(dst + "/imagesTr")])
json.dump({"channel_names": {"0": "CBCT"},
           "labels": {"background": 0, "mandibular_canal": 1},
           "numTraining": n, "file_ending": ".nii.gz"},
          open(dst + "/dataset.json", "w"), indent=4)
s = json.load(open(src + "/splits_final.json"))
hosp = sorted({c for f in s for c in f["train"] + f["val"] if c.startswith("HOSP_")})
out = []
for f in s:
    val = sorted(c for c in f["val"] if c.startswith("HOSP_"))
    out.append({"train": sorted(c for c in hosp if c not in set(val)), "val": val})
json.dump(out, open(dst + "/splits_final.json", "w"), indent=2)
print(f"  醫院訓練病人 {n} 位")
for i, f in enumerate(out):
    print(f"  fold {i}: train={len(f['train'])} val={len(f['val'])}")
assert n == 34, "醫院訓練病人應該是 34 位"
PY

echo
echo "=== 2. fingerprint + 沿用 504 的 plans ==="
nnUNetv2_extract_fingerprint -d 505 -np 8
nnUNetv2_move_plans_between_datasets -s 504 -t 505 -sp nnUNetPlans -tp nnUNetPlans

echo
echo "=== 3. 前處理 3d_fullres ==="
PREP=$ROOT/data/nnUNet_preprocessed/Dataset505_HospFT
# plan_experiment normally drops dataset.json here; move_plans does not, and the
# preprocessor refuses to start without it.
cp "$DST/dataset.json" "$PREP/"
nnUNetv2_preprocess -d 505 -c 3d_fullres -np 8
cp "$DST/splits_final.json" "$PREP/splits_final.json"
python - <<PY
import json
a = json.load(open("$ROOT/data/nnUNet_preprocessed/Dataset504_MixedCanalClean/nnUNetPlans.json"))["configurations"]["3d_fullres"]
b = json.load(open("$PREP/nnUNetPlans.json"))["configurations"]["3d_fullres"]
for k in ("patch_size", "spacing", "architecture"):
    assert a[k] == b[k], f"505 的 {k} 跟 504 不同,載入權重會失敗"
print("  patch / spacing / 架構 與 504 相同")
PY

echo
echo "=== 4. 安裝 trainer ==="
PKG=$(python -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))")
cp "$ROOT/scripts/nnunet_trainers/nnUNetTrainer_HospFT.py" "$PKG/training/nnUNetTrainer/variants/"
python - <<'PY'
import inspect
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainer_HospFT import nnUNetTrainer_HospFT
base = list(inspect.signature(nnUNetTrainer.__init__).parameters)
mine = list(inspect.signature(nnUNetTrainer_HospFT.__init__).parameters)
assert base == mine, f"trainer 參數跟這版 nnU-Net 不合: {base} vs {mine}"
print("  nnUNetTrainer_HospFT 安裝完成,參數相容")
PY

df -h "$ROOT" | tail -1
echo "=== 完成 $(date) ===  下一步:sbatch scripts/nano4/21_train_505_hospft.sh"
