"""Compare the two MCSTU development sub-cohorts to pin down the broken group's true spacing.

If the 1.0mm-header cases are the same scanner/protocol as the 0.18mm ones, the canal must
occupy the same number of VOXELS in both. Voxel counts are spacing-independent, so they
discriminate the true spacing: the same anatomy at 0.25mm would occupy (0.18/0.25)^3 = 0.37x
the voxels of a 0.18mm scan.
"""
import glob
import os
import re
import statistics as st

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt

imgd = "C:/Users/WIN11/mcstu_data/MCSTU/development_dataset/img_nii/"
lbld = "C:/Users/WIN11/mcstu_data/MCSTU/development_dataset/labels/"

fs = sorted(glob.glob(imgd + "*.nii.gz"))
groups = {"header_0.18": [], "header_1.0": []}
for f in fs:
    sp = float(nib.load(f).header.get_zooms()[0])
    key = "header_0.18" if abs(sp - 0.18) < 1e-3 else "header_1.0"
    groups[key].append(f)

print("=== 檔名格式 vs header spacing ===")
for k, v in groups.items():
    uid = sum(1 for f in v if re.match(r"^1\.2\.250", os.path.basename(f)))
    other = len(v) - uid
    print(f"  {k}: {len(v)} 位  (UID格式 {uid}, 其他格式 {other})")
    print(f"     範例: {', '.join(os.path.basename(f) for f in v[:3])}")

print()
print("=== label 前景 voxel 數 與 管半徑(以 voxel 為單位) ===")
print("   voxel 數與半徑都不依賴 header spacing,可用來反推真實解析度")
stats = {}
for k, v in groups.items():
    counts, radii = [], []
    for f in v[:10]:
        n = os.path.basename(f)
        m = np.asanyarray(nib.load(lbld + n).dataobj) > 0
        counts.append(int(m.sum()))
        idx = np.array(np.nonzero(m))
        lo, hi = idx.min(1), idx.max(1) + 1
        sub = np.pad(m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], 2)
        edt = distance_transform_edt(sub)  # sampling=1 -> 單位是 voxel
        radii.append(float(np.percentile(edt[sub], 99)))
    stats[k] = (counts, radii)
    print(f"  {k} (取前10位):")
    print(f"     voxel 數  平均 {st.mean(counts):.0f}  範圍 {min(counts)}~{max(counts)}")
    print(f"     半徑p99   平均 {st.mean(radii):.2f} voxel  範圍 {min(radii):.2f}~{max(radii):.2f}")

a, b = stats["header_0.18"], stats["header_1.0"]
ratio_cnt = st.mean(b[0]) / st.mean(a[0])
ratio_r = st.mean(b[1]) / st.mean(a[1])
print()
print(f"voxel 數比值 (壞/好) = {ratio_cnt:.3f}   半徑比值 = {ratio_r:.3f}")
print()
print("若壞組真實 spacing = S,則半徑比值應約等於 0.18/S:")
for S in (0.15, 0.18, 0.2, 0.25, 0.3):
    print(f"   S={S}mm -> 預期半徑比值 {0.18/S:.3f}")
