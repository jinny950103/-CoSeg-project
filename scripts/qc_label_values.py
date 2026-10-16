"""Survey the label value encoding across every MCSTU case.

Matters because the resample step thresholds interpolated labels at 0.5: that means
">=50% overlap" only if the foreground value is 1. The L-series labels encode foreground
as 4, so the same threshold admitted anything above 12.5% overlap and dilated the mask.

Slices are read with a stride so float64 volumes (1.3GB uncompressed each) are not fully
materialised.
"""
import glob
import os
import re
from collections import Counter

import nibabel as nib
import numpy as np

lbld = "C:/Users/WIN11/mcstu_data/MCSTU/development_dataset/labels/"
testd = "C:/Users/WIN11/mcstu_data/MCSTU/independent_test_dataset/"

STRIDE = 4


def values_of(path):
    img = nib.load(path)
    sub = img.dataobj[..., ::STRIDE]
    return np.unique(np.asarray(sub)), img.get_data_dtype()


print("=== development_dataset labels (218) ===")
combos = Counter()
for f in sorted(glob.glob(lbld + "*.nii.gz")):
    n = os.path.basename(f)
    vals, dt = values_of(f)
    group = "UID" if n.startswith("1.2.250") else "L-series"
    combos[(group, str(dt), tuple(float(v) for v in vals))] += 1

for (group, dt, vals), cnt in sorted(combos.items()):
    print(f"  {group:10s} dtype={dt:8s} values={vals}  -> {cnt} 位")

print()
print("=== independent_test_dataset labels (18) ===")
combos2 = Counter()
for d in sorted(glob.glob(testd + "*/")):
    vals, dt = values_of(d + "label.nii.gz")
    combos2[(str(dt), tuple(float(v) for v in vals))] += 1
for (dt, vals), cnt in sorted(combos2.items()):
    print(f"  dtype={dt:8s} values={vals}  -> {cnt} 位")

print()
print("=== hospital labels (23, 已轉好的) ===")
combos3 = Counter()
for f in sorted(glob.glob("C:/Users/WIN11/coseg-latest/data/hospital_nifti/labels/*.nii.gz")):
    vals, dt = values_of(f)
    combos3[(str(dt), tuple(float(v) for v in vals))] += 1
for (dt, vals), cnt in sorted(combos3.items()):
    print(f"  dtype={dt:8s} values={vals}  -> {cnt} 位")
