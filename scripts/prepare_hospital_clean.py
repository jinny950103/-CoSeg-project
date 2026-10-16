#!/usr/bin/env python3
"""Assemble the clean hospital cohort (data/hospital_nifti_clean) from the two image sources.

Sources:
    --old    data/hospital_nifti/{images,labels}/<PID>.nii.gz   the 23 DICOM conversions (D:\\0911)
    --nas    Downloads/nas/nas/<PID>_0000.nii.gz                 full hospital download (images only)
    --labels sphere_output_v2/<PID>/<PID>_sphere_v2.nii.gz       labels for both

Some hospital exports have the annotation burned into the image: the canal is painted
with the scanner's maximum value (3094-3095), so a model can find it by looking for the
brightest line. Every case is screened, and a burned-in case is never written. For four
patients the nas copy is burned in but the old D:\\0911 copy is clean, so the old copy
is preferred whenever one exists. Where the two are both clean they are voxel-identical.

    python scripts/prepare_hospital_clean.py
"""
import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

SAT = 3094  # burned-in annotations use 3094-3095; real anatomy rarely reaches it

# Excluded by the user on 2026-10-05 after manual review (missing slices, labelled images
# in the no_label folder, broken folder structure, single-side labels).
# Patient IDs are kept out of the repo: one ID per line in excluded_cases.local.txt (gitignored).
_EXCLUDED_FILE = Path(__file__).with_name("excluded_cases.local.txt")
USER_EXCLUDED = set(_EXCLUDED_FILE.read_text().split()) if _EXCLUDED_FILE.exists() else set()


def burn_score(img, lbl):
    """Fraction of saturated voxels lying on the canal (±3 voxels). Clean ~0, burned ~1."""
    s = img >= SAT
    if s.sum() == 0:
        return 0.0
    return float((s & binary_dilation(lbl, iterations=3)).sum() / s.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default="./data/hospital_nifti")
    ap.add_argument("--nas", default=r"C:\Users\WIN11\Downloads\nas\nas")
    ap.add_argument("--labels",
                    default=r"C:\Users\WIN11\Downloads\0923\sphere_output_v2\sphere_output_v2")
    ap.add_argument("--out", default="./data/hospital_nifti_clean")
    ap.add_argument("--max-burn", type=float, default=0.5)
    args = ap.parse_args()

    old, nas, lab, out = Path(args.old), Path(args.nas), Path(args.labels), Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    ids = sorted({p.name[: -len("_0000.nii.gz")] for p in nas.glob("*_0000.nii.gz")}
                 | {p.name[: -len(".nii.gz")] for p in (old / "images").glob("*.nii.gz")})

    kept, rejected = {}, {}
    for pid in ids:
        if pid in USER_EXCLUDED:
            rejected[pid] = "user_excluded"
            continue
        lbl_path = lab / pid / f"{pid}_sphere_v2.nii.gz"
        if not lbl_path.exists():
            rejected[pid] = "no_label"
            continue
        lnii = nib.load(lbl_path)
        g = np.asanyarray(lnii.dataobj) > 0

        candidates = [("old", old / "images" / f"{pid}.nii.gz"), ("nas", nas / f"{pid}_0000.nii.gz")]
        tried, chosen = [], None
        for src, path in candidates:
            if not path.exists():
                continue
            inii = nib.load(path)
            if tuple(inii.shape) != tuple(g.shape):
                tried.append(f"{src}:shape {inii.shape} vs label {g.shape}")
                continue
            if not np.allclose(inii.affine, lnii.affine, atol=1e-3):
                tried.append(f"{src}:affine mismatch")
                continue
            a = np.asanyarray(inii.dataobj)
            score = burn_score(a, g)
            if score > args.max_burn:
                tried.append(f"{src}:burned {score:.2f}")
                continue
            chosen = (src, path, inii, score)
            break

        if chosen is None:
            rejected[pid] = "; ".join(tried) or "no_image"
            print(f"  [skip] {pid}  {rejected[pid]}", flush=True)
            continue

        src, path, inii, score = chosen
        shutil.copyfile(path, out / "images" / f"{pid}.nii.gz")
        # One shared affine and a 0/1 uint8 label, same as the old conversions.
        nib.save(nib.Nifti1Image(g.astype(np.uint8), inii.affine), out / "labels" / f"{pid}.nii.gz")
        kept[pid] = {"image_source": src, "image": str(path), "label": str(lbl_path),
                     "burn_score": round(score, 3), "label_voxels": int(g.sum())}
        print(f"  [ok]   {pid}  from {src}  burn={score:.2f}", flush=True)

    with open(out / "manifest.json", "w") as f:
        json.dump({"kept": kept, "rejected": rejected}, f, indent=2)
    n_old = sum(1 for v in kept.values() if v["image_source"] == "old")
    print(f"\n可用 {len(kept)} 例 (舊 {n_old} / nas {len(kept) - n_old}),排除 {len(rejected)} 例")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
