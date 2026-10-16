#!/usr/bin/env python3
"""Build the mixed MCSTU + hospital nnU-Net dataset (Dataset503_MixedCanal) at a common spacing.

MCSTU is 0.18mm isotropic and the hospital cohort is 0.4mm isotropic. Everything is
brought to --target-spacing (default 0.4mm, the hospital/deployment resolution) so that
nnU-Net does not pick 0.18mm from the 218-vs-23 median and inflate every hospital volume.

Output:
    <out>/Dataset503_MixedCanal/
        imagesTr/{MCSTU_###,HOSP_<PID>}_0000.nii.gz
        labelsTr/{MCSTU_###,HOSP_<PID>}.nii.gz
        imagesTs/{MCSTU_TEST_VOL_<n>,HOSP_<PID>}_0000.nii.gz
        labelsTs_eval/...                 ground truth kept for metric computation
        dataset.json
        case_id_mapping.json              case id -> source path, plus cohort tags
        splits_final.json                 5-fold CV with hospital cases in every val fold
        build_report.json                 per-case geometry and label-volume checks

Resampling: images are Gaussian pre-filtered then linearly interpolated (the pre-filter
matters when downsampling 2.2x, otherwise fine bone detail aliases). Labels are resampled
linearly and re-thresholded at 0.5 rather than nearest-neighbour, which keeps thin canal
segments better; the report records foreground volume before and after so any loss is visible.
"""
import argparse
import json
import random
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, zoom

DATASET_DIRNAME = "Dataset503_MixedCanal"


def target_shape_for(shape, spacing, target_spacing):
    return tuple(int(round(s * sp / target_spacing)) for s, sp in zip(shape, spacing))


def resolve_spacing(shape, spacing, assume_spacing, fov_limit_mm):
    """Detect a missing spacing header and substitute the known true value.

    124 of the 218 MCSTU development cases ship with no voxel size written: NIfTI then
    reports the 1.0mm default. Taken at face value a 551-voxel axis implies a 551mm field
    of view and a 118,000mm^3 canal, both physically impossible for dental CBCT (~100mm
    FOV, ~650mm^3 canal). Those cases have the same shape and the same label voxel counts
    as the correctly-tagged ones, so the true spacing is the cohort's 0.18mm. Left
    uncorrected, nnU-Net would see the same anatomy at two scales 5.5x apart and derive a
    meaningless target spacing -- which fails silently rather than raising.
    """
    if assume_spacing is None:
        return spacing, False
    unset = all(abs(sp - 1.0) < 1e-6 for sp in spacing)
    implausible = max(s * sp for s, sp in zip(shape, spacing)) > fov_limit_mm
    if unset and implausible:
        return (assume_spacing,) * 3, True
    return spacing, False


def resample_pair(img_path, lbl_path, target_spacing, tol=1e-3,
                  assume_spacing=None, fov_limit_mm=300.0):
    """Return (image_int16, label_uint8, affine, info) resampled to an isotropic target."""
    img = nib.load(img_path)
    lbl = nib.load(lbl_path)

    if tuple(img.shape[:3]) != tuple(lbl.shape[:3]):
        raise ValueError(f"image/label shape mismatch: {img.shape[:3]} vs {lbl.shape[:3]}")

    raw_spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    shape = tuple(int(s) for s in img.shape[:3])
    spacing, overridden = resolve_spacing(shape, raw_spacing, assume_spacing, fov_limit_mm)

    # The label's own header is deliberately ignored: 9 cases carry a broken label spacing
    # alongside a correct image spacing. Image geometry governs, and both arrays are
    # written with one shared affine so they cannot drift apart.
    if overridden:
        fix = spacing[0] / raw_spacing[0]
        img = nib.Nifti1Image(np.asanyarray(img.dataobj),
                              img.affine @ np.diag([fix, fix, fix, 1.0]))

    a = np.asanyarray(img.dataobj, dtype=np.float32)

    # Binarise BEFORE resampling. 99 of the 218 MCSTU labels encode foreground as 4 rather
    # than 1 (and mix float64/uint16), so interpolating the raw values and thresholding at
    # 0.5 would mean ">=12.5% overlap" for those cases and dilate the canal by up to 18%.
    raw_label_values = np.unique(np.asanyarray(lbl.dataobj))
    m = (np.asanyarray(lbl.dataobj) > 0).astype(np.float32)
    vox_in = float(np.prod(spacing))
    fg_before = float((m > 0).sum()) * vox_in

    already = all(abs(sp - target_spacing) < tol for sp in spacing)
    if already:
        out_shape = shape
        affine = img.affine.copy()
        a_out, m_out = a, m
    else:
        out_shape = target_shape_for(shape, spacing, target_spacing)
        factors = tuple(o / s for o, s in zip(out_shape, shape))

        # Anti-alias only when shrinking; sigma scaled to the decimation factor per axis.
        sigmas = [max(0.0, (1.0 / f - 1.0) / 2.0) for f in factors]
        a_f = gaussian_filter(a, sigma=sigmas) if any(s > 0 for s in sigmas) else a
        a_out = zoom(a_f, factors, order=1)
        m_out = zoom(m, factors, order=1)

        # Keep image and label on one grid: the affine comes from the realised scale.
        scale = np.diag([shape[0] / out_shape[0], shape[1] / out_shape[1],
                         shape[2] / out_shape[2], 1.0])
        affine = img.affine @ scale

    a_out = np.rint(a_out).astype(np.int16)
    m_out = (m_out >= 0.5).astype(np.uint8)

    vox_out = float(abs(np.linalg.det(affine[:3, :3])))
    info = {
        "in_shape": list(shape),
        "in_spacing": [round(s, 4) for s in spacing],
        "header_spacing": [round(s, 4) for s in raw_spacing],
        "spacing_overridden": overridden,
        "raw_label_values": [float(v) for v in raw_label_values[:8]],
        "out_shape": [int(s) for s in a_out.shape],
        "out_spacing": [round(target_spacing, 4)] * 3,
        "resampled": not already,
        "label_mm3_before": round(fg_before, 1),
        "label_mm3_after": round(float((m_out > 0).sum()) * vox_out, 1),
    }
    info["label_mm3_change_pct"] = (
        round(100.0 * (info["label_mm3_after"] - fg_before) / fg_before, 1) if fg_before > 0 else None
    )
    return a_out, m_out, affine, info


def write_case(out_dir, case_id, arr, affine, spacing, is_label):
    nii = nib.Nifti1Image(arr, affine)
    nii.header.set_zooms(tuple(float(spacing) for _ in range(3)))
    suffix = f"{case_id}.nii.gz" if is_label else f"{case_id}_0000.nii.gz"
    nib.save(nii, out_dir / suffix)


def add_case(dirs, case_id, img_path, lbl_path, target_spacing, split, report, mapping, cohort,
             assume_spacing=None):
    arr, msk, affine, info = resample_pair(img_path, lbl_path, target_spacing,
                                           assume_spacing=assume_spacing)
    img_dir = dirs["imagesTr"] if split == "train" else dirs["imagesTs"]
    lbl_dir = dirs["labelsTr"] if split == "train" else dirs["labelsTs_eval"]
    write_case(img_dir, case_id, arr, affine, target_spacing, is_label=False)
    write_case(lbl_dir, case_id, msk, affine, target_spacing, is_label=True)

    info.update({"case": case_id, "cohort": cohort, "split": split})
    report.append(info)
    mapping[case_id] = {"cohort": cohort, "split": split,
                        "image": str(img_path), "label": str(lbl_path)}
    flag = " [spacing修正]" if info["spacing_overridden"] else ""
    chg = info["label_mm3_change_pct"]
    chg_s = f"  label {chg:+.1f}%" if chg is not None else ""
    print(f"  [{split:5s}] {case_id:24s} {info['in_shape']} -> {info['out_shape']}{chg_s}{flag}",
          flush=True)


def group_key_for(source_name):
    """Group together scans that may belong to one subject.

    119 MCSTU development files are named L<digits>-<n>; 20 of those prefixes carry 2-3
    files each, so 218 files cover at most ~196 subjects. A coarse image-correlation test
    could not separate same-prefix pairs from different-prefix controls (both ~0.7, since
    downsampled head CBCTs all look alike), so whether -1/-2 are repeat scans of one person
    is unresolved. Grouping them costs almost nothing and removes the risk of the same
    anatomy landing in both train and validation, which would inflate the fold's Dice.
    """
    stem = Path(source_name).name
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    m = re.match(r"^([A-Za-z]+\d+)-\d+$", stem)
    return m.group(1) if m else stem


def make_splits(mcstu_pairs, hosp_pairs, n_folds=5, seed=42):
    """5-fold CV, split by subject group, with hospital cases in every validation fold.

    Each argument is a list of (case_id, group_key). A plain random split over 218 MCSTU +
    17 hospital cases can leave a fold with no hospital case at all, making that fold's
    validation Dice say nothing about the cohort the 0.85 target is measured on, so the two
    cohorts are chunked separately.
    """
    rng = random.Random(seed)

    def chunk(pairs):
        by_group = {}
        for cid, gk in pairs:
            by_group.setdefault(gk, []).append(cid)
        groups = sorted(by_group)
        rng.shuffle(groups)
        return [[c for g in groups[i::n_folds] for c in by_group[g]] for i in range(n_folds)]

    chunks_a, chunks_b = chunk(mcstu_pairs), chunk(hosp_pairs)
    all_cases = [cid for cid, _ in mcstu_pairs] + [cid for cid, _ in hosp_pairs]

    splits = []
    for i in range(n_folds):
        val = chunks_a[i] + chunks_b[i]
        val_set = set(val)
        train = [c for c in all_cases if c not in val_set]
        splits.append({"train": sorted(train), "val": sorted(val)})
    return splits


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mcstu", default=r"C:\Users\WIN11\mcstu_data\MCSTU")
    p.add_argument("--hospital", default="./data/hospital_nifti")
    p.add_argument("--out", default="./data/nnUNet_raw")
    p.add_argument("--target-spacing", type=float, default=0.4)
    p.add_argument("--assume-mcstu-spacing", type=float, default=0.18,
                   help="true voxel size for MCSTU cases whose header spacing is the unset "
                        "1.0mm default (124 of 218 development cases); set 0 to disable")
    p.add_argument("--n-hospital-test", type=int, default=6)
    p.add_argument("--hospital-test-keep", default="",
                   help="comma-separated PIDs that must stay in the hospital test set (so earlier "
                        "test numbers stay comparable); the rest of --n-hospital-test is drawn at random")
    p.add_argument("--dataset-name", default=DATASET_DIRNAME)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-mcstu", type=int, default=None,
                   help="use only the first N MCSTU development cases (debugging)")
    args = p.parse_args()

    mcstu_root = Path(args.mcstu)
    hosp_root = Path(args.hospital)
    ds = Path(args.out) / args.dataset_name
    assume = args.assume_mcstu_spacing or None

    dirs = {n: ds / n for n in ("imagesTr", "labelsTr", "imagesTs", "labelsTs_eval")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    report, mapping = [], {}

    # ---- hospital: fixed test selection, remainder trains -------------------
    hosp_ids = sorted(p.stem.replace(".nii", "") for p in (hosp_root / "images").glob("*.nii.gz"))
    rng = random.Random(args.seed)
    keep = [h for h in args.hospital_test_keep.split(",") if h]
    missing = [h for h in keep if h not in hosp_ids]
    if missing:
        raise SystemExit(f"--hospital-test-keep 裡有病人不在 {hosp_root}: {missing}")
    pool = [h for h in hosp_ids if h not in set(keep)]
    n_draw = max(0, min(args.n_hospital_test - len(keep), len(pool)))
    hosp_test = sorted(keep + rng.sample(pool, n_draw))
    hosp_train = [h for h in hosp_ids if h not in set(hosp_test)]
    print(f"hospital: {len(hosp_train)} train / {len(hosp_test)} test")
    print(f"  test patients: {', '.join(hosp_test)}")

    hosp_train_cases, hosp_test_cases = [], []
    for pid in hosp_train:
        cid = f"HOSP_{pid}"
        add_case(dirs, cid, hosp_root / "images" / f"{pid}.nii.gz",
                 hosp_root / "labels" / f"{pid}.nii.gz",
                 args.target_spacing, "train", report, mapping, "hospital")
        hosp_train_cases.append((cid, pid))
    for pid in hosp_test:
        cid = f"HOSP_{pid}"
        add_case(dirs, cid, hosp_root / "images" / f"{pid}.nii.gz",
                 hosp_root / "labels" / f"{pid}.nii.gz",
                 args.target_spacing, "test", report, mapping, "hospital")
        hosp_test_cases.append(cid)

    # ---- MCSTU development -> train ----------------------------------------
    img_dir = mcstu_root / "development_dataset" / "img_nii"
    lbl_dir = mcstu_root / "development_dataset" / "labels"
    uids = sorted(q.name for q in img_dir.glob("*.nii.gz"))
    if args.limit_mcstu:
        uids = uids[: args.limit_mcstu]
    print(f"\nMCSTU development: {len(uids)} train")
    mcstu_train_cases = []
    for i, uid in enumerate(uids, start=1):
        cid = f"MCSTU_{i:03d}"
        add_case(dirs, cid, img_dir / uid, lbl_dir / uid,
                 args.target_spacing, "train", report, mapping, "public",
                 assume_spacing=assume)
        mcstu_train_cases.append((cid, group_key_for(uid)))

    # ---- MCSTU independent test -> test ------------------------------------
    test_root = mcstu_root / "independent_test_dataset"
    print(f"\nMCSTU independent test:")
    for vol in sorted(q for q in test_root.iterdir() if q.is_dir()):
        cid = f"MCSTU_TEST_{vol.name}"
        add_case(dirs, cid, vol / "image.nii.gz", vol / "label.nii.gz",
                 args.target_spacing, "test", report, mapping, "public",
                 assume_spacing=assume)

    # ---- metadata -----------------------------------------------------------
    with open(ds / "dataset.json", "w") as f:
        json.dump({
            "channel_names": {"0": "CBCT"},
            "labels": {"background": 0, "mandibular_canal": 1},
            "numTraining": len(mcstu_train_cases) + len(hosp_train_cases),
            "file_ending": ".nii.gz",
        }, f, indent=4)

    splits = make_splits(mcstu_train_cases, hosp_train_cases, seed=args.seed)
    n_groups = len({g for _, g in mcstu_train_cases})
    print(f"MCSTU: {len(mcstu_train_cases)} 檔 -> {n_groups} 個 subject 群組 (同前綴不拆開)")
    with open(ds / "splits_final.json", "w") as f:
        json.dump(splits, f, indent=2)

    with open(ds / "case_id_mapping.json", "w") as f:
        json.dump({"target_spacing": args.target_spacing,
                   "hospital_test_patients": hosp_test,
                   "cases": mapping}, f, indent=2)

    with open(ds / "build_report.json", "w") as f:
        json.dump(report, f, indent=2)

    n_tr = len(mcstu_train_cases) + len(hosp_train_cases)
    print(f"\ntraining cases : {n_tr}  ({len(mcstu_train_cases)} public + {len(hosp_train_cases)} hospital)")
    print(f"test cases     : {len(report) - n_tr}")
    for i, s in enumerate(splits):
        n_h = sum(1 for c in s["val"] if c.startswith("HOSP_"))
        print(f"  fold {i}: val={len(s['val']):3d} (hospital {n_h})")
    n_fixed = sum(1 for r in report if r.get("spacing_overridden"))
    print(f"spacing header 修正: {n_fixed} 個 case (header 寫 1.0mm -> 實際 {assume}mm)")
    changes = [r["label_mm3_change_pct"] for r in report if r.get("label_mm3_change_pct") is not None]
    if changes:
        print(f"label volume change: min {min(changes):+.1f}%  max {max(changes):+.1f}%  "
              f"mean {sum(changes)/len(changes):+.1f}%")
    print(f"\n-> {ds}")


if __name__ == "__main__":
    main()
