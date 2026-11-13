#!/usr/bin/env python3
"""Build the upload package for Dataset508_MCSTUEdge: MCSTU (218) edge-detection labels, optionally + hospital (v7).

2026-11-06: user wants MCSTU only first (a base model); hospital labels wait for the doctor's review,
so --hosp is optional and left out for now.

Both cohorts' labels come from the edge-detection pipeline (canal_sparse_label -> canal_track_fill ->
canal_extrap_fill), whose output values are
    0 measured background, 1 measured canal, 2 ignore,
    3 predicted canal (track), 4 predicted background shell,
    5 extrapolated canal,      6 extrapolated background shell.
User decision 2026-11-06: predicted and extrapolated canal are both taught as canal, so
    1,3,5 -> 1    0,4,6 -> 0    2 -> 2 (nnU-Net ignore label)

Images are not uploaded: on nano4 they are hard-linked from Dataset504_MixedCanalClean/imagesTr
(same files, MCSTU already resampled to 0.4mm). images_md5.txt is computed here from the local 504
copy so the nano4 script can verify every file.

Splits: 504's splits_final.json restricted to the cases present (cases in --skip are dropped), so the
hospital validation patients of every fold stay the same as 504/505/506.

    python scripts/build_508_mcstu_edge.py \
        --mcstu "outputs/mcstu_edge/extrap/MCSTU_*_extrap.nii.gz" --out outputs/upload508
    v8 (2026-11-13, caps + confirmation, scripts/canal_cap_guesses.py):
    python scripts/build_508_mcstu_edge.py --mcstu "outputs/v8_mcstu/MCSTU_*_capped.nii.gz" --out outputs/upload509 --name Dataset509_MCSTUEdgeV8
    (later, after the doctor's review, a mixed version adds --hosp "outputs/extrap_fill_v7/*_extrap.nii.gz")
"""
import argparse
import glob
import hashlib
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np

from case_lists import load_cases

DNAME = "Dataset508_MCSTUEdge"
SRC504 = "data/nnUNet_raw/Dataset504_MixedCanalClean"
LUT = np.array([0, 1, 2, 1, 0, 1, 0], dtype=np.uint8)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mcstu", required=True, help="glob of MCSTU_###_extrap.nii.gz or MCSTU_###_capped.nii.gz")
    ap.add_argument("--hosp", default="", help="glob of <pid>_extrap.nii.gz (hospital v7); empty = MCSTU only")
    ap.add_argument("--src504", default=SRC504)
    ap.add_argument("--skip", nargs="*", default=load_cases(), help="hospital IDs left out (misaligned labels)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=DNAME, help="dataset folder name, e.g. Dataset509_MCSTUEdgeV8 for the v8 labels")
    args = ap.parse_args()

    src = Path(args.src504)
    dst = Path(args.out) / args.name
    (dst / "labelsTr").mkdir(parents=True, exist_ok=True)

    jobs = []
    for f in sorted(glob.glob(args.mcstu)):
        cid = re.match(r"(MCSTU_\d+)_(extrap|capped)", Path(f).name).group(1)
        jobs.append((cid, f))
    n_mcstu = len(jobs)
    for f in (sorted(glob.glob(args.hosp)) if args.hosp else []):
        pid = Path(f).name.split("_")[0]
        if pid in args.skip:
            print(f"  skip HOSP_{pid} (misaligned)")
            continue
        jobs.append((f"HOSP_{pid}", f))
    n_hosp = len(jobs) - n_mcstu
    print(f"MCSTU {n_mcstu}, HOSP {n_hosp}")

    report, md5_lines, tot = [], [], np.zeros(7, np.int64)
    for cid, f in jobs:
        img_p = src / "imagesTr" / f"{cid}_0000.nii.gz"
        assert img_p.exists(), f"504 has no image for {cid}"
        lab = nib.load(f)
        img = nib.load(img_p)
        assert lab.shape == img.shape, f"{cid}: label {lab.shape} vs image {img.shape}"
        assert np.allclose(lab.affine, img.affine, atol=1e-3), f"{cid}: affine differs from 504 image"
        a = np.asanyarray(lab.dataobj).astype(np.int64)
        assert a.min() >= 0 and a.max() <= 6, f"{cid}: values outside 0..6"
        cnt = np.bincount(a.ravel(), minlength=7)
        tot += cnt
        out = LUT[a]
        # write with the IMAGE's header geometry so nnU-Net sees identical affines
        nib.save(nib.Nifti1Image(out, img.affine, img.header), dst / "labelsTr" / f"{cid}.nii.gz")
        md5_lines.append(f"{md5(img_p)}  {img_p.name}")
        report.append({"case": cid, "src": f, "counts_0to6": cnt.tolist(),
                       "canal_vox": int(cnt[1] + cnt[3] + cnt[5]), "ignore_vox": int(cnt[2])})
        print(f"  {cid}: measured {cnt[1]}  predicted {cnt[3]}  extrap {cnt[5]}  ignore {cnt[2]}")

    have = {c for c, _ in jobs}
    splits = []
    for fo in json.load(open(src / "splits_final.json")):
        val = sorted(c for c in fo["val"] if c in have)
        splits.append({"train": sorted(have - set(val)), "val": val})
    assert sorted(c for fo in splits for c in fo["val"]) == sorted(have), "every case must be in exactly one val fold"

    json.dump({"channel_names": {"0": "CBCT"},
               "labels": {"background": 0, "canal": 1, "ignore": 2},
               "numTraining": len(jobs), "file_ending": ".nii.gz"},
              open(dst / "dataset.json", "w"), indent=2)
    json.dump(splits, open(dst / "splits_final.json", "w"), indent=2)
    (dst / "images_md5.txt").write_text("\n".join(md5_lines) + "\n", newline="\n")
    json.dump({"lut": "1,3,5->1; 0,4,6->0; 2->2", "n_mcstu": n_mcstu, "n_hosp": n_hosp,
               "totals_0to6": tot.tolist(), "cases": report},
              open(dst / "build_report.json", "w"), indent=2)

    print(f"\ntotal voxels: measured canal {tot[1]}, predicted {tot[3]}, extrap {tot[5]}, ignore {tot[2]}")
    for i, fo in enumerate(splits):
        nh = sum(c.startswith("HOSP_") for c in fo["val"])
        print(f"  fold {i}: train={len(fo['train'])} val={len(fo['val'])} (hosp val {nh})")
    print(f"written {dst}")


if __name__ == "__main__":
    main()
