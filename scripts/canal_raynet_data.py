#!/usr/bin/env python3
"""Section patches + per-direction wall radii for the ray network (canal_raynet.py).

For every section of every doctor track (canal_sparse_label.py --save_rays): a small stack of
planes perpendicular to the track, centred on the labeller's start point (darkest point within
1.2mm of the click), and the 72 radii the strict labeller kept there (nan = not measured).
Only this hospital's images and our own edge-detection radii are used.

    python scripts/canal_raynet_data.py --rays outputs/sparse_label_v6_rays/chunk_0*/*_rays.npz --out outputs/raynet/data.npz
"""
import argparse
from case_lists import load_cases
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage



def load_case(npz):
    """Same as canal_gap_fill.load_case (kept here so nano4 needs only numpy/scipy/nibabel)."""
    z = np.load(npz)
    cid = Path(npz).name.replace("_rays.npz", "")
    sides = {}
    for k in z.files:
        if k != "sp":
            sd, name = k.split("_", 1)
            sides.setdefault(int(sd[1:]), {})[name] = z[k]
    return cid, float(z["sp"]), sides


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rays", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", nargs="*", default=load_cases())
    ap.add_argument("--n_px", type=int, default=64)
    ap.add_argument("--px_mm", type=float, default=0.1)
    ap.add_argument("--planes", type=int, default=5, help="沿管方向的平面數(間隔 0.4mm)")
    ap.add_argument("--img", default="data/hospital_nifti_clean/images/{}.nii.gz",
                    help="影像路徑格式;nano4: data/nnUNet_raw/Dataset504_MixedCanalClean/imagesTr/HOSP_{}_0000.nii.gz")
    ap.add_argument("--splits", default="data/nnUNet_raw/Dataset504_MixedCanalClean/splits_final.json")
    args = ap.parse_args()
    folds = json.load(open(args.splits))
    fold_of = {c[5:]: k for k, f in enumerate(folds) for c in f["val"] if c.startswith("HOSP_")}
    X, Y, meta = [], [], []
    for npz in sorted(args.rays):
        cid, sp, sides = load_case(npz)
        if cid in args.skip:
            continue
        vol = np.asanyarray(nib.load(args.img.format(cid)).dataobj).astype(np.float32)
        p1, p99 = np.percentile(vol[::4, ::4, ::4], [1, 99])
        vol = ndimage.gaussian_filter(vol, 0.2 / sp)
        g = (np.arange(args.n_px) - (args.n_px - 1) / 2) * args.px_mm / sp
        a, b = np.meshgrid(g, g, indexing="ij")
        offs = (np.arange(args.planes) - (args.planes - 1) / 2) * 0.4 / sp
        for s, d in sides.items():
            for i in range(len(d["c"])):
                c, u, w, t = d["c"][i], d["u"][i], d["w"][i], d["t"][i]
                xyz = (c[:, None, None, None] + t[:, None, None, None] * offs[None, :, None, None]
                       + u[:, None, None, None] * a[None, None] + w[:, None, None, None] * b[None, None])
                patch = ndimage.map_coordinates(vol, xyz.reshape(3, -1), order=1, mode="nearest")
                X.append(((patch.reshape(args.planes, args.n_px, args.n_px) - p1) / (p99 - p1)).astype(np.float16))
                Y.append(d["r_agree"][i].astype(np.float32))
                meta.append((int(cid), s, i, fold_of[cid]))
        print(cid[-4:], len(X), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, X=np.array(X), Y=np.array(Y), meta=np.array(meta, np.int64))


if __name__ == "__main__":
    main()
