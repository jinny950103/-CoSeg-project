#!/usr/bin/env python3
"""Turn sparse edge-detection labels into DENSE labels (every voxel near the canal gets an answer).

Why: the edge labels mark only measured walls; unmeasured directions and everything farther than a
0.4 mm shell outside the wall are 'ignore'. That is right for TRAINING (never teach a guess), but a
TEST answer with ~90% ignore around the canal inflates Dice (a prediction drawn into ignore is never
penalised). The literature tests on dense labels, so the test answer has to be completed — and,
because completing means guessing, the completed answer must be reviewed by a doctor before it is
used as ground truth.

Per cross-section (the same sections the edge pipeline used, centre from rays/track/extrap npz):
  1. read the per-direction wall radius from the final edge label (canal 1/3/5 until bg 0/4/6;
     running into ignore = unknown)
  2. fill unknown directions by circular linear interpolation between known neighbours
     (sections with fewer than --min_known known directions stay unknown)
  3. rasterise: every voxel within --far_mm of the track is assigned to its nearest section,
     projected onto that section's plane; inside the filled contour = canal, outside = background.
Voxels beyond the canal ends, or assigned to a section that could not be filled, stay ignore (2).
Voxels farther than --far_mm from the track are background (0), as in the edge labels.

Output values: 0 background, 1 canal, 2 ignore.

    python scripts/densify_edge_labels.py --rays "outputs/mcstu_edge/test_sparse/chunk_*/*_rays.npz" \
        --label "outputs/mcstu_edge/test_extrap/{}_extrap.nii.gz" \
        --track outputs/mcstu_edge/test_track --extrap outputs/mcstu_edge/test_extrap \
        --out outputs/mcstu_edge/test_dense
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import canal_gap_fill as G  # noqa: E402
from make_patient_pdf import N_DIR, label_radii, section_centres  # noqa: E402


def fill_circular(r, min_known):
    ok = np.isfinite(r)
    if ok.sum() < min_known:
        return None
    if ok.all():
        return r.copy()
    a = np.arange(len(r))
    ka = a[ok]
    kr = r[ok]
    # periodic interpolation: pad known points one turn on each side
    xa = np.concatenate([ka - len(r), ka, ka + len(r)])
    xr = np.concatenate([kr, kr, kr])
    return np.interp(a, xa, xr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rays", required=True)
    ap.add_argument("--label", required=True, help="edge label path pattern, {} = case id")
    ap.add_argument("--track", required=True)
    ap.add_argument("--extrap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--far_mm", type=float, default=6.0)
    ap.add_argument("--min_known", type=int, default=N_DIR // 8, help="known directions needed to close a section")
    ap.add_argument("--end_mm", type=float, default=0.4, help="beyond the last section by more than this = ignore")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    summary = []
    for npz in sorted(glob.glob(args.rays)):
        cid, sp, sides = G.load_case(npz)
        img = nib.load(args.label.format(cid))
        lab = np.asanyarray(img.dataobj).astype(np.uint8)
        dense = np.zeros(lab.shape, np.uint8)

        cents, us, ws, ts, rads, okk = [], [], [], [], [], []
        n_sec = n_closed = 0
        for s, d in sorted(sides.items()):
            def z(stage):
                p = Path(getattr(args, stage)) / f"{cid}_s{s}_{stage}.npz"
                return dict(np.load(p)) if p.exists() else None
            C = section_centres(lab, d, (z("track"), z("extrap")))
            for k in range(len(C)):
                r = label_radii(lab, C[k], d["u"][k], d["w"][k], canal=(1, 3, 5), stop=(0, 4, 6), sp=sp)
                f = fill_circular(r, args.min_known)
                n_sec += 1
                n_closed += f is not None
                cents.append(C[k]); us.append(d["u"][k]); ws.append(d["w"][k]); ts.append(d["t"][k])
                rads.append(f if f is not None else np.full(N_DIR, np.nan)); okk.append(f is not None)
        cents, us, ws, ts = map(np.asarray, (cents, us, ws, ts))
        rads, okk = np.asarray(rads), np.asarray(okk)
        # mark the two end sections of each side (to stop the canal at its ends)
        side_id = np.concatenate([[s] * len(d["c"]) for s, d in sorted(sides.items())])
        is_end = np.zeros(len(cents), bool)
        for s in np.unique(side_id):
            idx = np.flatnonzero(side_id == s)
            is_end[idx[0]] = is_end[idx[-1]] = True

        far_v = args.far_mm / sp
        lo = np.maximum(np.floor(cents.min(0) - far_v - 1).astype(int), 0)
        hi = np.minimum(np.ceil(cents.max(0) + far_v + 2).astype(int), lab.shape)
        grid = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
        tree = cKDTree(cents)
        dist, j = tree.query(grid, k=1)
        near = dist <= far_v
        g, j = grid[near], j[near]
        v = g - cents[j]
        along = np.einsum("ij,ij->i", v, ts[j] / np.linalg.norm(ts[j], axis=1, keepdims=True)) * sp
        x = np.einsum("ij,ij->i", v, us[j]) * sp
        y = np.einsum("ij,ij->i", v, ws[j]) * sp
        rad = np.hypot(x, y)
        ang = np.mod(np.arctan2(y, x), 2 * np.pi) / (2 * np.pi) * N_DIR
        i0 = np.floor(ang).astype(int) % N_DIR
        i1 = (i0 + 1) % N_DIR
        fr = ang - np.floor(ang)
        R = rads[j, i0] * (1 - fr) + rads[j, i1] * fr
        val = np.where(rad <= R, 1, 0).astype(np.uint8)
        val[~okk[j]] = 2                                                     # section could not be closed
        val[is_end[j] & (np.abs(along) > args.end_mm)] = 2                   # beyond the canal ends
        dense[g[:, 0], g[:, 1], g[:, 2]] = val
        nib.save(nib.Nifti1Image(dense, img.affine, img.header), out / f"{cid}_dense.nii.gz")
        cnt = np.bincount(dense.ravel(), minlength=3)
        summary.append({"case": cid, "sections": n_sec, "closed": n_closed,
                        "canal_vox": int(cnt[1]), "ignore_vox": int(cnt[2]),
                        "edge_canal_vox": int(np.isin(lab, (1, 3, 5)).sum())})
        print(f"{cid}: sections {n_sec} closed {n_closed / n_sec:.3f}  canal {cnt[1]} (edge {summary[-1]['edge_canal_vox']})  ignore {cnt[2]}", flush=True)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
