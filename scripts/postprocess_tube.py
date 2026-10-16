#!/usr/bin/env python3
"""A2: redraw the prediction the way the hospital label is drawn.

The hospital label is one point per slice (array axis 1) dilated into a 1.2mm sphere. This
takes the prediction's centroid on each slice and side, smooths that track over +-H slices,
and redraws spheres of radius r around it. Smoothing removes slice-to-slice wobble in the
prediction; r can be slightly above 1.2mm because, at a ~0.5mm centreline offset, a fatter
tube overlaps more of the label than it adds false positives.

Choose H and r on validation cases, then apply them unchanged to the test set:

    # 1) sweep on validation (prints mean hospital Dice for every H x r)
    python scripts/postprocess_tube.py --pred <val pred dir> --gt <labelsTr>
    # 2) apply the chosen pair to the test predictions and write them out
    python scripts/postprocess_tube.py --pred <test pred dir> --gt <labelsTs_eval> \
        --H 4 --r 1.3 --out <new dir>
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def tracks(p, ax):
    """{side: (slices, centroids)} from the prediction, one blob per side per slice."""
    q = np.moveaxis(p, ax, 0)
    pts = np.argwhere(q)
    if len(pts) == 0:
        return {}
    lat = 1 + int(np.argmax(pts[:, 1:].std(0)))
    mid = np.median(pts[:, lat])
    out = {0: [], 1: []}
    for i in np.unique(pts[:, 0]):
        lab, n = ndimage.label(q[i])
        if n == 0:
            continue
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        cents = ndimage.center_of_mass(np.ones_like(lab), lab, range(1, n + 1))
        best = {}
        for sz, c in zip(sizes, cents):
            side = int(c[lat - 1] > mid)
            if sz >= 4 and sz > best.get(side, (0, None))[0]:
                best[side] = (sz, c)
        for side, (_, c) in best.items():
            out[side].append((i, np.array(c)))
    return {s: (np.array([t[0] for t in T]), np.array([t[1] for t in T]))
            for s, T in out.items() if len(T) >= 5}


def smooth(s, c, H):
    if H == 0:
        return c
    out = np.empty_like(c)
    for k, si in enumerate(s):
        m = np.abs(s - si) <= H
        out[k] = c[m].mean(0)
    return out


def draw(shape, ax, tr, H, r_mm, spacing):
    q = np.zeros(np.moveaxis(np.zeros(shape, bool), ax, 0).shape, bool)
    sp = np.array([spacing[ax]] + [s for i, s in enumerate(spacing) if i != ax])
    rv = np.ceil(r_mm / sp).astype(int)
    zz, yy, xx = np.mgrid[-rv[0]:rv[0] + 1, -rv[1]:rv[1] + 1, -rv[2]:rv[2] + 1]
    for s, c in tr.values():
        cs = smooth(s, c, H)
        for si, (cy, cx) in zip(s, cs):
            iy, ix = int(round(cy)), int(round(cx))
            d2 = ((zz * sp[0]) ** 2 + ((yy + iy - cy) * sp[1]) ** 2 + ((xx + ix - cx) * sp[2]) ** 2)
            ball = d2 <= r_mm ** 2
            z0, y0, x0 = si - rv[0], iy - rv[1], ix - rv[2]
            sl = (slice(max(z0, 0), z0 + ball.shape[0]), slice(max(y0, 0), y0 + ball.shape[1]),
                  slice(max(x0, 0), x0 + ball.shape[2]))
            b = ball[max(-z0, 0):, max(-y0, 0):, max(-x0, 0):]
            tgt = q[sl]
            tgt |= b[:tgt.shape[0], :tgt.shape[1], :tgt.shape[2]]
    return np.moveaxis(q, 0, ax)


def dice(a, b):
    return 2 * np.sum(a & b) / max(a.sum() + b.sum(), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--prefix", default="HOSP_")
    ap.add_argument("--axis", type=int, default=1, help="醫生逐張點選的那一軸(醫院資料是 1)")
    ap.add_argument("--H", type=int, nargs="+", default=[0, 2, 4, 8])
    ap.add_argument("--r", type=float, nargs="+", default=[1.2, 1.3, 1.4])
    ap.add_argument("--out", default=None, help="給了就用第一組 H、r 寫出新的預測檔")
    args = ap.parse_args()

    preds = sorted(Path(args.pred).glob(f"{args.prefix}*.nii.gz"))
    if not preds:
        raise SystemExit(f"{args.pred} 裡沒有 {args.prefix}*.nii.gz")
    combos = [(H, r) for H in args.H for r in args.r]
    if args.out:
        combos = combos[:1]
        Path(args.out).mkdir(parents=True, exist_ok=True)
    hdr = f"{'case':22s} {'原始':>6s} " + " ".join(f"H{H}/r{r:g}".rjust(9) for H, r in combos)
    print(hdr)
    table = []
    for pf in preds:
        cid = pf.name[:-len(".nii.gz")]
        pi = nib.load(pf)
        p = np.asanyarray(pi.dataobj) > 0
        spacing = tuple(float(z) for z in pi.header.get_zooms()[:3])
        g = None
        if args.gt:
            g = np.asanyarray(nib.load(Path(args.gt) / pf.name).dataobj) > 0
        tr = tracks(p, args.axis)
        row = [dice(p, g) if g is not None else np.nan]
        for H, r in combos:
            q = draw(p.shape, args.axis, tr, H, r, spacing)
            row.append(dice(q, g) if g is not None else np.nan)
            if args.out:
                nib.save(nib.Nifti1Image(q.astype(np.uint8), pi.affine, pi.header),
                         str(Path(args.out) / pf.name))
        table.append(row)
        print(f"{cid:22s} " + " ".join(f"{v:9.4f}" if i else f"{v:6.4f}" for i, v in enumerate(row)),
              flush=True)
    if args.gt:
        m = np.mean(table, axis=0)
        print("-" * len(hdr))
        print(f"{'平均':22s} {m[0]:6.4f} " + " ".join(f"{v:9.4f}" for v in m[1:]))
        k = int(np.argmax(m[1:]))
        print(f"\n最好: H={combos[k][0]} r={combos[k][1]}  {m[1 + k]:.4f}  (原始 {m[0]:.4f}, "
              f"{m[1 + k] - m[0]:+.4f})")
    if args.out:
        print(f"-> {args.out}  (只寫了醫院病例;公開資料病例不受影響)")


if __name__ == "__main__":
    main()
