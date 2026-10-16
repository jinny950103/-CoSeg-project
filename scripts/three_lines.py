#!/usr/bin/env python3
"""Doctor's line vs model's line vs the darkest line in the canal, slice by slice.

The hospital label is one doctor click per slice dilated into a sphere, so on each slice
across the canal the label blob's centroid is the click. For every such slice and side we
take three in-plane points:

    label  centroid of the label blob                        (the doctor)
    pred   centroid of the overlapping prediction blob       (the model)
    dark   centroid of the darkest region near both          (the canal itself)

"dark" is searched in a 2mm window centred halfway between label and pred so neither is
favoured. Slices whose minimum sits on the window edge (no clear dark canal) are dropped.
Each side's three tracks are smoothed over +-4 slices, then the distances between them are
reported. If pred-dark is much smaller than label-dark, the model follows the canal and the
remaining Dice gap comes from where the doctor clicked.

    python scripts/three_lines.py \
        --pred   data/predictions/Dataset505_HospFT_fold0 \
        --gt     $nnUNet_raw/Dataset504_MixedCanalClean/labelsTs_eval \
        --images $nnUNet_raw/Dataset504_MixedCanalClean/imagesTs
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

W_MM = 2.0      # dark-centre search radius
SMOOTH = 4      # slices each side
MAX_PAIR_MM = 2.5


def dark_centre(img2d, c, w):
    cy, cx = np.round(c).astype(int)
    y0, x0 = cy - w, cx - w
    win = img2d[max(y0, 0):cy + w + 1, max(x0, 0):cx + w + 1]
    if win.shape != (2 * w + 1, 2 * w + 1):
        return None
    yy, xx = np.mgrid[-w:w + 1, -w:w + 1]
    disk = yy ** 2 + xx ** 2 <= w ** 2
    v = np.where(disk, win, np.inf)
    mi = np.unravel_index(np.argmin(v), v.shape)
    if yy[mi] ** 2 + xx[mi] ** 2 >= (w - 1) ** 2:
        return None
    vmin, vmed = v[mi], np.median(win[disk])
    lab, _ = ndimage.label(disk & (win <= vmin + 0.3 * (vmed - vmin)))
    return np.argwhere(lab == lab[mi]).mean(0) + [y0, x0]


def smooth(s, v):
    out = np.full_like(v, np.nan)
    for k, si in enumerate(s):
        m = np.abs(s - si) <= SMOOTH
        if m.sum() >= 5:
            out[k] = v[m].mean(0)
    return out


def analyse(img, g, p, spacing, ax):
    img, g, p = (np.moveaxis(a, ax, 0) for a in (img, g, p))
    sp = np.array([s for i, s in enumerate(spacing) if i != ax])
    w = int(round(W_MM / sp.mean()))
    lat = int(np.argmax(np.argwhere(g)[:, 1:].std(0)))   # in-plane left-right axis
    mid = np.median(np.argwhere(g)[:, 1 + lat])
    tracks = {0: [], 1: []}
    for i in np.unique(np.argwhere(g)[:, 0]):
        lg, ng = ndimage.label(g[i])
        lp, npp = ndimage.label(p[i])
        if ng == 0:
            continue
        pc = [np.argwhere(lp == k).mean(0) for k in range(1, npp + 1)]
        im = ndimage.gaussian_filter(img[max(i - 1, 0):i + 2].astype(np.float32).mean(0), 1.0)
        for k in range(1, ng + 1):
            gc = np.argwhere(lg == k).mean(0)
            if not pc:
                continue
            dist = [np.linalg.norm((c - gc) * sp) for c in pc]
            j = int(np.argmin(dist))
            if dist[j] > MAX_PAIR_MM:
                continue
            d = dark_centre(im, (gc + pc[j]) / 2, w)
            if d is None:
                continue
            tracks[int(gc[lat] > mid)].append((i, gc, pc[j], d))
    out = {"label_dark": [], "pred_dark": [], "label_pred": [], "n": 0}
    for side, T in tracks.items():
        if len(T) < 20:
            continue
        s = np.array([t[0] for t in T])
        G, P, D = (smooth(s, np.array([t[k] for t in T]) * sp) for k in (1, 2, 3))
        ok = ~np.isnan(G[:, 0])
        out["label_dark"] += list(np.linalg.norm(G[ok] - D[ok], axis=1))
        out["pred_dark"] += list(np.linalg.norm(P[ok] - D[ok], axis=1))
        out["label_pred"] += list(np.linalg.norm(G[ok] - P[ok], axis=1))
        out["n"] += int(ok.sum())
    return out, ax


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--prefix", default="HOSP_")
    ap.add_argument("--axis", type=int, default=1,
                    help="醫生逐張點選的那一軸(醫院資料是 1)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    preds = sorted(Path(args.pred).glob(f"{args.prefix}*.nii.gz"))
    if not preds:
        raise SystemExit(f"{args.pred} 裡沒有 {args.prefix}*.nii.gz")
    print(f"{'case':22s} {'軸':>2s} {'切片數':>6s} {'醫生-暗心':>9s} {'模型-暗心':>9s} {'醫生-模型':>9s}")
    rows, allv = [], {"label_dark": [], "pred_dark": [], "label_pred": []}
    for pf in preds:
        cid = pf.name[:-len(".nii.gz")]
        gi = nib.load(Path(args.gt) / pf.name)
        g = np.asanyarray(gi.dataobj) > 0
        p = np.asanyarray(nib.load(pf).dataobj) > 0
        img = np.asanyarray(nib.load(Path(args.images) / f"{cid}_0000.nii.gz").dataobj)
        ax = args.axis
        r, ax = analyse(img, g, p, tuple(float(z) for z in gi.header.get_zooms()[:3]), ax)
        if r["n"] == 0:
            print(f"{cid:22s} 沒有可用的切片")
            continue
        row = {"case": cid, "axis": ax, "n": r["n"],
               **{k: float(np.mean(r[k])) for k in allv}}
        rows.append(row)
        for k in allv:
            allv[k] += r[k]
        print(f"{cid:22s} {ax:2d} {r['n']:6d} {row['label_dark']:9.2f} "
              f"{row['pred_dark']:9.2f} {row['label_pred']:9.2f}", flush=True)
    if not rows:
        return
    print("\n全部切片(mm)       平均    中位    p90")
    for k, name in (("label_dark", "醫生-暗心"), ("pred_dark", "模型-暗心"), ("label_pred", "醫生-模型")):
        v = np.array(allv[k])
        print(f"  {name}        {v.mean():5.2f}   {np.median(v):5.2f}   {np.percentile(v, 90):5.2f}")
    closer = sum(r["pred_dark"] < r["label_dark"] for r in rows)
    print(f"\n模型比醫生更靠近暗心的病例: {closer}/{len(rows)}")
    out = Path(args.out) if args.out else Path(args.pred) / "three_lines.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
