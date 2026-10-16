#!/usr/bin/env python3
"""How much do the doctor's clicks jump between neighbouring slices?

The hospital label is one click per slice (array axis 1) drawn as a ~1.2mm ball. Eroding the
label with that ball gives back (roughly) the click positions. On each side, for every three
consecutive slices, the middle click is compared with the midpoint of its two neighbours:

    d_i = c_i - (c_{i-1} + c_{i+1}) / 2

Over 0.8mm of canal the true centre is practically a straight line, so d_i is mostly click
noise. With independent noise of std s per in-plane axis, var(d) = 1.5 s^2 per axis, so
s = rms(d) / sqrt(2 * 1.5). The mean click error size (2D) is about 1.25 s.

Assumption: the canal does not wiggle within 0.8mm. Slow drift (the doctor sitting on one
side of the canal for many slices) is invisible here, so this is a lower bound on click noise.

Control: the same numbers on public dense labels (centroid per slice, no clicks) show how
much the canal itself bends at this scale.

    python scripts/click_jitter.py --gt <labels dir> [--prefix HOSP_] [--out result.json]
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def ball(r_vox2):
    rv = int(np.ceil(np.sqrt(r_vox2)))
    zz, yy, xx = np.mgrid[-rv:rv + 1, -rv:rv + 1, -rv:rv + 1]
    return zz ** 2 + yy ** 2 + xx ** 2 <= r_vox2


def side_tracks(m, ax):
    """{side: (slices, centroids in voxels)}; side split at the median of the widest in-plane axis."""
    q = np.moveaxis(m, ax, 0)
    pts = np.argwhere(q)
    if len(pts) == 0:
        return {}
    lat = 1 + int(np.argmax(pts[:, 1:].std(0)))
    mid = np.median(pts[:, lat])
    out = {}
    for side in (0, 1):
        sel = pts[(pts[:, lat] > mid) == bool(side)]
        s = np.unique(sel[:, 0])
        c = np.array([sel[sel[:, 0] == i, 1:].mean(0) for i in s])
        if len(s) >= 5:
            out[side] = (s, c)
    return out


def second_diff(tr, sp):
    """|d_i| in mm for every run of three consecutive slices."""
    d = []
    for s, c in tr.values():
        ok = (s[2:] - s[1:-1] == 1) & (s[1:-1] - s[:-2] == 1)
        dd = c[1:-1] - (c[:-2] + c[2:]) / 2
        d.append(np.linalg.norm(dd[ok] * sp, axis=1))
    return np.concatenate(d) if d else np.array([])


def summarise(d):
    s = np.sqrt(np.mean(d ** 2) / 3) if len(d) else np.nan     # per-axis noise std
    return {"n": int(len(d)), "d_mean": float(d.mean()), "d_median": float(np.median(d)),
            "sigma_axis": float(s), "click_err_mean": float(1.25 * s)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, nargs="+", help="標註資料夾(可給多個)")
    ap.add_argument("--prefix", default="HOSP_")
    ap.add_argument("--axis", type=int, default=1, help="醫生逐張點選的那一軸")
    ap.add_argument("--dense", action="store_true", help="密集標註(公開資料):不做侵蝕,直接取每張切片中心")
    ap.add_argument("--r2", type=float, default=8, help="球半徑平方(體素^2);醫院的 1.2mm 球實際是 <=8")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(f for d in args.gt for f in Path(d).glob(f"{args.prefix}*.nii.gz"))
    if not files:
        raise SystemExit("找不到標註檔")
    b = ball(args.r2)
    per_case, all_d = {}, []
    print(f"{'case':22s} {'重建Dice':>8s} {'三連片數':>8s} {'|d|平均':>8s} {'|d|中位':>8s} {'點擊誤差估計':>10s}")
    for f in files:
        cid = f.name[:-len(".nii.gz")]
        im = nib.load(f)
        g = np.asanyarray(im.dataobj) > 0
        spacing = np.array(im.header.get_zooms()[:3], float)
        sp = np.array([s for i, s in enumerate(spacing) if i != args.axis])
        if args.dense:
            pts, rec = g, np.nan
        else:
            pts = ndimage.binary_erosion(g, b)
            back = ndimage.binary_dilation(pts, b)
            rec = 2 * np.sum(back & g) / (back.sum() + g.sum())
        d = second_diff(side_tracks(pts, args.axis), sp)
        if len(d) == 0:
            continue
        r = summarise(d)
        r["reconstruct_dice"] = float(rec)
        per_case[cid] = r
        all_d.append(d)
        print(f"{cid:22s} {rec:8.3f} {r['n']:8d} {r['d_mean']:8.3f} {r['d_median']:8.3f} "
              f"{r['click_err_mean']:10.3f}", flush=True)

    tot = summarise(np.concatenate(all_d))
    print("-" * 72)
    print(f"全部 {len(per_case)} 例  三連片 {tot['n']}  |d| 平均 {tot['d_mean']:.3f}mm  "
          f"中位 {tot['d_median']:.3f}mm")
    print(f"每軸雜訊 s = {tot['sigma_axis']:.3f}mm  →  平均點擊誤差約 {tot['click_err_mean']:.3f}mm")
    if args.out:
        Path(args.out).write_text(json.dumps({"total": tot, "cases": per_case}, indent=2,
                                             ensure_ascii=False), encoding="utf-8")
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
