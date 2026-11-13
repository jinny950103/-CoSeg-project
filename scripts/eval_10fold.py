#!/usr/bin/env python3
"""Score nnU-Net runs against the EXPERT labels (MCSTU) and the doctor's clicks (hospital).

The edge-detection labels are only used for training; the score uses answers the model never
learned from:
  cv     out-of-fold predictions of the 218 MCSTU development cases (each predicted by the fold that
         did not train on it) vs MCSTU's own expert full labels (Dataset504 labelsTr, 0/1).
         With --splits the mean is also given per fold (mean +- sd over folds, the usual k-fold number).
  test   ensemble predictions of the 18 MCSTU independent-test cases vs their expert labels
         (Dataset504 labelsTs_eval/MCSTU_TEST_*), and of the 11 hospital test patients vs the doctor's
         clicks (labelsTs_eval/HOSP_*): the doctor marked an approximate position only, so the hospital
         numbers are agreement with that position (distance from each clicked slice's centre to the
         prediction, share within 1 / 2 mm), NOT accuracy.
Metrics (MCSTU): Dice, HD95 and ASSD in mm (surface distances on the voxel grid, spacing from the header).
Two or more runs: the per-case difference to the first run, with a Wilcoxon signed-rank p-value.

    python scripts/eval_10fold.py \
        --cv v7_508=download508/Dataset508_MCSTUEdge_cv v11_510=download510/Dataset510_MCSTUEdgeV11_cv10 \
        --test v7_508=download508/Dataset508_MCSTUEdge_ens5 v11_510=download510/Dataset510_MCSTUEdgeV11_ens10 \
        --splits outputs/upload510/Dataset510_MCSTUEdgeV11/splits_10fold.json --out outputs/eval_510
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage, stats

GT = "data/nnUNet_raw/Dataset504_MixedCanalClean"


def surf(m):
    return m & ~ndimage.binary_erosion(m)


def metrics(p, g, sp):
    out = {"dice": float(2 * (p & g).sum() / (p.sum() + g.sum())) if (p.sum() + g.sum()) else float("nan")}
    if not p.any() or not g.any():
        out.update(hd95=float("nan"), assd=float("nan"))
        return out
    idx = np.argwhere(p | g); lo = np.maximum(idx.min(0) - 3, 0); hi = idx.max(0) + 4
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    p, g = p[sl], g[sl]
    sp_, sg = surf(p), surf(g)
    dg = ndimage.distance_transform_edt(~sg, sampling=sp)[sp_]       # pred surface -> gt surface
    dp = ndimage.distance_transform_edt(~sp_, sampling=sp)[sg]
    both = np.r_[dg, dp]
    out.update(hd95=float(np.percentile(both, 95)), assd=float(both.mean()))
    return out


def load(path):
    im = nib.load(path)
    return np.asanyarray(im.dataobj) > 0, tuple(float(x) for x in im.header.get_zooms()[:3])


def doctor_agreement(p, g, sp, axis=1):
    """Doctor clicks: one small ball per clicked slice along `axis`. Distance (mm) from each slice's
    click centre to the nearest predicted voxel."""
    if not p.any():
        return {"n_clicks": 0, "dist_median": float("nan"), "within1": float("nan"), "within2": float("nan")}
    dist = ndimage.distance_transform_edt(~p, sampling=sp)
    ds = []
    for k in np.unique(np.argwhere(g)[:, axis]):
        sl = [slice(None)] * 3; sl[axis] = k
        lab, n = ndimage.label(g[tuple(sl)])
        for i in range(1, n + 1):                          # left and right click on the same slice
            c = np.round(np.argwhere(lab == i).mean(0)).astype(int)
            q = list(c); q.insert(axis, k)
            ds.append(dist[tuple(q)])
    ds = np.array(ds)
    return {"n_clicks": int(len(ds)), "dist_median": float(np.median(ds)),
            "within1": float(np.mean(ds <= 1.0)), "within2": float(np.mean(ds <= 2.0))}


def parse(items):
    out = {}
    for it in items:
        name, path = it.split("=", 1)
        out[name] = Path(path)
    return out


def summarise(tag, rows, runs, keys, fold_of=None):
    print(f"\n== {tag} ==")
    names = list(runs)
    for name in names:
        x = [r for r in rows if r["run"] == name]
        if not x:
            print(f"  {name}: no cases"); continue
        line = f"  {name:12s} n={len(x):3d}"
        for k in keys:
            v = np.array([r[k] for r in x], float)
            line += f"  {k} {np.nanmean(v):.3f}+-{np.nanstd(v):.3f} (median {np.nanmedian(v):.3f})"
        print(line)
        if fold_of and "dice" in keys:
            per = {}
            for r in x:
                if r["case"] in fold_of:
                    per.setdefault(fold_of[r["case"]], []).append(r["dice"])
            fm = [np.nanmean(v) for _, v in sorted(per.items())]
            if len(fm) > 1:
                print(f"  {'':12s} Dice per fold: {' '.join(f'{m:.3f}' for m in fm)}  -> {np.mean(fm):.3f} +- {np.std(fm, ddof=1):.3f} over {len(fm)} folds")
    if len(names) > 1:
        base = {r["case"]: r for r in rows if r["run"] == names[0]}
        for name in names[1:]:
            for k in keys:
                pairs = [(r[k], base[r["case"]][k]) for r in rows if r["run"] == name and r["case"] in base]
                pairs = [(a, b) for a, b in pairs if np.isfinite(a) and np.isfinite(b)]
                if len(pairs) < 5:
                    continue
                d = np.array([a - b for a, b in pairs])
                p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                print(f"  {name} - {names[0]}  {k}: mean diff {d.mean():+.4f}  median {np.median(d):+.4f}  "
                      f"better in {np.mean(d > 0) if k not in ('hd95', 'assd', 'dist_median') else np.mean(d < 0):.0%} of {len(d)} cases  "
                      f"Wilcoxon p={p:.3g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv", nargs="*", default=[], help="name=folder of out-of-fold predictions MCSTU_###.nii.gz")
    ap.add_argument("--test", nargs="*", default=[], help="name=folder of test predictions (MCSTU_TEST_*, HOSP_*)")
    ap.add_argument("--gt", default=GT)
    ap.add_argument("--splits", default="", help="splits_10fold.json, for the per-fold mean")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    gt = Path(args.gt); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fold_of = {}
    if args.splits:
        for i, f in enumerate(json.load(open(args.splits))):
            for c in f["val"]:
                fold_of[c] = i
    res = {}

    cv = parse(args.cv); rows = []
    for name, d in cv.items():
        for pp in sorted(d.glob("MCSTU_[0-9]*.nii.gz")):
            cid = pp.name[:-7]
            g, sp = load(gt / "labelsTr" / pp.name)
            p, _ = load(pp)
            assert p.shape == g.shape, (name, cid, p.shape, g.shape)
            rows.append({"run": name, "case": cid, **metrics(p, g, sp)})
        print(f"cv {name}: {sum(r['run'] == name for r in rows)} cases", flush=True)
    if rows:
        summarise("CV: out-of-fold MCSTU development cases vs expert labels", rows, cv, ("dice", "hd95", "assd"), fold_of)
        res["cv"] = rows

    te = parse(args.test); rows_m, rows_h = [], []
    for name, d in te.items():
        for pp in sorted(d.glob("*.nii.gz")):
            cid = pp.name[:-7]
            gp = gt / "labelsTs_eval" / pp.name
            if not gp.exists():
                continue
            g, sp = load(gp); p, _ = load(pp)
            assert p.shape == g.shape, (name, cid, p.shape, g.shape)
            if cid.startswith("MCSTU_TEST"):
                rows_m.append({"run": name, "case": cid, **metrics(p, g, sp)})
            elif cid.startswith("HOSP_"):
                rows_h.append({"run": name, "case": cid, **doctor_agreement(p, g, sp)})
        print(f"test {name}: MCSTU {sum(r['run'] == name for r in rows_m)}, hospital {sum(r['run'] == name for r in rows_h)}", flush=True)
    if rows_m:
        summarise("TEST: 18 MCSTU independent cases vs expert labels", rows_m, te, ("dice", "hd95", "assd"))
        res["test_mcstu"] = rows_m
    if rows_h:
        summarise("HOSPITAL 11 (test only): agreement with the doctor's clicked positions, not accuracy",
                  rows_h, te, ("dist_median", "within1", "within2"))
        res["test_hosp"] = [{**r, "case": r["case"][-4:]} for r in rows_h]   # no full record numbers in the output
    (out / "scores.json").write_text(json.dumps(res, indent=0))
    print(f"\nwritten {out / 'scores.json'}")


if __name__ == "__main__":
    main()
