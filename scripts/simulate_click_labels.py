#!/usr/bin/env python3
"""Experiment 3: how much Dice does a ~0.5mm click wander cost on its own?

Turns dense public labels into hospital-style labels and asks what Dice a *perfect* model
could get against them. Per slice along the click axis, each side's GT centroid plays the
doctor's click; those points are drawn as 1.2mm spheres exactly like the hospital labels
(the "clean" label). Then the click track is moved by a smooth random wander of a chosen
mean size and redrawn (the "doctor" label). Dice(clean, doctor) is the ceiling for a model
that finds the true centre perfectly, because the wander carries no image information.

If a 0.5mm wander alone gives Dice ~0.72, the hospital result (offset 0.53mm, Dice 0.728)
is explained by the click wander, with no need for harder images or less data.

With --pred, model predictions are also scored against the clean and doctor labels.

    python scripts/simulate_click_labels.py --gt <labelsTs_eval>
    python scripts/simulate_click_labels.py --gt <labelsTs_eval> --pred <預測資料夾>
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter1d

from postprocess_tube import dice, draw, tracks


def wander(tr, mean_mm, corr, spacing_vox_mm, rng):
    """Same tracks, each centroid moved in-plane by a smooth random offset of mean size mean_mm."""
    out = {}
    for side, (s, c) in tr.items():
        if mean_mm == 0:
            out[side] = (s, c)
            continue
        n = rng.standard_normal((len(s), 2))
        if corr > 0:
            n = gaussian_filter1d(n, corr, axis=0, mode="nearest")
        # 依照平均位移長度縮放,所以每一條線的平均偏差剛好是 mean_mm
        n *= (mean_mm / spacing_vox_mm) / np.linalg.norm(n, axis=1).mean()
        out[side] = (s, c + n)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="密集標註的資料夾(公開資料)")
    ap.add_argument("--pred", default=None, help="模型預測資料夾(選填)")
    ap.add_argument("--prefix", default="MCSTU_")
    ap.add_argument("--axis", type=int, default=1, help="逐張點選的那一軸(醫院和公開資料都是 1)")
    ap.add_argument("--r", type=float, default=1.2, help="醫院標註的球半徑 mm")
    ap.add_argument("--wander", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1.0],
                    help="點擊平均偏差 mm")
    ap.add_argument("--corr", type=float, nargs="+", default=[8],
                    help="偏差的平滑程度(幾張切片);越大越像緩慢飄移")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=None, help="結果 json")
    args = ap.parse_args()

    gts = sorted(Path(args.gt).glob(f"{args.prefix}*.nii.gz"))
    if not gts:
        raise SystemExit(f"{args.gt} 裡沒有 {args.prefix}*.nii.gz")
    combos = [(w, c) for c in args.corr for w in args.wander]
    res = {"ceiling": {k: [] for k in combos}, "pred": {k: [] for k in combos},
           "format": [], "pred_dense": []}

    for gf in gts:
        cid = gf.name[:-len(".nii.gz")]
        gi = nib.load(gf)
        g = np.asanyarray(gi.dataobj) > 0
        spacing = tuple(float(z) for z in gi.header.get_zooms()[:3])
        sp_inplane = float(np.mean([s for i, s in enumerate(spacing) if i != args.axis]))
        tr = tracks(g, args.axis)
        clean = draw(g.shape, args.axis, tr, 0, args.r, spacing)
        res["format"].append(dice(clean, g))
        p = None
        if args.pred:
            p = np.asanyarray(nib.load(Path(args.pred) / gf.name).dataobj) > 0
            res["pred_dense"].append(dice(p, g))
        line = f"{cid:20s} 球形 vs 原標註 {res['format'][-1]:.3f}"
        for w, c in combos:
            ceil, pr = [], []
            for seed in range(args.seeds if w > 0 else 1):
                rng = np.random.default_rng(seed)
                doc = draw(g.shape, args.axis, wander(tr, w, c, sp_inplane, rng), 0, args.r, spacing)
                ceil.append(dice(clean, doc))
                if p is not None:
                    pr.append(dice(p, doc))
            res["ceiling"][(w, c)].append(float(np.mean(ceil)))
            if p is not None:
                res["pred"][(w, c)].append(float(np.mean(pr)))
        line += "  完美模型: " + " ".join(f"{np.mean(res['ceiling'][k][-1]):.3f}" for k in combos)
        print(line, flush=True)

    n = len(gts)
    print("\n" + "=" * 72)
    print(f"公開資料 n={n}，醫院式標註(每張切片一點 → {args.r}mm 球)")
    print(f"  球形標註 vs 原本密集標註 Dice: {np.mean(res['format']):.3f}"
          "  (只是換畫法造成的差異)")
    if args.pred:
        print(f"  模型 vs 原本密集標註 Dice: {np.mean(res['pred_dense']):.3f}")
    print("-" * 72)
    hdr = f"{'點擊平均偏差':>12s} {'平滑':>5s} {'完美模型 Dice':>14s}"
    if args.pred:
        hdr += f" {'實際模型 Dice':>14s}"
    print(hdr)
    for w, c in combos:
        v = np.array(res["ceiling"][(w, c)])
        row = f"{w:10.2f}mm {c:5g} {v.mean():8.3f} ± {v.std():.3f}"
        if args.pred:
            q = np.array(res["pred"][(w, c)])
            row += f" {q.mean():8.3f} ± {q.std():.3f}"
        print(row)
    print("-" * 72)
    print("對照:醫院實際  模型–醫生線偏差 0.53mm,Dice 0.728(A1)")
    print("=" * 72)

    if args.out:
        dump = {"n": n, "r_mm": args.r, "format_dice": res["format"], "pred_dense": res["pred_dense"],
                "rows": [{"wander_mm": w, "corr": c, "ceiling": res["ceiling"][(w, c)],
                          "pred": res["pred"][(w, c)]} for w, c in combos]}
        Path(args.out).write_text(json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
