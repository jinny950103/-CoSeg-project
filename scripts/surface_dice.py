#!/usr/bin/env python3
"""Surface Dice (Normalized Surface Distance, NSD) at several mm tolerances, per cohort.

Surface Dice is the fraction of both surfaces that lies within a tolerance tau of the other
surface (Nikolov et al. 2021, "Clinically applicable segmentation of head and neck anatomy").
Unlike volumetric Dice it does not punish a boundary that is off by less than tau. That suits
the hospital labels, which are dilated doctor clicks whose centreline itself wanders ~0.5mm
from the canal, so a sub-millimetre boundary disagreement is within label uncertainty.

Report it next to Dice, never instead of it, and always state the tolerance.

Usage:
    python scripts/surface_dice.py --pred <預測資料夾> --gt <labelsTs_eval>
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

TOLERANCES_MM = (0.4, 0.8, 1.0)


def surface(mask):
    return mask ^ binary_erosion(mask)


def surface_dice(pred, gt, spacing, taus):
    if not pred.any() and not gt.any():
        return {t: 1.0 for t in taus}
    if not pred.any() or not gt.any():
        return {t: 0.0 for t in taus}
    # 只在兩者聯集的外框(加上最大容許距離)裡算,整張 500³ 做距離轉換太慢
    margin = int(np.ceil(max(taus) / min(spacing))) + 2
    idx = np.argwhere(pred | gt)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = idx.max(0) + margin + 1
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    sp, sg = surface(pred[sl]), surface(gt[sl])
    # sampling=spacing -> 距離單位是 mm
    d_to_g = distance_transform_edt(~sg, sampling=spacing)[sp]
    d_to_p = distance_transform_edt(~sp, sampling=spacing)[sg]
    n = sp.sum() + sg.sum()
    # header 裡的 spacing 是 0.40000000596,剛好一格的距離會比 0.4 大一點點,要留浮點誤差
    eps = 1e-4
    return {t: float(((d_to_g <= t + eps).sum() + (d_to_p <= t + eps).sum()) / n) for t in taus}


def cohort_of(case_id):
    if case_id.startswith("HOSP_"):
        return "醫院"
    if case_id.startswith("MCSTU_"):
        return "公開"
    return "其他"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="nnUNetv2_predict 的輸出資料夾")
    ap.add_argument("--gt", default="./data/nnUNet_raw/Dataset504_MixedCanalClean/labelsTs_eval")
    ap.add_argument("--out", default=None, help="結果 json(預設放在 --pred 裡)")
    args = ap.parse_args()

    pred_dir, gt_dir = Path(args.pred), Path(args.gt)
    out_path = Path(args.out) if args.out else pred_dir / "surface_dice.json"
    gts = sorted(gt_dir.glob("*.nii.gz"))
    if not gts:
        raise SystemExit(f"找不到 ground truth: {gt_dir}")

    head = "".join(f"  SD@{t}mm" for t in TOLERANCES_MM)
    print(f"{'case':26s} {'cohort':6s}{head}")
    print("-" * (34 + 10 * len(TOLERANCES_MM)))
    rows, missing = [], []
    for g in gts:
        cid = g.name[: -len(".nii.gz")]
        p = pred_dir / g.name
        if not p.exists():
            missing.append(cid)
            continue
        gi = nib.load(g)
        gm = np.asanyarray(gi.dataobj) > 0
        pm = np.asanyarray(nib.load(p).dataobj) > 0
        if gm.shape != pm.shape:
            raise SystemExit(f"{cid}: 預測與 ground truth 尺寸不符 {pm.shape} vs {gm.shape}")
        spacing = tuple(float(z) for z in gi.header.get_zooms()[:3])
        sd = surface_dice(pm, gm, spacing, TOLERANCES_MM)
        rows.append({"case": cid, "cohort": cohort_of(cid),
                     **{f"sd_{t}mm": v for t, v in sd.items()}})
        print(f"{cid:26s} {cohort_of(cid):6s}" + "".join(f"  {sd[t]:8.4f}" for t in TOLERANCES_MM))

    if missing:
        print(f"\n!!! 缺少預測的 case ({len(missing)}): {', '.join(missing)}")

    by = defaultdict(list)
    for r in rows:
        by[r["cohort"]].append(r)
    summary = {}
    print("\n" + "=" * 70)
    for coh in ("醫院", "公開", "其他"):
        if coh not in by:
            continue
        print(f"[{coh}]  n={len(by[coh])}")
        summary[coh] = {}
        for t in TOLERANCES_MM:
            v = np.array([r[f"sd_{t}mm"] for r in by[coh]])
            summary[coh][f"sd_{t}mm"] = {"mean": float(v.mean()), "std": float(v.std()),
                                         "median": float(np.median(v)),
                                         "min": float(v.min()), "max": float(v.max())}
            print(f"  Surface Dice @{t}mm  平均 {v.mean():.4f} ± {v.std():.4f}   "
                  f"中位數 {np.median(v):.4f}   範圍 {v.min():.4f} ~ {v.max():.4f}")
    print("=" * 70)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"tolerances_mm": TOLERANCES_MM, "per_case": rows,
                   "summary": summary, "missing": missing}, f, indent=2, ensure_ascii=False)
    print(f"\n結果已存到 {out_path}")


if __name__ == "__main__":
    main()
