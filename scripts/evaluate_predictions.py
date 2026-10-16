#!/usr/bin/env python3
"""Score nnU-Net predictions with Dice, clDice and HD95, reported separately per cohort.

Reports hospital and public cases apart, because they are different targets: the hospital
set carries the Dice 0.85 goal while the public set is only there for comparison with the
literature. A single pooled average would be dominated by whichever cohort has more cases
and would hide the number that actually matters.

Metric choices worth knowing:
  Dice   overlap. Brutally sensitive for a ~2.4mm tube: at 0.4mm voxels a half-voxel
         systematic radius error already caps Dice near 0.85.
  clDice overlap of centrelines. The hospital ground truth *is* a dilated centreline, so
         this measures the thing the task actually turns on, and it tolerates small
         caliber error that Dice punishes.
  HD95   95th-percentile surface distance, in millimetres (spacing-aware, not voxels).
         Catches a single badly misplaced segment that Dice would average away.

Usage:
    python scripts/evaluate_predictions.py --pred <預測資料夾>
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

SMOOTH = 1e-5


def skeletonize_3d_safe(vol):
    """3D skeleton, degrading to slice-wise 2D if the 3D routine is unavailable."""
    try:
        from skimage.morphology import skeletonize
        return skeletonize(vol.astype(bool)).astype(np.uint8)
    except Exception:
        from skimage.morphology import skeletonize as sk2
        out = np.zeros_like(vol, dtype=np.uint8)
        for z in range(vol.shape[-1]):
            if vol[..., z].any():
                out[..., z] = sk2(vol[..., z].astype(bool)).astype(np.uint8)
        return out


def dice(pred, gt):
    inter = np.sum(pred & gt)
    return float((2 * inter + SMOOTH) / (pred.sum() + gt.sum() + SMOOTH))


def cldice(pred, gt):
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0
    sp, sg = skeletonize_3d_safe(pred), skeletonize_3d_safe(gt)
    if not sp.any() or not sg.any():
        return 0.0
    tprec = (sp & gt).sum() / (sp.sum() + SMOOTH)
    tsens = (sg & pred).sum() / (sg.sum() + SMOOTH)
    return float(2 * tprec * tsens / (tprec + tsens + SMOOTH))


def hd95(pred, gt, spacing):
    """95th-percentile symmetric surface distance in mm."""
    if not pred.any() or not gt.any():
        return float("nan")
    ps = pred ^ binary_erosion(pred)
    gs = gt ^ binary_erosion(gt)
    if not ps.any() or not gs.any():
        return float("nan")
    # sampling=spacing -> distances come out in mm rather than voxels
    dist_to_pred = distance_transform_edt(~pred, sampling=spacing)
    dist_to_gt = distance_transform_edt(~gt, sampling=spacing)
    both = np.concatenate([dist_to_gt[ps], dist_to_pred[gs]])
    return float(np.percentile(both, 95))


def components(mask):
    from scipy.ndimage import label
    return int(label(mask)[1])


def cohort_of(case_id):
    if case_id.startswith("HOSP_"):
        return "醫院"
    if case_id.startswith("MCSTU_"):
        return "公開"
    return "其他"


def summarise(values):
    v = [x for x in values if not np.isnan(x)]
    if not v:
        return None
    v = np.asarray(v)
    return {"n": len(v), "mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="nnUNetv2_predict 的輸出資料夾")
    ap.add_argument("--gt", default="./data/nnUNet_raw/Dataset503_MixedCanal/labelsTs_eval")
    ap.add_argument("--out", default=None, help="結果 json 的存放路徑(預設放在 --pred 裡)")
    args = ap.parse_args()

    pred_dir, gt_dir = Path(args.pred), Path(args.gt)
    out_path = Path(args.out) if args.out else pred_dir / "evaluation.json"

    gts = sorted(gt_dir.glob("*.nii.gz"))
    if not gts:
        raise SystemExit(f"找不到 ground truth: {gt_dir}")

    rows, missing = [], []
    print(f"{'case':26s} {'cohort':6s} {'Dice':>7s} {'clDice':>7s} {'HD95(mm)':>9s} {'元件數':>6s}")
    print("-" * 70)

    for g in gts:
        cid = g.name[: -len(".nii.gz")]
        p = pred_dir / g.name
        if not p.exists():
            missing.append(cid)
            continue

        gi, pi = nib.load(g), nib.load(p)
        gm = np.asanyarray(gi.dataobj) > 0
        pm = np.asanyarray(pi.dataobj) > 0
        if gm.shape != pm.shape:
            raise SystemExit(f"{cid}: 預測與 ground truth 尺寸不符 {pm.shape} vs {gm.shape}")
        spacing = tuple(float(z) for z in gi.header.get_zooms()[:3])

        r = {
            "case": cid,
            "cohort": cohort_of(cid),
            "dice": dice(pm, gm),
            "cldice": cldice(pm, gm),
            "hd95_mm": hd95(pm, gm, spacing),
            # 真實下頜管左右各一條,預測出很多碎塊代表有雜訊假陽性
            "n_components": components(pm),
            "gt_mm3": float(gm.sum() * np.prod(spacing)),
            "pred_mm3": float(pm.sum() * np.prod(spacing)),
        }
        rows.append(r)
        print(f"{cid:26s} {r['cohort']:6s} {r['dice']:7.4f} {r['cldice']:7.4f} "
              f"{r['hd95_mm']:9.2f} {r['n_components']:6d}")

    if missing:
        print(f"\n!!! 缺少預測的 case ({len(missing)}): {', '.join(missing)}")

    print("\n" + "=" * 70)
    print("分 cohort 總結")
    print("=" * 70)
    by = defaultdict(list)
    for r in rows:
        by[r["cohort"]].append(r)

    summary = {}
    for coh in ("醫院", "公開", "其他"):
        if coh not in by:
            continue
        g = by[coh]
        summary[coh] = {m: summarise([x[m] for x in g])
                        for m in ("dice", "cldice", "hd95_mm")}
        s = summary[coh]
        print(f"\n[{coh}]  n={len(g)}")
        for m, label in (("dice", "Dice    "), ("cldice", "clDice  "), ("hd95_mm", "HD95(mm)")):
            d = s[m]
            if d:
                print(f"  {label} 平均 {d['mean']:.4f} ± {d['std']:.4f}   "
                      f"中位數 {d['median']:.4f}   範圍 {d['min']:.4f} ~ {d['max']:.4f}")
        over = sum(1 for x in g if x["pred_mm3"] > 1.5 * x["gt_mm3"])
        frag = sum(1 for x in g if x["n_components"] > 4)
        if over:
            print(f"  ⚠ {over}/{len(g)} 位預測體積超過 ground truth 1.5 倍(過度分割)")
        if frag:
            print(f"  ⚠ {frag}/{len(g)} 位預測出超過 4 個分離元件(可能有假陽性碎塊)")

    if "醫院" in summary and summary["醫院"]["dice"]:
        d = summary["醫院"]["dice"]["mean"]
        print("\n" + "=" * 70)
        print(f"醫院 Dice = {d:.4f}   目標 0.85   -> {'達標' if d >= 0.85 else f'差 {0.85 - d:.4f}'}")
        print("=" * 70)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"per_case": rows, "summary": summary, "missing": missing},
                  f, indent=2, ensure_ascii=False)
    print(f"\n結果已存到 {out_path}")


if __name__ == "__main__":
    main()
