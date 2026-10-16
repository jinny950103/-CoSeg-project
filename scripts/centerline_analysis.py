#!/usr/bin/env python3
"""How far is the predicted canal centreline from the annotated one, and does that cap Dice?

The hospital ground truth is a doctor's centreline dilated to a fixed 1.2mm radius, so for
those cases Dice is decided almost entirely by how far the two centrelines sit apart. For
two tubes of radius r whose axes are offset by d, the cross-sections are circles and

    Dice(d) = [2 r^2 acos(d / 2r) - (d / 2) sqrt(4 r^2 - d^2)] / (pi r^2)

which gives 0.89 at d = 0.2mm, 0.85 at d ~ 0.28mm and 0.79 at d = 0.4mm (r = 1.2mm). This
script measures d for every case, converts it into the Dice that offset alone allows, and
compares it with the real Dice. If they agree, the remaining error is centreline placement
and nothing else (thickness, fragments, missing segments) needs fixing.

With --images it also reads the CT value along each centreline. The canal is a dark
channel in bone, so whichever centreline runs through darker voxels is the one sitting in
the real canal. That tells model error apart from annotation error.

    # validation cases of a trained fold
    python scripts/centerline_analysis.py \
        --pred  $nnUNet_results/Dataset504_MixedCanalClean/nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres/fold_0/validation \
        --gt    $nnUNet_raw/Dataset504_MixedCanalClean/labelsTr \
        --images $nnUNet_raw/Dataset504_MixedCanalClean/imagesTr
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

RADIUS_MM = 1.2      # hospital label radius, measured over 60 patients (std 0.008mm)
MATCH_MM = 2.0       # centreline points farther apart than this are a length/branch
                     # difference, not an offset, and are reported separately


def tube_dice(d, r=RADIUS_MM):
    """Dice of two parallel tubes of radius r whose axes are d apart."""
    d = min(max(d, 0.0), 2 * r)
    inter = 2 * r * r * np.arccos(d / (2 * r)) - (d / 2) * np.sqrt(4 * r * r - d * d)
    return float(inter / (np.pi * r * r))


def bbox(*masks, margin=8):
    any_ = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        any_ |= m
    idx = np.argwhere(any_)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin + 1, any_.shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def analyse(pred, gt, spacing, img=None):
    sl = bbox(pred, gt)
    p, g = pred[sl], gt[sl]
    sp, sg = skeletonize(p).astype(bool), skeletonize(g).astype(bool)

    # distance (mm) from every point of one centreline to the nearest point of the other
    d_g2p = distance_transform_edt(~sp, sampling=spacing)[sg]
    d_p2g = distance_transform_edt(~sg, sampling=spacing)[sp]
    both = np.concatenate([d_g2p, d_p2g])
    matched = both[both <= MATCH_MM]

    inter = np.sum(p & g)
    dice = 2 * inter / (p.sum() + g.sum())
    off = float(np.mean(matched)) if matched.size else float("nan")
    row = {
        "dice": float(dice),
        "offset_mean_mm": off,
        "offset_median_mm": float(np.median(matched)) if matched.size else float("nan"),
        "offset_p90_mm": float(np.percentile(matched, 90)) if matched.size else float("nan"),
        "dice_from_offset": tube_dice(off) if matched.size else float("nan"),
        # share of the annotated centreline the prediction misses entirely (gaps, short canal)
        "gt_unmatched_pct": float(100 * np.mean(d_g2p > MATCH_MM)) if d_g2p.size else float("nan"),
        # share of the predicted centreline with no annotation nearby (extra branches, noise)
        "pred_unmatched_pct": float(100 * np.mean(d_p2g > MATCH_MM)) if d_p2g.size else float("nan"),
        "vol_ratio": float(p.sum() / max(g.sum(), 1)),
    }
    if img is not None:
        im = img[sl]
        row["ct_on_gt_line"] = float(im[sg].mean())
        row["ct_on_pred_line"] = float(im[sp].mean())
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--images", default=None, help="資料夾內是 <case>_0000.nii.gz")
    ap.add_argument("--prefix", default="HOSP_", help="只分析這個開頭的 case(空字串 = 全部)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pred_dir, gt_dir = Path(args.pred), Path(args.gt)
    preds = sorted(p for p in pred_dir.glob(f"{args.prefix}*.nii.gz"))
    if not preds:
        raise SystemExit(f"{pred_dir} 裡沒有 {args.prefix}*.nii.gz")

    rows = []
    hdr = f"{'case':22s} {'Dice':>6s} {'偏移mm':>7s} {'中位':>6s} {'p90':>6s} {'偏移→Dice':>9s} {'漏段%':>6s} {'多段%':>6s} {'體積比':>6s}"
    if args.images:
        hdr += f" {'CT標註線':>8s} {'CT預測線':>8s}"
    print(hdr)
    print("-" * len(hdr) * 1)
    for pf in preds:
        cid = pf.name[: -len(".nii.gz")]
        gf = gt_dir / pf.name
        if not gf.exists():
            print(f"{cid:22s} 找不到 ground truth,略過")
            continue
        gi = nib.load(gf)
        g = np.asanyarray(gi.dataobj) > 0
        p = np.asanyarray(nib.load(pf).dataobj) > 0
        if p.shape != g.shape:
            raise SystemExit(f"{cid}: 尺寸不符 {p.shape} vs {g.shape}")
        spacing = tuple(float(z) for z in gi.header.get_zooms()[:3])
        img = None
        if args.images:
            img = np.asanyarray(nib.load(Path(args.images) / f"{cid}_0000.nii.gz").dataobj)
        r = analyse(p, g, spacing, img)
        r["case"] = cid
        rows.append(r)
        line = (f"{cid:22s} {r['dice']:6.3f} {r['offset_mean_mm']:7.2f} {r['offset_median_mm']:6.2f} "
                f"{r['offset_p90_mm']:6.2f} {r['dice_from_offset']:9.3f} {r['gt_unmatched_pct']:6.1f} "
                f"{r['pred_unmatched_pct']:6.1f} {r['vol_ratio']:6.2f}")
        if img is not None:
            line += f" {r['ct_on_gt_line']:8.0f} {r['ct_on_pred_line']:8.0f}"
        print(line, flush=True)

    if not rows:
        return
    m = lambda k: float(np.nanmean([r[k] for r in rows]))
    print("\n平均")
    print(f"  Dice 實際            {m('dice'):.3f}")
    print(f"  中心線平均偏移       {m('offset_mean_mm'):.2f} mm   (要 0.85 需要 <= 0.28 mm)")
    print(f"  只看偏移推得的 Dice  {m('dice_from_offset'):.3f}")
    print(f"  標註線沒被預測到     {m('gt_unmatched_pct'):.1f} %")
    print(f"  預測線多出來的       {m('pred_unmatched_pct'):.1f} %")
    if args.images:
        darker = sum(r["ct_on_pred_line"] < r["ct_on_gt_line"] for r in rows)
        print(f"  CT 值:預測線比標註線暗(更像在管腔裡)的病例 {darker}/{len(rows)}")
        print(f"         標註線平均 {m('ct_on_gt_line'):.0f},預測線平均 {m('ct_on_pred_line'):.0f}")

    gap = m("dice_from_offset") - m("dice")
    print("\n解讀")
    if abs(gap) < 0.05:
        print("  實際 Dice ≈ 偏移推得的 Dice -> 誤差幾乎全來自中心線位置,不是粗細或碎塊。")
    else:
        print(f"  實際 Dice 比偏移推得的低 {gap:.3f} -> 除了中心線偏移,還有漏段/多段/粗細問題,看上面兩欄 %。")

    out = Path(args.out) if args.out else pred_dir / "centerline_analysis.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
