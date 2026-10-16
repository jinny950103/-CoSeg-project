#!/usr/bin/env python3
"""Cross-section figures: CT + doctor's label + model prediction, to show which one sits in the canal.

The hospital labels were made by clicking one point per slice along array axis 1, so axis-1
slices cut the canal across: it appears as a small dark disc inside bright bone. Each panel
shows one such cross-section, cropped around the canal, with the doctor's label outlined in
green and the prediction in red. The '+' marks show each outline's centre. If the red centre
keeps landing on the dark lumen while the green one leans onto the bright wall, the remaining
Dice gap is in the label, not the model.

Output (default --out <pred>/figures):
    <case>.png      2 rows (left/right canal) x 4 positions along the canal (20/40/60/80%)
    summary.png     one mid-canal panel per case, for slides

Text is in English on purpose: the cluster has no CJK font, so Chinese would render as boxes.

    python scripts/make_overlay_figures.py \
        --pred   data/predictions/Dataset505_HospFT_fold0 \
        --gt     data/nnUNet_raw/Dataset504_MixedCanalClean/labelsTs_eval \
        --images data/nnUNet_raw/Dataset504_MixedCanalClean/imagesTs
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import label

AXIS = 1            # slice axis the doctor clicked along
HALF = 20           # crop half-width in voxels (20 x 0.4mm = 8mm each side)
POSITIONS = (0.2, 0.4, 0.6, 0.8)


def centroid(mask2d):
    idx = np.argwhere(mask2d)
    return idx.mean(0) if len(idx) else None


def panel(ax, img, gt, pred, y, cx, cz, spacing, title):
    """Draw one cropped cross-section at axis-1 index y, centred on (cx, cz)."""
    x0, x1 = max(int(cx) - HALF, 0), min(int(cx) + HALF, img.shape[0])
    z0, z1 = max(int(cz) - HALF, 0), min(int(cz) + HALF, img.shape[2])
    im = img[x0:x1, y, z0:z1].astype(np.float32)
    g = gt[x0:x1, y, z0:z1]
    p = pred[x0:x1, y, z0:z1]

    lo, hi = np.percentile(im, [1, 99.5])
    # transpose + origin=lower: horizontal = axis 0, vertical = axis 2
    ax.imshow(im.T, cmap="gray", vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
    if g.any():
        ax.contour(g.T.astype(float), levels=[0.5], colors="lime", linewidths=1.4, origin="lower")
    if p.any():
        ax.contour(p.T.astype(float), levels=[0.5], colors="red", linewidths=1.4, origin="lower")

    info = []
    cg, cp = centroid(g), centroid(p)
    if cg is not None:
        ax.plot(cg[0], cg[1], "+", color="lime", ms=10, mew=2)
        info.append(f"GT CT {im[g].mean():.0f}")
    if cp is not None:
        ax.plot(cp[0], cp[1], "+", color="red", ms=10, mew=2)
        info.append(f"Pred CT {im[p].mean():.0f}")
    if cg is not None and cp is not None:
        d = np.linalg.norm((cg - cp) * np.array([spacing[0], spacing[2]]))
        info.append(f"shift {d:.2f}mm")
    ax.set_title(title + "\n" + "  ".join(info), fontsize=7.5)
    ax.set_xticks([])
    ax.set_yticks([])


def canal_slices(gt):
    """For each canal (connected component), the axis-1 indices at POSITIONS along it."""
    lab, n = label(gt)
    out = []
    for c in range(1, n + 1):
        comp = lab == c
        if comp.sum() < 200:          # ignore specks
            continue
        ys = np.where(comp.any(axis=(0, 2)))[0]
        picks = [int(ys[int(round(q * (len(ys) - 1)))]) for q in POSITIONS]
        centre_x = np.argwhere(comp)[:, 0].mean()
        out.append((centre_x, comp, picks))
    out.sort(key=lambda t: t[0])      # stable left/right order by axis-0 position
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--prefix", default="HOSP_")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pred_dir = Path(args.pred)
    out = Path(args.out) if args.out else pred_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    cases = sorted(p.name[: -len(".nii.gz")] for p in pred_dir.glob(f"{args.prefix}*.nii.gz"))
    if not cases:
        raise SystemExit(f"{pred_dir} 裡沒有 {args.prefix}*.nii.gz")
    summary = []
    for cid in cases:
        gi = nib.load(Path(args.gt) / f"{cid}.nii.gz")
        gt = np.asanyarray(gi.dataobj) > 0
        pred = np.asanyarray(nib.load(pred_dir / f"{cid}.nii.gz").dataobj) > 0
        img = np.asanyarray(nib.load(Path(args.images) / f"{cid}_0000.nii.gz").dataobj)
        spacing = tuple(float(z) for z in gi.header.get_zooms()[:3])
        dice = 2 * (gt & pred).sum() / (gt.sum() + pred.sum())

        canals = canal_slices(gt)
        fig, axes = plt.subplots(len(canals), len(POSITIONS),
                                 figsize=(3 * len(POSITIONS), 3.8 * len(canals)), squeeze=False)
        for r, (_, comp, picks) in enumerate(canals):
            for c, y in enumerate(picks):
                cg = centroid(comp[:, y, :])
                panel(axes[r][c], img, gt, pred, y, cg[0], cg[1], spacing,
                      f"canal {r + 1}  {int(POSITIONS[c] * 100)}%  (slice {y})")
        fig.suptitle(f"{cid}   Dice {dice:.3f}   green = doctor label, red = model", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"{cid}.png", dpi=130)
        plt.close(fig)

        _, comp, picks = canals[0]
        y = picks[len(picks) // 2]
        summary.append((cid, dice, img, gt, pred, y, centroid(comp[:, y, :]), spacing))
        print(f"  {cid}  Dice {dice:.3f}  -> {out / (cid + '.png')}", flush=True)

    ncol = 4
    nrow = int(np.ceil(len(summary) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3.6 * nrow), squeeze=False)
    for i, (cid, dice, img, gt, pred, y, cg, spacing) in enumerate(summary):
        panel(axes[i // ncol][i % ncol], img, gt, pred, y, cg[0], cg[1], spacing,
              f"{cid}  Dice {dice:.3f}")
    for j in range(len(summary), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Mid-canal cross-sections   green = doctor label, red = model, + = centre",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=130)
    print(f"\n-> {out}  ({len(summary)} 位病人 + summary.png)")


if __name__ == "__main__":
    main()
