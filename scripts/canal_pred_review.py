#!/usr/bin/env python3
"""What the sparse-label model (Dataset506) draws where the wall is unclear.

Sections perpendicular to the doctor click track every 0.4mm (same as canal_sparse_label.py).
Per section:
  labelled   the sparse label has canal (1) in this section -> the wall was clear here
  pred       the model draws canal within --near_mm of the track centre
Per case it writes a straightened view of each side (the canal pulled straight: one row per
section, cut through the centre along u and along w) with the sparse canal in red and the model
in green, and a gallery of unlabelled sections; plus one line of numbers per case.

    python scripts/canal_pred_review.py --cases <case_id> ... --out outputs/pred506/review
"""
import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
import canal_edge_gt as E  # noqa: E402

IMG = "data/hospital_nifti_clean/images/{}.nii.gz"
DOC = "data/hospital_nifti_clean/labels/{}.nii.gz"
SPARSE = "outputs/sparse_label_v5_train34/chunk_*/{}_agree.nii.gz"
PRED = "outputs/pred506/Dataset506_CanalSparse_cv/HOSP_{}.nii.gz"


def section(vol, c, u, w, half, px, order):
    g = np.arange(-half, half + 1e-6, px)
    a, b = np.meshgrid(g, g, indexing="ij")
    xyz = c[:, None, None] + u[:, None, None] * a + w[:, None, None] * b
    return ndimage.map_coordinates(vol, xyz, order=order, mode="nearest")


def review(cid, args):
    im = nib.load(IMG.format(cid))
    sp = float(im.header.get_zooms()[0]); mm = lambda x: x / sp
    doc = np.asanyarray(nib.load(DOC.format(cid)).dataobj) > 0
    lo = np.maximum(np.argwhere(doc).min(0) - int(mm(8)), 0)
    hi = np.minimum(np.argwhere(doc).max(0) + int(mm(8)) + 1, doc.shape)
    crop = tuple(slice(a, b) for a, b in zip(lo, hi))
    vol = np.asanyarray(im.dataobj)[crop].astype(np.float32)
    lab = np.asanyarray(nib.load(glob.glob(SPARSE.format(cid))[0]).dataobj)[crop]
    pred = (np.asanyarray(nib.load(PRED.format(cid)).dataobj)[crop] > 0).astype(np.uint8)

    half, px = mm(args.half_mm), mm(0.1)
    n = len(np.arange(-half, half + 1e-6, px)); mid = n // 2
    g = (np.arange(n) - mid) * 0.1
    near = np.hypot(*np.meshgrid(g, g, indexing="ij")) <= args.near_mm
    rows, gal = [], []
    strips = {}
    for side, pts in E.click_tracks(doc[crop]).items():
        cl, tan = E.resample_track(pts, mm(0.4), 1.2, sp)
        prev_u = None
        S = {k: [] for k in ("iu", "iw", "lu", "lw", "pu", "pw")}
        for i in range(len(cl)):
            u, w = E.plane_basis(tan[i], prev_u); prev_u = u
            si = section(vol, cl[i], u, w, half, px, 1)
            sl = section(lab, cl[i], u, w, half, px, 0)
            sq = section(pred, cl[i], u, w, half, px, 0)
            labelled = bool((sl == 1).any())
            drawn = bool((sq[near] > 0).any())
            diam = np.nan
            if drawn:
                cc, _ = ndimage.label(sq > 0)
                keep = np.isin(cc, np.unique(cc[near & (cc > 0)]))
                diam = 2 * np.sqrt(keep.sum() * 0.01 / np.pi)     # equal-area diameter, mm
            rows.append((side, i, labelled, drawn, diam))
            for k, a in (("i", si), ("l", sl), ("p", sq)):
                S[k + "u"].append(a[:, mid]); S[k + "w"].append(a[mid, :])
            if not labelled:
                gal.append((side, i, si, sq, sl))
        strips[side] = {k: np.array(v) for k, v in S.items()}
    return rows, gal, strips


def overlay(ax, img, lab, pred, aspect):
    ax.imshow(img, cmap="gray", aspect=aspect, interpolation="nearest")
    m = np.zeros(img.shape + (4,)); m[lab == 1] = [1, 0, 0, .55]
    ax.imshow(m, aspect=aspect, interpolation="nearest")
    if pred.any():
        ax.contour(pred.astype(float), [.5], colors="lime", linewidths=.8)


def save_figs(cid, rows, gal, strips, out, args):
    tag = cid[-4:]
    fig, axes = plt.subplots(2 * len(strips), 1, figsize=(16, 3.2 * 2 * len(strips)))
    axes = np.atleast_1d(axes)
    for k, (side, S) in enumerate(sorted(strips.items())):
        r = np.array([(lb, dr) for s, i, lb, dr, _ in rows if s == side])
        for j, cut in enumerate("uw"):
            ax = axes[2 * k + j]
            overlay(ax, S["i" + cut].T, S["l" + cut].T, S["p" + cut].T, "auto")
            # top strip: dark = labelled section, white = unclear (ignore)
            ax.imshow(np.where(r[:, 0], 0.15, 1.0)[None], cmap="gray", vmin=0, vmax=1, aspect="auto",
                      extent=(-.5, len(r) - .5, -.5, 3.5))
            ax.set_ylim(S["iu"].shape[1] - .5, -.5)
            ax.set_title(f"{tag} side {side} cut along {cut}   x = section (0.4mm each), "
                         f"y = {2 * args.half_mm:.0f}mm across;  red = sparse canal  green = model;  "
                         f"top bar dark = section has canal label, white = unclear", fontsize=9)
            ax.set_yticks([])
    plt.tight_layout(); plt.savefig(out / f"{tag}_straight.png", dpi=70); plt.close()

    if not gal:
        return
    pick = [gal[i] for i in np.linspace(0, len(gal) - 1, min(args.n_gal, len(gal))).astype(int)]
    nc = 8; nr = int(np.ceil(len(pick) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(2.2 * nc, 2.4 * nr))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (side, i, si, sq, sl) in zip(np.ravel(axes), pick):
        overlay(ax, si.T, sl.T, sq.T, "equal")
        c = (si.shape[0] - 1) / 2
        ax.plot(c, c, "y+", ms=6)
        ax.set_title(f"s{side} #{i}", fontsize=8)
    fig.suptitle(f"{tag} unclear sections (no canal label)  {2 * args.half_mm:.0f}mm  green = model  "
                 f"+ = doctor track", fontsize=9)
    plt.tight_layout(); plt.savefig(out / f"{tag}_unclear.png", dpi=70); plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--half_mm", type=float, default=4.0)
    ap.add_argument("--near_mm", type=float, default=1.5, help="模型的管離截面中心多近才算「這張有畫」")
    ap.add_argument("--n_gal", type=int, default=24)
    ap.add_argument("--figs", nargs="*", default=None, help="只對這些病例做圖(預設全部)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print("case  sections  labelled  drawn|labelled  drawn|unclear  width_labelled  width_unclear  >3.8mm|unclear")
    for cid in args.cases:
        rows, gal, strips = review(cid, args)
        r = np.array([(lb, dr, d) for _, _, lb, dr, d in rows], float)
        lb, dr, d = r[:, 0] > 0, r[:, 1] > 0, r[:, 2]
        du = d[~lb & dr]
        print(f"{cid[-4:]}  {len(r):5d}  {lb.mean():.2f}  {dr[lb].mean():.2f}  {dr[~lb].mean():.2f}  "
              f"{np.nanmedian(d[lb & dr]):.2f}  {np.nanmedian(du) if len(du) else np.nan:.2f}  "
              f"{(du > 3.8).mean() if len(du) else np.nan:.2f}", flush=True)
        if args.figs is None or cid in args.figs or cid[-4:] in args.figs:
            save_figs(cid, rows, gal, strips, out, args)


if __name__ == "__main__":
    main()
