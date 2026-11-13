#!/usr/bin/env python3
"""One PDF page per patient: original CBCT, doctor's label, our edge-detection label.

Top: the canal straightened along the doctor's track (one strip per side, 6 mm either side of
the track), three strips each: image / doctor / ours. Bottom: cross-sections at 4 places per side
(20/40/60/80% along the canal), the same three panels each. Ours is coloured by how it was made
(measured / predicted / extrapolated).

    <venv>/python scripts/make_patient_pdf.py --out outputs/patient_review_20261102.pdf
"""
import argparse
from case_lists import load_cases
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import canal_gap_fill as G  # noqa: E402

for f in ("C:/Windows/Fonts/msjh.ttc",):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()

DOC = (0.15, 0.85, 0.25)
OURS = {1: (0.16, 0.43, 0.86), 3: (0.94, 0.55, 0.12), 5: (0.86, 0.16, 0.63)}
HALF_MM, STEP = 6.0, 0.2
N_DIR, R_MAX_MM = 72, 4.0


def sample(vol, pts, order):
    return ndimage.map_coordinates(vol, pts.reshape(-1, 3).T, order=order, mode="constant").reshape(pts.shape[:-1])


def overlay(ax, img, mask_rgb, extent, lo, hi, alpha=0.45, aspect="auto"):
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, extent=extent, aspect=aspect, interpolation="bilinear")
    if mask_rgb is not None:
        ax.imshow(mask_rgb, extent=extent, aspect=aspect, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])


def rgba(lab, colors, alpha=0.5):
    out = np.zeros(lab.shape + (4,))
    for v, c in colors.items():
        out[lab == v] = (*c, alpha)
    return out


ANG = np.linspace(0, 2 * np.pi, N_DIR, endpoint=False)
DSTEP = np.arange(0.0, R_MAX_MM + 1e-6, 0.05)


def label_radii(lab, centre, u, w, canal, stop, sp=0.4):
    """Per-direction radius (mm) of the labelled canal in one cross-section.

    Walk out from `centre` along N_DIR directions in the (u, w) plane; the radius is where the label
    first leaves `canal`. If it leaves into a value not in `stop` (i.e. into 'ignore'), the wall in
    that direction is unknown -> NaN. Reads the final label, so it shows exactly what the dataset holds.
    """
    dirs = np.cos(ANG)[:, None] * u + np.sin(ANG)[:, None] * w                 # (dir, 3)
    pts = centre + dirs[:, None, :] * (DSTEP[None, :, None] / sp)              # (dir, step, 3)
    v = sample(lab, pts, 0).astype(np.int64)
    r = np.full(N_DIR, np.nan)
    if not np.isin(v[:, 0], canal).mean() > 0.5:                              # centre not in canal
        return r
    for i in range(N_DIR):
        out = np.flatnonzero(~np.isin(v[i], canal))
        if len(out) and v[i, out[0]] in stop:
            r[i] = DSTEP[out[0]]
    return r


def diam_profile(lab, centres, U, W, canal, stop):
    """Per section: median diameter, 10/90th pct of opposite-ray diameters (mm); NaN if <25% known."""
    n = len(centres)
    med, lo, hi = (np.full(n, np.nan) for _ in range(3))
    for k in range(n):
        r = label_radii(lab, centres[k], U[k], W[k], canal, stop)
        if np.isfinite(r).mean() < 0.25:
            continue
        d = r[: N_DIR // 2] + r[N_DIR // 2:]
        med[k] = 2 * np.nanmedian(r)
        if np.isfinite(d).sum() >= 3:
            lo[k], hi[k] = np.nanpercentile(d, [10, 90])
    return med, lo, hi


def section_centres(ours, d, extra):
    """Centre used for each section: the extrap/track centre where that stage filled it, else the ray centre."""
    c = d["c"].copy()
    for z in extra:                                                             # track first, extrap last
        if z is not None and "c" in z and "r" in z:
            filled = np.isfinite(z["r"]).any(1)
            c[filled] = z["c"][filled]
    return c


def page(pdf, cid, sides, img, doc, ours, note, extra_by_side, png=None):
    lo, hi = np.percentile(img[img > np.percentile(img, 5)], [1, 99.5])
    fig = plt.figure(figsize=(16.5, 14.0))
    fig.suptitle(f"{cid}    原始影像 / 醫生標註 / 我們的 edge detection{note}", fontsize=16, x=0.02, ha="left")
    off = np.arange(-HALF_MM, HALF_MM + 1e-6, 0.1)
    gs = fig.add_gridspec(10, 12, left=0.065, right=0.99, top=0.94, bottom=0.04, hspace=0.38, wspace=0.08,
                          height_ratios=[1, 1, 1, 1.5, 1, 1, 1, 1.5, 2.2, 2.2])
    for si, (s, d) in enumerate(sorted(sides.items())):
        K, U, W, n = d["click"], d["u"], d["w"], len(d["click"])
        pts = K[:, None] + U[:, None] * (off[None, :, None] / 0.4)        # (n, off, 3)
        I = sample(img, pts, 1).T; D = sample(doc, pts, 0).T; O = sample(ours, pts, 0).T
        L = n * 0.4; ext = [0, L, HALF_MM, -HALF_MM]
        picks = [int(round(f * (n - 1))) for f in (0.2, 0.4, 0.6, 0.8)]
        row0 = si * 4
        for r, (m, name) in enumerate(((None, "影像"), (rgba(D, {1: DOC}), "醫生"), (rgba(O, OURS), "我們"))):
            ax = fig.add_subplot(gs[row0 + r, :])
            overlay(ax, I, m, ext, lo, hi)
            ax.set_ylabel(f"側{s} {name}", fontsize=10, rotation=0, ha="right", va="center")
            for p in picks:
                ax.axvline(p * 0.4, color="yellow", lw=0.6, ls=":")
            ax.set_xlim(0, L)

        # diameter along the canal: ours (per kind, with direction spread) vs the doctor's label
        C = section_centres(ours, d, extra_by_side.get(s, ()))
        om, olo, ohi = diam_profile(ours, C, U, W, canal=(1, 3, 5), stop=(0, 4, 6))
        dm, _, _ = diam_profile(doc, K, U, W, canal=(1,), stop=(0,))
        kind = np.zeros(n, int)
        for k in range(n):
            v = sample(ours, C[k][None], 0)[0]
            kind[k] = v if v in (1, 3, 5) else 0
        x = (np.arange(n) + 0.5) * 0.4
        ax = fig.add_subplot(gs[row0 + 3, :])
        ax.fill_between(x, olo, ohi, color=OURS[1], alpha=0.15, lw=0, step="mid")
        ax.plot(x, dm, color=DOC, lw=1.8, label="醫生標註")
        for v in (1, 3, 5):
            yy = np.where(kind == v, om, np.nan)
            ax.plot(x, yy, color=OURS[v], lw=2.2, marker=".", ms=3)
        ax.set_xlim(0, L); ax.set_ylim(0, 5)
        ax.set_xticks(np.arange(0, L, 10)); ax.tick_params(labelsize=7)
        ax.set_ylabel(f"側{s} 管徑\n(mm)", fontsize=10, rotation=0, ha="right", va="center")
        ax.grid(axis="y", color="0.85", lw=0.6)
        for p in picks:
            ax.axvline(p * 0.4, color="goldenrod", lw=0.6, ls=":")
        if np.isfinite(om).any():
            q = np.nanpercentile(om, [5, 95])
            txt = f"我們 {q[0]:.1f}–{q[1]:.1f} mm"
            if np.isfinite(dm).any():
                qd = np.nanpercentile(dm, [5, 95])
                txt += f"    醫生 {qd[0]:.1f}–{qd[1]:.1f} mm"
            ax.text(0.005, 0.93, txt + "（5–95%）", transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(facecolor="white", alpha=0.8, lw=0))

        # cross-sections
        g = np.arange(-4, 4.01, 0.1)
        for pi, k in enumerate(picks):
            P = K[k][None, None] + (g[:, None, None] * U[k] + g[None, :, None] * W[k]) / 0.4
            Ik, Dk, Ok = sample(img, P, 1), sample(doc, P, 0), sample(ours, P, 0)
            for r, m in enumerate((None, rgba(Dk, {1: DOC}), rgba(Ok, OURS))):
                ax = fig.add_subplot(gs[8 + si, pi * 3 + r])
                overlay(ax, Ik, m, [-4, 4, 4, -4], lo, hi, aspect="equal")
                if r == 0:
                    ax.set_title(f"側{s}  {k * 0.4:.0f} mm", fontsize=9, loc="left")
                elif r == 1 and np.isfinite(dm[k]):
                    ax.set_title(f"Ø{dm[k]:.1f}", fontsize=9, loc="left", color=(0.1, 0.55, 0.2))
                elif r == 2 and np.isfinite(om[k]):
                    ax.set_title(f"Ø{om[k]:.1f}", fontsize=9, loc="left", color=OURS.get(kind[k], OURS[1]))
    fig.legend(handles=[Patch(color=DOC, label="醫生標註"), Patch(color=OURS[1], label="我們：量到"),
                        Patch(color=OURS[3], label="我們：內插（前後接起來）"), Patch(color=OURS[5], label="我們：外插")],
               loc="upper right", ncol=4, fontsize=11, frameon=False)
    fig.text(0.02, 0.008, "拉直圖：沿醫生軌跡拉直的管（上下各 6 mm，橫軸 mm），黃虛線＝下方截面位置。"
             "管徑：每張截面 72 個方向的直徑中位數，淡藍帶＝不同方向的直徑範圍（10–90%），綠線＝醫生標註。"
             "截面：每個位置三格依序為 影像 / 醫生 / 我們，每格 8×8 mm", fontsize=9.5)
    if png:
        fig.savefig(png, dpi=90)
    pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rays", default="outputs/sparse_label_v6_rays/chunk_0*/*_rays.npz")
    ap.add_argument("--ours", default="outputs/extrap_fill/{}_extrap.nii.gz")
    ap.add_argument("--doc", default="data/hospital_nifti_clean/labels/{}.nii.gz")
    ap.add_argument("--flag", nargs="*", default=load_cases())
    ap.add_argument("--track", default="outputs/track_fill", help="folder with <cid>_s<side>_track.npz (section centres)")
    ap.add_argument("--extrap", default="outputs/extrap_fill", help="folder with <cid>_s<side>_extrap.npz")
    ap.add_argument("--out", default="outputs/patient_review_20261102.pdf")
    ap.add_argument("--cases", nargs="*", default=[], help="only these case IDs (for a quick check)")
    ap.add_argument("--png_dir", default="", help="also save each page as PNG here")
    args = ap.parse_args()
    files = sorted(glob.glob(args.rays), key=lambda f: (Path(f).name.split("_")[0] in args.flag, f))
    if args.cases:
        files = [f for f in files if Path(f).name.split("_")[0] in args.cases]
    if args.png_dir:
        Path(args.png_dir).mkdir(parents=True, exist_ok=True)

    def npz_or_none(p):
        return dict(np.load(p)) if Path(p).exists() else None

    with PdfPages(args.out) as pdf:
        for npz in files:
            cid, sp, sides = G.load_case(npz)
            img = np.asanyarray(nib.load(G.IMG.format(cid)).dataobj).astype(np.float32)
            doc = np.asanyarray(nib.load(args.doc.format(cid)).dataobj).astype(np.uint8)
            ours = np.asanyarray(nib.load(args.ours.format(cid)).dataobj).astype(np.uint8)
            note = "    （醫生標註上下錯開約 14 mm，未放入資料集）" if cid in args.flag else ""
            extra = {s: (npz_or_none(f"{args.track}/{cid}_s{s}_track.npz"),
                         npz_or_none(f"{args.extrap}/{cid}_s{s}_extrap.npz")) for s in sides}
            png = f"{args.png_dir}/{cid}.png" if args.png_dir else None
            page(pdf, cid, sides, img, doc, ours, note, extra, png)
            print(cid, flush=True)


if __name__ == "__main__":
    main()
