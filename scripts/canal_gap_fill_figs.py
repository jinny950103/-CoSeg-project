#!/usr/bin/env python3
"""Pictures of the gap fill (canal_gap_fill.py): sections that had no wall, with the new rays.

Each panel is centred on the doctor track (yellow +), 8x8mm, same plane as the labeller.
red = canal from the new rays, blue = wall background, cyan x = predicted centre.
Title: section, how the location was found (interp / path), share of directions measured.

    python scripts/canal_gap_fill_figs.py --fill outputs/gap_fill --cases <case_id> ...
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
import canal_gap_fill as G  # noqa: E402

HOW = {1: "interp", 2: "path", 3: "lab", 4: "bridge"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fill", required=True)
    ap.add_argument("--rays", default="outputs/sparse_label_v6_rays/chunk_*/{}_rays.npz")
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--half_mm", type=float, default=4.0)
    args = ap.parse_args()
    for cid in args.cases:
        cid_, sp, sides = G.load_case(glob.glob(args.rays.format(cid))[0])
        vol = np.asanyarray(nib.load(G.IMG.format(cid)).dataobj).astype(np.float32)
        panels = []
        for s, d in sides.items():
            f = np.load(Path(args.fill) / f"{cid}_s{s}_fill.npz")
            known = np.isfinite(d["r_agree"]).any(1)
            for i in np.where(~known & (f["how"] > 0) & np.isfinite(f["r"]).any(1))[0]:
                panels.append((s, i, d, f["c"][i], f["r"][i], int(f["how"][i])))
        if not panels:
            continue
        pick = [panels[k] for k in np.linspace(0, len(panels) - 1, min(args.n, len(panels))).astype(int)]
        nc = 8; nr = int(np.ceil(len(pick) / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(2.3 * nc, 2.5 * nr), squeeze=False)
        for ax in axs.ravel():
            ax.axis("off")
        h = args.half_mm
        for ax, (s, i, d, c, r, how) in zip(axs.ravel(), pick):
            u, w, k = d["u"][i], d["w"][i], d["click"][i]
            sec, g = E.sample_section(vol, k, u, w, h / sp, 0.1 / sp)
            ax.imshow(sec, cmap="gray", extent=[-h, h, h, -h])
            if np.isfinite(c).all():
                off = np.array([(c - k) @ u, (c - k) @ w]) * sp
                if np.isfinite(r).any():
                    n = sec.shape[0]
                    yy, xx = np.mgrid[:n, :n] * 0.1 - h
                    a_, b_ = yy - off[0], xx - off[1]
                    rr = np.hypot(a_, b_); th = np.mod(np.arctan2(b_, a_), 2 * np.pi)
                    L = G.SL.label_from_polar(rr, th, r, 0.4)
                    ov = np.zeros(L.shape + (4,))
                    ov[L == 1] = (1, .1, .1, .45); ov[L == 0] = (.2, .5, 1, .45)
                    ax.imshow(ov, extent=[-h, h, h, -h])
                ax.plot(off[1], off[0], "cx", ms=6)
            ax.plot(0, 0, "y+", ms=7)
            ax.set_title(f"s{s} #{i} {HOW.get(how, '-')} {np.isfinite(r).mean():.0%}", fontsize=8)
        fig.suptitle(f"{cid[-4:]} gap sections  8x8mm  red = canal from new rays  blue = wall  "
                     f"cyan x = predicted centre  yellow + = doctor track", fontsize=9)
        fig.tight_layout(); fig.savefig(Path(args.fill) / f"{cid[-4:]}_gapfill.png", dpi=75); plt.close(fig)


if __name__ == "__main__":
    main()
