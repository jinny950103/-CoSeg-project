#!/usr/bin/env python3
"""Sections that one labeller run leaves empty and another fills (canal_sparse_label.py --save_rays).

    python scripts/canal_label_diff_figs.py --base outputs/sparse_label_v6_rays --new outputs/sparse_label_v6_nods \
        --cases <case_id> --out outputs/label_diff
Each panel: 8x8mm centred on the doctor track (yellow +), the new run's canal (red) and wall (blue).
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_enclose", type=float, default=0, help="只畫 enclosure 至少這麼多的截面(0=全部)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    h = 4.0
    for cid in args.cases:
        _, sp, B = G.load_case(glob.glob(f"{args.base}/chunk_*/{cid}_rays.npz")[0])
        _, _, N = G.load_case(glob.glob(f"{args.new}/chunk_*/{cid}_rays.npz")[0])
        vol = np.asanyarray(nib.load(G.IMG.format(cid)).dataobj).astype(np.float32)
        vs = ndimage.gaussian_filter(vol, 0.2 / sp) if args.min_enclose > 0 else vol
        panels = []
        for s in B:
            kb = np.isfinite(B[s]["r_agree"]).any(1); kn = np.isfinite(N[s]["r_agree"]).any(1)
            for i in np.where(~kb & kn)[0]:
                d = N[s]
                if args.min_enclose > 0 and G.enclosure(vs, d["c"][i], d["u"][i], d["w"][i], sp) < args.min_enclose:
                    continue
                panels.append((s, i))
        if not panels:
            continue
        pick = sorted(rng.choice(len(panels), min(args.n, len(panels)), replace=False))
        nc = 8; nr = int(np.ceil(len(pick) / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(2.3 * nc, 2.5 * nr), squeeze=False)
        for ax in axs.ravel():
            ax.axis("off")
        for ax, q in zip(axs.ravel(), pick):
            s, i = panels[q]; d = N[s]
            u, w, k, c, r = d["u"][i], d["w"][i], d["click"][i], d["c"][i], d["r_agree"][i]
            sec, _ = E.sample_section(vol, k, u, w, h / sp, 0.1 / sp)
            ax.imshow(sec, cmap="gray", extent=[-h, h, h, -h])
            off = np.array([(c - k) @ u, (c - k) @ w]) * sp
            n = sec.shape[0]
            yy, xx = np.mgrid[:n, :n] * 0.1 - h
            a_, b_ = yy - off[0], xx - off[1]
            L = G.SL.label_from_polar(np.hypot(a_, b_), np.mod(np.arctan2(b_, a_), 2 * np.pi), r, 0.4)
            ov = np.zeros(L.shape + (4,)); ov[L == 1] = (1, .1, .1, .45); ov[L == 0] = (.2, .5, 1, .45)
            ax.imshow(ov, extent=[-h, h, h, -h]); ax.plot(0, 0, "y+", ms=7)
            ax.set_title(f"s{s} #{i} {np.isfinite(r).mean():.0%}", fontsize=8)
        fig.suptitle(f"{cid[-4:]} sections empty in {Path(args.base).name}, filled in {Path(args.new).name}  8x8mm",
                     fontsize=9)
        fig.tight_layout(); fig.savefig(out / f"{cid[-4:]}_{Path(args.new).name}{'_enc' if args.min_enclose else ''}.png", dpi=75); plt.close(fig)
        print(cid[-4:], "new sections", len(panels), flush=True)


if __name__ == "__main__":
    main()
