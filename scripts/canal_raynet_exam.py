#!/usr/bin/env python3
"""Fair exam for the ray network: make a clear wall faint, then see if the network still finds it.

On measured directions the network simply sees the wall (it copies the strict labeller), so
canal_raynet_eval.py cannot tell whether it can predict where the wall is NOT visible. Here, as
in canal_gap_fill.py --exam --degrade: take a block of L clear sections (>= 50% of directions
measured), lower the contrast and add noise around it so the strict ray method fails too, rebuild
the network's input from the degraded image and predict with the fold model that never saw the
patient. Scored against the strict radii from before degrading, next to the bridge (linear
interpolation from the clear sections at both ends of the block, which never looks at the image).

Per section: radius error on the measured directions (same centre), and Dice of the star contours
(truth = strict radii, missing directions filled from neighbouring directions for rasterising only).

    <venv>/python scripts/canal_raynet_exam.py --L 10 20 --out outputs/raynet/exam.json
"""
import argparse
from case_lists import load_cases
import glob
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
import canal_edge_gt as E  # noqa: E402
import canal_gap_fill as G  # noqa: E402
import canal_raynet as RN  # noqa: E402
from canal_raynet_eval import bridge_radius  # noqa: E402

ANG = G.ANG


def patches(vs, d, idx, sp, p1, p99, n_px=64, px_mm=0.1, planes=5):
    """Same sampling as canal_raynet_data.py, on an already smoothed volume."""
    g = (np.arange(n_px) - (n_px - 1) / 2) * px_mm / sp
    a, b = np.meshgrid(g, g, indexing="ij")
    offs = (np.arange(planes) - (planes - 1) / 2) * 0.4 / sp
    out = []
    for i in idx:
        c, u, w, t = d["c"][i], d["u"][i], d["w"][i], d["t"][i]
        xyz = (c[:, None, None, None] + t[:, None, None, None] * offs[None, :, None, None]
               + u[:, None, None, None] * a[None, None] + w[:, None, None, None] * b[None, None])
        p = ndimage.map_coordinates(vs, xyz.reshape(3, -1), order=1, mode="nearest")
        out.append((p.reshape(planes, n_px, n_px) - p1) / (p99 - p1))
    return np.array(out, np.float32)


def star_dice(r1, r2, n=121, half=3.0):
    yy, xx = (np.mgrid[:n, :n] - (n - 1) / 2) * (2 * half / (n - 1))
    rr = np.hypot(yy, xx); th = np.mod(np.arctan2(xx, yy), 2 * np.pi)
    k = np.round(th / (2 * np.pi) * len(ANG)).astype(int) % len(ANG)
    a = rr <= r1[k]; b = rr <= r2[k]
    return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rays", default="outputs/sparse_label_v6_rays/chunk_0*/*_rays.npz")
    ap.add_argument("--nets", default="outputs/raynet/cv/net_fold{}.pt")
    ap.add_argument("--splits", default="data/nnUNet_raw/Dataset504_MixedCanalClean/splits_final.json")
    ap.add_argument("--img", default=G.IMG)
    ap.add_argument("--skip", nargs="*", default=load_cases())
    ap.add_argument("--L", type=int, nargs="+", default=[10, 20], help="藏起的區塊長度(截面數,0.4mm/張)")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--min_frac", type=float, default=0.5)
    ap.add_argument("--degrade_contrast", type=float, default=0.35)
    ap.add_argument("--degrade_noise", type=float, default=40)
    ap.add_argument("--degrade_r_mm", type=float, default=4.0)
    ap.add_argument("--no_degrade", action="store_true", help="對照:不弄淡,網路看得到管壁")
    ap.add_argument("--out", default="outputs/raynet/exam.json")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = json.load(open(args.splits))
    fold_of = {c[5:]: k for k, f in enumerate(folds) for c in f["val"] if c.startswith("HOSP_")}
    nets = {}
    for f in range(5):
        net = RN.Net(5).to(dev); net.load_state_dict(torch.load(args.nets.format(f), map_location=dev)); nets[f] = net
    grid, r = RN.polar_grid(64, 0.1, dev)
    rng = np.random.default_rng(0)
    gargs = argparse.Namespace(degrade_contrast=args.degrade_contrast, degrade_noise=args.degrade_noise,
                               degrade_r_mm=args.degrade_r_mm)
    rows = []
    for npz in sorted(glob.glob(args.rays)):
        cid, sp, sides = G.load_case(npz)
        if cid in args.skip:
            continue
        vol = np.asanyarray(nib.load(args.img.format(cid)).dataobj).astype(np.float32)
        p1, p99 = np.percentile(vol[::4, ::4, ::4], [1, 99])
        vs = ndimage.gaussian_filter(vol, 0.2 / sp)
        net = nets[fold_of[cid]]
        for s, d in sides.items():
            R = d["r_agree"]; n = len(R)
            good = np.isfinite(R).mean(1) >= args.min_frac
            gi = np.where(good)[0]
            for L in args.L:
                i = 2
                while i + L + 2 <= n:
                    blk = np.arange(i, i + L)
                    if not good[i - 2:i + L + 2].all():
                        i += 1; continue
                    v = vs if args.no_degrade else G.degrade(vs, d["c"][blk], sp, gargs, rng)
                    X = patches(v, d, blk, sp, p1, p99)
                    pr, pv, _ = RN.predict(net, X, grid, r, dev)
                    known = np.setdiff1d(gi, blk)
                    for j, k in enumerate(blk):
                        ok = np.isfinite(R[k]); truth = E.fill_angles(ANG, R[k])
                        _, r0, _ = E.ray_contour(v, d["c"][k], d["u"][k], d["w"][k], G.N_ANG, 0.3 / sp,
                                                 G.HALF / sp, 0.05 / sp, 100)
                        rb = bridge_radius(d, k, sp, known)
                        row = {"case": cid[-4:], "L": L, "sec": int(k),
                               "strict_on_degraded": float(np.isfinite(r0).mean()),
                               "net_err": float(np.median(np.abs(pr[j] - R[k])[ok])),
                               "net_dice": float(star_dice(pr[j], truth)), "net_vis": float(pv[j].mean()),
                               "net_r": float(np.median(pr[j]))}
                        if rb is not None:
                            row["bridge_err"] = float(np.median(np.abs(rb - R[k])[ok]))
                            row["bridge_dice"] = float(star_dice(rb, truth))
                        rows.append(row)
                    i += L + args.stride
        print(cid[-4:], len(rows), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows))
    for L in args.L:
        x = [q for q in rows if q["L"] == L]; b = [q for q in x if "bridge_err" in q]
        med = lambda k, s: np.median([q[k] for q in s])
        print(f"L={L} ({L * 0.4:.0f}mm) sections {len(x)}  strict still finds {med('strict_on_degraded', x):.2f}  |  "
              f"net err {med('net_err', b):.3f} Dice {med('net_dice', b):.2f} vis {med('net_vis', b):.2f}  |  "
              f"bridge err {med('bridge_err', b):.3f} Dice {med('bridge_dice', b):.2f}  (n={len(b)})")


if __name__ == "__main__":
    main()
