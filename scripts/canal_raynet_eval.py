#!/usr/bin/env python3
"""Score the ray network's cross-validated predictions (canal_raynet.py) against our strict radii,
next to the bridge (interpolation from this patient's own neighbouring sections).

Per direction, radius error (mm) where the strict labeller measured that direction, from the
same centre:
  clear    sections with >= 50% of directions measured
  partial  sections with 1-50% measured (the wall is only partly visible: closest to a real gap
           that still has an answer); here the bridge is scored too, from the nearest >= 50%
           sections on both sides within 8mm
And on the empty sections (no answer): predicted width and how irregular it is.

    python scripts/canal_raynet_eval.py --pred outputs/raynet/cv/cv_pred_f*.npz
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import canal_edge_gt as E  # noqa: E402
import canal_gap_fill as G  # noqa: E402

ANG = G.ANG


def bridge_radius(d, k, sp, good):
    """Bridge contour for section k, re-expressed as radii from section k's own centre."""
    R, C, U, W, K = d["r_agree"], d["c"], d["u"], d["w"], d["click"]
    L = good[good < k]; Rr = good[good > k]
    if not len(L) or not len(Rr) or (Rr[0] - L[-1] - 1) * 0.4 > 8:
        return None
    a, b = L[-1], Rr[0]; t = (k - a) / (b - a)
    ca = G.wall_centre(C[a], U[a], W[a], R[a], sp); cb = G.wall_centre(C[b], U[b], W[b], R[b], sp)
    c = K[k] + (1 - t) * (ca - K[a]) + t * (cb - K[b])
    rk = (1 - t) * E.fill_angles(ANG, R[a]) + t * E.fill_angles(ANG, R[b])
    o = np.array([(c - C[k]) @ U[k], (c - C[k]) @ W[k]]) * sp
    py = o[0] + np.cos(ANG) * rk; px = o[1] + np.sin(ANG) * rk          # contour in the plane, mm
    th = np.mod(np.arctan2(px, py), 2 * np.pi); rr = np.hypot(px, py)
    order = np.argsort(th)
    return np.interp(ANG, th[order], rr[order], period=2 * np.pi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--rays", default="outputs/sparse_label_v6_rays/chunk_0*/{}_rays.npz")
    args = ap.parse_args()
    P = V = None
    for f in args.pred:                   # one file per fold job: merge the rows each one predicted
        z = np.load(f)
        if P is None:
            P, V, Y, meta = z["r"].copy(), z["v"].copy(), z["Y"], z["meta"]
        got = np.isfinite(z["r"]).all(1)
        P[got], V[got] = z["r"][got], z["v"][got]
    keep = np.isfinite(P).all(1)
    P, V, Y, meta = P[keep], V[keep], Y[keep], meta[keep]
    cov = np.isfinite(Y).mean(1)
    m = np.isfinite(Y)
    e = np.abs(P - Y)
    clear = cov >= 0.5; part = (cov > 0) & (cov < 0.5); empty = cov == 0
    for name, sel in (("clear", clear), ("partial", part)):
        x = e[sel][m[sel]]
        print(f"{name:8s} sections {sel.sum():5d}  raynet |r err| median {np.median(x):.3f}  <=0.2mm {np.mean(x <= .2):.2f}")
    # bridge on partial sections
    eb, en = [], []
    cache = {}
    for j in np.where(part)[0]:
        cid, s, i = f"{meta[j, 0]:010d}", int(meta[j, 1]), int(meta[j, 2])
        if cid not in cache:
            cache[cid] = G.load_case(glob.glob(args.rays.format(cid))[0])
        _, sp, sides = cache[cid]
        d = sides[s]
        good = np.where(np.isfinite(d["r_agree"]).mean(1) >= 0.5)[0]
        rb = bridge_radius(d, i, sp, good)
        if rb is None:
            continue
        ok = np.isfinite(Y[j])
        eb += list(np.abs(rb - Y[j])[ok]); en += list(e[j][ok])
    eb, en = np.array(eb), np.array(en)
    print(f"partial, where a bridge exists: raynet median {np.median(en):.3f} (<=0.2 {np.mean(en <= .2):.2f}) | "
          f"bridge median {np.median(eb):.3f} (<=0.2 {np.mean(eb <= .2):.2f})  n_dirs={len(eb)}")
    # empty sections: what the network draws where we have no answer
    for name, sel in (("measured", cov > 0), ("empty", empty)):
        rm = np.median(P[sel], 1)
        ratio = P[sel].max(1) / np.maximum(P[sel].min(1), 1e-3)
        print(f"{name:8s} median radius {np.median(rm):.2f} mm  max/min>3 {np.mean(ratio > 3):.2f}  "
              f"visible-prob mean {np.mean(V[sel]):.2f}")


if __name__ == "__main__":
    main()
