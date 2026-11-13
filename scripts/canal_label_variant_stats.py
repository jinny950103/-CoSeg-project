#!/usr/bin/env python3
"""Compare labeller runs (canal_sparse_label.py --save_rays) on the sections the base run left empty.

For each run: share of sections with a wall, mean share of directions, and for sections that
are empty in the base run but measured in this run ("new"):
  r_med       median radius (mm)
  ratio>3     share of sections whose largest radius is > 3x the smallest (wedges / fans)
  bridge_dice section Dice between the new wall and the bridge prediction from this patient's own
              measured neighbours (base run, per-direction interpolation across the gap); only
              for gaps <= 8mm. Independent of the new measurement, so agreement = corroboration.

    python scripts/canal_label_variant_stats.py outputs/sparse_label_v6_rays outputs/sparse_label_v6_nods ...
"""
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import canal_edge_gt as E  # noqa: E402
import canal_gap_fill as G  # noqa: E402

ANG = G.ANG
g = np.arange(-3, 3.0001, 0.05)
YY, XX = np.meshgrid(g, g, indexing="ij")


def mask(off, r):
    if not np.isfinite(r).any():
        return np.zeros(YY.shape, bool)
    rf = E.fill_angles(ANG, r); a = YY - off[0]; b = XX - off[1]
    k = np.round(np.mod(np.arctan2(b, a), 2 * np.pi) / (2 * np.pi) * 72).astype(int) % 72
    return np.hypot(a, b) <= rf[k]


def bridge(d, i, sp, max_gap=20):
    """Base-run prediction for section i from the nearest measured sections on both sides."""
    R, C, U, W, K = d["r_agree"], d["c"], d["u"], d["w"], d["click"]
    known = np.isfinite(R).any(1)
    a = i - 1
    while a >= 0 and not known[a]:
        a -= 1
    b = i + 1
    while b < len(R) and not known[b]:
        b += 1
    if a < 0 or b >= len(R) or b - a - 1 > max_gap:
        return None
    t = (i - a) / (b - a)
    ca = G.wall_centre(C[a], U[a], W[a], R[a], sp); cb = G.wall_centre(C[b], U[b], W[b], R[b], sp)
    c = K[i] + (1 - t) * (ca - K[a]) + t * (cb - K[b])
    return c, (1 - t) * R[a] + t * R[b]


from case_lists import load_cases
SKIP = set(load_cases())


def main():
    base = sys.argv[1]
    for run in sys.argv[1:]:
        tot = got = 0; dirs = []; rr = []; ra = []; bd = []
        for f in sorted(glob.glob(f"{base}/chunk_*/*_rays.npz")):
            cid = Path(f).name.replace("_rays.npz", "")
            if cid in SKIP:
                continue
            _, sp, B = G.load_case(f)
            _, _, N = G.load_case(glob.glob(f"{run}/chunk_*/{cid}_rays.npz")[0])
            for s in B:
                kb = np.isfinite(B[s]["r_agree"]).any(1); RN = N[s]["r_agree"]; kn = np.isfinite(RN).any(1)
                tot += len(kb); got += kn.sum(); dirs.append(np.isfinite(RN).mean(1))
                for i in np.where(~kb & kn)[0]:
                    r = RN[i]; rr.append(np.nanmean(r)); ra.append(np.nanmax(r) / np.nanmin(r))
                    p = bridge(B[s], i, sp)
                    if p is None or not np.isfinite(p[1]).any():
                        continue
                    c, rk = p; d = N[s]
                    rel = lambda q: np.array([(q - d["c"][i]) @ d["u"][i], (q - d["c"][i]) @ d["w"][i]]) * sp
                    m1 = mask(rel(d["c"][i]), r); m2 = mask(rel(c), rk)
                    bd.append(2 * (m1 & m2).sum() / max(m1.sum() + m2.sum(), 1))
        ra = np.array(ra); bd = np.array(bd)
        print(f"{Path(run).name:24s} sections_with_wall={got / tot:.3f} dirs={np.concatenate(dirs).mean():.3f} "
              f"new={len(rr)} r_med={np.median(rr) if rr else np.nan:.2f} ratio>3={np.mean(ra > 3) if len(ra) else np.nan:.2f} "
              f"bridge_dice med={np.median(bd) if len(bd) else np.nan:.2f} (n={len(bd)}, >=0.7: {np.mean(bd >= .7) if len(bd) else np.nan:.2f})",
              flush=True)


if __name__ == "__main__":
    main()
