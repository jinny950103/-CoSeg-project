#!/usr/bin/env python3
"""Problem 3, idea 2 (2026-11-13): left/right symmetry as an independent confirmer of long guessed runs.

The two mandibular canals are roughly mirror images. Where one side is invisible, the other side
may be clear at the same place. The other side's measured centre line is mirrored (x -> -x) and
rigidly aligned (ICP, Kabsch) onto this side's measured centres OUTSIDE the gap. A guessed section
is confirmed when its centre lies within --agree_mm of the mirrored, aligned path at a point where
the other side was measured, and its mean radius is within --r_mm of the other side's there.

  exam  hide L clear sections, guess them with the track (as canal_track_fill), and report how far
        the mirrored path is from the measured centre (how good symmetry is at all) and the guess
        Dice on symmetry-confirmed vs other sections.

    ~/venvs/dentalseg/Scripts/python scripts/canal_symmetry.py --out outputs/sym_exam
"""
import argparse
import glob
import json
import sys
import zlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from case_lists import load_cases  # noqa: E402
import canal_edge_gt as E  # noqa: E402
import canal_gap_fill as G  # noqa: E402
import canal_track_hmm as H  # noqa: E402
from canal_raynet_exam import star_dice  # noqa: E402



def centres(d, R, sp, min_cov=0.5):
    """Measured wall centres (voxels) and mean radius (mm) of sections with >= min_cov coverage."""
    cov = np.isfinite(R).mean(1)
    ks = np.where(cov >= min_cov)[0]
    C = np.array([G.wall_centre(d["c"][k], d["u"][k], d["w"][k], R[k], sp) for k in ks]).reshape(-1, 3)
    r = np.array([np.nanmean(R[k]) for k in ks])
    return ks, C, r


def kabsch(A, B):
    """Rigid R, t minimising |R A + t - B| (rows are points)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rm = Vt.T @ D @ U.T
    return Rm, cb - Rm @ ca


def mirror_align(Cother, Cthis, sp, iters=40, trim=0.8):
    """Mirror the other side's centres and ICP-align them onto this side's (mm). Returns the mapped
    other-side points (mm) and the RMS residual (mm) on the trimmed matches."""
    B = Cthis * sp
    lr = int(np.argmax(np.abs(Cother.mean(0) - Cthis.mean(0))))   # left-right axis = where the two sides differ most
    A = Cother * sp; A = A.copy(); A[:, lr] *= -1
    A = A - A.mean(0) + B.mean(0)                       # start: centroids together
    tree = cKDTree(B); X = A.copy(); res = np.inf
    for _ in range(iters):
        dist, idx = tree.query(X)
        keep = dist <= np.quantile(dist, trim)
        Rm, t = kabsch(X[keep], B[idx[keep]])
        X = X @ Rm.T + t
        res = float(np.sqrt(np.mean(dist[keep] ** 2)))
    return X, res


def sym_check(d_this, R_this, gap, d_other, R_other, gC, gr, sp, args):
    """gap: section indices of this side; gC (len(gap), 3) guessed centres (voxels), gr mean guessed radius (mm).
    Returns per gap section (dist_mm, radius_diff_mm, ok) and the ICP residual."""
    ks, C, _ = centres(d_this, R_this, sp)
    keep = ~np.isin(ks, gap)
    ko, Co, ro = centres(d_other, R_other, sp)
    if keep.sum() < 20 or len(Co) < 20:
        return None, np.inf
    X, res = mirror_align(Co, C[keep], sp)
    tree = cKDTree(X)
    dist, idx = tree.query(gC * sp)
    rdiff = np.abs(gr - ro[idx])
    ok = (dist <= args.agree_mm) & (rdiff <= args.r_mm) & (res <= args.max_res_mm)
    return (dist, rdiff, ok), res


def exam(args):
    no = int(round(4.0 / 0.4)); rows = []; resid = []
    for npz in sorted(glob.glob(args.rays)):
        cid, sp, sides = G.load_case(npz)
        if cid in args.skip or len(sides) < 2:
            continue
        _, _, _, _, v0, sides = G.prep_volume(cid, sides, sp)
        rng = np.random.default_rng(zlib.crc32(cid.encode()))
        keys = list(sides)
        for s in keys:
            d = sides[s]; o = sides[[k for k in keys if k != s][0]]
            R = d["r_agree"]; n = len(R); good = np.isfinite(R).mean(1) >= 0.5
            mg = no + 2
            for L in args.L:
                i = mg
                while i + L + mg <= n:
                    if not good[i - mg:i + L + mg].all():
                        i += 1; continue
                    blk = np.arange(i, i + L)
                    Rh = R.copy(); Rh[blk] = np.nan; dh = {**d, "r_agree": Rh}
                    for cc in args.contrast:
                        v = v0 if cc >= 1 else G.degrade(v0, d["c"][blk], sp, argparse.Namespace(
                            degrade_contrast=cc, degrade_noise=40, degrade_r_mm=4.0), rng)
                        cen, ra, rb, _ = H.ends(dh, i, i + L, sp, no)
                        Ev = np.stack([H.evidence(v, cen[j], d["u"][k], d["w"][k], sp, 60) for j, k in enumerate(blk)])
                        mean, sd = H.track(Ev, ra, rb, 0.08)
                        r = np.where((sd <= 0.25) & (mean <= 1.9), mean, np.nan)
                        kept = np.isfinite(r).sum(1) >= 3
                        if not kept.any():
                            continue
                        gC = np.array([G.wall_centre(cen[j], d["u"][k], d["w"][k], r[j], sp) if kept[j] else cen[j]
                                       for j, k in enumerate(blk)])
                        gr = np.array([np.nanmean(r[j]) if kept[j] else np.nan for j in range(L)])
                        out, res = sym_check(d, Rh, blk, o, o["r_agree"], gC, gr, sp, args)
                        if out is None:
                            continue
                        resid.append(res)
                        dist, rdiff, ok = out
                        # mirrored path vs the TRUE (hidden) centre
                        tC = np.array([G.wall_centre(d["c"][k], d["u"][k], d["w"][k], R[k], sp) for k in blk])
                        _, tdist_ok = None, None
                        ok_run = np.zeros(L, bool)
                        x = 0
                        while x < L:                       # steady runs >= 3
                            if not (ok[x] and kept[x]):
                                x += 1; continue
                            y = x
                            while y < L and ok[y] and kept[y]:
                                y += 1
                            if y - x >= 3:
                                ok_run[x:y] = True
                            x = y
                        for j, k in enumerate(blk):
                            if not kept[j]:
                                continue
                            pr = H.radii_from(d["c"][k], cen[j], E.fill_angles(H.ANG, r[j]), d["u"][k], d["w"][k], sp)
                            rows.append({"case": cid[-4:], "L": L, "c": cc, "pos": min(j + 1, L - j), "res": res,
                                         "guess": float(star_dice(pr, E.fill_angles(H.ANG, R[k]))),
                                         "guess_err_mm": float(np.linalg.norm((gC[j] - tC[j]) * sp)),
                                         "sym_dist_mm": float(dist[j]), "confirmed": bool(ok_run[j])})
                    i += L + 10
        print(cid[-4:], len(rows), flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "exam_rows.json").write_text(json.dumps(rows))
    r_ = np.array(resid)
    print(f"ICP residual on clear sections: median {np.median(r_):.2f}mm  p90 {np.percentile(r_, 90):.2f}mm  (n={len(r_)})")
    for L in args.L:
        for cc in args.contrast:
            x = [q for q in rows if q["L"] == L and q["c"] == cc and q["pos"] > 6]
            if not x:
                continue
            cf = [q for q in x if q["confirmed"]]; nc = [q for q in x if not q["confirmed"]]
            f = lambda sel: f"{np.mean([q['guess'] for q in sel]):.3f} (n={len(sel)})" if sel else "  -  "
            print(f"L={L} ({L * 0.4:.0f}mm) contrast {cc:.2f} middle: guess all {f(x)}  sym-confirmed {f(cf)}  not {f(nc)}  "
                  f"| mirrored path to guess: median {np.median([q['sym_dist_mm'] for q in x]):.2f}mm")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rays", default="outputs/sparse_label_v7_rays/chunk_0*/*_rays.npz")
    ap.add_argument("--skip", nargs="*", default=load_cases())
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, nargs="+", default=[20, 40])
    ap.add_argument("--contrast", type=float, nargs="+", default=[1.0, 0.35])
    ap.add_argument("--agree_mm", type=float, default=0.5)
    ap.add_argument("--r_mm", type=float, default=0.3)
    ap.add_argument("--max_res_mm", type=float, default=1.0, help="ICP 殘差太大(左右不對稱)就不用")
    exam(ap.parse_args())


if __name__ == "__main__":
    main()
