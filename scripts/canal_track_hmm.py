#!/usr/bin/env python3
"""Bayesian tracking of the wall radius through a gap (per direction, forward-backward HMM).

The bridge interpolates each direction's radius linearly between the clear sections at both ends
of a gap and never looks at the image. Here each direction is tracked across the gap instead:
state = wall radius (0.3..2.5mm), both ends clamped to the radii measured there, a small random
step allowed per section (--step_mm), and the image as evidence: a dark-to-bright rise along the
ray at radius r makes r more likely. Where the image shows nothing the evidence is flat and the
answer falls back to the bridge; where a faint wall is there it pulls the radius onto it. The
posterior spread says how sure it is (--max_sd: more unsure than that -> leave unknown).
Same centre as the bridge (doctor's track plus the end offsets interpolated). Only this patient.

Exam: hide blocks of L clear sections, fade the image there (as canal_raynet_exam.py), track
through, score against the strict radii next to the bridge.

    <venv>/python scripts/canal_track_hmm.py --out outputs/track_hmm
"""
import argparse
from case_lists import load_cases
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
import canal_edge_gt as E  # noqa: E402
import canal_gap_fill as G  # noqa: E402
from canal_raynet_exam import star_dice  # noqa: E402

ANG = G.ANG
RS = np.arange(0.3, 2.501, 0.05)                 # radius states, mm


def end_profile(d, e, step, n_out, sp):
    """End contour at section e: each direction from the nearest section (e, e+step, ... n_out away)
    where it was measured, as 3D wall points; centre = their mean, radii in e's plane from it."""
    R, C, U, W = d["r_agree"], d["c"], d["u"], d["w"]
    pts = []
    for j in range(len(ANG)):
        for k in range(e, e + step * (n_out + 1), step):
            if 0 <= k < len(R) and np.isfinite(R[k][j]):
                dj = np.cos(ANG[j]) * U[k] + np.sin(ANG[j]) * W[k]
                pts.append(C[k] + dj * R[k][j] / sp)
                break
    if len(pts) < 3:
        return G.wall_centre(C[e], U[e], W[e], R[e], sp), E.fill_angles(ANG, R[e])
    pts = np.array(pts); cen = pts.mean(0)
    rel = pts - cen
    a_, b_ = rel @ U[e] * sp, rel @ W[e] * sp
    th = np.mod(np.arctan2(b_, a_), 2 * np.pi); rr = np.hypot(a_, b_)
    order = np.argsort(th)
    return cen, np.interp(ANG, th[order], rr[order], period=2 * np.pi)


def ends(d, k0, k1, sp, n_out=0):
    """Bridge centres for sections k0..k1 (exclusive ends a=k0-1, b=k1) and the end radii.

    n_out > 0: each end direction may come from up to n_out sections further out (end_profile)."""
    R, C, U, W, K = d["r_agree"], d["c"], d["u"], d["w"], d["click"]
    a, b = k0 - 1, k1
    if n_out:
        ca, ra = end_profile(d, a, -1, n_out, sp); cb, rb = end_profile(d, b, 1, n_out, sp)
    else:
        ca = G.wall_centre(C[a], U[a], W[a], R[a], sp); cb = G.wall_centre(C[b], U[b], W[b], R[b], sp)
        ra, rb = E.fill_angles(ANG, R[a]), E.fill_angles(ANG, R[b])
    ks = np.arange(k0, k1); t = (ks - a) / (b - a)
    cen = K[ks] + (1 - t)[:, None] * (ca - K[a]) + t[:, None] * (cb - K[b])
    return cen, ra, rb, t


def evidence(v, cen, u, w, sp, tau, dr=0.15):
    """(A, S) likelihood of each radius state from the dark-to-bright rise along each ray."""
    dirs = np.cos(ANG)[:, None] * u + np.sin(ANG)[:, None] * w        # (A, 3)
    rr = np.concatenate([RS - dr, RS + dr])
    pts = cen[None, None] + dirs[:, None] * (rr[None, :, None] / sp)
    I = ndimage.map_coordinates(v, pts.reshape(-1, 3).T, order=1, mode="nearest").reshape(len(ANG), -1)
    rise = I[:, len(RS):] - I[:, :len(RS)]
    return np.exp(np.clip(rise, 0, 5 * tau) / tau)


def track(Ev, ra, rb, step_mm):
    """Ev (n, A, S) evidence for the n gap sections; returns posterior mean and sd (n, A)."""
    n = len(Ev)
    T = np.exp(-0.5 * ((RS[:, None] - RS[None]) / step_mm) ** 2); T /= T.sum(1, keepdims=True)
    pa = np.exp(-0.5 * ((RS[None] - ra[:, None]) / 0.05) ** 2); pa /= pa.sum(1, keepdims=True)
    pb = np.exp(-0.5 * ((RS[None] - rb[:, None]) / 0.05) ** 2)
    fw = np.zeros_like(Ev); p = pa
    for i in range(n):
        p = (p @ T) * Ev[i]; p /= p.sum(1, keepdims=True); fw[i] = p
    bw = np.zeros_like(Ev); q = pb @ T.T; q /= q.sum(1, keepdims=True)
    for i in range(n - 1, -1, -1):
        bw[i] = q
        q = (q * Ev[i]) @ T.T; q /= q.sum(1, keepdims=True)
    post = fw * bw; post /= post.sum(2, keepdims=True)
    mean = (post * RS).sum(2)
    sd = np.sqrt(np.maximum((post * RS ** 2).sum(2) - mean ** 2, 0))
    return mean, sd


def radii_from(c_from, cen, rk, u, w, sp):
    o = np.array([(cen - c_from) @ u, (cen - c_from) @ w]) * sp
    py = o[0] + np.cos(ANG) * rk; px = o[1] + np.sin(ANG) * rk
    th = np.mod(np.arctan2(px, py), 2 * np.pi); rr = np.hypot(px, py)
    order = np.argsort(th)
    return np.interp(ANG, th[order], rr[order], period=2 * np.pi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rays", default="outputs/sparse_label_v6_rays/chunk_0*/*_rays.npz")
    ap.add_argument("--skip", nargs="*", default=load_cases())
    ap.add_argument("--out", default="outputs/track_hmm")
    ap.add_argument("--L", type=int, nargs="+", default=[10, 20, 40])
    ap.add_argument("--contrast", type=float, nargs="+", default=[1.0, 0.35, 0.15, 0.0], help="1 = 不弄淡")
    ap.add_argument("--noise_hu", type=float, default=40)
    ap.add_argument("--step_mm", type=float, default=0.08, help="每張截面半徑可以變多少")
    ap.add_argument("--tau", type=float, default=60, help="亮度上升多少 HU 算一倍證據")
    ap.add_argument("--max_sd", type=float, default=0.25, help="後驗標準差超過這個就不收")
    ap.add_argument("--end_keep", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="考試:兩端截面只留一段連續的方向(比例在 LO..HI 之間),模擬只看得到部分管壁的端點")
    ap.add_argument("--end_mm", type=float, default=0.0, help="每個方向往缺口外最多找多遠的截面當端點(0=只用旁邊那張)")
    ap.add_argument("--same_arc", action="store_true", help="考試:端點外面幾張都缺同一段方向(較難);預設每張各自隨機")
    ap.add_argument("--real_masks", action="store_true",
                    help="考試:端點和外面幾張直接套用真實缺口旁(看得到 25-50%% 方向的)截面的「哪些方向有量到」")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    no = int(round(args.end_mm / 0.4))
    pool = []                 # real visibility patterns: partial gap end + the sections beyond it
    if args.real_masks:
        for npz in sorted(glob.glob(args.rays)):
            cid, _, sides = G.load_case(npz)
            if cid in args.skip:
                continue
            for d in sides.values():
                fin = np.isfinite(d["r_agree"]); cov = fin.mean(1); good = np.where(cov >= 0.25)[0]
                for a, b in zip(good[:-1], good[1:]):
                    if not 1 <= b - a - 1 <= 40 or not (cov[a + 1:b] == 0).any():
                        continue
                    for e, stp in ((a, -1), (b, 1)):
                        ks = list(range(e, e + stp * (no + 1), stp))
                        if cov[e] < 0.5 and all(0 <= k < len(fin) for k in ks):
                            pool.append(fin[ks])
        print("real end patterns", len(pool), flush=True)
    for npz in sorted(glob.glob(args.rays)):
        cid, sp, sides = G.load_case(npz)
        if cid in args.skip:
            continue
        _, _, _, _, v0, sides = G.prep_volume(cid, sides, sp)
        rng = np.random.default_rng(int(cid))
        for s, d in sides.items():
            R = d["r_agree"]; n = len(R)
            good = np.isfinite(R).mean(1) >= 0.5
            mg = 2 + no
            for L in args.L:
                i = mg
                while i + L + mg <= n:
                    blk = np.arange(i, i + L)
                    if not good[i - mg:i + L + mg].all():
                        i += 1; continue
                    de = d
                    if args.end_keep:            # the two end sections only partly visible
                        Re = R.copy()
                        for e0, stp in ((i - 1, -1), (i + L, 1)):
                            keep = None
                            if pool:
                                while True:      # the end itself must keep >= 3 measured directions
                                    pat = pool[int(rng.integers(len(pool)))]
                                    if (pat[0] & np.isfinite(R[e0])).sum() >= 3:
                                        break
                                for j, e in enumerate(range(e0, e0 + stp * (no + 1), stp)):
                                    Re[e][~pat[j]] = np.nan
                                continue
                            for e in range(e0, e0 + stp * (no + 1), stp):   # the end and the sections beyond it
                                while keep is None or not args.same_arc:   # arc with >= 3 measured directions
                                    k0 = int(rng.integers(len(ANG))); m = int(round(rng.uniform(*args.end_keep) * len(ANG)))
                                    keep = np.zeros(len(ANG), bool); keep[(k0 + np.arange(m)) % len(ANG)] = True
                                    if (keep & np.isfinite(R[e])).sum() >= 3:
                                        break
                                Re[e][~keep] = np.nan
                        de = {**d, "r_agree": Re}
                    cen, ra, rb, t = ends(de, i, i + L, sp, no)
                    for cc in args.contrast:
                        v = v0 if cc >= 1 else G.degrade(v0, d["c"][blk], sp, argparse.Namespace(
                            degrade_contrast=cc, degrade_noise=args.noise_hu, degrade_r_mm=4.0), rng)
                        Ev = np.stack([evidence(v, cen[j], d["u"][k], d["w"][k], sp, args.tau) for j, k in enumerate(blk)])
                        mean, sd = track(Ev, ra, rb, args.step_mm)
                        for j, k in enumerate(blk):
                            ok = np.isfinite(R[k]); truth = E.fill_angles(ANG, R[k])
                            pr = radii_from(d["c"][k], cen[j], mean[j], d["u"][k], d["w"][k], sp)
                            br = radii_from(d["c"][k], cen[j], (1 - t[j]) * ra + t[j] * rb, d["u"][k], d["w"][k], sp)
                            sure = sd[j] <= args.max_sd
                            rows.append({"case": cid[-4:], "L": L, "c": cc, "sec": int(k),
                                         "hmm_err": float(np.median(np.abs(pr - R[k])[ok])),
                                         "hmm_dice": float(star_dice(pr, truth)),
                                         "hmm_sure": float(sure.mean()),
                                         "hmm_err_sure": float(np.median(np.abs(mean[j] - (1 - t[j]) * ra - t[j] * rb)[sure])) if sure.any() else None,
                                         "bridge_err": float(np.median(np.abs(br - R[k])[ok])),
                                         "bridge_dice": float(star_dice(br, truth))})
                    i += L + 10
        print(cid[-4:], len(rows), flush=True)
    (out / "exam.json").write_text(json.dumps(rows))
    for L in args.L:
        for cc in args.contrast:
            x = [q for q in rows if q["L"] == L and q["c"] == cc]
            med = lambda k: np.median([q[k] for q in x])
            print(f"L={L} ({L * 0.4:.0f}mm) contrast {cc:.2f} n={len(x)}  track err {med('hmm_err'):.3f} Dice {med('hmm_dice'):.2f}"
                  f" sure {med('hmm_sure'):.2f}  |  bridge err {med('bridge_err'):.3f} Dice {med('bridge_dice'):.2f}")


if __name__ == "__main__":
    main()
