#!/usr/bin/env python3
"""Problem 3, idea 3 (2026-11-13): "how canals usually run", learned from MCSTU's expert labels.

Library: the canal centre lines of the MCSTU development set (expert full labels; d["c"] of the
MCSTU rays files = label centroid per 0.4 mm section), each also mirrored (left <-> right).
Query: a gap of L sections with F measured sections on each side (the flanks). Every library
stretch of F + L + F sections is rigidly aligned (Kabsch) onto the two flanks; the K stretches with
the smallest flank residual vote for the gap path (median of their aligned middle parts).
A guessed section is confirmed when its centre is within --agree_mm of that path and the K votes
agree with each other (spread <= --spread_mm). Position only; the radius is not checked here
(the canal size must come from the patient's own image, see memory no-public-data-for-gap-filling).

  exam  hospital: hide L clear sections, guess with the track, report the shape-prior path error vs
        the true centre and the guess Dice on confirmed vs other sections.

    ~/venvs/dentalseg/Scripts/python scripts/canal_shape_prior.py --out outputs/shape_exam
"""
import argparse
import glob
import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from case_lists import load_cases  # noqa: E402
import canal_edge_gt as E  # noqa: E402
import canal_gap_fill as G  # noqa: E402
import canal_track_hmm as H  # noqa: E402
from canal_raynet_exam import star_dice  # noqa: E402

LIB = "outputs/shape_lib.npz"
MCSTU_RAYS = "outputs/mcstu_edge/sparse/chunk_*/*_rays.npz"


def build_library(path=LIB, rays=MCSTU_RAYS):
    if Path(path).exists():
        z = np.load(path, allow_pickle=True)
        return list(z["curves"]), list(z["ids"])
    curves, ids = [], []
    for npz in sorted(glob.glob(rays)):
        cid, sp, sides = G.load_case(npz)
        for s, d in sides.items():
            c = d["c"].astype(float) * sp
            if len(c) < 60:
                continue
            for mir in (False, True):
                cc = c.copy()
                if mir:
                    cc[:, 0] *= -1
                curves.append(cc); ids.append(cid)
    np.savez(path, curves=np.array(curves, dtype=object), ids=np.array(ids))
    return curves, ids


def batched_kabsch(A, B):
    """A (N, P, 3) candidates, B (P, 3) target. Returns aligned A and RMS residual (N,)."""
    ca = A.mean(1, keepdims=True); cb = B.mean(0, keepdims=True)
    M = np.einsum("npi,pj->nij", A - ca, B - cb)
    U, _, Vt = np.linalg.svd(M)
    det = np.sign(np.linalg.det(np.einsum("nji,nkj->nik", Vt, U)))
    D = np.ones((len(A), 3)); D[:, 2] = det
    Rm = np.einsum("nji,nj,nkj->nik", Vt, D, U)
    return Rm, ca, cb


class ShapePrior:
    def __init__(self, exclude=None, step=3):
        self.step = step
        curves, ids = build_library()
        self.curves = [c for c, i in zip(curves, ids) if i != exclude]

    def predict(self, flankA, flankB, L, K=10):
        """flankA (F,3) mm sections before the gap (in order), flankB (F,3) after. Returns the
        predicted gap path (L,3) mm, the spread (L,) mm, and the best flank residual."""
        F = len(flankA); W = 2 * F + L
        cand = []
        for c in self.curves:
            for cr in (c, c[::-1]):                        # library curves can run either way
                for j in range(0, len(cr) - W + 1, self.step):
                    cand.append(cr[j:j + W])
        if not cand:
            return None
        C = np.stack(cand).astype(np.float32)              # (N, W, 3)
        fl = np.r_[0:F, F + L:W]
        tgt = np.vstack([flankA, flankB]).astype(np.float32)
        Rm, ca, cb = batched_kabsch(C[:, fl], tgt)
        Af = np.einsum("nij,nwj->nwi", Rm, C[:, fl] - ca) + cb
        res = np.sqrt(((Af - tgt) ** 2).sum(2).mean(1))
        best = np.argsort(res)[:K]
        mid = (np.einsum("nij,nwj->nwi", Rm[best], C[best][:, F:F + L] - ca[best]) + cb).astype(float)   # (K, L, 3)
        path = np.median(mid, 0)
        spread = np.median(np.linalg.norm(mid - path, axis=2), 0)
        return path, spread, float(res[best[0]])


def flank_centres(d, R, a, b, F, sp):
    """Measured wall centres (mm) of the F sections ending at a and starting at b, or None."""
    n = len(d["c"])
    ka, kb = np.arange(a - F + 1, a + 1), np.arange(b, b + F)
    if ka[0] < 0 or kb[-1] >= n:
        return None, None
    f = lambda ks: np.array([(G.wall_centre(d["c"][k], d["u"][k], d["w"][k], R[k], sp) if np.isfinite(R[k]).sum() >= 3
                              else d["c"][k]) for k in ks]) * sp
    return f(ka), f(kb)


def shape_check(SP, d, R, a, b, gC, sp, args):
    """Gap = a+1..b-1 (a, b measured). gC (L,3) guessed centres in voxels. Returns ok (L,), dist (L,)."""
    L = b - a - 1
    fa, fb = flank_centres(d, R, a, b, args.flank, sp)
    if fa is None:
        return None
    out = SP.predict(fa, fb, L, args.K)
    if out is None:
        return None
    path, spread, res = out
    dist = np.linalg.norm(gC * sp - path, axis=1)
    ok = (dist <= args.agree_mm) & (spread <= args.spread_mm) & (res <= args.max_res_mm)
    return ok, dist, path


def exam(args):
    SP = ShapePrior()
    no = int(round(4.0 / 0.4)); rows = []
    for npz in sorted(glob.glob(args.rays)):
        cid, sp, sides = G.load_case(npz)
        if cid in args.skip:
            continue
        _, _, _, _, v0, sides = G.prep_volume(cid, sides, sp)
        rng = np.random.default_rng(zlib.crc32(cid.encode()))
        for s, d in sides.items():
            R = d["r_agree"]; n = len(R); good = np.isfinite(R).mean(1) >= 0.5
            mg = max(no, args.flank) + 2
            for L in args.L:
                i = mg
                while i + L + mg <= n:
                    if not good[i - mg:i + L + mg].all():
                        i += 1; continue
                    blk = np.arange(i, i + L); a, b = i - 1, i + L
                    Rh = R.copy(); Rh[blk] = np.nan; dh = {**d, "r_agree": Rh}
                    for cc in args.contrast:
                        v = v0 if cc >= 1 else G.degrade(v0, d["c"][blk], sp, argparse.Namespace(
                            degrade_contrast=cc, degrade_noise=40, degrade_r_mm=4.0), rng)
                        cen, ra, rb, _ = H.ends(dh, i, i + L, sp, no)
                        Ev = np.stack([H.evidence(v, cen[j], d["u"][k], d["w"][k], sp, 60) for j, k in enumerate(blk)])
                        mean, sd = H.track(Ev, ra, rb, 0.08)
                        r = np.where((sd <= 0.25) & (mean <= 1.9), mean, np.nan)
                        kept = np.isfinite(r).sum(1) >= 3
                        gC = np.array([G.wall_centre(cen[j], d["u"][k], d["w"][k], r[j], sp) if kept[j] else cen[j]
                                       for j, k in enumerate(blk)])
                        o = shape_check(SP, d, Rh, a, b, gC, sp, args)
                        if o is None:
                            continue
                        ok, dist, path = o
                        tC = np.array([G.wall_centre(d["c"][k], d["u"][k], d["w"][k], R[k], sp) for k in blk]) * sp
                        okk = ok & kept; run = np.zeros(L, bool); x = 0
                        while x < L:
                            if not okk[x]:
                                x += 1; continue
                            y = x
                            while y < L and okk[y]:
                                y += 1
                            if y - x >= 3:
                                run[x:y] = True
                            x = y
                        for j, k in enumerate(blk):
                            row = {"case": cid[-4:], "L": L, "c": cc, "pos": min(j + 1, L - j),
                                   "prior_err_mm": float(np.linalg.norm(path[j] - tC[j]))}
                            if kept[j]:
                                pr = H.radii_from(d["c"][k], cen[j], E.fill_angles(H.ANG, r[j]), d["u"][k], d["w"][k], sp)
                                row.update(guess=float(star_dice(pr, E.fill_angles(H.ANG, R[k]))),
                                           guess_err_mm=float(np.linalg.norm(gC[j] * sp - tC[j])), confirmed=bool(run[j]))
                            rows.append(row)
                    i += L + 10
        print(cid[-4:], len(rows), flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "exam_rows.json").write_text(json.dumps(rows))
    for L in args.L:
        for cc in args.contrast:
            x = [q for q in rows if q["L"] == L and q["c"] == cc and q["pos"] > 6]
            if not x:
                continue
            g = [q for q in x if "guess" in q]; cf = [q for q in g if q["confirmed"]]; nc = [q for q in g if not q["confirmed"]]
            f = lambda sel: f"{np.mean([q['guess'] for q in sel]):.3f} (n={len(sel)})" if sel else "  -  "
            print(f"L={L} ({L * 0.4:.0f}mm) contrast {cc:.2f} middle: shape-prior path error median "
                  f"{np.median([q['prior_err_mm'] for q in x]):.2f}mm (track guess {np.median([q['guess_err_mm'] for q in g]):.2f}mm)  "
                  f"guess all {f(g)}  confirmed {f(cf)}  not {f(nc)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rays", default="outputs/sparse_label_v7_rays/chunk_0*/*_rays.npz")
    ap.add_argument("--skip", nargs="*", default=load_cases())
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, nargs="+", default=[20, 40])
    ap.add_argument("--contrast", type=float, nargs="+", default=[1.0, 0.35])
    ap.add_argument("--flank", type=int, default=20, help="兩側各用幾張量到的截面當條件(0.4mm/張)")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--agree_mm", type=float, default=0.5)
    ap.add_argument("--spread_mm", type=float, default=0.8)
    ap.add_argument("--max_res_mm", type=float, default=0.6)
    exam(ap.parse_args())


if __name__ == "__main__":
    main()
