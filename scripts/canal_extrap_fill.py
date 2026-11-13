#!/usr/bin/env python3
"""Extrapolate into the sections that are still empty after canal_track_fill.py (target: 100%).

Every still-empty section takes its wall from the NEAREST known section (measured or predicted by
the track) on each side and is pushed outward one direction at a time with a one-sided track:
state = wall radius, start = the radii at the known end, a small random step per section, the
image (dark-to-bright rise) as evidence. Centre = doctor's track + the end offset held constant.
Head/tail runs have one known side; interior runs use both sides, each section the nearer one
(no clamping between them: that is the interpolating track, already done for gaps <= 16mm).
No sd / width filter by default (the point is to see what 100% costs); every section gets a label.

Exam (--exam): hide the last L clear sections of a clear run, extrapolate from the clear side,
score per-section Dice against the measured radii vs the doctor-position circle (radius = the end
median), by distance from the known end.

Output (--out): <cid>_extrap.nii.gz = track_fill labels + 5 canal (extrapolated) / 6 background shell,
sections.json, summary.json.

    <venv>/python scripts/canal_extrap_fill.py --exam --out outputs/extrap_fill
    <venv>/python scripts/canal_extrap_fill.py --out outputs/extrap_fill
"""
import argparse
from case_lists import load_cases
import glob
import json
import zlib
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import canal_edge_gt as E  # noqa: E402
import canal_gap_fill as G  # noqa: E402
import canal_track_hmm as H  # noqa: E402
from canal_raynet_exam import star_dice  # noqa: E402

ANG, RS = H.ANG, H.RS


def one_sided(Ev, r0, step_mm):
    """Forward filter only from the end radii r0 (A,); returns mean, sd (n, A)."""
    T = np.exp(-0.5 * ((RS[:, None] - RS[None]) / step_mm) ** 2); T /= T.sum(1, keepdims=True)
    p = np.exp(-0.5 * ((RS[None] - r0[:, None]) / 0.05) ** 2); p /= p.sum(1, keepdims=True)
    mean = np.zeros(Ev.shape[:2]); sd = np.zeros(Ev.shape[:2])
    for i in range(len(Ev)):
        p = (p @ T) * Ev[i]; p /= p.sum(1, keepdims=True)
        mean[i] = (p * RS).sum(1)
        sd[i] = np.sqrt(np.maximum((p * RS ** 2).sum(1) - mean[i] ** 2, 0))
    return mean, sd


def extrapolate(v, d, R, e, ks, sp, n_out, args):
    """Sections ks (ordered moving away from known end e). Returns centres (n,3), radii (n,A), sd."""
    stp = 1 if ks[0] > e else -1
    ce, re_ = H.end_profile({**d, "r_agree": R}, e, -stp, n_out, sp)
    off = ce - d["click"][e]
    cen = d["click"][ks] + off
    Ev = np.stack([H.evidence(v, cen[j], d["u"][k], d["w"][k], sp, args.tau) for j, k in enumerate(ks)])
    mean, sd = one_sided(Ev, re_, args.step_mm)
    return cen, mean, sd, re_


def known_radii(d, tr):
    """Measured radii, with the track-filled sections (from <cid>_s<side>_track.npz) added."""
    R = d["r_agree"].copy()
    if tr is not None:
        r = tr["r"]; pred = np.isfinite(r).any(1) & ~np.isfinite(R).any(1)
        R[pred] = r[pred]
    return R


def run_fill(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    no = int(round(args.end_mm / 0.4))
    secs, summary = [], []
    for npz in sorted(glob.glob(args.rays)):
        cid, sp, sides = G.load_case(npz)
        base = nib.load(f"{args.track}/{cid}_track.nii.gz")
        full = np.asanyarray(base.dataobj).astype(np.uint8)
        if cid in args.skip:
            nib.save(nib.Nifti1Image(full, base.affine), out / f"{cid}_extrap.nii.gz")
            continue
        _, _, crop, lo, v, sides = G.prep_volume(cid, sides, sp)
        lab = full[crop].copy()
        st = {"case": cid, "sections": 0, "known": 0, "extrap": 0, "empty_left": 0}
        for s, d in sides.items():
            tf = Path(args.track) / f"{cid}_s{s}_track.npz"
            tr = None
            if tf.exists():
                z = np.load(tf); tr = {"r": z["r"]}
            R = known_radii(d, tr)
            n = len(R); known = np.isfinite(R).sum(1) >= 3
            newC = np.full((n, 3), np.nan); newR = np.full((n, len(ANG)), np.nan); dist = np.zeros(n)
            for a, b in G.gaps(known):
                left, right = a - 1, b          # known neighbours (may be out of range)
                ks = np.arange(a, b)
                parts = []
                if left >= 0 and right < n:
                    mid = (a + b) // 2 if (b - a) % 2 == 0 else (a + b + 1) // 2
                    parts = [(left, ks[ks < mid]), (right, ks[ks >= mid][::-1])]
                elif left >= 0:
                    parts = [(left, ks)]
                elif right < n:
                    parts = [(right, ks[::-1])]
                for e, kk in parts:
                    if not len(kk):
                        continue
                    cen, mean, sd, _ = extrapolate(v, d, R, e, kk, sp, no, args)
                    for j, k in enumerate(kk):
                        r = mean[j].copy()
                        if args.max_sd:
                            r[sd[j] > args.max_sd] = np.nan
                        if args.r_cap_mm:
                            r[r > args.r_cap_mm] = np.nan
                        if np.isfinite(r).sum() >= 3:
                            newC[k] = cen[j]; newR[k] = r; dist[k] = abs(k - e) * 0.4
            add = G.raster(lab.shape, d, newC, newR, sp, args.wall_mm)
            put = (lab == 2) & (add != 2)
            lab[put] = np.where(add[put] == 1, 5, 6)
            ext = np.isfinite(newR).any(1)
            for k in range(n):
                kind = "known" if known[k] else "extrap" if ext[k] else "empty"
                secs.append({"case": cid[-4:], "side": int(s), "sec": k, "kind": kind,
                             "dist_mm": float(dist[k]), "frac": float(np.isfinite(newR[k]).mean()) if ext[k] else 0.0})
            st["sections"] += n; st["known"] += int(known.sum()); st["extrap"] += int(ext.sum())
            st["empty_left"] += int((~known & ~ext).sum())
            np.savez_compressed(out / f"{cid}_s{s}_extrap.npz", c=newC + lo, r=newR, dist=dist)
        full[crop] = lab
        nib.save(nib.Nifti1Image(full, base.affine), out / f"{cid}_extrap.nii.gz")
        summary.append(st)
        print(f"{cid[-4:]} sections {st['sections']} known {st['known'] / st['sections']:.2f} "
              f"+extrap {st['extrap'] / st['sections']:.2f} empty {st['empty_left'] / st['sections']:.2f}", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "sections.json").write_text(json.dumps(secs))
    T = {k: sum(x[k] for x in summary) for k in ("sections", "known", "extrap", "empty_left")}
    print(f"total {T['sections']}: known {T['known'] / T['sections']:.1%}  extrapolated {T['extrap'] / T['sections']:.1%}  "
          f"still empty {T['empty_left'] / T['sections']:.1%}")


def run_exam(args):
    """Hide the L sections just past a clear end and extrapolate into them from that end."""
    no = int(round(args.end_mm / 0.4)); rows = []
    for npz in sorted(glob.glob(args.rays)):
        cid, sp, sides = G.load_case(npz)
        if cid in args.skip:
            continue
        _, _, _, _, v0, sides = G.prep_volume(cid, sides, sp)
        rng = np.random.default_rng(int(cid) if cid.isdigit() else zlib.crc32(cid.encode()))
        for s, d in sides.items():
            R = d["r_agree"]; n = len(R); good = np.isfinite(R).mean(1) >= 0.5
            i = no + 2
            while i + args.L <= n:
                e = i - 1; ks = np.arange(i, i + args.L)
                if not good[e - no:i + args.L].all():
                    i += 1; continue
                for stp in (1, -1):                       # extrapolate forward and backward
                    if stp == 1:
                        ee, kk = e, ks
                        Rh = R.copy(); Rh[ks] = np.nan; Rh[ks[-1] + 1:] = np.nan
                    else:
                        ee, kk = ks[-1] + 1, ks[::-1]
                        if ee + no >= n:
                            continue
                        Rh = R.copy(); Rh[:ks[0]] = np.nan
                    for cc in args.contrast:
                        v = v0 if cc >= 1 else G.degrade(v0, d["c"][ks], sp, argparse.Namespace(
                            degrade_contrast=cc, degrade_noise=40, degrade_r_mm=4.0), rng)
                        cen, mean, sd, re_ = extrapolate(v, d, Rh, ee, kk, sp, no, args)
                        circ = np.full(len(ANG), np.nanmedian(re_))
                        for j, k in enumerate(kk):
                            truth = E.fill_angles(ANG, R[k])
                            pr = H.radii_from(d["c"][k], cen[j], mean[j], d["u"][k], d["w"][k], sp)
                            hold = H.radii_from(d["c"][k], cen[j], re_, d["u"][k], d["w"][k], sp)
                            doc = H.radii_from(d["c"][k], d["click"][k], circ, d["u"][k], d["w"][k], sp)
                            rows.append({"case": cid[-4:], "c": cc, "dist": (j + 1) * 0.4,
                                         "extrap": float(star_dice(pr, truth)), "hold": float(star_dice(hold, truth)),
                                         "doctor": float(star_dice(doc, truth)),
                                         "sure": float((sd[j] <= 0.25).mean())})
                i += args.L + 10
        print(cid[-4:], len(rows), flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "exam.json").write_text(json.dumps(rows))
    for cc in args.contrast:
        for lo_, hi_ in ((0, 2), (2, 4), (4, 8), (8, 16), (16, 40)):
            x = [r for r in rows if r["c"] == cc and lo_ < r["dist"] <= hi_]
            if not x:
                continue
            m = lambda k: np.median([r[k] for r in x])
            print(f"contrast {cc:.2f} dist {lo_:>2}-{hi_:<2}mm n={len(x):5d}  extrap Dice {m('extrap'):.2f}  "
                  f"hold-end {m('hold'):.2f}  doctor-circle {m('doctor'):.2f}  sure {m('sure'):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rays", default="outputs/sparse_label_v6_rays/chunk_0*/*_rays.npz")
    ap.add_argument("--track", default="outputs/track_fill")
    ap.add_argument("--skip", nargs="*", default=load_cases())
    ap.add_argument("--out", default="outputs/extrap_fill")
    ap.add_argument("--end_mm", type=float, default=4.0)
    ap.add_argument("--step_mm", type=float, default=0.08)
    ap.add_argument("--tau", type=float, default=60)
    ap.add_argument("--max_sd", type=float, default=0.0, help="0 = 不丟(要 100%%)")
    ap.add_argument("--r_cap_mm", type=float, default=0.0, help="0 = 不設上限")
    ap.add_argument("--wall_mm", type=float, default=0.4)
    ap.add_argument("--exam", action="store_true")
    ap.add_argument("--L", type=int, default=90, help="考試藏幾張(90 = 36mm,和最長空白一樣)")
    ap.add_argument("--contrast", type=float, nargs="+", default=[1.0, 0.0])
    args = ap.parse_args()
    run_exam(args) if args.exam else run_fill(args)


if __name__ == "__main__":
    main()
