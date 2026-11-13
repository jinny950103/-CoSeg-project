#!/usr/bin/env python3
"""v8 exam (to-do 4): are the guessed sections that v8 keeps at least as good as the doctor's circle?

v8 = edge labels (v7 pipeline) + caps from canal_cap_guesses.py (interp <= 12 sections, extrap <= 10,
longer only where the longitudinal sections or the 508 prediction confirm).

  interp  hide L clear sections (L up to the cap), image faded there (--contrast), guess them with the
          track exactly as canal_track_fill.py does (end_mm 4, max_sd 0.25, r_cap 1.9), score each
          section (star Dice vs the measured radii) next to the doctor's circle (centre = click,
          radius = median of the two end radii), and the share of sections the track would label.
  stats   lengths of the guessed runs in a capped output (--sections): anything still over the cap
          must be confirmed.
Extrapolation: canal_extrap_fill.py --exam --L 10 (same scoring, doctor circle included).

    ~/venvs/dentalseg/Scripts/python scripts/canal_v8_exam.py --out outputs/v8_exam_hosp --sections outputs/v8_hosp/sections.json
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


def interp_exam(args):
    no = int(round(args.end_mm / 0.4)); rows = []
    files = sorted(glob.glob(args.rays))
    for npz in files[:args.limit] if args.limit else files:
        cid, sp, sides = G.load_case(npz)
        if cid in args.skip:
            continue
        _, _, _, _, v0, sides = G.prep_volume(cid, sides, sp)
        rng = np.random.default_rng(zlib.crc32(cid.encode()))
        for s, d in sides.items():
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
                        circ = np.full(len(H.ANG), np.nanmedian(np.concatenate([ra, rb])))
                        for j, k in enumerate(blk):
                            truth = E.fill_angles(H.ANG, R[k])
                            r = np.where((sd[j] <= 0.25) & (mean[j] <= 1.9), mean[j], np.nan)
                            kept = np.isfinite(r).sum() >= 3
                            row = {"case": cid[-4:], "L": L, "c": cc, "kept": bool(kept),
                                   "doctor": float(star_dice(H.radii_from(d["c"][k], d["click"][k], circ, d["u"][k], d["w"][k], sp), truth))}
                            if kept:
                                pr = H.radii_from(d["c"][k], cen[j], E.fill_angles(H.ANG, r), d["u"][k], d["w"][k], sp)
                                row["guess"] = float(star_dice(pr, truth))
                            rows.append(row)
                    i += L + 10
        print(cid[-4:], len(rows), flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "interp_rows.json").write_text(json.dumps(rows))
    print("interpolation exam (hidden clear sections; median Dice over the sections the track labels)")
    for L in args.L:
        for cc in args.contrast:
            x = [r for r in rows if r["L"] == L and r["c"] == cc]
            k = [r for r in x if r["kept"]]
            if not k:
                continue
            g = np.median([r["guess"] for r in k]); dc = np.median([r["doctor"] for r in k])
            print(f"  hide {L:2d} ({L * 0.4:.1f}mm) contrast {cc:.2f}  n={len(x):5d}  labelled {len(k) / len(x):.0%}  "
                  f"guess {g:.3f}  doctor-circle {dc:.3f}  {'PASS' if g >= dc else 'FAIL'}  "
                  f"(guess >= doctor in {np.mean([r['guess'] >= r['doctor'] for r in k]):.0%} of sections)")


def run_stats(path, max_i, max_e):
    S = json.load(open(path))
    by = {}
    for s in S:
        by.setdefault((s["case"], s["side"]), []).append(s)
    lens = {"interp": [], "extrap": []}; over = {"interp": [], "extrap": []}
    for key, ss in by.items():
        ss.sort(key=lambda x: x["sec"])
        for kind, cap in (("interp", max_i), ("extrap", max_e)):
            i = 0
            while i < len(ss):
                if ss[i]["kind"] != kind:
                    i += 1; continue
                j = i
                while j < len(ss) and ss[j]["kind"] == kind:
                    j += 1
                unconf = j - i if kind == "extrap" else 0
                if kind == "interp":                          # longest unconfirmed stretch
                    run = 0
                    for x in ss[i:j]:
                        run = 0 if x["confirmed"] else run + 1; unconf = max(unconf, run)
                lens[kind].append(j - i)
                if kind == "interp" and unconf > cap:
                    over[kind].append((key, i, j, unconf))
                i = j
    ext = [s["dist_mm"] for s in S if s["kind"] == "extrap"]
    tot = len(S); cnt = lambda k: sum(1 for s in S if s["kind"] == k)
    print(f"stats {path}: {tot} sections  measured {cnt('measured') / tot:.1%}  interp {cnt('interp') / tot:.1%}  "
          f"extrap {cnt('extrap') / tot:.1%}  ignore {(cnt('ignored_interp') + cnt('ignored_extrap')) / tot:.1%}")
    li = np.array(lens["interp"])
    print(f"  interp runs {len(li)}: median {np.median(li):.0f}  mean {li.mean():.1f} sections  > {max_i}: {np.sum(li > max_i)} "
          f"(all confirmed: unconfirmed stretch > {max_i} = {len(over['interp'])})  longest {li.max()} ({li.max() * 0.4:.1f}mm)")
    print(f"  extrap sections {len(ext)}: farthest {max(ext) if ext else 0:.1f}mm from the known end (cap {max_e * 0.4:.1f}mm)")
    return len(over["interp"]) == 0 and (max(ext) if ext else 0) <= max_e * 0.4 + 1e-6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rays", default="outputs/sparse_label_v7_rays/chunk_0*/*_rays.npz")
    ap.add_argument("--skip", nargs="*", default=load_cases())
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sections", default="", help="capped sections.json for the length stats")
    ap.add_argument("--L", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--contrast", type=float, nargs="+", default=[1.0, 0.35, 0.15])
    ap.add_argument("--end_mm", type=float, default=4.0)
    ap.add_argument("--max_interp", type=int, default=12)
    ap.add_argument("--max_extrap", type=int, default=10)
    ap.add_argument("--no_exam", action="store_true")
    args = ap.parse_args()
    if args.sections:
        ok = run_stats(args.sections, args.max_interp, args.max_extrap)
        print(f"  caps respected: {'YES' if ok else 'NO'}")
    if not args.no_exam:
        interp_exam(args)


if __name__ == "__main__":
    main()
