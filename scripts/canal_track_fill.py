#!/usr/bin/env python3
"""Fill the empty sections with the Bayesian track (canal_track_hmm.py) and write the labels.

Gaps = runs of sections between two sections where >= --end_frac of the directions were measured,
at most --max_gap_mm long, holding at least one empty section. Each end direction may come from
up to --end_mm further out (end_profile). Only the EMPTY sections in the gap are filled; partly
measured sections keep what they have. Directions the track is unsure of (posterior sd > --max_sd)
or that come out wider than --r_cap_mm stay unknown (drop only, never clamped).

Output per case (<cid>_track.nii.gz), on top of the strict sparse label:
  0 background (measured)   1 canal (measured)   2 ignore
  3 canal (predicted)       4 background shell around the predicted canal (predicted)
plus sections.json (per section: measured / predicted / empty) and summary.json.

    <venv>/python scripts/canal_track_fill.py --out outputs/track_fill
"""
import argparse
from case_lists import load_cases
import glob
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import canal_gap_fill as G  # noqa: E402
import canal_track_hmm as H  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rays", default="outputs/sparse_label_v6_rays/chunk_0*/*_rays.npz")
    ap.add_argument("--base", default="outputs/sparse_label_v6_rays/chunk_*/{}_agree.nii.gz")
    ap.add_argument("--skip", nargs="*", default=load_cases(), help="醫生標註錯位:只複製原本的稀疏標註")
    ap.add_argument("--out", default="outputs/track_fill")
    ap.add_argument("--end_frac", type=float, default=0.25, help="端點截面至少要量到這麼多方向")
    ap.add_argument("--end_mm", type=float, default=4.0, help="每個方向往缺口外最多找多遠")
    ap.add_argument("--max_gap_mm", type=float, default=16.0)
    ap.add_argument("--step_mm", type=float, default=0.08)
    ap.add_argument("--tau", type=float, default=60)
    ap.add_argument("--max_sd", type=float, default=0.25)
    ap.add_argument("--r_cap_mm", type=float, default=1.9, help="超過就不收(文獻最大管徑 3.8mm)")
    ap.add_argument("--wall_mm", type=float, default=0.4)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    no = int(round(args.end_mm / 0.4)); max_l = int(round(args.max_gap_mm / 0.4))
    summary, secs = [], []
    for npz in sorted(glob.glob(args.rays)):
        cid, sp, sides = G.load_case(npz)
        base = nib.load(glob.glob(args.base.format(cid))[0])
        full = np.asanyarray(base.dataobj).astype(np.uint8)
        if cid in args.skip:
            nib.save(nib.Nifti1Image(full, base.affine), out / f"{cid}_track.nii.gz")
            continue
        _, shape, crop, lo, v, sides = G.prep_volume(cid, sides, sp)
        lab = full[crop].copy()
        st = {"case": cid, "sections": 0, "measured": 0, "predicted": 0, "empty_left": 0, "dirs_dropped": 0, "dirs_pred": 0}
        for s, d in sides.items():
            R = d["r_agree"]; n = len(R); cov = np.isfinite(R).mean(1)
            newC = np.full((n, 3), np.nan); newR = np.full((n, len(H.ANG)), np.nan)
            ends_ok = np.where(cov >= args.end_frac)[0]
            for a, b in zip(ends_ok[:-1], ends_ok[1:]):
                if not 1 <= b - a - 1 <= max_l or not (cov[a + 1:b] == 0).any():
                    continue
                cen, ra, rb, _ = H.ends(d, a + 1, b, sp, no)
                ks = np.arange(a + 1, b)
                Ev = np.stack([H.evidence(v, cen[j], d["u"][k], d["w"][k], sp, args.tau) for j, k in enumerate(ks)])
                mean, sd = H.track(Ev, ra, rb, args.step_mm)
                for j, k in enumerate(ks):
                    if cov[k] > 0:
                        continue
                    r = np.where((sd[j] <= args.max_sd) & (mean[j] <= args.r_cap_mm), mean[j], np.nan)
                    st["dirs_dropped"] += int(np.isnan(r).sum()); st["dirs_pred"] += int(np.isfinite(r).sum())
                    if np.isfinite(r).sum() >= 3:
                        newC[k] = cen[j]; newR[k] = r
            add = G.raster(lab.shape, d, newC, newR, sp, args.wall_mm)
            put = (lab == 2) & (add != 2)
            lab[put] = np.where(add[put] == 1, 3, 4)
            pred = np.isfinite(newR).any(1)
            for k in range(n):
                kind = "measured" if cov[k] > 0 else "predicted" if pred[k] else "empty"
                secs.append({"case": cid[-4:], "side": int(s), "sec": k, "kind": kind,
                             "pred_frac": float(np.isfinite(newR[k]).mean()) if pred[k] else 0.0})
            st["sections"] += n; st["measured"] += int((cov > 0).sum())
            st["predicted"] += int(pred.sum()); st["empty_left"] += int(((cov == 0) & ~pred).sum())
            np.savez_compressed(out / f"{cid}_s{s}_track.npz", c=newC + lo, r=newR)
        full[crop] = lab
        nib.save(nib.Nifti1Image(full, base.affine), out / f"{cid}_track.nii.gz")
        summary.append(st)
        print(f"{cid[-4:]} sections {st['sections']}  measured {st['measured'] / st['sections']:.2f}  "
              f"+predicted {st['predicted'] / st['sections']:.2f}  empty {st['empty_left'] / st['sections']:.2f}", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "sections.json").write_text(json.dumps(secs))
    T = {k: sum(x[k] for x in summary) for k in ("sections", "measured", "predicted", "empty_left", "dirs_dropped", "dirs_pred")}
    print(f"total {T['sections']} sections: measured {T['measured']} ({T['measured'] / T['sections']:.1%}), "
          f"predicted {T['predicted']} ({T['predicted'] / T['sections']:.1%}), still empty {T['empty_left']} "
          f"({T['empty_left'] / T['sections']:.1%}); predicted directions dropped {T['dirs_dropped'] / max(T['dirs_dropped'] + T['dirs_pred'], 1):.1%}")


if __name__ == "__main__":
    main()
