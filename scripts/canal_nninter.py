#!/usr/bin/env python3
"""Problem 3 (2026-11-13): a second, independent confirmer for long guessed runs: nnInteractive.

nnInteractive (Isensee et al. 2025, arXiv 2503.08373; weights CC BY-NC-SA 4.0, academic use only)
is a promptable 3D segmentation model trained on 120+ public 3D datasets; it never saw our data.
Prompt = positive points at the measured canal centres just before and just after a gap. Its 3D
mask is compared with our guessed contour section by section (the same Dice as the 508 check,
canal_cap_guesses.sec_dice); >= --agree for >= 3 sections in a row = confirmed.

  exam  hide L clear sections (image faded by --contrast), guess them with the track exactly as
        canal_track_fill, prompt nnInteractive at the ends only, and report
          - nnInteractive vs the measured contour (is it any good by itself?)
          - our guess vs the measured contour, on the sections nnInteractive confirms
  real  the long runs still turned into ignore in a capped output (--capped summary.json): how many
        of their sections would nnInteractive confirm.

    ~/venvs/dentalseg/Scripts/python scripts/canal_nninter.py --mode exam --limit 10 --out outputs/nninter_exam
    ~/venvs/dentalseg/Scripts/python scripts/canal_nninter.py --mode real --capped outputs/v9_hosp --out outputs/nninter_real
"""
import argparse
import glob
import json
import sys
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from case_lists import load_cases  # noqa: E402
import canal_edge_gt as E  # noqa: E402
import canal_gap_fill as G  # noqa: E402
import canal_track_hmm as H  # noqa: E402
from canal_cap_guesses import sec_dice, steady  # noqa: E402

MODEL = "outputs/nninteractive_model/nnInteractive_v1.0"


def make_session():
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    s = nnInteractiveInferenceSession(device=torch.device("cuda"), use_torch_compile=False, verbose=False,
                                      torch_n_threads=8, do_autozoom=True)
    s.initialize_from_trained_model_folder(MODEL, use_fold=0)
    return s


def prompt_points(d, R, a, b, sp, n_each=3, step=2):
    """Measured wall centres of the clear sections just before a and just after b."""
    n = len(d["c"]); pts = []
    for ks in (range(a, max(a - n_each * step, -1), -step), range(b, min(b + n_each * step, n), step)):
        for k in ks:
            if np.isfinite(R[k]).sum() >= 3:
                pts.append(G.wall_centre(d["c"][k], d["u"][k], d["w"][k], R[k], sp))
    return np.array(pts)


def segment(sess, v, d, a, b, pts, sp, margin_mm=8.0):
    """nnInteractive mask (full crop shape, bool) from positive points, run in a box around a..b."""
    ks = np.arange(max(a - 6, 0), min(b + 7, len(d["c"])))
    allp = np.vstack([d["c"][ks], pts])
    m = int(round(margin_mm / sp))
    lo = np.maximum(np.floor(allp.min(0)).astype(int) - m, 0)
    hi = np.minimum(np.ceil(allp.max(0)).astype(int) + m + 1, np.array(v.shape))
    box = v[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    sess.set_image(box[None])
    tgt = np.zeros(box.shape, np.uint8)
    sess.set_target_buffer(tgt)
    for i, p in enumerate(pts):
        q = tuple(int(round(x)) for x in np.clip(p - lo, 0, np.array(box.shape) - 1))
        sess.add_point_interaction(q, include_interaction=True, run_prediction=True)
    out = np.zeros(v.shape, bool)
    out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = tgt > 0
    sess.reset_interactions()
    return out


def exam(args, sess):
    no = int(round(4.0 / 0.4)); rows = []
    files = sorted(glob.glob(args.rays))
    done = 0
    for npz in files:
        cid, sp, sides = G.load_case(npz)
        if cid in args.skip:
            continue
        if args.limit and done >= args.limit:
            break
        done += 1
        _, _, crop, _, v0, sides = G.prep_volume(cid, sides, sp)
        rng = np.random.default_rng(zlib.crc32(cid.encode()))
        FM = None
        if args.mask:                                     # a fixed, precomputed prediction instead of nnInteractive
            mp = Path(args.mask.format(cid))
            if not mp.exists():
                continue
            FM = np.asanyarray(nib.load(mp).dataobj)[crop] > 0
        for s, d in sides.items():
            R = d["r_agree"]; n = len(R); good = np.isfinite(R).mean(1) >= 0.5
            Rf = np.array([E.fill_angles(G.ANG, r) if np.isfinite(r).sum() >= 3 else np.full(len(G.ANG), np.nan) for r in R])
            mg = no + 2
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
                        trC = np.full((n, 3), np.nan); trR = np.full((n, len(G.ANG)), np.nan)
                        trC[blk] = cen
                        trR[blk] = np.where((sd <= 0.25) & (mean <= 1.9), mean, np.nan)
                        pts = prompt_points(d, Rh, a, b, sp)
                        if len(pts) < 2:
                            continue
                        M = FM if FM is not None else segment(sess, v, d, a, b, pts, sp)
                        agree = np.zeros(n, bool); ag = {}
                        for k in blk:
                            if np.isfinite(trR[k]).sum() < 3:
                                continue
                            gR = E.fill_angles(G.ANG, trR[k]); tmp = trR.copy(); tmp[k] = gR
                            ag[k] = sec_dice(M, d, trC, tmp, k, v.shape, sp, 0.4)
                            agree[k] = ag[k] >= args.agree
                        conf = steady(agree, 3)
                        for j, k in enumerate(blk):
                            row = {"case": cid[-4:], "L": L, "c": cc, "pos": min(j + 1, L - j),
                                   "nni_vs_truth": sec_dice(M, d, d["c"], Rf, k, v.shape, sp, 0.4)}
                            if k in ag:
                                pr = H.radii_from(d["c"][k], cen[j], E.fill_angles(H.ANG, trR[k]), d["u"][k], d["w"][k], sp)
                                row.update(guess=float(star_dice(pr, E.fill_angles(H.ANG, R[k]))),
                                           nni_vs_guess=ag[k], confirmed=bool(conf[k]))
                            rows.append(row)
                    i += L + 10
        print(cid[-4:], len(rows), flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "exam_rows.json").write_text(json.dumps(rows))
    for L in args.L:
        for cc in args.contrast:
            x = [r for r in rows if r["L"] == L and r["c"] == cc and r["pos"] > 6]
            if not x:
                continue
            g = [r for r in x if "guess" in r]; cf = [r for r in g if r["confirmed"]]; nc = [r for r in g if not r["confirmed"]]
            f = lambda sel, key: f"{np.nanmean([r[key] for r in sel]):.3f} (n={len(sel)})" if sel else "  -  "
            print(f"L={L} ({L * 0.4:.0f}mm) contrast {cc:.2f} middle sections: nnInteractive vs truth {f(x, 'nni_vs_truth')}  "
                  f"guess all {f(g, 'guess')}  guess on nnI-confirmed {f(cf, 'guess')}  not confirmed {f(nc, 'guess')}")


def real(args, sess):
    summ = json.load(open(Path(args.capped) / "summary.json"))
    todo = {x["case"]: x["drop_runs_interp"] for x in summ if x["drop_runs_interp"]}
    rows = []
    for npz in sorted(glob.glob(args.rays)):
        cid, sp, sides = G.load_case(npz)
        if cid not in todo or cid in args.skip:
            continue
        _, _, crop, lo, v, sides = G.prep_volume(cid, sides, sp)
        Mcase = np.zeros(v.shape, bool)
        for s_, x, y in todo[cid]:
            d = sides[s_] if s_ in sides else sides[str(s_)]
            R = d["r_agree"]; n = len(R); cov = np.isfinite(R).mean(1)
            a = max([k for k in range(x) if cov[k] >= 0.25], default=None)
            b = min([k for k in range(y, n) if cov[k] >= 0.25], default=None)
            if a is None or b is None:
                continue
            tz = np.load(Path(args.track) / f"{cid}_s{s_}_track.npz")
            trC, trR = tz["c"] - lo, tz["r"]
            pts = prompt_points(d, R, a, b, sp)
            M = segment(sess, v, d, a, b, pts, sp)
            Mcase |= M
            agree = np.zeros(n, bool)
            for k in range(x, y):
                if np.isfinite(trR[k]).sum() >= 3:
                    tmp = trR.copy(); tmp[k] = E.fill_angles(G.ANG, trR[k])
                    agree[k] = sec_dice(M, d, trC, tmp, k, v.shape, sp, 0.4) >= args.agree
            conf = steady(agree, 3)
            rest, run = 0, 0
            for k in range(x, y):                        # longest stretch still unconfirmed
                run = 0 if conf[k] else run + 1; rest = max(rest, run)
            rows.append({"case": cid[-4:], "side": int(s_), "len": y - x, "confirmed": int(conf[x:y].sum()),
                         "longest_left": rest, "now_ok": rest <= args.max_interp})
            print(cid[-4:], rows[-1], flush=True)
        src = nib.load(f"{args.extrap}/{cid}_extrap.nii.gz")     # full grid, for canal_cap_guesses --pred2
        full = np.zeros(src.shape, np.uint8); full[crop] = Mcase
        (Path(args.out) / "masks").mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(full, src.affine), Path(args.out) / "masks" / f"{cid}.nii.gz")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "real_rows.json").write_text(json.dumps(rows, indent=0))
    print(f"long runs {len(rows)}: sections {sum(r['len'] for r in rows)}, nnI-confirmed {sum(r['confirmed'] for r in rows)}; "
          f"runs that would be short enough after confirmation: {sum(r['now_ok'] for r in rows)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["exam", "real"], required=True)
    ap.add_argument("--rays", default="outputs/sparse_label_v7_rays/chunk_0*/*_rays.npz")
    ap.add_argument("--track", default="outputs/track_fill_v7")
    ap.add_argument("--extrap", default="outputs/extrap_fill_v7", help="real: grid for the saved masks")
    ap.add_argument("--capped", default="outputs/v9_hosp")
    ap.add_argument("--skip", nargs="*", default=load_cases())
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, nargs="+", default=[20, 40])
    ap.add_argument("--contrast", type=float, nargs="+", default=[1.0, 0.35])
    ap.add_argument("--agree", type=float, default=0.8)
    ap.add_argument("--mask", default="", help="exam a fixed prediction instead (e.g. DentalSegmentator), {} = case id; "
                                             "it saw the original image, so only --contrast 1 is meaningful")
    ap.add_argument("--max_interp", type=int, default=12)
    args = ap.parse_args()
    sess = None if args.mask else make_session()
    exam(args, sess) if args.mode == "exam" else real(args, sess)


if __name__ == "__main__":
    from canal_raynet_exam import star_dice  # noqa: E402
    main()
