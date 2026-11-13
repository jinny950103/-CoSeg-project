#!/usr/bin/env python3
"""Cap the guessed stretches of the edge labels (paper standard, 2026-11-12) -> 'ignore'.

Standard: a guessed section must not be worse than the doctor-position circle (Dice .77-.79).
From the hidden-section exams that holds for interpolation up to 12 sections (4.8 mm) and for
extrapolation up to 10 sections (4 mm) from the known end. Longer guesses become ignore (2),
except where the longitudinal sections (canal_longitudinal.py) see the canal walls and agree
with the guess.

Input: the v7 pipeline outputs (canal_track_fill -> canal_extrap_fill), labels 0-6:
  0/1 measured bg/canal, 2 ignore, 3/4 interpolated canal/shell, 5/6 extrapolated canal/shell.

Rules per side (sections 0.4 mm apart):
  interp  runs of consecutive interpolated sections. A run longer than --max_interp is checked on
          the longitudinal sections (--confirm): a section is confirmed when, in the up/down (V) or
          buccal/lingual (H) unrolled view, both walls rise by >= --min_rise HU and both lie within
          --agree_mm of the guessed contour (--lower_only: in V the lower wall alone is enough).
          Confirmed sections in runs of >= --min_confirm_run act as new ends; every unconfirmed
          stretch still longer than --max_interp becomes ignore.
  extrap  sections more than --max_extrap sections from their known end become ignore, and so do
          extrapolated sections whose known end was an interpolated section that was dropped.
Only voxels written by a dropped section (values 3/4 or 5/6) are changed; measured voxels never are.

Output (--out): <cid>_capped.nii.gz (same values, dropped -> 2), sections.json, summary.json.

    ~/venvs/dentalseg/Scripts/python scripts/canal_cap_guesses.py --out outputs/capped_v7
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from case_lists import load_cases  # noqa: E402
import canal_gap_fill as G  # noqa: E402
import canal_longitudinal as LG  # noqa: E402
import canal_track_hmm as H  # noqa: E402


def runs(mask):
    """[(a, b)] runs of True; b exclusive."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        out.append((i, j)); i = j
    return out


def confirm_gap(v, d, a, b, trR, sp, args):
    """Sections a+1..b-1 (a, b = ends with >= 25% coverage, as canal_track_fill). Returns bool (b-a-1,).
    --frame both = canal OR pano."""
    if getattr(args, "frame", "canal") == "both":
        return (confirm_gap(v, d, a, b, trR, sp, argparse.Namespace(**{**vars(args), "frame": "canal"}))
                | confirm_gap(v, d, a, b, trR, sp, argparse.Namespace(**{**vars(args), "frame": "pano"})))
    no = int(round(args.end_mm / 0.4))
    cen, _, _, _ = H.ends(d, a + 1, b, sp, no)               # the centres the track used
    R_ends = d["r_agree"]
    lo = LG.longitudinal(v, d, a, b, cen, sp, args, R_ends)
    ks = np.arange(a + 1, b)
    V, Hh = lo["V"], lo["Hh"]
    ok = np.zeros(len(ks), bool)
    for ai, AX in enumerate((V, Hh)):
        if ai == 1 and getattr(args, "frame", "canal") == "pano" and getattr(args, "pano_only_v", False):
            continue
        off = np.einsum("ij,ij->i", cen - lo["Cg"], AX) * sp   # guess centre seen from the view's centre line
        yp, ym, _, _, _ = lo[f"walls{ai}"]
        rp, rm = lo[f"rise{ai}"]
        yp, ym, rp, rm = yp[1:-1], ym[1:-1], rp[1:-1], rm[1:-1]
        for j, k in enumerate(ks):
            r = trR[k]
            if np.isfinite(r).sum() < 3:
                continue
            tp, tm = LG.ray_radius(r, d["u"][k], d["w"][k], AX[j])   # guessed walls along +AX / -AX
            up_ok = rp[j] >= args.min_rise and abs(yp[j] - (off[j] + tp)) <= args.agree_mm
            dn_ok = rm[j] >= args.min_rise and abs(ym[j] - (off[j] - tm)) <= args.agree_mm
            if up_ok and dn_ok or (args.lower_only and ai == 0 and dn_ok):
                ok[j] = True
    return ok


def sec_dice(P, d, C, Rr, k, shape, sp, wall_mm):
    """Dice of the 508 prediction P vs one section's contour (centre C[k], radii Rr[k]), inside that
    section's footprint (contour + wall shell, 0.4 mm thick along the canal)."""
    n = len(C)
    Cm = np.full((n, 3), np.nan); Rm = np.full_like(Rr, np.nan)
    Cm[k], Rm[k] = C[k], Rr[k]
    add = G.raster(shape, d, Cm, Rm, sp, wall_mm)
    g = add == 1; p = P & (add != 2)
    den = g.sum() + p.sum()
    return float(2 * (p & g).sum() / den) if den else np.nan


def model_tips(pc, P, d, sp, args):
    """Tip of the model's predicted canal at both ends of this side's track (to-do 2, method A).
    Prediction voxels = components of P that touch the track clicks (within 3 voxels). Tip = centroid
    of the voxels within --tip_mm of the farthest one outward, searched within --tip_near_mm of the
    track end (same rule as canal_foramen.label_end). Returns [(tip or None, t_out)] for ends 0, n-1."""
    from canal_foramen import label_end
    n = len(d["c"]); nd = int(round(4.0 / 0.4))
    m = np.zeros(P.shape, bool)
    ci = np.clip(np.round(d["click"]).astype(int), 0, np.array(P.shape) - 1)
    m[tuple(ci.T)] = True
    ids = np.unique(pc[ndimage.binary_dilation(m, iterations=3) & P]); ids = ids[ids > 0]
    side_pred = np.isin(pc, ids)
    res = []
    for end, sg in ((0, -1), (n - 1, 1)):
        t_out = d["c"][end] - d["c"][end - sg * nd]; t_out = t_out / np.linalg.norm(t_out)
        tip = None
        if side_pred.any():
            tp = label_end(side_pred, d["click"][end], t_out, sp, near_mm=args.tip_near_mm, tip_mm=0.8)
            if not np.allclose(tp, d["click"][end]):
                tip = tp
        res.append((tip, t_out))
    return res


def steady(ok, min_run):
    out = np.zeros_like(ok)
    for x, y in runs(ok):
        if y - x >= min_run:
            out[x:y] = True
    return out


def exam(args):
    """Is a longitudinally confirmed guess good enough? Hide L clear sections (image faded by
    --contrast), guess them with the track exactly as canal_track_fill does, run the confirmation,
    and score the guess (star Dice vs the measured radii) on confirmed vs unconfirmed sections."""
    import canal_edge_gt as E
    from canal_raynet_exam import star_dice
    no = int(round(args.end_mm / 0.4)); rows = []
    for npz in sorted(glob.glob(args.rays)):
        cid, sp, sides = G.load_case(npz)
        if cid in args.skip or (args.cases and cid[-4:] not in args.cases):
            continue
        _, _, _, _, v0, sides = G.prep_volume(cid, sides, sp)
        rng = np.random.default_rng(int(cid))
        for s, d in sides.items():
            R = d["r_agree"]; n = len(R); good = np.isfinite(R).mean(1) >= 0.5
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
                        Ev = np.stack([H.evidence(v, cen[j], d["u"][k], d["w"][k], sp, args.tau) for j, k in enumerate(blk)])
                        mean, sd = H.track(Ev, ra, rb, 0.08)
                        trR = np.full_like(R, np.nan)
                        trR[blk] = np.where((sd <= 0.25) & (mean <= 1.9), mean, np.nan)
                        res = {}
                        for lo_ in (False, True):
                            res[lo_] = steady(confirm_gap(v, dh, a, b, trR, sp, argparse.Namespace(**{**vars(args), "lower_only": lo_})),
                                              args.min_confirm_run)
                        for j, k in enumerate(blk):
                            if np.isfinite(trR[k]).sum() < 3:
                                continue
                            pr = H.radii_from(d["c"][k], cen[j], E.fill_angles(H.ANG, trR[k]), d["u"][k], d["w"][k], sp)
                            rows.append({"case": cid[-4:], "L": L, "c": cc, "pos": min(j + 1, L - j),
                                         "dice": float(star_dice(pr, E.fill_angles(H.ANG, R[k]))),
                                         "both": bool(res[False][j]), "lower": bool(res[True][j])})
                    i += L + 10
        print(cid[-4:], len(rows), flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "exam_rows.json").write_text(json.dumps(rows))
    for L in args.L:
        for cc in args.contrast:
            x = [r for r in rows if r["L"] == L and r["c"] == cc and r["pos"] > args.max_interp // 2]
            if not x:
                continue
            m = lambda sel: (f"{np.mean([r['dice'] for r in sel]):.3f} (n={len(sel)})" if sel else "  -  ")
            print(f"L={L} ({L * 0.4:.0f}mm) contrast {cc:.2f} | middle sections (>{args.max_interp // 2} from both ends): "
                  f"all {m(x)}  confirmed-both {m([r for r in x if r['both']])}  "
                  f"lower-only {m([r for r in x if r['lower']])}  not-confirmed(lower) {m([r for r in x if not r['lower']])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exam", action="store_true", help="考試:藏清楚段,看確認過的猜測準不準")
    ap.add_argument("--L", type=int, nargs="+", default=[20, 40])
    ap.add_argument("--contrast", type=float, nargs="+", default=[1.0, 0.35, 0.15])
    ap.add_argument("--rays", default="outputs/sparse_label_v7_rays/chunk_0*/*_rays.npz")
    ap.add_argument("--track", default="outputs/track_fill_v7")
    ap.add_argument("--extrap", default="outputs/extrap_fill_v7")
    ap.add_argument("--skip", nargs="*", default=load_cases())
    ap.add_argument("--cases", nargs="*", default=[], help="只跑這些病歷號末 4 碼")
    ap.add_argument("--out", default="outputs/capped_v7")
    ap.add_argument("--max_interp", type=int, default=12, help="內插最多幾張(0.4mm/張)")
    ap.add_argument("--max_extrap", type=int, default=10, help="外插離已知端最多幾張")
    ap.add_argument("--confirm", type=int, default=1, help="1 = 長內插段用縱向斷面確認")
    ap.add_argument("--lower_only", action="store_true", help="縱向 V 斷面只要下緣清楚且一致就算確認")
    ap.add_argument("--agree_mm", type=float, default=0.4, help="縱向管壁和推測輪廓差多少以內算一致")
    ap.add_argument("--min_confirm_run", type=int, default=3, help="連續確認幾張才當新端點")
    ap.add_argument("--end_mm", type=float, default=4.0, help="和 canal_track_fill 相同")
    ap.add_argument("--wall_mm", type=float, default=0.4)
    # longitudinal settings (same defaults as canal_longitudinal.py)
    ap.add_argument("--ctx_mm", type=float, default=4.0)
    ap.add_argument("--half_mm", type=float, default=3.5)
    ap.add_argument("--slab_mm", type=float, default=0.4)
    ap.add_argument("--frame", choices=["canal", "pano", "both"], default="canal", help="縱向斷面:沿管子 / 真正環景圖 / 兩者取聯集")
    ap.add_argument("--pano_smooth_mm", type=float, default=5.0)
    ap.add_argument("--pano_only_v", action="store_true", help="pano: 只用上下(V)那張,不用水平那張")
    ap.add_argument("--slab_mode", choices=["mean", "minip"], default="mean", help="厚斷面投影:平均或最暗值(環景厚斷面)")
    ap.add_argument("--dy_mm", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=60)
    ap.add_argument("--min_rise", type=float, default=60)
    ap.add_argument("--scale_lim", type=float, nargs=2, default=[0.7, 1.4])
    ap.add_argument("--pred", default="", help="A2: 508 預測路徑, {} = 病歷號(空 = 不用)")
    ap.add_argument("--pred_dice", type=float, default=0.7, help="A2: 508 和推測輪廓 Dice 達到這個才算一致")
    ap.add_argument("--pred_baseline", type=int, default=10, help="A2: 每隔幾張量到的截面算一次 508 基準(0 = 不算)")
    ap.add_argument("--pred2", nargs="*", default=[], help="第二個確認者的 mask 路徑, {} = 病歷號(例如 canal_nninter.py 存的 nnInteractive mask)")
    ap.add_argument("--shape", action="store_true", help="公開資料管子形狀(canal_shape_prior.py)當確認者;只確認位置")
    ap.add_argument("--sym", action="store_true", help="左右對稱當確認者(另一側鏡像對齊)")
    ap.add_argument("--sym_agree_mm", type=float, default=0.5)
    ap.add_argument("--sym_r_mm", type=float, default=0.3)
    ap.add_argument("--sym_max_res_mm", type=float, default=1.0)
    ap.add_argument("--tip_cut", action="store_true",
                    help="找孔(方法 A):頭尾的外插截面超出模型管尖 --tip_margin_mm 以上就改 ignore(要 --pred)")
    ap.add_argument("--tip_margin_mm", type=float, default=2.0)
    ap.add_argument("--tip_near_mm", type=float, default=20.0)
    args = ap.parse_args()
    if args.exam:
        return exam(args)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    secs, summary = [], []
    T = dict(measured=0, interp=0, extrap=0, ignored_interp=0, ignored_extrap=0, empty=0, confirmed=0,
             confirmed_long=0, confirmed_508=0)
    base, gapd = [], []                                   # 508 vs measured (by coverage) / vs guess in long runs
    for npz in sorted(glob.glob(args.rays)):
        cid, sp, sides = G.load_case(npz)
        if args.cases and cid[-4:] not in args.cases:
            continue
        src = nib.load(f"{args.extrap}/{cid}_extrap.nii.gz")
        full = np.asanyarray(src.dataobj).astype(np.uint8)
        if cid in args.skip:
            nib.save(nib.Nifti1Image(full, src.affine), out / f"{cid}_capped.nii.gz")
            continue
        _, _, crop, lo, v, sides = G.prep_volume(cid, sides, sp)
        lab = full[crop].copy()
        P = None
        if args.pred:
            P = np.asanyarray(nib.load(args.pred.format(cid)).dataobj)[crop] == 1
            pc = ndimage.label(P)[0] if args.tip_cut else None
        P2s = [np.asanyarray(nib.load(q.format(cid)).dataobj)[crop] > 0
               for q in args.pred2 if Path(q.format(cid)).exists()]
        st = {"case": cid, **{k: 0 for k in T}, "drop_runs_interp": [], "drop_runs_extrap": [], "tip_cut": 0, "tips": []}
        for s, d in sides.items():
            R = d["r_agree"]; n = len(R); cov = np.isfinite(R).mean(1)
            nanC, nanR = np.full((n, 3), np.nan), np.full((n, len(G.ANG)), np.nan)
            tf, ef = Path(args.track) / f"{cid}_s{s}_track.npz", Path(args.extrap) / f"{cid}_s{s}_extrap.npz"
            tz = np.load(tf) if tf.exists() else {"c": nanC + lo, "r": nanR}
            ez = np.load(ef) if ef.exists() else {"c": nanC + lo, "r": nanR, "dist": np.zeros(n)}
            trC, trR = tz["c"] - lo, tz["r"]; exC, exR, dist = ez["c"] - lo, ez["r"], ez["dist"]
            pred = np.isfinite(trR).any(1) & (cov == 0)
            ext = np.isfinite(exR).any(1)
            conf = np.zeros(n, bool); conf_l = np.zeros(n, bool); conf_p = np.zeros(n, bool)
            drop_i = np.zeros(n, bool)
            if P is not None and args.pred_baseline:
                import canal_edge_gt as E
                Rf = np.array([E.fill_angles(G.ANG, r) if np.isfinite(r).sum() >= 3 else np.full(len(G.ANG), np.nan) for r in R])
                for k in range(0, n, args.pred_baseline):
                    if cov[k] >= 0.25:
                        base.append((float(cov[k]), sec_dice(P, d, d["c"], Rf, k, lab.shape, sp, args.wall_mm)))
            for a, b in runs(pred):
                if b - a <= args.max_interp:
                    continue
                c = np.zeros(n, bool)
                if args.confirm:
                    ea = max([k for k in range(a) if cov[k] >= 0.25], default=None)
                    eb = min([k for k in range(b, n) if cov[k] >= 0.25], default=None)
                    if ea is not None and eb is not None:
                        ok = confirm_gap(v, d, ea, eb, trR, sp, args)
                        cl = np.zeros(n, bool); cl[ea + 1:eb] = ok
                        c |= cl & pred; conf_l |= steady(cl & pred, args.min_confirm_run)
                if P is not None:
                    cp = np.zeros(n, bool)
                    for k in range(a, b):
                        dk = sec_dice(P, d, trC, trR, k, lab.shape, sp, args.wall_mm)
                        gapd.append(dk)
                        cp[k] = dk >= args.pred_dice
                    c |= cp; conf_p |= steady(cp, args.min_confirm_run)
                if args.sym and len(sides) == 2:              # left/right symmetry (canal_symmetry.py)
                    import canal_symmetry as SY
                    o = [dd for kk, dd in sides.items() if kk != s][0]
                    gap = np.arange(a, b); kept = np.isfinite(trR[a:b]).sum(1) >= 3
                    gC = np.array([G.wall_centre(trC[k], d["u"][k], d["w"][k], trR[k], sp) if kept[j] else trC[k]
                                   for j, k in enumerate(gap)])
                    gr = np.array([np.nanmean(trR[k]) if kept[j] else np.nan for j, k in enumerate(gap)])
                    res_, _ = SY.sym_check(d, R, gap, o, o["r_agree"], gC, gr, sp, argparse.Namespace(
                        agree_mm=args.sym_agree_mm, r_mm=args.sym_r_mm, max_res_mm=args.sym_max_res_mm))
                    if res_ is not None:
                        cs = np.zeros(n, bool); cs[a:b] = res_[2] & kept
                        c |= cs; st["confirmed_sym"] = st.get("confirmed_sym", 0) + int(steady(cs, args.min_confirm_run).sum())
                if args.shape:                                # how canals usually run (MCSTU expert centre lines)
                    import canal_shape_prior as SH
                    if not hasattr(main, "_sp") or main._sp_ex != cid:
                        main._sp = SH.ShapePrior(exclude=cid); main._sp_ex = cid      # leave this case out
                    ea2 = max([k for k in range(a) if cov[k] >= 0.25], default=None)
                    eb2 = min([k for k in range(b, n) if cov[k] >= 0.25], default=None)
                    if ea2 is not None and eb2 is not None:
                        gap = np.arange(ea2 + 1, eb2); kept = np.isfinite(trR[gap]).sum(1) >= 3
                        gC = np.array([G.wall_centre(trC[k], d["u"][k], d["w"][k], trR[k], sp) if kept[j] else trC[k]
                                       for j, k in enumerate(gap)])
                        o = SH.shape_check(main._sp, d, R, ea2, eb2, gC, sp, argparse.Namespace(
                            flank=20, K=10, agree_mm=0.5, spread_mm=0.8, max_res_mm=0.6))
                        if o is not None:
                            cs = np.zeros(n, bool); cs[gap] = o[0] & kept; cs &= pred
                            c |= cs; st["confirmed_shape"] = st.get("confirmed_shape", 0) + int(steady(cs, args.min_confirm_run).sum())
                for P2 in P2s:                               # more independent confirmers (nnInteractive, DentalSegmentator, ...)
                    cp2 = np.zeros(n, bool)
                    for k in range(a, b):
                        cp2[k] = sec_dice(P2, d, trC, trR, k, lab.shape, sp, args.wall_mm) >= args.pred_dice
                    c |= cp2; st["confirmed_pred2"] = st.get("confirmed_pred2", 0) + int(steady(cp2, args.min_confirm_run).sum())
                for x, y in runs(c):                         # only steady runs count as new ends
                    if y - x >= args.min_confirm_run and x >= a and y <= b:
                        conf[x:y] = True
                for x, y in runs(pred & ~conf & (np.arange(n) >= a) & (np.arange(n) < b)):
                    if y - x > args.max_interp:
                        drop_i[x:y] = True
                        st["drop_runs_interp"].append([int(s), x, y])
            drop_e = np.zeros(n, bool)
            for k in np.where(ext)[0]:
                step = int(round(dist[k] / 0.4))
                if step > args.max_extrap:
                    drop_e[k] = True; continue
                for e in (k - step, k + step):                # known end it was pushed from
                    if 0 <= e < n and drop_i[e] and not np.isfinite(R[e]).any():
                        drop_e[k] = True
            if args.tip_cut and P is not None:              # extrap past the model's canal tip -> ignore
                known = np.where((cov > 0) | pred)[0]
                tips = model_tips(pc, P, d, sp, args)
                for ei, (tip, t_out) in enumerate(tips):
                    end = 0 if ei == 0 else n - 1
                    st["tips"].append({"side": int(s), "end": ei, "found": tip is not None,
                                       "track_end_past_tip_mm": None if tip is None else float((d["click"][end] - tip) @ t_out * sp)})
                    if tip is None or not len(known):
                        continue
                    ks = np.where(ext & ((np.arange(n) < known.min()) if ei == 0 else (np.arange(n) > known.max())))[0]
                    for k in ks:
                        if not drop_e[k] and (exC[k] - tip) @ t_out * sp > args.tip_margin_mm:
                            drop_e[k] = True; st["tip_cut"] += 1
            for x, y in runs(drop_e):
                st["drop_runs_extrap"].append([int(s), x, y])
            # wipe the dropped sections' voxels (only the values they wrote)
            for mask, C, Rr, vals in ((drop_i, trC, trR, (3, 4)), (drop_e, exC, exR, (5, 6))):
                if not mask.any():
                    continue
                Cm = np.where(mask[:, None], C, np.nan); Rm = np.where(mask[:, None], Rr, np.nan)
                add = G.raster(lab.shape, d, Cm, Rm, sp, args.wall_mm)
                hit = (add != 2) & np.isin(lab, vals)
                lab[hit] = 2
            for k in range(n):
                if ext[k]:
                    kind = "ignored_extrap" if drop_e[k] else "extrap"
                elif pred[k]:
                    kind = "ignored_interp" if drop_i[k] else "interp"
                elif cov[k] > 0:
                    kind = "measured"
                else:
                    kind = "empty"
                st[kind] += 1
                secs.append({"case": cid[-4:], "side": int(s), "sec": k, "kind": kind,
                             "confirmed": bool(conf[k]), "dist_mm": float(dist[k]) if ext[k] else 0.0})
            st["confirmed"] += int(conf.sum()); st["confirmed_long"] += int(conf_l.sum()); st["confirmed_508"] += int(conf_p.sum())
        full[crop] = lab
        nib.save(nib.Nifti1Image(full, src.affine), out / f"{cid}_capped.nii.gz")
        summary.append(st)
        for k in T:
            T[k] += st[k]
        print(f"{cid[-4:]} interp {st['interp']} (-{st['ignored_interp']}, confirmed {st['confirmed']})  "
              f"extrap {st['extrap']} (-{st['ignored_extrap']})", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "sections.json").write_text(json.dumps(secs))
    tot = sum(T[k] for k in ("measured", "interp", "extrap", "ignored_interp", "ignored_extrap", "empty"))
    f = lambda k: T[k] / max(tot, 1)
    pi = sum(1 for x in summary if x["drop_runs_interp"]); pe = sum(1 for x in summary if x["drop_runs_extrap"])
    ni = sum(len(x["drop_runs_interp"]) for x in summary); ne = sum(len(x["drop_runs_extrap"]) for x in summary)
    print(f"interp<={args.max_interp} extrap<={args.max_extrap} confirm={args.confirm} lower_only={args.lower_only}: "
          f"{tot} sections  measured {f('measured'):.1%}  interp {f('interp'):.1%}  extrap {f('extrap'):.1%}  "
          f"ignore {f('ignored_interp') + f('ignored_extrap'):.1%} (interp {T['ignored_interp']} in {ni} runs / {pi} patients, "
          f"extrap {T['ignored_extrap']} in {ne} runs / {pe} patients)  empty {f('empty'):.1%}  "
          f"confirmed {T['confirmed']} (longitudinal {T['confirmed_long']}, 508 {T['confirmed_508']})")
    if args.tip_cut:
        tp = [t for x in summary for t in x["tips"]]
        a = np.array([t["track_end_past_tip_mm"] for t in tp if t["found"]])
        print(f"tip cut (margin {args.tip_margin_mm}mm): {sum(x['tip_cut'] for x in summary)} extrap sections -> ignore "
              f"in {sum(1 for x in summary if x['tip_cut'])} patients; tips found {len(a)}/{len(tp)}; "
              f"track end past the model tip: median {np.median(a):+.2f}mm  >2mm {np.mean(a > 2):.0%}  <-2mm {np.mean(a < -2):.0%}")
    if base:
        b = np.array(base)
        for lo_, hi_ in ((0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
            x = b[(b[:, 0] >= lo_) & (b[:, 0] < hi_), 1]; x = x[np.isfinite(x)]
            if len(x):
                print(f"508 vs measured contour, coverage {lo_:.2f}-{min(hi_, 1):.2f}: n={len(x)} Dice median {np.median(x):.3f}  "
                      f"p25 {np.percentile(x, 25):.3f}  >= {args.pred_dice}: {np.mean(x >= args.pred_dice):.0%}")
    if gapd:
        x = np.array(gapd); x = x[np.isfinite(x)]
        print(f"508 vs guess in over-cap runs: n={len(x)} Dice median {np.median(x):.3f}  >= {args.pred_dice}: {np.mean(x >= args.pred_dice):.0%}")


if __name__ == "__main__":
    main()
