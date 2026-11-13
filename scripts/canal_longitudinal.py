#!/usr/bin/env python3
"""Fill a gap with help from longitudinal sections (professor's suggestion A, 2026-11).

The bridge (and the HMM track) keeps the centre on the straight line between the two clear ends
and only looks at each cross-section. Here the canal is unrolled along the bridge into two
longitudinal sections (a curved-planar reformat), each a thin slab (--slab_mm) averaged across:
  V: up/down    (in-plane direction closest to the image S axis; like sagittal / panoramic)
  H: buccal/lingual (t x V; like axial)
In each, the two walls (dark canal -> bright wall, half-max rise) are tracked as smooth paths
(dynamic programming, at most --dy_mm change per section) from the clear sections before the gap,
through it, to the clear sections after it. Both ends are pinned to the walls measured there.
From the four walls per section: centre shift (mid of upper/lower, mid of buccal/lingual) and
size (height, width) -> the bridge contour is moved and scaled (scale clipped to --scale_lim).
A section keeps the image answer only where both walls of that view rose by >= --min_rise HU;
otherwise that axis stays at the bridge.

Exam: hide L clear sections (optionally fade the image there, as canal_track_hmm.py), score
bridge vs longitudinal (star Dice vs strict radii, centre error).
Real: run on the real gaps longer than --real_min_mm and save figures.

    <venv>/python scripts/canal_longitudinal.py --rays "outputs/sparse_label_v7_rays/chunk_0*/*_rays.npz" \
        --out outputs/longitudinal --L 10 20 40 --contrast 1.0 0.35 0.15
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from case_lists import load_cases  # noqa: E402
import canal_edge_gt as E  # noqa: E402
import canal_gap_fill as G  # noqa: E402
import canal_track_hmm as H  # noqa: E402
from canal_raynet_exam import star_dice  # noqa: E402

ANG = G.ANG
S_AXIS = np.array([0.0, 0.0, 1.0])


def frames(T):
    """Per section: V = S axis projected off the tangent, Hh = t x V."""
    V = S_AXIS[None] - (T @ S_AXIS)[:, None] * T
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    Hh = np.cross(T, V)
    return V, Hh


def pano_frames(d, ks, C, smooth_mm):
    """True panoramic frame (to-do 3): the canal path seen from the side of the jaw, like a dental
    panoramic image. The in-plane (x, y) path of this side's whole track is smoothed with a Gaussian of
    `smooth_mm` (arch-like curve, local track wiggles removed); the surface is that curve extruded
    straight up/down. V = image S axis exactly, Hh = horizontal normal of the curve.
    Returns centres on the surface (z kept from C) and V, Hh for rows ks."""
    xy = ndimage.gaussian_filter1d(d["c"][:, :2].astype(float), smooth_mm / 0.4, axis=0, mode="nearest")
    Cp = np.asarray(C, float).copy(); Cp[:, :2] = xy[ks]
    txy = np.gradient(xy, axis=0)[ks]
    T = np.c_[txy, np.zeros(len(ks))]; T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    V = np.repeat(S_AXIS[None], len(ks), 0)
    Hh = np.cross(T, V); Hh /= np.linalg.norm(Hh, axis=1, keepdims=True) + 1e-9
    return Cp, V, Hh


def ray_radius(r, u, w, axis):
    """Radius of a contour (72 rays in u,w) along in-plane unit axis (and -axis)."""
    ang = np.mod(np.arctan2(axis @ w, axis @ u), 2 * np.pi)
    rf = E.fill_angles(ANG, r)
    return np.interp(ang, ANG, rf, period=2 * np.pi), np.interp(ang + np.pi, ANG, rf, period=2 * np.pi)


def unroll(v, cen, ax, side, sp, half_mm, slab_mm, step=0.05, mode="mean"):
    """(n, Y) image: rows = sections, cols = offset along ax from -half..half mm, projected over a slab
    along `side` (the other in-plane direction): mode 'mean' (average) or 'minip' (darkest value)."""
    ys = np.arange(-half_mm, half_mm + 1e-6, step)
    ss = np.arange(-slab_mm, slab_mm + 1e-6, 0.1) if slab_mm > 0 else np.zeros(1)
    pts = cen[:, None, None] + (ys[None, :, None, None] * ax[:, None, None] + ss[None, None, :, None] * side[:, None, None]) / sp
    I = ndimage.map_coordinates(v, pts.reshape(-1, 3).T, order=1, mode="nearest").reshape(len(cen), len(ys), len(ss))
    return (I.min(2) if mode == "minip" else I.mean(2)), ys


def rise_map(img, ys, sign, dr=0.15, step=0.05):
    """Dark-inside -> bright-outside rise at each offset y for a wall on the +sign side."""
    k = int(round(dr / step))
    pad = np.pad(img, ((0, 0), (k, k)), mode="edge")
    out_ = pad[:, 2 * k:] if sign > 0 else pad[:, :-2 * k]
    in_ = pad[:, :-2 * k] if sign > 0 else pad[:, 2 * k:]
    return out_ - in_


def dp_path(score, ys, y0, y1, dy_max, pin_w=0.15):
    """Max-score path through rows, |dy| <= dy_max per row, rows 0 and -1 pinned near y0, y1."""
    n, m = score.shape
    step = ys[1] - ys[0]
    kmax = max(1, int(round(dy_max / step)))
    sc = score.copy()
    pin = np.exp(-0.5 * ((ys - y0) / pin_w) ** 2) * 1e3
    pin1 = np.exp(-0.5 * ((ys - y1) / pin_w) ** 2) * 1e3
    sc[0] += pin; sc[-1] += pin1
    acc = sc[0].copy(); back = np.zeros((n, m), int)
    for i in range(1, n):
        best = np.full(m, -np.inf); arg = np.zeros(m, int)
        for d in range(-kmax, kmax + 1):
            sh = np.full(m, -np.inf)
            if d >= 0:
                sh[d:] = acc[:m - d] if d else acc
            else:
                sh[:d] = acc[-d:]
            better = sh > best
            best[better] = sh[better]; arg[better] = (np.arange(m) - d)[better]
        acc = best + sc[i]; back[i] = arg
    path = np.zeros(n, int); path[-1] = int(np.argmax(acc))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return ys[path], score[np.arange(n), path]


def end_walls(d, k, cen_k, ax, sp):
    """Positions (mm, along ax from cen_k) of the + and - walls measured at section k."""
    c = G.wall_centre(d["c"][k], d["u"][k], d["w"][k], d["r_agree"][k], sp)
    rp, rm = ray_radius(d["r_agree"][k], d["u"][k], d["w"][k], ax)
    off = (c - cen_k) @ ax * sp
    return off + rp, off - rm


def longitudinal(v, d, a, b, cen, sp, args, R_ends):
    """Gap = sections a+1..b-1 (a, b clear). cen: bridge centres for a+1..b-1.
    Returns per gap section: shift vector (voxels), scale along V and Hh, and debug info."""
    no = int(round(args.ctx_mm / 0.4))
    n = len(d["c"])
    ka = max(a - no, 0); kb = min(b + no, n - 1)
    ks = np.arange(ka, kb + 1)
    # centres along the whole run: measured wall centres outside, bridge inside
    C = np.array([G.wall_centre(d["c"][k], d["u"][k], d["w"][k], R_ends[k], sp)
                  if (k <= a or k >= b) and np.isfinite(R_ends[k]).sum() >= 3 else d["c"][k] for k in ks], float)
    gi = (ks > a) & (ks < b)
    C[gi] = cen
    if getattr(args, "frame", "canal") == "pano":
        C, V, Hh = pano_frames(d, ks, C, args.pano_smooth_mm)
    else:
        V, Hh = frames(d["t"][ks])
    out = {"ks": ks, "gi": gi, "Cg": C[gi], "V": V[gi], "Hh": Hh[gi]}
    shift = np.zeros((gi.sum(), 3)); scale = np.ones((gi.sum(), 2)); used = np.zeros((gi.sum(), 2), bool)
    for ai, (ax, side) in enumerate(((V, Hh), (Hh, V))):
        img, ys = unroll(v, C, ax, side, sp, args.half_mm, args.slab_mm, mode=getattr(args, "slab_mode", "mean"))
        walls = {}
        for sign in (1, -1):
            ia, ib = np.where(ks == a)[0][0], np.where(ks == b)[0][0]
            ea = end_walls({**d, "r_agree": R_ends}, a, C[ia], ax[ia], sp)
            eb = end_walls({**d, "r_agree": R_ends}, b, C[ib], ax[ib], sp)
            y0 = ea[0] if sign > 0 else ea[1]; y1 = eb[0] if sign > 0 else eb[1]
            sc = rise_map(img, ys, sign)
            # only the wall on its own side of the centre, at least 0.3mm out
            ok = (ys * sign >= 0.3) & (ys * sign <= args.half_mm - 0.3)
            sc = np.where(ok[None], np.clip(sc, 0, None), -1e3)
            sub = slice(ia, ib + 1)
            yp, rise = dp_path(sc[sub] / args.tau, ys, y0, y1, args.dy_mm)
            walls[sign] = (yp, rise * args.tau)
        yp_p, rp = walls[1]; yp_m, rm = walls[-1]
        t = np.linspace(0, 1, len(yp_p))
        br_p = (1 - t) * yp_p[0] + t * yp_p[-1]; br_m = (1 - t) * yp_m[0] + t * yp_m[-1]
        mid = (yp_p + yp_m) / 2; size = yp_p - yp_m
        br_mid = (br_p + br_m) / 2; br_size = br_p - br_m
        good = (rp >= args.min_rise) & (rm >= args.min_rise)
        inner = slice(1, -1)
        gax = ax[gi]
        dm = np.where(good, mid - br_mid, 0.0)[inner]
        shift += dm[:, None] * gax / sp
        scale[:, ai] = np.where(good[inner], np.clip(size / np.maximum(br_size, 0.3), *args.scale_lim)[inner], 1.0)
        used[:, ai] = good[inner]
        out[f"img{ai}"] = img; out["ys"] = ys
        out[f"walls{ai}"] = (yp_p, yp_m, br_p, br_m, good)
        out[f"rise{ai}"] = (rp, rm)                         # HU rise of the + / - wall per row
    out["shift"], out["scale"], out["used"] = shift, scale, used
    return out


def apply(cen, rk, u, w, V, Hh, shift, scale):
    """Move the contour (radii rk around cen in u,w) and scale it along V and Hh."""
    p = np.cos(ANG)[:, None] * u + np.sin(ANG)[:, None] * w
    pts = p * rk[:, None]                                   # mm, relative
    a = pts @ V; b = pts @ Hh
    pts = (a * scale[0])[:, None] * V + (b * scale[1])[:, None] * Hh
    return cen + shift, pts


def radii_vs(c_ref, cen_new, pts_mm, u, w, sp):
    o = (cen_new - c_ref) * sp
    q = o[None] + pts_mm
    a_, b_ = q @ u, q @ w
    th = np.mod(np.arctan2(b_, a_), 2 * np.pi); rr = np.hypot(a_, b_)
    order = np.argsort(th)
    return np.interp(ANG, th[order], rr[order], period=2 * np.pi)


def exam(args):
    rows = []
    for npz in sorted(glob.glob(args.rays)):
        cid, sp, sides = G.load_case(npz)
        if cid in args.skip or (args.cases and cid[-4:] not in args.cases):
            continue
        _, _, _, _, v0, sides = G.prep_volume(cid, sides, sp)
        rng = np.random.default_rng(int(cid))
        for s, d in sides.items():
            R = d["r_agree"]; n = len(R)
            good = np.isfinite(R).mean(1) >= 0.5
            mg = 2 + int(round(args.ctx_mm / 0.4))
            for L in args.L:
                i = mg
                while i + L + mg <= n:
                    blk = np.arange(i, i + L)
                    if not good[i - mg:i + L + mg].all():
                        i += 1; continue
                    a, b = i - 1, i + L
                    cen, ra, rb, t = H.ends(d, i, i + L, sp, 0)
                    Rh = R.copy(); Rh[blk] = np.nan
                    for cc in args.contrast:
                        v = v0 if cc >= 1 else G.degrade(v0, d["c"][blk], sp, argparse.Namespace(
                            degrade_contrast=cc, degrade_noise=args.noise_hu, degrade_r_mm=4.0), rng)
                        lo = longitudinal(v, d, a, b, cen, sp, args, Rh)
                        V, Hh = frames(d["t"][blk])
                        for j, k in enumerate(blk):
                            ok = np.isfinite(R[k]); truth = E.fill_angles(ANG, R[k])
                            tc = G.wall_centre(d["c"][k], d["u"][k], d["w"][k], R[k], sp)
                            rbr = (1 - t[j]) * ra + t[j] * rb
                            br = H.radii_from(d["c"][k], cen[j], rbr, d["u"][k], d["w"][k], sp)
                            c2, pts = apply(cen[j], rbr, d["u"][k], d["w"][k], V[j], Hh[j], lo["shift"][j], lo["scale"][j])
                            lr = radii_vs(d["c"][k], c2, pts, d["u"][k], d["w"][k], sp)
                            # shift only (no scaling)
                            c3, pts3 = apply(cen[j], rbr, d["u"][k], d["w"][k], V[j], Hh[j], lo["shift"][j], np.ones(2))
                            sr = radii_vs(d["c"][k], c3, pts3, d["u"][k], d["w"][k], sp)
                            rows.append({"case": cid[-4:], "side": int(s), "L": L, "c": cc, "sec": int(k),
                                         "br_dice": float(star_dice(br, truth)),
                                         "lo_dice": float(star_dice(lr, truth)),
                                         "sh_dice": float(star_dice(sr, truth)),
                                         "br_cerr": float(np.linalg.norm(cen[j] - tc) * sp),
                                         "lo_cerr": float(np.linalg.norm(c2 - tc) * sp),
                                         "usedV": bool(lo["used"][j, 0]), "usedH": bool(lo["used"][j, 1])})
                    i += L + args.stride
        print(cid[-4:], len(rows), flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out / "exam_rows.json", "w"))
    summarize(rows)


def summarize(rows):
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["L"], r["c"])].append(r)
    for (L, c), rs in sorted(g.items()):
        f = lambda k: np.mean([r[k] for r in rs])
        print(f"L={L:2d} ({L*0.4:4.1f}mm) contrast {c:.2f} n={len(rs):4d} | bridge Dice {f('br_dice'):.3f} cerr {f('br_cerr'):.2f}"
              f" | longit Dice {f('lo_dice'):.3f} cerr {f('lo_cerr'):.2f} | shift-only Dice {f('sh_dice'):.3f}"
              f" | used V {f('usedV'):.2f} H {f('usedH'):.2f}", flush=True)


def real(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(args.out) / "real"; out.mkdir(parents=True, exist_ok=True)
    min_l = int(np.ceil(args.real_min_mm / 0.4))
    stats = []
    for npz in sorted(glob.glob(args.rays)):
        cid, sp, sides = G.load_case(npz)
        if cid in args.skip or (args.cases and cid[-4:] not in args.cases):
            continue
        _, _, _, _, v, sides = G.prep_volume(cid, sides, sp)
        for s, d in sides.items():
            R = d["r_agree"]; cov = np.isfinite(R).mean(1)
            ends_ok = np.where(cov >= 0.25)[0]
            for a, b in zip(ends_ok[:-1], ends_ok[1:]):
                L = b - a - 1
                if L < min_l or L > 40 or not (cov[a + 1:b] == 0).any():
                    continue
                cen, ra, rb, t = H.ends(d, a + 1, b, sp, 0)
                lo = longitudinal(v, d, a, b, cen, sp, args, R)
                sh = np.linalg.norm(lo["shift"], axis=1) * sp
                stats.append({"case": cid[-4:], "side": int(s), "a": int(a), "b": int(b), "len_mm": L * 0.4,
                              "usedV": float(lo["used"][:, 0].mean()), "usedH": float(lo["used"][:, 1].mean()),
                              "shift_med": float(np.median(sh)), "shift_max": float(sh.max()),
                              "scaleV_med": float(np.median(lo["scale"][:, 0])), "scaleH_med": float(np.median(lo["scale"][:, 1]))})
                fig, axs = plt.subplots(2, 1, figsize=(9, 5.5))
                ys = lo["ys"]; ks = lo["ks"]
                for ai, (ttl, lab) in enumerate((("up/down (V, like sagittal)", "up"), ("buccal/lingual (H, like axial)", "+"))):
                    ax = axs[ai]
                    im_ = lo[f"img{ai}"]; vlo, vhi = np.percentile(im_, [1, 99])
                    ax.imshow(im_.T, cmap="gray", origin="lower", aspect="auto", vmin=vlo, vmax=vhi,
                              extent=[ks[0] - 0.5, ks[-1] + 0.5, ys[0], ys[-1]])
                    yp, ym, bp, bm, good = lo[f"walls{ai}"]
                    xs = np.arange(a, b + 1)
                    ax.plot(xs, bp, "c--", lw=1); ax.plot(xs, bm, "c--", lw=1, label="bridge")
                    ax.plot(xs, yp, "y-", lw=1); ax.plot(xs, ym, "y-", lw=1, label="tracked wall")
                    ax.scatter(xs[good], np.full(good.sum(), ys[0] + 0.2), s=6, c="lime", label="image used")
                    ax.axvline(a + 0.5, color="r", lw=1); ax.axvline(b - 0.5, color="r", lw=1)
                    ax.set_title(f"{cid[-4:]} side {s}  gap {L*0.4:.1f}mm  {ttl}", fontsize=9)
                    ax.set_ylabel("mm"); ax.legend(fontsize=7, loc="upper right")
                axs[1].set_xlabel("section (0.4 mm each); between red lines = gap")
                fig.tight_layout(); fig.savefig(out / f"{cid[-4:]}_s{s}_{a}_{b}.png", dpi=110); plt.close(fig)
        print(cid[-4:], len(stats), flush=True)
    json.dump(stats, open(out / "real_stats.json", "w"), indent=1)
    if stats:
        f = lambda k: np.median([x[k] for x in stats])
        print(f"real gaps >= {args.real_min_mm}mm: {len(stats)}  usedV {f('usedV'):.2f} usedH {f('usedH'):.2f}"
              f"  shift median {f('shift_med'):.2f}mm  max(median over gaps) {f('shift_max'):.2f}mm"
              f"  scaleV {f('scaleV_med'):.2f} scaleH {f('scaleH_med'):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rays", default="outputs/sparse_label_v7_rays/chunk_0*/*_rays.npz")
    ap.add_argument("--skip", nargs="*", default=load_cases())
    ap.add_argument("--cases", nargs="*", default=[], help="只跑這些病歷號末 4 碼")
    ap.add_argument("--out", default="outputs/longitudinal")
    ap.add_argument("--mode", choices=["exam", "real"], default="exam")
    ap.add_argument("--L", type=int, nargs="+", default=[10, 20, 40])
    ap.add_argument("--contrast", type=float, nargs="+", default=[1.0, 0.35, 0.15])
    ap.add_argument("--noise_hu", type=float, default=40)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--ctx_mm", type=float, default=4.0, help="缺口前後各攤開多長")
    ap.add_argument("--half_mm", type=float, default=3.5, help="縱向斷面上下各看多遠")
    ap.add_argument("--slab_mm", type=float, default=0.4, help="斷面厚度(兩側各),平均降雜訊")
    ap.add_argument("--frame", choices=["canal", "pano"], default="canal",
                    help="canal = 沿管子 3D 彎曲攤開(原本); pano = 真正的環景圖(水平路徑平滑、垂直切)")
    ap.add_argument("--pano_smooth_mm", type=float, default=5.0, help="pano: 水平路徑平滑程度")
    ap.add_argument("--slab_mode", choices=["mean", "minip"], default="mean", help="厚斷面投影:平均或最暗值")
    ap.add_argument("--dy_mm", type=float, default=0.1, help="管壁每張截面最多移動多少")
    ap.add_argument("--tau", type=float, default=60)
    ap.add_argument("--min_rise", type=float, default=60, help="兩邊管壁都上升這麼多 HU 才採用影像")
    ap.add_argument("--scale_lim", type=float, nargs=2, default=[0.7, 1.4])
    ap.add_argument("--real_min_mm", type=float, default=5.0)
    args = ap.parse_args()
    exam(args) if args.mode == "exam" else real(args)


if __name__ == "__main__":
    main()
