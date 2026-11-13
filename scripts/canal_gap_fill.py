#!/usr/bin/env python3
"""Fill the sections where the sparse labeller found no wall: predict WHERE the canal is,
then MEASURE how wide it is from this patient's own image.

Input: the per-section rays saved by canal_sparse_label.py --save_rays (centre, plane, final
per-ray radius for every section along each doctor track).

  location  a gap (run of sections with no wall) between two measured sections is bridged by
            - short gaps: the lumen centre interpolated between the two measured ends
            - long gaps:  the darkest connected path through the bone from one end to the other
                          (minimal path, kept within path_tube_mm of the doctor track)
  width     at the predicted centre the ray method is run again with a lower rise threshold
            (the wall is faint there, but we now know where to look); the same drop-only checks
            as the labeller (lumen brightness, ray runs, radius jumps, cross-section agreement)
            plus an anatomical cap on the radius. Rays that still find nothing stay unknown.
            Nothing is interpolated, smoothed or borrowed from other patients or public data:
            every kept radius is an edge in this image.

Exam (--exam): in sections where the wall IS clear, hide a block of L sections, predict it,
and compare with what the strict labeller found there. --degrade lowers the contrast and adds
noise inside the hidden block so the strict method would fail too (closer to a real gap).

    python scripts/canal_gap_fill.py --rays outputs/sparse_label_v6_rays/chunk_*/*_rays.npz --out outputs/gap_fill
"""
import argparse
from case_lists import load_cases
import glob
import os
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.graph import route_through_array

sys.path.insert(0, str(Path(__file__).parent))
import canal_edge_gt as E  # noqa: E402
import canal_sparse_label as SL  # noqa: E402
import canal_edge_compare as C3  # noqa: E402

ANG, N_ANG, PX, HALF = SL.ANG, SL.N_ANG, SL.PX, SL.HALF
IMG = os.environ.get("CANAL_IMG", "data/hospital_nifti_clean/images/{}.nii.gz")  # image path template, {} = case id


def load_case(npz):
    z = np.load(npz)
    cid = Path(npz).name.replace("_rays.npz", "")
    sides = {}
    for k in z.files:
        if k == "sp":
            continue
        sd, name = k.split("_", 1)
        sides.setdefault(int(sd[1:]), {})[name] = z[k]
    return cid, float(z["sp"]), sides


def gaps(known):
    """[(a, b)] index runs of unknown sections; a = first unknown, b = one past the last."""
    out, i, n = [], 0, len(known)
    while i < n:
        if known[i]:
            i += 1
            continue
        j = i
        while j < n and not known[j]:
            j += 1
        out.append((i, j))
        i = j
    return out


def wall_centre(c, u, w, r, sp):
    """Centre of the measured wall points (voxels); falls back to the ray origin."""
    ok = np.isfinite(r)
    if ok.sum() < 3:
        return c
    d = np.cos(ANG[ok])[:, None] * u + np.sin(ANG[ok])[:, None] * w
    return (c + d * (r[ok] / sp)[:, None]).mean(0)


def path_centres(v, a, b, clicks, planes, sp, args):
    """Darkest connected path from voxel a to b; one point per gap section (nan if none)."""
    pts = np.vstack([a, b, clicks])
    pad = int(np.ceil(args.path_tube_mm / sp)) + 2
    lo = np.maximum(np.floor(pts.min(0)).astype(int) - pad, 0)
    hi = np.minimum(np.ceil(pts.max(0)).astype(int) + pad + 1, v.shape)
    sub = v[tuple(slice(l, h) for l, h in zip(lo, hi))]
    p5, p95 = np.percentile(sub, [5, 95])
    cost = np.clip((sub - p5) / (p95 - p5 + 1e-6), 0, 1) ** 2 + args.path_eps
    # stay near the doctor track: the clicks of the gap sections, joined
    tr = np.zeros(sub.shape, bool)
    line = np.vstack([a, clicks, b])
    for p, q in zip(line[:-1], line[1:]):
        for t in np.linspace(0, 1, int(np.linalg.norm(q - p)) * 2 + 2):
            k = np.round(p + t * (q - p) - lo).astype(int)
            if np.all(k >= 0) and np.all(k < sub.shape):
                tr[tuple(k)] = True
    far = ndimage.distance_transform_edt(~tr) * sp > args.path_tube_mm
    cost[far] = 1e3
    s = tuple(np.clip(np.round(a - lo).astype(int), 0, np.array(sub.shape) - 1))
    e = tuple(np.clip(np.round(b - lo).astype(int), 0, np.array(sub.shape) - 1))
    idx, _ = route_through_array(cost, s, e, fully_connected=True, geometric=True)
    path = np.array(idx, float) + lo
    out = []
    for click, t in planes:
        d = (path - click) @ t                       # signed distance to the section plane
        near = np.linalg.norm(path - click, axis=1) * sp <= args.path_tube_mm + 1
        cand = np.where(near)[0]
        if not len(cand):
            out.append(np.full(3, np.nan)); continue
        j = cand[np.argmin(np.abs(d[cand]))]
        out.append(path[j] - d[j] * t)                # onto the plane
    return np.array(out)


def relaxed_rays(v, c, u, w, sp, args):
    """Ray method with the lower rise threshold, after the anatomical cap."""
    _, r, _ = E.ray_contour(v, c, u, w, N_ANG, 0.3 / sp, HALF / sp, 0.05 / sp, args.min_rise)
    r = E.clean_rays(r * sp)
    r[r > args.r_cap_mm] = np.nan
    return r


def snap(v, c, u, w, rk, sp, win_mm, rise):
    """Per ray: the strongest dark->bright rise within win_mm of the predicted radius rk, if the
    image rises there by at least `rise` (0.3mm outside minus 0.3mm inside); else nan."""
    dr = 0.05
    r = np.arange(0, HALF + 1e-6, dr)
    d = np.cos(ANG)[:, None, None] * u + np.sin(ANG)[:, None, None] * w
    p = ndimage.map_coordinates(v, (c + d * (r / sp)[None, :, None]).reshape(-1, 3).T, order=1,
                                mode="nearest").reshape(N_ANG, len(r))
    gr = np.gradient(ndimage.gaussian_filter1d(p, 1.0, axis=1), dr, axis=1)
    k = int(round(0.3 / dr))
    out = np.full(N_ANG, np.nan)
    for a in np.where(np.isfinite(rk))[0]:
        lo = max(int((rk[a] - win_mm) / dr), k); hi = min(int((rk[a] + win_mm) / dr) + 1, len(r) - k)
        if hi <= lo:
            continue
        j = lo + int(np.argmax(gr[a, lo:hi]))
        if p[a, j + k] - p[a, j - k] >= rise:
            out[a] = r[j]
    return out


def enclosure(v, c, u, w, sp, rise=50, r_max=2.0):
    """Share of all 72 directions in which the image gets brighter than the centre by `rise`
    within r_max mm. A canal lumen is closed off (wall, cortex or trabecular bone) on most sides;
    a fan into a large dark marrow space is not."""
    r = np.arange(0, HALF + 1e-6, 0.05)
    d = np.cos(ANG)[:, None, None] * u + np.sin(ANG)[:, None, None] * w
    p = ndimage.map_coordinates(v, (c + d * (r / sp)[None, :, None]).reshape(-1, 3).T, order=1,
                                mode="nearest").reshape(N_ANG, len(r))
    return float(((p[:, r <= r_max] - p[:, :4].mean()) >= rise).any(1).mean())


def checks(v, c, u, w, r, sp, args):
    """Drop-only checks of the labeller: lumen brightness, wall rise, runs, jumps."""
    lum_mean, _, wall = SL.lumen_ratio(v, c, u, w, r, sp)
    bad = (lum_mean > args.lum_frac) | (np.isfinite(r) & (wall < args.wall_rise))
    r = np.where(bad, np.nan, r)
    return SL.filter_rays(r, args.min_sec_frac, args.min_run, args.max_jump_mm)


def polar3d_side(v, d, sp, args, key):
    """polar3d (optimal surface over the whole side) on this volume; returns a callable per section."""
    P = C3.PolarDP3D(rise_frac=args.p3_rise_frac)
    base = dict(vol=v, sp=sp, mm=lambda x: x / sp, min_rise=100, cid=key, side=0,
                track=(d["click"], d["t"]), search_mm=1.2)

    def at(i, c, u, w):
        return P({**base, "i": i, "c": c, "u": u, "w": w})
    return at


def fill_side(v, d, sp, args, hide=None, key="x"):
    """Predicted centres and radii (n x 72, mm) for the unknown sections of one side.

    loc     labeller = the labeller's own start (darkest point within 1.2mm of the click)
            interp   = wall centres of the measured ends, interpolated (short gaps) / minimal path
    method  ray      = relaxed ray method
            agree    = relaxed ray kept only where polar3d (whole-side surface) agrees within agree_mm
            p3       = polar3d alone
    """
    C, U, W, T, R, K = d["c"], d["u"], d["w"], d["t"], d["r_agree"].copy(), d["click"]
    mm = lambda x: x / sp
    if hide is not None:
        R[hide] = np.nan
    n = len(C)
    known = np.isfinite(R).any(1)
    newC = np.full((n, 3), np.nan); newR = np.full((n, N_ANG), np.nan); how = np.zeros(n, int)
    P3 = polar3d_side(v, d, sp, args, key) if args.method in ("agree", "p3") else None
    snapped = np.zeros((n, N_ANG), bool)
    for a0, b0 in gaps(known):
        idx = np.arange(a0, b0)
        if args.method == "interp":
            # bridge interior gaps up to max_gap_mm from this patient's own measured ends:
            # centre follows the doctor track plus the measured offset at each end (lerp),
            # radius per direction lerped between the two ends (only directions measured at both)
            if a0 == 0 or b0 == n or (b0 - a0) * args.step_mm > args.max_gap_mm:
                continue
            a, b = a0 - 1, b0
            ca = wall_centre(C[a], U[a], W[a], R[a], sp); cb = wall_centre(C[b], U[b], W[b], R[b], sp)
            # per direction, the nearest section on each side (within end_mm) that measured it
            ke = int(round(args.end_mm / args.step_mm))
            jL = np.full(N_ANG, -1); jR = np.full(N_ANG, -1)
            for j in range(a, max(a - ke, 0) - 1, -1):
                jL[(jL < 0) & np.isfinite(R[j])] = j
            for j in range(b, min(b + ke, n - 1) + 1):
                jR[(jR < 0) & np.isfinite(R[j])] = j
            al = np.arange(N_ANG)
            rL = np.where(jL >= 0, R[np.maximum(jL, 0), al], np.nan)
            rR = np.where(jR >= 0, R[np.maximum(jR, 0), al], np.nan)
            # an end measured all round (>= fill_end_frac of directions) gives every direction,
            # its missing ones read off its neighbouring directions (exam: same error as direct)
            # (the nearest such section within end_mm on each side)
            if args.fill_end_frac > 0:
                cov = np.isfinite(R).mean(1)
                for j in range(a, max(a - ke, 0) - 1, -1):
                    if cov[j] >= args.fill_end_frac:
                        rL = E.fill_angles(ANG, R[j]); jL = np.full(N_ANG, j); break
                for j in range(b, min(b + ke, n - 1) + 1):
                    if cov[j] >= args.fill_end_frac:
                        rR = E.fill_angles(ANG, R[j]); jR = np.full(N_ANG, j); break
            ok = (jL >= 0) & (jR >= 0)
            rL = np.where(ok, rL, np.nan); rR = np.where(ok, rR, np.nan)
            for i in idx:
                t = (i - a) / (b - a)
                c = K[i] + (1 - t) * (ca - K[a]) + t * (cb - K[b])
                c = c - ((c - K[i]) @ T[i]) * T[i]
                ta = np.where(ok, (i - jL) / np.maximum(jR - jL, 1), np.nan)
                rk = (1 - ta) * rL + ta * rR
                if args.snap_mm > 0:
                    rs = snap(v, c, U[i], W[i], rk, sp, args.snap_mm, args.snap_rise)
                    snapped[i] = np.isfinite(rs)
                    rk = np.where(snapped[i], rs, rk)
                if args.lum_frac > 0:
                    lum_mean, _, _ = SL.lumen_ratio(v, c, U[i], W[i], rk, sp)
                    rk = np.where(lum_mean > args.lum_frac, np.nan, rk)
                newC[i], newR[i] = c, rk
            how[idx] = 4
            continue
        if args.loc == "labeller" or a0 == 0 or b0 == n:
            if args.loc != "labeller" and not args.open_ends:
                continue
            P = np.array([SL.darkest_centre(v, K[i], U[i], W[i], mm(1.2), mm(PX)) for i in idx])
            how[idx] = 3
        else:
            a, b = a0 - 1, b0
            ca = wall_centre(C[a], U[a], W[a], R[a], sp); cb = wall_centre(C[b], U[b], W[b], R[b], sp)
            if (b0 - a0) * args.step_mm <= args.short_mm:
                f = (idx - a) / (b - a)
                P = ca + f[:, None] * (cb - ca)
                P = P - (((P - K[idx]) * T[idx]).sum(1))[:, None] * T[idx]     # onto each plane
                how[idx] = 1
            else:
                P = path_centres(v, ca, cb, K[idx], list(zip(K[idx], T[idx])), sp, args)
                how[idx] = 2
            P = np.array([SL.darkest_centre(v, p, U[i], W[i], mm(args.recentre_mm), mm(PX))
                          if np.isfinite(p).all() else p for p, i in zip(P, idx)])
        for j, i in enumerate(idx):
            c = P[j]
            if not np.isfinite(c).all():
                continue
            if args.method == "p3":
                r = P3(i, c, U[i], W[i])
                r[r > args.r_cap_mm] = np.nan
            else:
                r = relaxed_rays(v, c, U[i], W[i], sp, args)
                if args.method == "agree":
                    r = np.where(np.abs(r - P3(i, c, U[i], W[i])) <= args.agree_mm, r, np.nan)
            newC[i], newR[i] = c, checks(v, c, U[i], W[i], r, sp, args)
    # cross-section agreement over measured + new rays together; only new rays can be dropped
    if args.method == "interp":
        return newC, newR, how, snapped
    if args.xsec_mm > 0:
        allR = np.where(np.isfinite(newR), newR, R)
        allC = np.where(np.isfinite(newC), newC, C)
        drop, _, _ = SL.xsec_drop(allR, allC, U, W, T, sp, 3, args.xsec_mm, 2)
        newR[drop & np.isfinite(newR)] = np.nan
        newR = np.array([SL.filter_rays(r, 0, args.min_run, args.max_jump_mm) for r in newR])
    return newC, newR, how, snapped


def wall_points(c, u, w, r, sp):
    d = np.cos(ANG)[:, None] * u + np.sin(ANG)[:, None] * w
    return c * sp + d * r[:, None]


def exam_side(v0, d, sp, L, args, rng):
    """Hide blocks of L clear sections, predict them, compare with the strict rays."""
    R = d["r_agree"]; n = len(R)
    good = np.isfinite(R).mean(1) >= args.exam_min_frac
    res = []
    i = 2
    while i + L + 2 <= n:
        blk = np.arange(i, i + L)
        if good[i - 2:i + L + 2].all():
            v = v0
            if args.degrade:
                v = degrade(v0, d["c"][blk], sp, args, rng)
            hide = np.zeros(n, bool); hide[blk] = True
            newC, newR, how, _ = fill_side(v, d, sp, args, hide, key=f"exam{L}_{i}")
            for k in blk:
                tc = wall_centre(d["c"][k], d["u"][k], d["w"][k], R[k], sp) * sp
                row = {"L": L, "how": int(how[k]), "sec": int(k),
                       "doc_err": float(np.linalg.norm(d["click"][k] * sp - tc)),
                       "strict_frac": float(np.isfinite(R[k]).mean())}
                if args.degrade:     # would the strict labeller still see the wall after degrading?
                    _, r0, _ = E.ray_contour(v, d["c"][k], d["u"][k], d["w"][k], N_ANG, 0.3 / sp, HALF / sp,
                                             0.05 / sp, 100)
                    row["strict_on_degraded"] = float(np.isfinite(r0).mean())
                if np.isfinite(newC[k]).all():
                    row["centre_err"] = float(np.linalg.norm(newC[k] * sp - tc))
                    ok = np.isfinite(newR[k])
                    row["found_frac"] = float(ok.mean())
                    if ok.any():
                        # each new wall point vs the nearest strict wall point of that section
                        pn = wall_points(newC[k], d["u"][k], d["w"][k], newR[k], sp)[ok]
                        ps = wall_points(d["c"][k], d["u"][k], d["w"][k], R[k], sp)[np.isfinite(R[k])]
                        dd = np.linalg.norm(pn[:, None] - ps[None], axis=2).min(1)
                        row["wall_err_med"] = float(np.median(dd))
                        row["wall_err_p90"] = float(np.percentile(dd, 90))
                        row["r_new"] = float(np.nanmean(newR[k])); row["r_strict"] = float(np.nanmean(R[k]))
                res.append(row)
            i += L + args.exam_stride
        else:
            i += 1
    return res


def degrade(v0, centres, sp, args, rng):
    """Lower contrast and add noise in a ball around the hidden sections."""
    v = v0.copy()
    lo = np.maximum(np.floor(centres.min(0) - args.degrade_r_mm / sp).astype(int), 0)
    hi = np.minimum(np.ceil(centres.max(0) + args.degrade_r_mm / sp).astype(int) + 1, v.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    sub = v[sl]
    m = ndimage.gaussian_filter(sub, 2.0 / sp)
    sub = m + args.degrade_contrast * (sub - m) + rng.normal(0, args.degrade_noise, sub.shape)
    v[sl] = ndimage.gaussian_filter(sub, 0.2 / sp)
    return v


def prep_volume(cid, sides, sp):
    im = nib.load(IMG.format(cid))
    vol = np.asanyarray(im.dataobj).astype(np.float32)
    pts = np.vstack([d["c"] for d in sides.values()])
    lo = np.maximum(np.floor(pts.min(0) - 10 / sp).astype(int), 0)
    hi = np.minimum(np.ceil(pts.max(0) + 10 / sp).astype(int) + 1, vol.shape)
    crop = tuple(slice(a, b) for a, b in zip(lo, hi))
    v = ndimage.gaussian_filter(vol[crop], 0.2 / sp)          # same smoothing as the labeller
    sides = {s: {**d, "c": d["c"] - lo, "click": d["click"] - lo} for s, d in sides.items()}
    return im, vol.shape, crop, lo, v, sides


def raster(shape, d, newC, newR, sp, wall_mm):
    """canal / wall-background from the new rays only, nearest-ray, no filling between rays."""
    lab = np.full(shape, 2, np.uint8)
    for i in np.where(np.isfinite(newR).any(1))[0]:
        c, u, w, t, r = newC[i], d["u"][i], d["w"][i], d["t"][i], newR[i]
        rad = int(np.ceil((np.nanmax(r) + wall_mm) / sp)) + 1
        lo = np.maximum(np.floor(c).astype(int) - rad, 0); hi = np.minimum(np.floor(c).astype(int) + rad + 2, shape)
        g = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
        dd = g - c
        along = np.abs(dd @ t)
        a_ = dd @ u; b_ = dd @ w
        rr = np.hypot(a_, b_) * sp
        th = np.mod(np.arctan2(b_, a_), 2 * np.pi)
        rk = r[np.round(th / (2 * np.pi) * N_ANG).astype(int) % N_ANG]
        on = (along <= 0.2 / sp + 0.5) & np.isfinite(rk)
        sel = g[on & (rr <= rk)]; lab[tuple(sel.T)] = np.where(lab[tuple(sel.T)] == 2, 1, lab[tuple(sel.T)])
        sel = g[on & (rr > rk) & (rr <= rk + wall_mm)]; lab[tuple(sel.T)] = np.where(lab[tuple(sel.T)] == 2, 0, lab[tuple(sel.T)])
    return lab


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rays", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", nargs="*", default=load_cases(), help="醫生標註錯位的病例")
    ap.add_argument("--step_mm", type=float, default=0.4)
    ap.add_argument("--short_mm", type=float, default=2.0, help="缺口不超過這個長度用內插找位置,更長用最短路徑")
    ap.add_argument("--path_tube_mm", type=float, default=2.5, help="最短路徑離醫生軌跡的上限")
    ap.add_argument("--path_eps", type=float, default=0.02)
    ap.add_argument("--recentre_mm", type=float, default=0.6, help="預測中心附近找最暗點的範圍")
    ap.add_argument("--min_rise", type=float, default=50, help="放寬的管壁亮度上升門檻(原本 100)")
    ap.add_argument("--r_cap_mm", type=float, default=1.9, help="半徑上限(文獻最大管徑 3.8mm)")
    ap.add_argument("--lum_frac", type=float, default=0.3)
    ap.add_argument("--wall_rise", type=float, default=30)
    ap.add_argument("--min_sec_frac", type=float, default=0.25)
    ap.add_argument("--min_run", type=int, default=3)
    ap.add_argument("--max_jump_mm", type=float, default=0.4)
    ap.add_argument("--xsec_mm", type=float, default=0.5)
    ap.add_argument("--wall_mm", type=float, default=0.4)
    ap.add_argument("--loc", choices=["labeller", "interp"], default="interp", help="中心:標註程式原本的起點,或前後內插/最短路徑")
    ap.add_argument("--open_ends", action="store_true", help="頭尾沒有兩端可接的缺口也用原本的起點去量")
    ap.add_argument("--method", choices=["ray", "agree", "p3", "interp"], default="ray",
                    help="interp = 用這位病人自己缺口兩端量到的管壁內插(不看別人的資料)")
    ap.add_argument("--max_gap_mm", type=float, default=8.0, help="interp:超過這個長度的缺口不補")
    ap.add_argument("--snap_mm", type=float, default=0.0, help="interp:內插的管壁附近這個範圍內有淡管壁就改用影像的位置(0=關)")
    ap.add_argument("--snap_rise", type=float, default=25)
    ap.add_argument("--fill_end_frac", type=float, default=0.0, help="interp:端點截面抓到這麼多方向就把它的缺的方向用相鄰方向補齊(0=關)")
    ap.add_argument("--end_mm", type=float, default=2.0, help="interp:每個方向往缺口外最多找多遠的截面來當端點")
    ap.add_argument("--p3_rise_frac", type=float, default=0.25)
    ap.add_argument("--agree_mm", type=float, default=0.3)
    ap.add_argument("--exam", type=int, nargs="*", default=[], help="考試:藏起的區塊長度(截面數)")
    ap.add_argument("--exam_min_frac", type=float, default=0.5, help="考試只用抓到這麼多方向的清楚截面")
    ap.add_argument("--exam_stride", type=int, default=10)
    ap.add_argument("--degrade", action="store_true")
    ap.add_argument("--degrade_contrast", type=float, default=0.35)
    ap.add_argument("--degrade_noise", type=float, default=40)
    ap.add_argument("--degrade_r_mm", type=float, default=4.0)
    ap.add_argument("--write", action="store_true", help="寫出補完的標註")
    ap.add_argument("--base", default="outputs/sparse_label_v6_rays/chunk_*/{}_agree.nii.gz")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    exam_rows, fill_rows = [], []
    for npz in sorted(args.rays):
        cid, sp, sides = load_case(npz)
        if cid in args.skip:
            continue
        im, shape, crop, lo, v, sides = prep_volume(cid, sides, sp)
        for L in args.exam:
            for s, d in sides.items():
                for row in exam_side(v, d, sp, L, args, rng):
                    exam_rows.append({"case": cid[-4:], "side": s, **row})
        if args.exam:
            print(f"{cid[-4:]} exam rows {len(exam_rows)}", flush=True)
            continue
        lab = None
        if args.write:
            lab = np.asanyarray(nib.load(glob.glob(args.base.format(cid))[0]).dataobj)[crop].copy()
            src = (lab == 1).astype(np.uint8)                 # 1 = measured by the strict labeller
        for s, d in sides.items():
            newC, newR, how, snapped = fill_side(v, d, sp, args, key=f"{cid}_{s}")
            R = d["r_agree"]; known = np.isfinite(R).any(1)
            got = np.isfinite(newR).any(1)
            for i in np.where(~known)[0]:
                fill_rows.append({"case": cid[-4:], "side": s, "sec": int(i), "how": int(how[i]),
                                  "found_frac": float(np.isfinite(newR[i]).mean()),
                                  "snapped": float((snapped[i] & np.isfinite(newR[i])).mean()),
                                  "r_mean": float(np.nanmean(newR[i])) if got[i] else None,
                                  "r_ratio": float(np.nanmax(newR[i]) / np.nanmin(newR[i])) if got[i] else None,
                                  "doc_dist": float(np.linalg.norm(newC[i] - d["click"][i]) * sp) if np.isfinite(newC[i]).all() else None})
            if lab is not None:
                add = raster(lab.shape, d, newC, newR, sp, args.wall_mm)
                put = (lab == 2) & (add != 2)
                lab[put] = add[put]; src[put & (add == 1)] = 2  # 2 = predicted location, measured width
            np.savez_compressed(out / f"{cid}_s{s}_fill.npz", c=newC + lo, r=newR, how=how, snapped=snapped)
        if lab is not None:
            base = nib.load(glob.glob(args.base.format(cid))[0])
            outl = np.asanyarray(base.dataobj).copy(); outl[crop] = lab
            nib.save(nib.Nifti1Image(outl, base.affine), out / f"{cid}_filled.nii.gz")
            fs = np.zeros(shape, np.uint8); fs[crop] = src
            nib.save(nib.Nifti1Image(fs, base.affine), out / f"{cid}_source.nii.gz")
        n_un = sum(1 for r in fill_rows if r["case"] == cid[-4:])
        n_got = sum(1 for r in fill_rows if r["case"] == cid[-4:] and r["found_frac"] > 0)
        print(f"{cid[-4:]} empty {n_un} filled {n_got}", flush=True)
    if exam_rows:
        (out / "exam.json").write_text(json.dumps({"args": vars(args), "rows": exam_rows}), encoding="utf-8")
    if fill_rows:
        (out / "fill.json").write_text(json.dumps({"args": vars(args), "rows": fill_rows}), encoding="utf-8")


if __name__ == "__main__":
    main()
