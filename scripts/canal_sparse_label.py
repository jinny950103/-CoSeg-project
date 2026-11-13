#!/usr/bin/env python3
"""Sparse per-direction canal labels for nnU-Net v2 (background / canal / ignore).

The whole-section rule (keep a section only if >=75% of its rays find the wall) throws away
most of the wall the ray method did find. Here every ray counts on its own:

  ray found the wall at radius r   ->  centre..r          = canal (1)
                                       r..r + wall_mm      = background (0, the bright wall)
  ray found nothing                ->  that direction      = ignore (2)
  farther than far_mm from the doctor track                = background (0)
  everything else near the canal                           = ignore (2)

No filling between rays, no smoothing along the canal, no circle fit: unknown stays unknown.
Rays that are probably wrong are dropped (section with too few rays, lone rays, radius jumps;
see filter_rays), and a small disc at the lumen centre is labelled canal so the middle of the
canal has no ignore holes.
Two more checks only turn rays into ignore (nothing is filled or smoothed):
  lumen check    the stretch centre..edge on a ray must stay dark like the lumen; a ray whose
                 inside is clearly brighter than the centre ran through marrow or along a plate
  cross-section  the wall point of a ray must lie near the wall found in the sections just
                 before and after (+-k * 0.4mm); a ray that disagrees with its neighbours is dropped
  location       (--ds_dir) the wall point must lie within ds_mm of the canal predicted by the
                 public DentalSegmentator model; used for location only, never for the boundary
With --agree_mm the ray must also match polar dynamic programming (scripts/canal_edge_compare.py)
on the same angle, per section (--agree_with polardp) or solved over the whole canal at once
(--agree_with polar3d); both variants are written so they can be compared.

Sections, centres and rays are the same as in canal_edge_compare.py (perpendicular to the
click track every 0.4mm; start = darkest point within 1.2mm of the click).
nnU-Net v2 needs the ignore label to be the highest one: labels {"background":0,"canal":1,"ignore":2}.

    python scripts/canal_sparse_label.py --image a.nii.gz ... --label a_lab.nii.gz ... --out outputs/sparse_label
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import canal_edge_gt as E  # noqa: E402
import canal_edge_compare as C  # noqa: E402

BG, CANAL, IGN = 0, 1, 2
ANG, N_ANG, PX, HALF = C.ANG, C.N_ANG, C.PX, C.HALF


def section_label_2d(rad, wall_mm, n, core=np.nan):
    """2D label of one section from per-ray radii (mm), same rule as the 3D rasterisation."""
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2
    rr = np.hypot(yy, xx) * PX
    th = np.mod(np.arctan2(xx, yy), 2 * np.pi)
    out = label_from_polar(rr, th, rad, wall_mm)
    out[(out == IGN) & (rr <= core)] = CANAL
    return out


def label_from_polar(rr, th, rad, wall_mm):
    """rr (mm), th (rad) -> label using the nearest ray's radius; no interpolation between rays."""
    k = np.round(th / (2 * np.pi) * N_ANG).astype(int) % N_ANG
    r = rad[k]
    out = np.full(rr.shape, IGN, np.uint8)
    ok = np.isfinite(r)
    out[ok & (rr <= r)] = CANAL
    out[ok & (rr > r) & (rr <= r + wall_mm)] = BG
    return out


def filter_rays(rad, min_sec_frac, min_run, max_jump_mm):
    """Drop rays that are probably wrong, without changing any kept radius.

    - whole section dropped if fewer than min_sec_frac of its rays found a wall
    - neighbouring rays whose radii jump more than max_jump_mm are not one piece of wall
    - a ray is kept only inside a run of >= min_run neighbouring, continuous rays
      (a lone ray or two is usually a leak into marrow or onto a bone plate)
    """
    ok = np.isfinite(rad)
    if ok.mean() < min_sec_frac:
        return np.full_like(rad, np.nan)
    n = len(rad)
    link = ok & np.roll(ok, -1) & (np.abs(rad - np.roll(rad, -1)) <= max_jump_mm)   # ray a joins a+1
    if link.all():
        return rad
    if not link.any():
        return np.full_like(rad, np.nan) if min_run > 1 else rad
    out = np.full_like(rad, np.nan)
    start = (int(np.argmin(link)) + 1) % n            # a ray just after a break: walk the circle once
    run = [start]
    for s in range(1, n + 1):
        a = (start + s) % n
        if link[(a - 1) % n] and s < n:
            run.append(a)
            continue
        if len(run) >= min_run and ok[run].all():
            out[run] = rad[run]
        run = [a]
    return out


def lumen_ratio(v, c, u, w, rad_mm, sp):
    """Per ray: how bright the stretch centre..edge is, relative to the wall rise.

    0 = as dark as the centre, 1 = as bright as the wall peak. Same rays as the ray method.
    Returns (mean ratio, max ratio, wall peak minus centre), nan where the ray has no edge.
    A peak that is not brighter than the centre means the ray first dropped into a darker region
    (marrow) and the "wall" it found is only the way back up from there.
    """
    dr = 0.05
    r = np.arange(0, HALF + 1e-6, dr)
    d = np.cos(ANG)[:, None, None] * u + np.sin(ANG)[:, None, None] * w
    xyz = c + d * (r / sp)[None, :, None]
    prof = ndimage.map_coordinates(v, xyz.reshape(-1, 3).T, order=1, mode="nearest").reshape(N_ANG, len(r))
    ref = prof[:, 0].mean()
    mean_r, max_r, wall = (np.full(N_ANG, np.nan) for _ in range(3))
    for a in np.where(np.isfinite(rad_mm))[0]:
        inside = prof[a, r < rad_mm[a]]
        peak = prof[a, (r >= rad_mm[a]) & (r <= rad_mm[a] + 0.6)].max()
        wall[a] = peak - ref
        if wall[a] > 0:
            mean_r[a] = (inside.mean() - ref) / wall[a]
            max_r[a] = (inside.max() - ref) / wall[a]
    return mean_r, max_r, wall


def xsec_drop(R, Cc, U, W, T, sp, k, max_d, min_nb):
    """Rays (n_sec x N_ANG, mm) whose wall point is far from the wall in neighbouring sections.

    Wall points are compared in 3D, not as radii: the lumen centre of each section moves a little
    from section to section, which changes every radius without the wall moving. For each
    neighbour section within +-k: distance (across the canal, along-canal offset removed) from
    this ray's wall point to the nearest wall point found in that section. A ray is dropped when
    the median of those distances is > max_d. Rays with fewer than min_nb usable neighbours are
    left as they are.
    """
    n = len(R)
    dirs = np.cos(ANG)[None, :, None] * U[:, None] + np.sin(ANG)[None, :, None] * W[:, None]
    P = Cc[:, None] * sp + R[..., None] * dirs                     # (n, A, 3) mm, nan where no wall
    D = np.full((2 * k, n, N_ANG), np.nan)
    for q, o in enumerate([o for o in range(-k, k + 1) if o]):
        for i in range(max(0, -o), min(n, n - o)):
            pj = P[i + o][np.isfinite(R[i + o])]
            if not len(pj):
                continue
            d = P[i][:, None] - pj[None]                           # (A, B, 3)
            d -= (d @ T[i])[..., None] * T[i]
            D[q, i] = np.linalg.norm(d, axis=2).min(1)
    nb = np.isfinite(D).sum(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                     # all-nan rows -> nan
        med = np.nanmedian(D, 0)
    found = np.isfinite(R)
    drop = found & (nb >= min_nb) & (med > max_d)
    unchecked = found & (nb < min_nb)
    return drop, unchecked, np.where(found, med, np.nan)


def ds_far(ds_dist, c, u, w, rad_mm, sp, args):
    """Rays whose wall point is more than ds_mm from an independent model's canal (location only).

    ds_dist: distance (mm) to the DentalSegmentator canal mask. That model learned smooth
    public-style labels, so it only says roughly where the canal is; kept radii are not changed.
    Sections where the model has no canal within ds_centre_mm of the lumen centre are not checked
    (the model missed that part; no evidence either way).
    """
    out = np.zeros(N_ANG, bool)
    if ds_dist is None:
        return out
    if ndimage.map_coordinates(ds_dist, c[:, None], order=1, mode="nearest")[0] > args.ds_centre_mm:
        return out
    ok = np.isfinite(rad_mm)
    d = np.cos(ANG)[:, None] * u + np.sin(ANG)[:, None] * w
    p = c + d * (np.where(ok, rad_mm, 0) / sp)[:, None]
    dist = ndimage.map_coordinates(ds_dist, p.T, order=1, mode="nearest")
    return ok & (dist > args.ds_mm)


def rnd(a):
    return [None if not np.isfinite(x) else round(float(x), 3) for x in a]


def core_radius(rad, core_mm):
    """Small disc around the lumen centre labelled canal, never past the nearest found wall."""
    return min(core_mm, float(np.nanmin(rad))) if np.isfinite(rad).any() else np.nan


def darkest_centre(v, c0, u, w, search, px):
    """Neutral start: darkest point (lumen) within search voxels of c0 in the (u, w) plane."""
    big, g = E.sample_section(v, c0, u, w, search, px)
    yy, xx = np.meshgrid(g, g, indexing="ij")
    big = np.where(np.hypot(yy, xx) <= search, ndimage.gaussian_filter(big, 2), np.inf)
    k = np.unravel_index(np.argmin(big), big.shape)
    return c0 + g[k[0]] * u + g[k[1]] * w


def hessian_volume(v, sigma):
    """Six second derivatives of the image at the canal scale (Gaussian derivatives, voxel units)."""
    out = {}
    for a in range(3):
        for b in range(a, 3):
            order = [0, 0, 0]; order[a] += 1; order[b] += 1
            out[(a, b)] = ndimage.gaussian_filter(v, sigma, order=order)
    return out


def hessian_tangents(v, H, cl, tan, mm, args):
    """Canal direction from the image (method 2), one per section.

    At the lumen centre a dark tube has two large positive Hessian eigenvalues (across the tube)
    and one near zero (along it); the eigenvector of the near-zero one is the canal direction.
    The doctor track only decides where to look. The direction is kept only where the point is
    really tube-like and within max_tilt of the track; elsewhere the track direction is used.
    Directions are lightly averaged over neighbouring sections (cutting plane only; the wall
    itself is still found per section and per ray).
    """
    n = len(cl)
    dirs = tan.copy()
    used = np.zeros(n, bool)
    for i in range(n):
        u, w = E.plane_basis(tan[i])
        c = darkest_centre(v, cl[i], u, w, mm(args.search_mm), mm(PX))
        m = np.zeros((3, 3))
        for (a, b), h in H.items():
            m[a, b] = m[b, a] = ndimage.map_coordinates(h, c[:, None], order=1, mode="nearest")[0]
        lam, vec = np.linalg.eigh(m)                       # ascending
        k = int(np.argmin(np.abs(lam)))
        across = np.delete(lam, k)
        tube = across.min() > 0 and abs(lam[k]) < args.tube_ratio * across.min()
        d = vec[:, k] * np.sign(vec[:, k] @ tan[i] + 1e-12)
        if tube and np.degrees(np.arccos(np.clip(d @ tan[i], -1, 1))) <= args.max_tilt_deg:
            dirs[i] = d; used[i] = True
    if args.dir_sigma_mm > 0:
        dirs = ndimage.gaussian_filter1d(dirs, args.dir_sigma_mm / args.step_mm, axis=0, mode="nearest")
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    tilt = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", dirs, tan), -1, 1)))
    print(f"  hessian direction used {used.mean():.0%}, tilt median {np.median(tilt):.1f} deg, p90 {np.percentile(tilt, 90):.1f}", flush=True)
    return dirs, tilt


def process(ip, lp, args):
    im = nib.load(ip)
    vol = np.asanyarray(im.dataobj).astype(np.float32)
    lab = np.asanyarray(nib.load(lp).dataobj) > 0
    sp = float(im.header.get_zooms()[0]); mm = lambda x: x / sp
    lo = np.maximum(np.argwhere(lab).min(0) - int(mm(8)), 0)
    hi = np.minimum(np.argwhere(lab).max(0) + int(mm(8)) + 1, vol.shape)
    crop = tuple(slice(a, b) for a, b in zip(lo, hi))
    v = ndimage.gaussian_filter(vol[crop], mm(0.2))
    cid = Path(ip).name.replace(".nii.gz", "")

    ds_dist = None
    if args.ds_dir:
        ds = np.asanyarray(nib.load(Path(args.ds_dir) / f"{cid}_dentalseg_canal.nii.gz").dataobj)[crop] > 0
        ds_dist = ndimage.distance_transform_edt(~ds) * sp if ds.any() else None

    agree_fn = C.m_polardp if args.agree_with == "polardp" else C.PolarDP3D(rise_frac=args.agree_rise_frac)
    variants = ("ray", "agree")
    lab_c ={k: np.full(v.shape, IGN, np.uint8) for k in variants}
    track = np.zeros(v.shape, bool)
    secs, figs = [], []
    rays = {}                                   # per side, for --save_rays (canal_gap_fill.py)
    H = None
    for side, pts in E.click_tracks(lab[crop]).items():
        cl, tan = E.resample_track(pts, mm(args.step_mm), 1.2, sp)
        ci = np.clip(np.round(cl).astype(int), 0, np.array(v.shape) - 1)
        track[tuple(ci.T)] = True
        if args.direction == "hessian":
            if H is None:
                H = hessian_volume(v, mm(args.hess_sigma_mm))
            tan, tilt = hessian_tangents(v, H, cl, tan, mm, args)
        else:
            tilt = np.zeros(len(cl))
        prev_u = None
        S = {"c": [], "u": [], "w": [], "t": [], "pre": {k: [] for k in variants}, "lum": {k: [] for k in variants},
             "ds": {k: [] for k in variants}}
        sec0, fig0 = len(secs), len(figs)
        for i in range(len(cl)):
            u, w = E.plane_basis(tan[i], prev_u); prev_u = u
            c0 = cl[i]
            c = darkest_centre(v, c0, u, w, mm(args.search_mm), mm(PX))
            ctx = dict(vol=v, c=c, u=u, w=w, sp=sp, mm=mm, min_rise=args.min_rise, cid=cid, side=int(side), i=i,
                       track=(cl, tan), search_mm=args.search_mm)
            r_ray = E.clean_rays(C.m_ray(ctx))
            r_dp = agree_fn(ctx)
            r_agree = np.where(np.abs(r_ray - r_dp) <= args.agree_mm, r_ray, np.nan)
            raw = {"ray": r_ray, "agree": r_agree}
            lum_mean, lum_max, wall = lumen_ratio(v, c, u, w, r_ray, sp)
            bright = np.zeros(N_ANG, bool)
            if args.lum_frac > 0:
                bright |= lum_mean > args.lum_frac
            if args.wall_rise > 0:
                bright |= np.isfinite(r_ray) & (wall < args.wall_rise)
            far_ds = ds_far(ds_dist, c, u, w, r_ray, sp, args)
            for kk in variants:
                S["lum"][kk].append(int((np.isfinite(raw[kk]) & bright).sum()))
                S["ds"][kk].append(int((np.isfinite(raw[kk]) & far_ds & ~bright).sum()))
                S["pre"][kk].append(filter_rays(np.where(bright | far_ds, np.nan, raw[kk]),
                                                args.min_sec_frac, args.min_run, args.max_jump_mm))
            S["c"].append(c); S["u"].append(u); S["w"].append(w); S["t"].append(tan[i])
            secs.append({"side": int(side), "i": i,
                         "ray_frac_raw": float(np.isfinite(r_ray).mean()),
                         "agree_frac_raw": float(np.isfinite(r_agree).mean()),
                         "dp_frac": float(np.isfinite(r_dp).mean()),
                         "tilt_deg": float(tilt[i]),
                         "ray_rad_raw": rnd(r_ray), "lum_mean": rnd(lum_mean), "lum_max": rnd(lum_max),
                         "wall_rise": rnd(wall)})
            focus = f"{cid}:{side}:{i}" in args.focus
            if i % args.fig_every == 0 or focus:
                sec, _ = E.sample_section(v, c, u, w, mm(HALF), mm(PX))
                v2 = filter_rays(r_ray, args.min_sec_frac, args.min_run, args.max_jump_mm)    # no checks
                figs.append([side, i, sec, {"pre": v2}, np.array([(c0 - c) @ u, (c0 - c) @ w]) * sp, focus])

        # cross-section check on the filtered rays, then the run filter again (a ray left alone
        # after the drop goes too); both checks only remove rays
        S["rad"], S["core"] = {}, {}
        for kk in variants:
            P = np.array(S["pre"][kk])
            if args.xsec_mm > 0:
                drop, unchk, med = xsec_drop(P, np.array(S["c"]), np.array(S["u"]), np.array(S["w"]),
                                             np.array(S["t"]), sp, args.xsec_k, args.xsec_mm, args.xsec_min_nb)
            else:
                drop = unchk = np.zeros(P.shape, bool); med = np.full(P.shape, np.nan)
            # section-fraction rule already judged the raw rays; here only the run rule again
            R = [filter_rays(np.where(drop[i], np.nan, P[i]), 0, args.min_run, args.max_jump_mm)
                 for i in range(len(P))]
            S["rad"][kk] = R
            S["core"][kk] = [core_radius(r_, args.core_mm) for r_ in R]
            for i, st in enumerate(secs[sec0:]):
                st[f"{kk}_frac_prefilter"] = float(np.isfinite(P[i]).mean())
                st[f"{kk}_frac"] = float(np.isfinite(R[i]).mean())
                st[f"{kk}_lum_drop"] = S["lum"][kk][i]
                st[f"{kk}_ds_drop"] = S["ds"][kk][i]
                st[f"{kk}_xsec_drop"] = int(drop[i].sum())
                st[f"{kk}_xsec_unchecked"] = int(unchk[i].sum())
            if kk == "ray":
                for i, st in enumerate(secs[sec0:]):
                    st["r_mean_mm"] = float(np.nanmean(R[i])) if np.isfinite(R[i]).any() else None
                    st["xsec_med"] = rnd(med[i])
        for f in figs[fig0:]:
            f[3].update(ray=S["rad"]["ray"][f[1]], agree=S["rad"]["agree"][f[1]])

        rays[int(side)] = {"c": np.array(S["c"]) + lo, "u": np.array(S["u"]), "w": np.array(S["w"]),
                           "t": np.array(S["t"]), "click": cl + lo,
                           **{f"r_{kk}": np.array(S["rad"][kk]) for kk in variants}}

        # 3D: each voxel near this track is read by its own ray in its nearest section; if that
        # ray found nothing, the second-nearest section is tried (sections are only 0.4mm apart)
        Cc = np.array(S["c"]); U = np.array(S["u"]); W = np.array(S["w"]); T = np.array(S["t"])
        tree = cKDTree(Cc)
        band = np.zeros(v.shape, bool)
        band[tuple(np.clip(np.round(Cc).astype(int), 0, np.array(v.shape) - 1).T)] = True
        band = ndimage.binary_dilation(band, E.ball(mm(HALF) ** 2))
        vox = np.argwhere(band).astype(float)
        _, J = tree.query(vox, k=2)
        vi = vox.astype(int)
        for kk in variants:
            R = np.array(S["rad"][kk]); Rc = np.array(S["core"][kk])
            lv = np.full(len(vox), IGN, np.uint8)
            for j in (J[:, 0], J[:, 1]):
                d = vox - Cc[j]
                along = np.abs(np.einsum("ij,ij->i", d, T[j]))
                a = np.einsum("ij,ij->i", d, U[j]); b = np.einsum("ij,ij->i", d, W[j])
                rr = np.hypot(a, b) * sp
                th = np.mod(np.arctan2(b, a), 2 * np.pi)
                near = along <= mm(args.step_mm) + 0.5            # voxel lies on that section's slab
                r = R[j, np.round(th / (2 * np.pi) * N_ANG).astype(int) % N_ANG]
                todo = near & (lv == IGN)
                ok = todo & np.isfinite(r)
                lv[ok & (rr <= r)] = CANAL
                lv[ok & (rr > r) & (rr <= r + args.wall_mm)] = BG
                lv[todo & (rr <= Rc[j])] = CANAL
            cur = lab_c[kk][tuple(vi.T)]
            # merge with the other side / overlapping slabs: canal > background > ignore
            new = np.where((lv == CANAL) | (cur == CANAL), CANAL, np.where((lv == BG) | (cur == BG), BG, IGN))
            lab_c[kk][tuple(vi.T)] = new

    far = ndimage.distance_transform_edt(~track) * sp > args.far_mm
    stats = {"case": cid, "n_sections": len(secs)}
    outs = {}
    for kk in variants:
        lab_c[kk][far & (lab_c[kk] == IGN)] = BG
        full = np.zeros(vol.shape, np.uint8)                     # outside the crop: >8mm from clicks
        full[crop] = lab_c[kk]
        outs[kk] = full
        nb = ndimage.distance_transform_edt(~track) * sp <= args.far_mm
        x = lab_c[kk][nb]
        stats[kk] = {"canal_vox": int((full == CANAL).sum()),
                     "wall_bg_vox": int(((lab_c[kk] == BG) & nb).sum()),
                     "ignore_vox": int((full == IGN).sum()),
                     "known_frac_near_track": float((x != IGN).mean()),
                     "known_frac_r1p5mm": float((lab_c[kk][ndimage.distance_transform_edt(~track) * sp <= 1.5] != IGN).mean())}
    fr = np.array([s["ray_frac_raw"] for s in secs])
    for k_ in ("ray_frac_raw", "ray_frac_prefilter", "ray_frac", "agree_frac_raw", "agree_frac"):
        stats[k_ + "_mean"] = float(np.mean([s[k_] for s in secs]))
    nraw = max(sum(x is not None for s in secs for x in s["ray_rad_raw"]), 1)
    for k_ in ("lum_drop", "ds_drop", "xsec_drop", "xsec_unchecked"):
        stats[f"ray_{k_}_of_raw"] = float(sum(s[f"ray_{k_}"] for s in secs) / nraw)
    stats["whole_section_rule_frac"] = float((fr >= 0.75).mean())     # old rule, for comparison
    stats["sections_kept"] = float(np.mean([s["ray_frac"] > 0 for s in secs]))
    rm = [s["r_mean_mm"] for s in secs if s["r_mean_mm"] is not None]
    stats["r_mean_mm_median"] = float(np.median(rm)) if rm else None
    stats["tilt_deg_median"] = float(np.median([s["tilt_deg"] for s in secs]))
    return cid, im, outs, stats, secs, figs, crop, v, rays


def save_figs(out, cid, figs, outs, crop, v, args):
    # sections: canal red, background-wall blue, ignore = no colour
    reg = [f for f in figs if f[1] % args.fig_every == 0]
    pick = [reg[k] for k in np.linspace(0, len(reg) - 1, min(args.n_figs, len(reg))).astype(int)]
    pick = sorted({(f[0], f[1]): f for f in pick + [f for f in figs if f[5]]}.values(), key=lambda f: (f[0], f[1]))
    rows = ("pre", "ray", "agree")                          # pre = ray before the two checks
    fig, axs = plt.subplots(len(rows), len(pick), figsize=(2.3 * len(pick), 2.5 * len(rows)), squeeze=False)
    for ci, (side, i, sec, rads, click, _) in enumerate(pick):
        for ri, kk in enumerate(rows):
            ax = axs[ri, ci]
            ax.imshow(sec, cmap="gray", extent=[-HALF, HALF, HALF, -HALF])
            L = section_label_2d(rads[kk], args.wall_mm, sec.shape[0], core_radius(rads[kk], args.core_mm))
            ov = np.zeros(L.shape + (4,))
            ov[L == CANAL] = (1, 0.1, 0.1, 0.45)
            ov[L == BG] = (0.2, 0.5, 1, 0.45)
            ax.imshow(ov, extent=[-HALF, HALF, HALF, -HALF])
            ax.plot(click[1], click[0], "y+", ms=7)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"s{side} #{i} {kk} {np.isfinite(rads[kk]).mean():.0%}", fontsize=8)
    fig.suptitle(f"{cid}  rows: before checks / ray / agree  red=canal  blue=background(wall)  no colour=ignore  yellow+=click  6x6mm", fontsize=9)
    fig.tight_layout(); fig.savefig(out / f"{cid}_sections.png", dpi=110); plt.close(fig)

    # 3D check: slices of the written label volume across the click axis
    L = outs["ray"][crop]
    zs = np.where((L == CANAL).any((0, 2)))[0]
    if len(zs):
        zs = zs[np.linspace(0, len(zs) - 1, 6).astype(int)]
        fig, axs = plt.subplots(1, len(zs), figsize=(3 * len(zs), 3.2))
        for ax, z in zip(np.atleast_1d(axs), zs):
            sl = L[:, z, :]
            cc, nc = ndimage.label(sl == CANAL)               # left and right canal: centre on the larger
            big = np.argmax(np.bincount(cc.ravel())[1:]) + 1
            yx = np.argwhere(cc == big).mean(0).astype(int)
            h = int(round(6 / 0.4))
            ys = slice(max(yx[0] - h, 0), yx[0] + h); xs = slice(max(yx[1] - h, 0), yx[1] + h)
            ax.imshow(v[ys, z, xs], cmap="gray")
            s = sl[ys, xs]
            ov = np.zeros(s.shape + (4,))
            ov[s == CANAL] = (1, 0.1, 0.1, 0.45); ov[s == BG] = (0.2, 0.5, 1, 0.25)
            ax.imshow(ov); ax.set_title(f"slice {z + crop[1].start}", fontsize=8); ax.axis("off")
        fig.suptitle(f"{cid}  label volume (ray variant), 12x12mm; blue includes far background", fontsize=9)
        fig.tight_layout(); fig.savefig(out / f"{cid}_volume.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", nargs="+", required=True)
    ap.add_argument("--label", nargs="+", required=True, help="醫生點擊球形標註,順序對應 --image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step_mm", type=float, default=0.4)
    ap.add_argument("--search_mm", type=float, default=1.2)
    ap.add_argument("--min_rise", type=float, default=100)
    ap.add_argument("--wall_mm", type=float, default=0.4, help="內緣外標成背景的厚度(亮管壁)")
    ap.add_argument("--far_mm", type=float, default=6.0, help="離醫生軌跡超過此距離=背景")
    ap.add_argument("--agree_mm", type=float, default=0.3, help="agree 版本:射線法與 polardp 同方向差距上限")
    ap.add_argument("--agree_with", choices=["polardp", "polar3d"], default="polardp",
                    help="agree 版本拿來比的方法:逐截面 polardp,或整條管一起解的 3D 版")
    ap.add_argument("--agree_rise_frac", type=float, default=0.25, help="polar3d 收射線的亮度上升門檻(×min_rise)")
    ap.add_argument("--min_sec_frac", type=float, default=0.25, help="截面抓到管壁的射線比例低於此值整張不收")
    ap.add_argument("--min_run", type=int, default=3, help="至少連續幾條相鄰射線都抓到才收")
    ap.add_argument("--max_jump_mm", type=float, default=0.4, help="相鄰射線半徑差超過此值視為不連續")
    ap.add_argument("--core_mm", type=float, default=0.4, help="中心這一小圈直接標成管(不超過最近的管壁)")
    ap.add_argument("--lum_frac", type=float, default=0.3, help="管內亮度檢查:中心到內緣平均亮度超過管壁上升幅度的這個比例就改 ignore(0=關)")
    ap.add_argument("--wall_rise", type=float, default=50, help="管壁峰至少要比管中心亮這麼多,否則是先掉進更暗的骨髓再爬上來,改 ignore(0=關)")
    ap.add_argument("--xsec_mm", type=float, default=0.5, help="前後截面交叉檢查:管壁點離前後截面管壁的中位距離超過此值就改 ignore(0=關)")
    ap.add_argument("--xsec_k", type=int, default=3, help="前後各看幾張截面(每張 0.4mm)")
    ap.add_argument("--xsec_min_nb", type=int, default=2, help="至少有幾張前後截面可比才檢查,不夠就不動")
    ap.add_argument("--ds_dir", default="", help="DentalSegmentator 管預測資料夾(scripts/run_dentalsegmentator.py);空=不檢查")
    ap.add_argument("--ds_mm", type=float, default=1.0, help="管壁點離 DentalSegmentator 的管超過此距離就改 ignore")
    ap.add_argument("--ds_centre_mm", type=float, default=2.0, help="截面中心離它的管超過此距離=模型漏掉這段,不檢查")
    ap.add_argument("--focus", nargs="*", default=[], help="一定要畫的截面,格式 病例:側:截面編號")
    ap.add_argument("--direction", choices=["track", "hessian"], default="track", help="切截面方向:醫生軌跡,或影像 Hessian 估的管子走向(辦法 2)")
    ap.add_argument("--hess_sigma_mm", type=float, default=0.8, help="Hessian 尺度,約管子半徑")
    ap.add_argument("--tube_ratio", type=float, default=0.5, help="沿管方向特徵值須小於橫向的這個比例才算管狀")
    ap.add_argument("--max_tilt_deg", type=float, default=35, help="和醫生軌跡夾角超過就不用影像方向")
    ap.add_argument("--dir_sigma_mm", type=float, default=0.8, help="切面方向沿管的輕微平均")
    ap.add_argument("--save_rays", action="store_true", help="另存每張截面的中心、方向和最後的射線半徑(npz,給 canal_gap_fill.py)")
    ap.add_argument("--fig_every", type=int, default=10)
    ap.add_argument("--n_figs", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    allst = []
    for ip, lp in zip(args.image, args.label):
        cid, im, outs, st, secs, figs, crop, v, rays = process(ip, lp, args)
        if args.save_rays:
            np.savez_compressed(out / f"{cid}_rays.npz", sp=float(im.header.get_zooms()[0]),
                                **{f"s{sd}_{k}": a for sd, d in rays.items() for k, a in d.items()})
        for kk, arr in outs.items():
            nib.save(nib.Nifti1Image(arr, im.affine), out / f"{cid}_{kk}.nii.gz")
        (out / f"{cid}_sections.json").write_text(json.dumps(secs), encoding="utf-8")
        if args.n_figs > 0:
            save_figs(out, cid, figs, outs, crop, v, args)
        print(json.dumps(st), flush=True)
        allst.append(st)
    (out / "summary.json").write_text(json.dumps({"args": vars(args), "cases": allst}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
