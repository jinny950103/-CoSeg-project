#!/usr/bin/env python3
"""Build a mandibular-canal mask from image edges, using the doctor's clicks only as a rough guide.

The hospital label is one click per slice (array axis 1) drawn as a ~1.2mm ball. Here the
clicks only say *where to look*; the canal wall itself comes from the image:

  1. clicks  -> eroded back to one point per slice, split into left / right canal
  2. points  -> lightly smoothed track, used ONLY for the cutting direction (tangent) and as
                the ray origin; it never decides where the wall is
  3. every ~0.4mm along the track, cut a 2D cross-section perpendicular to the tangent
  4. ray method: 72 rays from the centre, each stops at the first strong dark->bright rise
     (lumen -> cortical wall). Each section keeps its own radius per angle: no circle fit,
     no averaging with neighbouring sections, so thickness changes survive.
  5. check method: morphological geodesic active contour with balloon on the same section
     (same idea as potpov/IAN_annotation_tool's snake, re-implemented with scikit-image).
     The Dice between the two is stored per section as a quality flag.
  6. voxels are filled by projecting each voxel onto its nearest section and comparing its
     radius with that section's ray radius at the same angle.

Sections where too many rays find no wall (weak contrast, open wall near the foramen) are
marked invalid and left empty rather than guessed.

    python scripts/canal_edge_gt.py --image <img.nii.gz> --label <clicks.nii.gz> --out <dir>
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import find_contours
from skimage.segmentation import inverse_gaussian_gradient, morphological_geodesic_active_contour


def ball(r_vox2):
    rv = int(np.ceil(np.sqrt(r_vox2)))
    zz, yy, xx = np.mgrid[-rv:rv + 1, -rv:rv + 1, -rv:rv + 1]
    return zz ** 2 + yy ** 2 + xx ** 2 <= r_vox2


def click_tracks(lab, ax=1, r2=8):
    """{side: (N,3) click points in voxel coords}, one per clicked slice, ordered along ax."""
    pts = ndimage.binary_erosion(lab, ball(r2))
    if not pts.any():
        pts = lab
    idx = np.argwhere(pts)
    other = [a for a in range(3) if a != ax]
    lat = other[int(np.argmax(idx[:, other].std(0)))]
    # split left/right at the widest empty band across the midline; the median put the front end
    # of the side with more points onto the other side, whose track then crossed the chin
    v = np.unique(idx[:, lat])
    gi = int(np.argmax(np.diff(v))) if len(v) > 1 else 0
    mid = (v[gi] + v[gi + 1]) / 2 if len(v) > 1 and v[gi + 1] - v[gi] >= 10 else np.median(idx[:, lat])
    out = {}
    for side in (0, 1):
        sel = idx[(idx[:, lat] > mid) == bool(side)]
        s = np.unique(sel[:, ax])
        if len(s) < 5:
            continue
        out[side] = np.array([sel[sel[:, ax] == i].mean(0) for i in s])
    return out


def resample_track(p, step, sigma_mm, sp):
    """Smooth (for direction only) and resample a click track at fixed arc length (voxels)."""
    sig = max(sigma_mm / sp, 0.5)
    q = ndimage.gaussian_filter1d(p, sig, axis=0, mode="nearest")
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.arange(0, s[-1], step)
    c = np.stack([np.interp(t, s, q[:, k]) for k in range(3)], 1)
    tan = np.gradient(c, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9
    return c, tan


def plane_basis(t, prev_u=None):
    """Orthonormal (u, v) perpendicular to t; u kept close to the previous u so sections don't spin."""
    ref = prev_u if prev_u is not None else (np.array([0, 0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0, 0]))
    u = ref - t * (ref @ t)
    u /= np.linalg.norm(u)
    return u, np.cross(t, u)


def sample_section(vol, c, u, v, half, px):
    """2D section (n x n) centred at c, pixel size px voxels; returns image and grid offsets."""
    g = np.arange(-half, half + 1e-6, px)
    a, b = np.meshgrid(g, g, indexing="ij")
    xyz = c[:, None, None] + u[:, None, None] * a + v[:, None, None] * b
    return ndimage.map_coordinates(vol, xyz, order=1, mode="nearest"), g


def ray_contour(vol, c, u, v, n_rays, r_min, r_max, dr, min_rise):
    """Per-angle wall radius (voxels) or nan.

    On CBCT the canal wall is a thin bright ring around a darker lumen, often with darker marrow
    outside it. Along each ray: take the first intensity peak (the wall) that rises at least
    min_rise above the darkest lumen value before it; the edge is where the profile crosses
    half-way between that lumen minimum and the peak (half-max, sub-sample by interpolation).
    Taking the *first* peak keeps the ray from jumping to the outer cortical plate.
    """
    ang = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    r = np.arange(0, r_max + 1e-6, dr)
    d = np.cos(ang)[:, None, None] * u + np.sin(ang)[:, None, None] * v      # (A,1,3)
    xyz = c + d * r[None, :, None]                                            # (A,R,3)
    prof = ndimage.map_coordinates(vol, xyz.reshape(-1, 3).T, order=1, mode="nearest").reshape(len(ang), len(r))
    radius = np.full(len(ang), np.nan)
    rise = np.zeros(len(ang))
    i0 = int(np.searchsorted(r, r_min))
    for a in range(len(ang)):
        p = prof[a]
        run_min, jmin = p[0], 0
        for j in range(1, len(r) - 1):
            if p[j] < run_min:
                run_min, jmin = p[j], j
            if j < i0 or not (p[j] >= p[j - 1] and p[j] >= p[j + 1]):
                continue
            if p[j] - run_min < min_rise:
                continue
            half = 0.5 * (p[j] + run_min)
            k = jmin + int(np.argmax(p[jmin:j + 1] >= half))     # first crossing after the minimum
            f = (half - p[k - 1]) / (p[k] - p[k - 1]) if k > 0 and p[k] != p[k - 1] else 0.0
            rh = r[k - 1] + np.clip(f, 0, 1) * dr if k > 0 else r[k]
            if rh >= r_min:
                radius[a] = rh
                rise[a] = p[j] - run_min
            break
    return ang, radius, rise


def clean_rays(radius, max_ratio=2.0):
    """Drop rays that shot far past the rest (leaked through a gap). No smoothing of valid rays."""
    ok = np.isfinite(radius)
    if ok.sum() < 3:
        return radius
    med = np.median(radius[ok])
    out = radius.copy()
    out[ok & (radius > max_ratio * med)] = np.nan
    return out


def fill_angles(ang, radius):
    """Linear (periodic) interpolation over missing rays, used only for rasterising."""
    ok = np.isfinite(radius)
    a = np.concatenate([ang[ok] - 2 * np.pi, ang[ok], ang[ok] + 2 * np.pi])
    r = np.tile(radius[ok], 3)
    return np.interp(ang, a, r)


def gac_mask(sec, px, r_init, args_gac_iter=25):
    """Balloon geodesic active contour on a section (the IAN_annotation_tool idea)."""
    n = sec.shape[0]
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2
    init = (yy ** 2 + xx ** 2) <= (r_init / px) ** 2
    s = (sec - sec.min()) / (np.ptp(sec) + 1e-6)
    gimg = inverse_gaussian_gradient(s, alpha=100, sigma=1.5)
    m = morphological_geodesic_active_contour(gimg, args_gac_iter, init, smoothing=2, balloon=1, threshold="auto")
    lab, nl = ndimage.label(m)
    if nl > 1:
        m = lab == lab[n // 2, n // 2] if lab[n // 2, n // 2] else lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return m.astype(bool)


def polar_mask(n, px, ang, radius):
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2
    rr = np.hypot(yy, xx) * px
    th = np.mod(np.arctan2(xx, yy), 2 * np.pi)      # matches sample_section: axis0=u, axis1=v
    rf = np.interp(th, np.concatenate([ang, [2 * np.pi]]), np.concatenate([radius, radius[:1]]))
    return rr <= rf


def dice(a, b):
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else np.nan


def process(img_path, lab_path, out_dir, args):
    im = nib.load(img_path)
    vol = np.asanyarray(im.dataobj).astype(np.float32)
    lab = np.asanyarray(nib.load(lab_path).dataobj) > 0
    sp = float(im.header.get_zooms()[0])
    assert np.allclose(im.header.get_zooms()[:3], sp), "expects isotropic spacing"
    mm = lambda x: x / sp

    # crop around the clicks to keep things fast
    lo = np.maximum(np.argwhere(lab).min(0) - int(mm(8)), 0)
    hi = np.minimum(np.argwhere(lab).max(0) + int(mm(8)) + 1, vol.shape)
    crop = tuple(slice(a, b) for a, b in zip(lo, hi))
    v = ndimage.gaussian_filter(vol[crop], mm(args.sigma_mm))
    labc = lab[crop]

    tracks = click_tracks(labc, ax=args.axis)
    mask = np.zeros(v.shape, bool)
    sections = []
    figs = []
    for side, pts in tracks.items():
        cl, tan = resample_track(pts, mm(args.step_mm), args.track_sigma_mm, sp)
        prev_u = None
        sec_c, sec_u, sec_v, sec_t, sec_r, sec_ok = [], [], [], [], [], []
        for i, (c0, t) in enumerate(zip(cl, tan)):
            u, w = plane_basis(t, prev_u)
            prev_u = u
            # the click may be off the canal: try centres within max_shift of it and keep the one
            # whose rays find a wall most often; then one centroid pass to centre it in the lumen
            best = None
            g = np.arange(-mm(args.max_shift_mm), mm(args.max_shift_mm) + 1e-6, mm(0.4))
            for da in g:
                for db in g:
                    if np.hypot(da, db) > mm(args.max_shift_mm) + 1e-6:
                        continue
                    cc = c0 + da * u + db * w
                    ang, rad, rise = ray_contour(v, cc, u, w, args.rays // 2, mm(args.r_min_mm),
                                                 mm(args.r_max_mm), mm(0.05), args.min_rise)
                    sc = np.isfinite(clean_rays(rad)).mean() - 0.05 * np.hypot(da, db) / mm(1)
                    if best is None or sc > best[0]:
                        best = (sc, cc)
            c = best[1]
            for it in range(2):
                ang, rad, rise = ray_contour(v, c, u, w, args.rays, mm(args.r_min_mm), mm(args.r_max_mm),
                                             mm(0.05), args.min_rise)
                rad = clean_rays(rad)
                ok = np.isfinite(rad)
                if it == 0 and ok.sum() >= args.rays // 2:
                    off = np.array([np.nanmean(rad * np.cos(ang)), np.nanmean(rad * np.sin(ang))])
                    if np.linalg.norm(c + off[0] * u + off[1] * w - c0) <= mm(args.max_shift_mm) * 1.5:
                        c = c + off[0] * u + off[1] * w
            frac = ok.mean()
            valid = frac >= args.min_valid_frac
            rec = {"side": int(side), "i": i, "valid": bool(valid), "valid_ray_frac": float(frac),
                   "shift_mm": float(np.linalg.norm(c - c0) * sp)}
            if valid:
                rf = fill_angles(ang, rad)
                rec.update(r_mean_mm=float(np.mean(rf) * sp), r_min_mm=float(rf.min() * sp),
                           r_max_mm=float(rf.max() * sp), area_mm2=float(0.5 * np.sum(rf ** 2) * (2 * np.pi / len(rf)) * sp ** 2))
                if i % args.gac_every == 0:
                    sec, g = sample_section(v, c, u, w, mm(args.r_max_mm), mm(0.1))
                    px = mm(0.1)
                    gm = gac_mask(sec, px, mm(0.4))
                    rm = polar_mask(sec.shape[0], px, ang, rf)
                    rec["dice_ray_vs_gac"] = float(dice(rm, gm))
                    figs.append((side, i, sec, rm, gm, rec.copy(), np.array([(c0 - c) @ u, (c0 - c) @ w]) * sp))
            else:
                rf = None
            sections.append(rec)
            sec_c.append(c); sec_u.append(u); sec_v.append(w); sec_t.append(t); sec_r.append(rf); sec_ok.append(valid)

        # rasterise: each voxel near the track goes to its nearest *valid* section
        okc = np.array(sec_ok)
        if okc.sum() == 0:
            continue
        C = np.array(sec_c)[okc]; U = np.array(sec_u)[okc]; W = np.array(sec_v)[okc]; T = np.array(sec_t)[okc]
        R = np.array([r for r, o in zip(sec_r, sec_ok) if o])
        tree = cKDTree(C)
        band = np.zeros(v.shape, bool)
        ci = np.clip(np.round(C).astype(int), 0, np.array(v.shape) - 1)
        band[tuple(ci.T)] = True
        band = ndimage.binary_dilation(band, ball(mm(args.r_max_mm) ** 2), iterations=1)
        vox = np.argwhere(band).astype(float)
        dist, j = tree.query(vox)
        d = vox - C[j]
        along = np.abs(np.einsum("ij,ij->i", d, T[j]))
        a = np.einsum("ij,ij->i", d, U[j]); b = np.einsum("ij,ij->i", d, W[j])
        rr = np.hypot(a, b)
        th = np.mod(np.arctan2(b, a), 2 * np.pi)
        A = len(ang)
        k = th / (2 * np.pi) * A
        k0 = np.floor(k).astype(int) % A; k1 = (k0 + 1) % A; f = k - np.floor(k)
        rwall = R[j, k0] * (1 - f) + R[j, k1] * f
        inside = (rr <= rwall) & (along <= mm(args.step_mm))
        hit = vox[inside].astype(int)
        mask[tuple(hit.T)] = True

    full = np.zeros(vol.shape, np.uint8)
    full[crop] = mask
    out_dir.mkdir(parents=True, exist_ok=True)
    cid = Path(img_path).name.replace(".nii.gz", "")
    nib.save(nib.Nifti1Image(full, im.affine, im.header), out_dir / f"{cid}_edge.nii.gz")

    vs = [s for s in sections if s["valid"]]
    dg = [s["dice_ray_vs_gac"] for s in vs if "dice_ray_vs_gac" in s]
    summ = {"case": cid, "n_sections": len(sections), "n_valid": len(vs),
            "r_mean_mm": float(np.mean([s["r_mean_mm"] for s in vs])) if vs else None,
            "r_mean_p5_p95_mm": [float(np.percentile([s["r_mean_mm"] for s in vs], q)) for q in (5, 95)] if vs else None,
            "dice_ray_vs_gac_median": float(np.median(dg)) if dg else None,
            "voxels": int(full.sum()), "doctor_label_voxels": int(lab.sum())}
    (out_dir / f"{cid}_sections.json").write_text(json.dumps({"summary": summ, "sections": sections}, indent=1), encoding="utf-8")

    # figure 1: sample sections (ray = red, GAC = cyan, click = yellow cross)
    if figs:
        figs = [figs[k] for k in np.linspace(0, len(figs) - 1, min(args.n_figs, len(figs))).astype(int)]
        nc = len(figs)
        fig, axs = plt.subplots(1, nc, figsize=(3 * nc, 3.3))
        axs = np.atleast_1d(axs)
        for ax_, (side, i, sec, rm, gm, rec, click) in zip(axs, figs):
            ext = sec.shape[0] * 0.1 / 2
            ax_.imshow(sec, cmap="gray", extent=[-ext, ext, ext, -ext])
            for m, col in ((rm, "r"), (gm, "c")):
                for cc in find_contours(m.astype(float), 0.5):
                    ax_.plot(cc[:, 1] * 0.1 - ext, cc[:, 0] * 0.1 - ext, col, lw=1)
            ax_.plot(click[1], click[0], "y+", ms=8)
            ax_.set_title(f"side{side} #{i}\nr={rec['r_mean_mm']:.2f}mm Dice={rec['dice_ray_vs_gac']:.2f}", fontsize=8)
            ax_.axis("off")
        fig.suptitle(f"{cid}  red=ray edge  cyan=GAC  yellow=doctor click", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"{cid}_sections.png", dpi=130)
        plt.close(fig)

    # figure 2: radius along the canal (shows thickness variation; gaps = invalid sections)
    fig, ax_ = plt.subplots(figsize=(8, 3))
    for side in sorted({s["side"] for s in sections}):
        ss = [s for s in sections if s["side"] == side]
        x = np.array([s["i"] for s in ss]) * args.step_mm
        y = np.array([s.get("r_mean_mm", np.nan) for s in ss])
        lo_ = np.array([s.get("r_min_mm", np.nan) for s in ss]); hi_ = np.array([s.get("r_max_mm", np.nan) for s in ss])
        l, = ax_.plot(x, y, label=f"side{side} mean r")
        ax_.fill_between(x, lo_, hi_, color=l.get_color(), alpha=0.2)
    ax_.set_xlabel("distance along canal (mm)"); ax_.set_ylabel("radius (mm)"); ax_.legend(fontsize=8)
    ax_.set_title(f"{cid}  per-section radius (band = min..max over angles)", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / f"{cid}_radius.png", dpi=130); plt.close(fig)
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", nargs="+", required=True)
    ap.add_argument("--label", nargs="+", required=True, help="醫生點擊球形標註,順序對應 --image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--axis", type=int, default=1, help="醫生逐張點選的那一軸")
    ap.add_argument("--step_mm", type=float, default=0.4, help="截面間距")
    ap.add_argument("--track_sigma_mm", type=float, default=1.2, help="只用於切面方向的軌跡平滑")
    ap.add_argument("--sigma_mm", type=float, default=0.2, help="影像去雜訊高斯")
    ap.add_argument("--rays", type=int, default=72)
    ap.add_argument("--r_min_mm", type=float, default=0.3)
    ap.add_argument("--r_max_mm", type=float, default=3.0)
    ap.add_argument("--min_rise", type=float, default=100, help="管壁兩側最小亮度差(影像原始單位)")
    ap.add_argument("--min_valid_frac", type=float, default=0.75)
    ap.add_argument("--max_shift_mm", type=float, default=1.2, help="重新置中最多移動")
    ap.add_argument("--gac_every", type=int, default=5, help="每幾張截面做一次 GAC 對照")
    ap.add_argument("--n_figs", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    allsum = []
    for ip, lp in zip(args.image, args.label):
        s = process(ip, lp, out, args)
        print(json.dumps(s, ensure_ascii=False), flush=True)
        allsum.append(s)
    (out / "summary.json").write_text(json.dumps(allsum, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
