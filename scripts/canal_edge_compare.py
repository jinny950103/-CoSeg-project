#!/usr/bin/env python3
"""Compare edge-detection methods for the mandibular canal wall on the same CBCT cross-sections.

Every method sees the same sections (perpendicular to the doctor-click track, every --every
sections) and the same start point: the darkest pixel within --search_mm of the click (lumen),
so no method gets the ray method's own re-centring. Each method returns a wall radius for each
of 72 angles (nan where it finds nothing):

  ray       first bright-ring peak, half-max edge            (ours, scripts/canal_edge_gt.py)
  gac       balloon geodesic active contour                  (potpov/IAN_annotation_tool idea, skimage)
  acwe      morphological Chan-Vese, region based            (scikit-image)
  canny     Canny edge map, first edge along each ray          (scikit-image)
  hed       HED deep edge map (BSDS500 weights)              (s9xie/hed via sniklaus/pytorch-hed)
  pidinet   PiDiNet deep edge map (BSDS500 weights)          (hellozhuo/pidinet)
  otsu2d    Otsu per section + 2D region growing of the lumen  (Li et al. 2026 hybrid, no hand barrier)
  otsu3d    Otsu per case + 3D region growing of the lumen     (same, 3D as in the paper)

For the edge-map methods the radius is the first edge response along the ray above a threshold,
so they all share one read-out rule. No ground truth exists, so the report gives: fraction of
valid sections, radius statistics, whether the doctor's click falls inside, how strongly the
contour sits on an image edge (gradient on contour / gradient in section), pairwise agreement
(Dice) and run time.

    python scripts/canal_edge_compare.py --image a.nii.gz b.nii.gz --label a_lab.nii.gz b_lab.nii.gz \
        --out outputs/edge_compare --ext <dir with pytorch-hed and pidinet clones>
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.segmentation import (inverse_gaussian_gradient, morphological_chan_vese,
                                  morphological_geodesic_active_contour)

sys.path.insert(0, str(Path(__file__).parent))
import canal_edge_gt as E  # noqa: E402

PX = 0.1          # section pixel size (mm)
HALF = 3.0        # section half width (mm)
N_ANG = 72
R_MIN = 0.3       # mm
ANG = np.linspace(0, 2 * np.pi, N_ANG, endpoint=False)


# ---------------------------------------------------------------- read-outs
def radii_from_mask(m):
    """Per-angle distance (mm) from the section centre to where the region containing it ends."""
    n = m.shape[0]
    c = (n - 1) / 2
    lab, _ = ndimage.label(m)
    if lab[int(c), int(c)] == 0:
        return np.full(N_ANG, np.nan)
    m = lab == lab[int(c), int(c)]
    r = np.arange(0, HALF, PX / 2)
    out = np.full(N_ANG, np.nan)
    for a, t in enumerate(ANG):
        y = np.clip(np.round(c + r * np.cos(t) / PX).astype(int), 0, n - 1)
        x = np.clip(np.round(c + r * np.sin(t) / PX).astype(int), 0, n - 1)
        inside = m[y, x]
        k = np.argmin(inside) if not inside.all() else None
        if k is not None and k > 0:
            out[a] = r[k]
    return out


def radii_from_edgemap(e, thr):
    """Per-angle distance (mm) to the first edge response >= thr beyond R_MIN (local max)."""
    n = e.shape[0]
    c = (n - 1) / 2
    r = np.arange(0, HALF, PX / 2)
    yy = c + np.outer(np.cos(ANG), r) / PX
    xx = c + np.outer(np.sin(ANG), r) / PX
    prof = ndimage.map_coordinates(e.astype(float), [yy, xx], order=1, mode="nearest")
    out = np.full(N_ANG, np.nan)
    i0 = int(np.searchsorted(r, R_MIN))
    for a in range(N_ANG):
        p = prof[a]
        for j in range(max(i0, 1), len(r) - 1):
            if p[j] >= thr and p[j] >= p[j - 1] and p[j] >= p[j + 1]:
                out[a] = r[j]
                break
    return out


def radius_mask(rad, n):
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2
    rr = np.hypot(yy, xx) * PX
    th = np.mod(np.arctan2(xx, yy), 2 * np.pi)
    rf = E.fill_angles(ANG, rad)
    rf = np.interp(th, np.concatenate([ANG, [2 * np.pi]]), np.concatenate([rf, rf[:1]]))
    return rr <= rf


def to_u8(sec):
    lo, hi = np.percentile(sec, [1, 99])
    return (np.clip((sec - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------- methods
def m_ray(ctx):
    _, rad, _ = E.ray_contour(ctx["vol"], ctx["c"], ctx["u"], ctx["w"], N_ANG, ctx["mm"](R_MIN),
                              ctx["mm"](HALF), ctx["mm"](0.05), ctx["min_rise"])
    return rad * ctx["sp"]


def m_gac(ctx):
    sec = ctx["sec"]
    n = sec.shape[0]
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2
    init = np.hypot(yy, xx) <= 0.4 / PX
    s = (sec - sec.min()) / (np.ptp(sec) + 1e-6)
    m = morphological_geodesic_active_contour(inverse_gaussian_gradient(s, alpha=100, sigma=1.5), 25, init,
                                              smoothing=2, balloon=1, threshold="auto")
    return radii_from_mask(m.astype(bool))


def m_acwe(ctx):
    sec = ctx["sec"]
    n = sec.shape[0]
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2
    init = np.hypot(yy, xx) <= 0.6 / PX
    s = (sec - sec.min()) / (np.ptp(sec) + 1e-6)
    m = morphological_chan_vese(s, 40, init_level_set=init, smoothing=2, lambda1=1, lambda2=1)
    return radii_from_mask(m.astype(bool))


def m_canny(ctx):
    """Canny on the section (scikit-image implementation of the classic detector; hysteresis
    thresholds as quantiles of the gradient so it adapts to each section's contrast)."""
    from skimage.feature import canny
    e = canny(ctx["sec"].astype(float), sigma=1.5, low_threshold=0.5, high_threshold=0.8, use_quantiles=True)
    return radii_from_edgemap(e.astype(float), 0.5)


class DeepEdge:
    """Wraps a natural-image edge network: section -> 8-bit RGB, upsampled, -> edge probability."""

    def __init__(self, kind, ext):
        import torch
        self.torch = torch
        torch.set_grad_enabled(False)
        if kind == "hed":
            spec = importlib.util.spec_from_file_location("hed_run", Path(ext) / "pytorch-hed" / "run.py")
            mod = importlib.util.module_from_spec(spec)
            argv, sys.argv = sys.argv, [sys.argv[0]]
            try:
                spec.loader.exec_module(mod)
            finally:
                sys.argv = argv
            self.net = mod.Network().eval()
            self.prep = lambda x: x                                   # expects RGB in [0,1], BGR mean inside
        else:
            sys.path.insert(0, str(Path(ext) / "pidinet"))
            import models
            from models.convert_pidinet import convert_pidinet
            a = argparse.Namespace(config="carv4", sa=True, dil=True)
            net = models.pidinet(a)
            ck = torch.load(Path(ext) / "pidinet" / "trained_models" / "table5_pidinet.pth", map_location="cpu")
            sd = {k.replace("module.", ""): v for k, v in ck["state_dict"].items()}
            net.load_state_dict(sd)
            self.net = net.eval()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.prep = lambda x: (x - mean) / std
        self.kind = kind
        self.rel_thr = 0.4

    def __call__(self, ctx):
        """Run on a 12mm window (the nets need context and mark the image border as an edge),
        then cut out the central 6mm. Threshold is relative: these nets were trained on photos
        and give weaker absolute scores on CBCT."""
        import cv2
        mm = ctx["mm"]
        big, _ = E.sample_section(ctx["vol"], ctx["c"], ctx["u"], ctx["w"], mm(2 * HALF), mm(PX))
        u8 = to_u8(big)
        up = cv2.resize(u8, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        x = self.torch.from_numpy(np.repeat(up[None, None].astype(np.float32) / 255.0, 3, 1))
        y = self.net(self.prep(x))
        if isinstance(y, (list, tuple)):
            y = y[-1]
        e = cv2.resize(y[0, 0].numpy(), u8.shape[::-1], interpolation=cv2.INTER_AREA)
        n = ctx["sec"].shape[0]
        o = (e.shape[0] - n) // 2
        e = e[o:o + n, o:o + n]
        ctx.setdefault("edgemaps", {})[self.kind] = e
        return radii_from_edgemap(e, self.rel_thr * e.max())


def polar_profiles(ctx, r_max_mm=HALF, dr_mm=0.05):
    """Intensity along N_ANG rays from the centre, sampled in the (smoothed) volume."""
    mm = ctx["mm"]
    r = np.arange(0, r_max_mm + 1e-6, dr_mm)
    d = np.cos(ANG)[:, None, None] * ctx["u"] + np.sin(ANG)[:, None, None] * ctx["w"]
    xyz = ctx["c"] + d * mm(r)[None, :, None]
    p = ndimage.map_coordinates(ctx["vol"], xyz.reshape(-1, 3).T, order=1, mode="nearest").reshape(N_ANG, len(r))
    return r, p


def m_polardp(ctx, gcap=500.0, lam=0.3, max_jump=2):
    """Optimal closed contour in the polar image (dynamic programming / graph search, as used for
    vessel-wall and IVUS borders). Cost favours a strong dark->bright rise (capped, so the outer
    cortical plate does not win) and, mildly, a smaller radius (the inner edge). Neighbouring
    angles may differ by at most max_jump samples (0.1mm per 5 deg), which keeps the contour
    connected without making it round. A ray counts only if the image really rises there."""
    r, p = polar_profiles(ctx)
    dr = r[1] - r[0]
    cost = polar_cost(r, p, gcap, lam)
    A, R = cost.shape
    c3 = np.concatenate([cost, cost, cost], 0)                                   # unwrap the circle
    acc = c3[0].copy()
    back = np.zeros((3 * A, R), int)
    for a in range(1, 3 * A):
        best = np.full(R, np.inf); arg = np.zeros(R, int)
        for s in range(-max_jump, max_jump + 1):
            sh = np.roll(acc, s)
            if s > 0: sh[:s] = np.inf
            if s < 0: sh[s:] = np.inf
            better = sh < best
            best[better] = sh[better]; arg[better] = (np.arange(R) - s)[better]
        acc = best + c3[a]; back[a] = arg
    path = np.zeros(3 * A, int); path[-1] = int(np.argmin(acc))
    for a in range(3 * A - 1, 0, -1):
        path[a - 1] = back[a, path[a]]
    j = path[A:2 * A]
    rad = r[j].astype(float)
    rad[wall_rise(p, j, dr) < 0.5 * ctx["min_rise"]] = np.nan
    return rad


def polar_cost(r, p, gcap=500.0, lam=0.3):
    """polardp cost per (angle, radius): strong dark->bright rise is cheap, a larger radius mildly costly."""
    dr = r[1] - r[0]
    g = np.gradient(ndimage.gaussian_filter1d(p, 1.0, axis=-1), dr, axis=-1)    # intensity / mm
    cost = -np.clip(g, 0, gcap) / gcap + lam * r / r[-1]
    cost[..., r < R_MIN] = 10.0
    return cost


def wall_rise(p, j, dr):
    """Intensity 0.3mm outside minus 0.3mm inside the chosen radius index j (per angle)."""
    k = max(1, int(round(0.3 / dr)))
    R = p.shape[-1]
    a = np.arange(p.shape[0])
    return p[a, np.minimum(j + k, R - 1)] - p[a, np.maximum(j - k, 0)]


class PolarDP3D:
    """polardp on the whole canal at once: one closed surface over all sections of a side, found as
    a minimum closed set by max-flow (optimal surface graph search, Wu & Chen 2006 / Li et al. 2006,
    LOGISMOS). Same cost and same angle limit as polardp; in addition the wall at a given angle may
    move at most max_jump_sec samples between neighbouring sections (0.4mm apart). So a faint
    section takes its wall from the faint rise that best fits its clearer neighbours. Every section
    of the track is used (not only every --every), around the darkest-point centres smoothed along
    the track so that centre jitter does not read as a radius change. A ray still counts only if
    the image rises there by rise_frac * min_rise."""

    def __init__(self, gcap=500.0, lam=0.3, max_jump=2, max_jump_sec=4, rise_frac=0.5, centre_sigma=2.0):
        self.gcap, self.lam, self.max_jump, self.max_jump_sec = gcap, lam, max_jump, max_jump_sec
        self.rise_frac, self.centre_sigma = rise_frac, centre_sigma
        self.key = None

    def prepare(self, ctx):
        key = (ctx["cid"], ctx["side"])
        if self.key == key:
            return
        v, mm, (cl, tan) = ctx["vol"], ctx["mm"], ctx["track"]
        us, ws, cs = [], [], []
        prev_u = None
        for i in range(len(cl)):
            u, w = E.plane_basis(tan[i], prev_u); prev_u = u
            big, g = E.sample_section(v, cl[i], u, w, mm(ctx["search_mm"]), mm(PX))
            yy, xx = np.meshgrid(g, g, indexing="ij")
            big = np.where(np.hypot(yy, xx) <= mm(ctx["search_mm"]), ndimage.gaussian_filter(big, 2), np.inf)
            k = np.unravel_index(np.argmin(big), big.shape)
            us.append(u); ws.append(w); cs.append(np.array([g[k[0]], g[k[1]]]))
        off = ndimage.gaussian_filter1d(np.array(cs), self.centre_sigma, axis=0, mode="nearest")
        us, ws = np.array(us), np.array(ws)
        self.centre = cl + off[:, :1] * us + off[:, 1:] * ws
        self.u, self.w = us, ws
        r = np.arange(0, HALF + 1e-6, 0.05)
        d = np.cos(ANG)[None, :, None, None] * us[:, None, None] + np.sin(ANG)[None, :, None, None] * ws[:, None, None]
        xyz = self.centre[:, None, None] + d * mm(r)[None, None, :, None]
        p = ndimage.map_coordinates(v, xyz.reshape(-1, 3).T, order=1, mode="nearest").reshape(len(cl), N_ANG, len(r))
        j = self.surface(polar_cost(r, p, self.gcap, self.lam))
        rad = r[j].astype(float)
        for i in range(len(cl)):
            rad[i, wall_rise(p[i], j[i], r[1] - r[0]) < self.rise_frac * ctx["min_rise"]] = np.nan
        self.rad, self.key = rad, key

    def surface(self, cost):
        """argmin over surfaces j(i, a) of sum cost(i, a, j) with |j(i,a) - j(i,a+-1)| <= max_jump
        (angles wrap) and |j(i,a) - j(i+-1,a)| <= max_jump_sec, by s-t min cut."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import breadth_first_order, maximum_flow
        I, A, R = cost.shape
        S = 100                                                           # integer capacities
        c = np.round(cost * S).astype(np.int64)
        wgt = np.diff(c, axis=2, prepend=0)                               # node weight = cost step
        nid = np.arange(I * A * R).reshape(I, A, R)
        src, dst = I * A * R, I * A * R + 1
        INF = 2 ** 30
        e = [(nid[:, :, 1:], nid[:, :, :-1])]                             # column: j -> j-1
        jj = np.arange(R)
        for da in (1, -1):                                                # angle neighbours (wrap)
            e.append((nid, np.roll(nid, -da, axis=1)[:, :, np.maximum(jj - self.max_jump, 0)]))
        e.append((nid[1:], nid[:-1][:, :, np.maximum(jj - self.max_jump_sec, 0)]))   # section neighbours
        e.append((nid[:-1], nid[1:][:, :, np.maximum(jj - self.max_jump_sec, 0)]))
        rows = [a.ravel() for a, _ in e]; cols = [b.ravel() for _, b in e]
        caps = [np.full(a.size, INF, np.int64) for a, _ in e]
        wf = wgt.ravel()
        base = nid[:, :, 0].ravel()                                       # every column starts at j=0
        neg = np.flatnonzero(wf < 0); pos = np.flatnonzero(wf > 0)
        rows += [np.full(base.size, src), np.full(neg.size, src), pos]
        cols += [base, neg, np.full(pos.size, dst)]
        caps += [np.full(base.size, INF, np.int64), -wf[neg], wf[pos]]
        n = I * A * R + 2
        G = csr_matrix((np.concatenate(caps).astype(np.int32), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(n, n))
        G.sum_duplicates()
        flow = maximum_flow(G, src, dst, method="dinic").flow
        res = (G - flow).tocsr()
        res.data = (res.data > 0).astype(np.int8); res.eliminate_zeros()
        reach = np.zeros(n, bool); reach[breadth_first_order(res, src, directed=True, return_predecessors=False)] = True
        inside = reach[:I * A * R].reshape(I, A, R)
        return R - 1 - np.argmax(inside[:, :, ::-1], axis=2)              # top of the closed set

    def __call__(self, ctx):
        self.prepare(ctx)
        i, c = ctx["i"], ctx["c"]
        mm = ctx["mm"]
        rad = self.rad[i]
        rf = E.fill_angles(ANG, rad) if np.isfinite(rad).any() else None
        if rf is None:
            return np.full(N_ANG, np.nan)
        # wall polygon in the section plane, relative to the compare centre c (same plane, other centre)
        o = self.centre[i] - c
        oy, ox = o @ self.u[i] * ctx["sp"], o @ self.w[i] * ctx["sp"]
        py, px_ = oy + rf * np.cos(ANG), ox + rf * np.sin(ANG)
        from matplotlib.path import Path as MPath
        poly = MPath(np.c_[py, px_])
        rr = np.arange(0, HALF, 0.025)
        out = np.full(N_ANG, np.nan)
        for a, t in enumerate(ANG):
            pts = np.c_[rr * np.cos(t), rr * np.sin(t)]
            ins = poly.contains_points(pts)
            if not ins[0] or ins.all():
                continue
            k = int(np.argmin(ins))
            hit = pts[k]
            near = int(np.argmin(np.hypot(py - hit[0], px_ - hit[1])))
            if np.isfinite(rad[near]):
                out[a] = rr[k]
        return out


class Grad3D:
    """3D gradient edges with 3D non-maximum suppression (a Canny-style detector in 3D, with a
    sub-sample peak fit). Simplified stand-in for the ULPGC sub-voxel 3D detector (MATLAB only):
    the edge is judged along the 3D surface normal, so a wall that is oblique to the section is
    not penalised. Along each ray: the first outward (dark->bright) 3D edge above threshold."""

    def __init__(self):
        self.key = None

    def prepare(self, ctx):
        if self.key == id(ctx["vol"]):
            return
        v, sp = ctx["vol"], ctx["sp"]
        g = np.stack(np.gradient(v), 0) / sp                                     # intensity / mm
        mag = np.linalg.norm(g, axis=0)
        n = g / (mag + 1e-6)
        idx = np.indices(v.shape).astype(np.float32)
        fwd = ndimage.map_coordinates(mag, idx + n, order=1, mode="nearest")
        bwd = ndimage.map_coordinates(mag, idx - n, order=1, mode="nearest")
        self.g, self.edge = g, (mag >= fwd) & (mag >= bwd)
        self.key = id(ctx["vol"])

    def __call__(self, ctx, thr=None):
        self.prepare(ctx)
        mm = ctx["mm"]
        thr = ctx["min_rise"] / 0.6 if thr is None else thr                        # ~min_rise over 0.6mm
        r = np.arange(0, HALF + 1e-6, 0.05)
        out = np.full(N_ANG, np.nan)
        for a, t in enumerate(ANG):
            d = np.cos(t) * ctx["u"] + np.sin(t) * ctx["w"]
            xyz = (ctx["c"][:, None] + d[:, None] * mm(r))
            gd = sum(ndimage.map_coordinates(self.g[k], xyz, order=1, mode="nearest") * d[k] for k in range(3))
            ed = ndimage.map_coordinates(self.edge.astype(np.float32), xyz, order=1, mode="nearest") > 0.5
            for j in range(1, len(r) - 1):
                if r[j] < R_MIN or gd[j] < thr or not ed[j]:
                    continue
                if gd[j] >= gd[j - 1] and gd[j] >= gd[j + 1]:
                    den = gd[j - 1] - 2 * gd[j] + gd[j + 1]
                    off = 0.5 * (gd[j - 1] - gd[j + 1]) / den if den != 0 else 0.0
                    out[a] = r[j] + np.clip(off, -0.5, 0.5) * 0.05
                    break
        return out


class Vesselness:
    """Sato tubeness filter for dark tubes (scikit-image, 3D), the standard vessel enhancer.
    The canal region = voxels whose tubeness is at least half of the value at the centre."""

    def __init__(self):
        self.key = None

    def prepare(self, ctx):
        if self.key == id(ctx["vol"]):
            return
        from skimage.filters import sato
        mm = ctx["mm"]
        self.t = sato(ctx["vol"], sigmas=[mm(s) for s in (0.6, 0.9, 1.2, 1.6)], black_ridges=True)
        self.key = id(ctx["vol"])

    def __call__(self, ctx):
        self.prepare(ctx)
        mm = ctx["mm"]
        sec, _ = E.sample_section(self.t, ctx["c"], ctx["u"], ctx["w"], mm(HALF), mm(PX))
        n = sec.shape[0]
        yy, xx = np.mgrid[:n, :n] - (n - 1) / 2
        centre = sec[np.hypot(yy, xx) <= 0.5 / PX].max()
        if centre <= 0:
            return np.full(N_ANG, np.nan)
        return radii_from_mask(sec >= 0.5 * centre)


def m_otsu2d(ctx):
    """Otsu threshold on the section, then 2D region growing of the dark class from the centre.
    Hybrid idea of Li et al. 2026 (Int Dent J 76:111140) turned around for a dark lumen, without
    their hand-placed growing boundary: where the wall is missing the region leaks, and rays that
    leave the 3mm section come out nan."""
    from skimage.filters import threshold_otsu
    sec = ctx["sec"]
    return radii_from_mask(sec < threshold_otsu(sec))


class Otsu3D:
    """Same, but as in the paper: one Otsu threshold per case (voxels within 3mm of the doctor's
    clicks) and 3D region growing (connected dark component) from the section centre."""

    def __init__(self):
        self.key = None

    def prepare(self, ctx):
        if self.key == ctx["cid"]:
            return
        from skimage.filters import threshold_otsu
        near = ndimage.distance_transform_edt(~ctx["lab"], sampling=ctx["sp"]) <= HALF
        self.thr = float(threshold_otsu(ctx["vol"][near]))
        self.comp, _ = ndimage.label(ctx["vol"] < self.thr)
        self.key = ctx["cid"]

    def __call__(self, ctx):
        self.prepare(ctx)
        i = tuple(np.clip(np.round(ctx["c"]).astype(int), 0, np.array(self.comp.shape) - 1))
        if self.comp[i] == 0:
            return np.full(N_ANG, np.nan)
        mm = ctx["mm"]
        sec, _ = E.sample_section((self.comp == self.comp[i]).astype(np.float32), ctx["c"], ctx["u"], ctx["w"],
                                  mm(HALF), mm(PX))
        return radii_from_mask(sec > 0.5)


class NNInteractive:
    """Reads a precomputed nnInteractive mask (MIC-DKFZ/nnInteractive, doctor clicks as positive
    point prompts; run separately because it needs its own environment). Not an edge detector:
    a promptable segmentation network, so where the wall is faint its border reflects what it
    learned from other datasets' labels."""

    def __init__(self, mask_dir):
        self.dir, self.key, self.m = Path(mask_dir), None, None

    def __call__(self, ctx):
        if self.key != ctx["cid"]:
            f = self.dir / f"{ctx['cid']}.nii.gz"
            self.m = (np.asanyarray(nib.load(f).dataobj)[ctx["crop"]] > 0).astype(np.float32) if f.exists() else None
            self.key = ctx["cid"]
        if self.m is None:
            return np.full(N_ANG, np.nan)
        mm = ctx["mm"]
        sec, _ = E.sample_section(self.m, ctx["c"], ctx["u"], ctx["w"], mm(HALF), mm(PX))
        return radii_from_mask(sec > 0.5)


# ---------------------------------------------------------------- scoring
def contour_edge_strength(sec, rad):
    """Mean gradient magnitude on the contour / mean gradient in the section (1 = no better than random)."""
    g = ndimage.gaussian_gradient_magnitude(sec, 1.0)
    ok = np.isfinite(rad)
    if ok.sum() == 0:
        return np.nan
    c = (sec.shape[0] - 1) / 2
    y = c + rad[ok] * np.cos(ANG[ok]) / PX
    x = c + rad[ok] * np.sin(ANG[ok]) / PX
    return float(ndimage.map_coordinates(g, [y, x], order=1).mean() / (g.mean() + 1e-6))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", nargs="+", required=True)
    ap.add_argument("--label", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ext", required=True, help="pytorch-hed 與 pidinet 的 clone 所在資料夾")
    ap.add_argument("--methods", default="ray,gac,acwe,canny,hed,pidinet")
    ap.add_argument("--every", type=int, default=4, help="每幾張截面取一張")
    ap.add_argument("--search_mm", type=float, default=1.2)
    ap.add_argument("--min_rise", type=float, default=100)
    ap.add_argument("--min_valid_frac", type=float, default=0.75)
    ap.add_argument("--n_figs", type=int, default=6)
    ap.add_argument("--nni_dir", default="outputs/nninteractive", help="nnInteractive 預先算好的遮罩")
    ap.add_argument("--deep_thr", type=float, default=0.2, help="HED/PiDiNet 邊緣門檻(相對於截面最大值)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    names = args.methods.split(",")
    fns = {"ray": m_ray, "gac": m_gac, "acwe": m_acwe, "canny": m_canny, "polardp": m_polardp,
           "grad3d": Grad3D(), "sato": Vesselness(), "nninter": NNInteractive(args.nni_dir),
           "otsu2d": m_otsu2d, "otsu3d": Otsu3D(), "polar3d": PolarDP3D(),
           "polar3d_lo": PolarDP3D(rise_frac=0.25)}
    for k in ("hed", "pidinet"):
        if k in names:
            fns[k] = DeepEdge(k, args.ext)
            fns[k].rel_thr = args.deep_thr
    rows, timing = [], {k: 0.0 for k in names}
    fig_secs = []

    for ip, lp in zip(args.image, args.label):
        im = nib.load(ip)
        vol = np.asanyarray(im.dataobj).astype(np.float32)
        lab = np.asanyarray(nib.load(lp).dataobj) > 0
        sp = float(im.header.get_zooms()[0]); mm = lambda x: x / sp
        lo = np.maximum(np.argwhere(lab).min(0) - int(mm(8)), 0)
        hi = np.minimum(np.argwhere(lab).max(0) + int(mm(8)) + 1, vol.shape)
        crop = tuple(slice(a, b) for a, b in zip(lo, hi))
        v = ndimage.gaussian_filter(vol[crop], mm(0.2))
        cid = Path(ip).name.replace(".nii.gz", "")
        for side, pts in E.click_tracks(lab[crop]).items():
            cl, tan = E.resample_track(pts, mm(0.4), 1.2, sp)
            prev_u = None
            for i in range(len(cl)):
                u, w = E.plane_basis(tan[i], prev_u); prev_u = u
                if i % args.every:
                    continue
                c0 = cl[i]
                # neutral start: darkest point (lumen) within search_mm of the click
                big, g = E.sample_section(v, c0, u, w, mm(args.search_mm), mm(PX))
                yy, xx = np.meshgrid(g, g, indexing="ij")
                big = np.where(np.hypot(yy, xx) <= mm(args.search_mm), ndimage.gaussian_filter(big, 2), np.inf)
                k = np.unravel_index(np.argmin(big), big.shape)
                c = c0 + g[k[0]] * u + g[k[1]] * w
                sec, _ = E.sample_section(v, c, u, w, mm(HALF), mm(PX))
                ctx = dict(vol=v, c=c, u=u, w=w, sp=sp, mm=mm, sec=sec, min_rise=args.min_rise, cid=cid, crop=crop,
                           lab=lab[crop], side=int(side), i=i, track=(cl, tan), search_mm=args.search_mm)
                click = np.array([(c0 - c) @ u, (c0 - c) @ w]) * sp
                rec = {"case": cid, "side": int(side), "i": i}
                rads = {}
                for k_ in names:
                    t0 = time.perf_counter()
                    try:
                        rad = fns[k_](ctx)
                    except Exception as ex:  # one failing method must not stop the comparison
                        print(f"{k_} failed on {cid} {side} {i}: {ex}", flush=True)
                        rad = np.full(N_ANG, np.nan)
                    timing[k_] += time.perf_counter() - t0
                    rad = E.clean_rays(rad)
                    ok = np.isfinite(rad)
                    valid = ok.mean() >= args.min_valid_frac
                    rads[k_] = rad if valid else None
                    r = {"valid": bool(valid), "ray_frac": float(ok.mean())}
                    if valid:
                        rf = E.fill_angles(ANG, rad)
                        cr = np.hypot(*click)
                        ca = np.mod(np.arctan2(click[1], click[0]), 2 * np.pi)
                        r.update(r_mean=float(rf.mean()), r_min=float(rf.min()), r_max=float(rf.max()),
                                 click_inside=bool(cr <= np.interp(ca, np.r_[ANG, 2 * np.pi], np.r_[rf, rf[0]])),
                                 edge_strength=contour_edge_strength(sec, rad))
                    rec[k_] = r
                n = sec.shape[0]
                masks = {k_: radius_mask(r_, n) for k_, r_ in rads.items() if r_ is not None}
                rec["pair_dice"] = {f"{a}|{b}": float(E.dice(masks[a], masks[b]))
                                    for ai, a in enumerate(names) for b in names[ai + 1:] if a in masks and b in masks}
                rows.append(rec)
                fig_secs.append((cid, side, i, sec, rads, click))
        print(f"{cid}: {sum(r['case'] == cid for r in rows)} sections", flush=True)

    # ---------------- summary
    summ = {}
    for k_ in names:
        vs = [r[k_] for r in rows if r[k_]["valid"]]
        summ[k_] = {
            "valid_frac": len(vs) / len(rows),
            "r_mean_median_mm": float(np.median([x["r_mean"] for x in vs])) if vs else None,
            "r_mean_p5_p95_mm": [float(np.percentile([x["r_mean"] for x in vs], q)) for q in (5, 95)] if vs else None,
            "within_section_spread_mm": float(np.median([x["r_max"] - x["r_min"] for x in vs])) if vs else None,
            "click_inside_frac": float(np.mean([x["click_inside"] for x in vs])) if vs else None,
            "edge_strength_median": float(np.nanmedian([x["edge_strength"] for x in vs])) if vs else None,
            "ms_per_section": 1000 * timing[k_] / len(rows),
        }
    pair = {}
    for r in rows:
        for kk, d in r["pair_dice"].items():
            pair.setdefault(kk, []).append(d)
    pair = {kk: {"median": float(np.median(d)), "n": len(d)} for kk, d in pair.items()}
    (out / "compare_summary.json").write_text(json.dumps({"n_sections": len(rows), "methods": summ,
                                                          "pair_dice": pair}, indent=1), encoding="utf-8")
    (out / "compare_sections.json").write_text(json.dumps(rows), encoding="utf-8")

    # ---------------- figure: sections where at least half the methods found a contour
    cand = [f for f in fig_secs if sum(r is not None for r in f[4].values()) >= len(names) / 2]
    pick = [cand[k] for k in np.linspace(0, len(cand) - 1, min(args.n_figs, len(cand))).astype(int)] if cand else []
    if pick:
        fig, axs = plt.subplots(len(pick), len(names), figsize=(2.2 * len(names), 2.3 * len(pick)), squeeze=False)
        for ri, (cid, side, i, sec, rads, click) in enumerate(pick):
            for ci, k_ in enumerate(names):
                ax_ = axs[ri, ci]
                ax_.imshow(sec, cmap="gray", extent=[-HALF, HALF, HALF, -HALF])
                rad = rads[k_]
                if rad is not None:
                    rf = np.r_[E.fill_angles(ANG, rad), E.fill_angles(ANG, rad)[:1]]
                    t = np.r_[ANG, ANG[:1]]
                    ax_.plot(rf * np.sin(t), rf * np.cos(t), "r", lw=1)
                else:
                    ax_.text(0, -HALF + 0.5, "invalid", color="y", ha="center", fontsize=8)
                ax_.plot(click[1], click[0], "y+", ms=7)
                ax_.set_xticks([]); ax_.set_yticks([])
                if ri == 0:
                    ax_.set_title(k_, fontsize=10)
                if ci == 0:
                    ax_.set_ylabel(f"{cid[-4:]} s{side} #{i}", fontsize=8)
        fig.suptitle("same cross-sections, each method's wall (red); yellow + = doctor click; 6x6mm", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "compare_sections.png", dpi=110)
        plt.close(fig)
    print(json.dumps(summ, indent=1))
    print(json.dumps(pair, indent=1))


if __name__ == "__main__":
    main()
