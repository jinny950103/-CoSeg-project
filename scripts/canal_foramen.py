#!/usr/bin/env python3
"""Find where the canal leaves the bone (mandibular foramen / mental foramen) from the image.

Why (to-do 2, 2026-11-13): near both ends the sections often find no wall and are extrapolated.
If the foramen can be found in the image, (a) sections past it are not canal and can be dropped,
(b) the stretch between the last measured section and the foramen is short and bounded.

Method: start from a section inside the canal, follow the dark lumen outward in 0.4 mm steps
(direction from the last --dir_mm of the track, re-centred each step on the darkest point within
--recentre_mm in the section plane, direction updated with momentum). At each step measure the
enclosure: the share of 72 directions in which the image rises by >= --rise HU above the centre
within --r_max mm (bone around the lumen). Inside the canal it is ~0.9-1.0; where the canal opens
out of the bone it drops. Foramen = first step where the mean over --win steps is < --thr.

Exam (default): answer = the end of the original label on that side (MCSTU: the public dataset's
own annotation). The track is cut back by --cut_mm (0 = start at the label end) or started from
the last section with >= 25%% measured directions ('known'), and the detected point is compared
with the answer (3D distance). Ends are reported as mandibular (upper) / mental (lower).

    CANAL_IMG="outputs/mcstu_edge/in/images/{}.nii.gz" ~/venvs/dentalseg/Scripts/python scripts/canal_foramen.py \
        --rays "outputs/mcstu_edge/sparse/chunk_*/*_rays.npz" --labels "outputs/mcstu_edge/in/labels/{}.nii.gz" \
        --out outputs/foramen_mcstu
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
import canal_edge_gt as E  # noqa: E402
import canal_gap_fill as G  # noqa: E402


def label_end(lab_side, p_end, t_out, sp, near_mm=6.0, tip_mm=0.8):
    """Centroid of the label voxels at the very tip (within tip_mm of the farthest one along t_out)."""
    idx = np.argwhere(lab_side)
    idx = idx[np.linalg.norm((idx - p_end) * sp, axis=1) <= near_mm]
    if not len(idx):
        return p_end
    proj = (idx - p_end) @ t_out * sp
    return idx[proj >= proj.max() - tip_mm].mean(0)


def follow(v, vs, p0, d0, sp, args):
    """Follow the lumen from p0 along d0. Returns points (M,3) and enclosure (M,)."""
    pos, dr = p0.astype(float), d0 / np.linalg.norm(d0)
    rr = np.arange(-args.recentre_mm, args.recentre_mm + 1e-6, 0.2)
    pts, enc = [], []
    for _ in range(int(round(args.max_mm / 0.4))):
        p = pos + dr * 0.4 / sp
        u, w = E.plane_basis(dr)
        A, B = np.meshgrid(rr, rr)
        ok = np.hypot(A, B) <= args.recentre_mm
        cand = p + (A[ok][:, None] * u + B[ok][:, None] * w) / sp
        val = ndimage.map_coordinates(vs, cand.T, order=1, mode="nearest")
        q = cand[int(np.argmin(val))]
        step = q - pos
        if np.linalg.norm(step) > 0:
            dr = 0.8 * dr + 0.2 * step / np.linalg.norm(step); dr /= np.linalg.norm(dr)
        pos = q
        pts.append(pos.copy()); enc.append(G.enclosure(v, pos, u, w, sp, rise=args.rise, r_max=args.r_max))
    return np.array(pts), np.array(enc)


def exit_path(v, vs, p0, t_out, sp, args):
    """Cheapest way out of the bone from p0 (inside the canal), going forward (along t_out).
    Cost per voxel = 1 + exp((I - I_canal) / tau): through the dark canal it is cheap, through
    bone very expensive. 'Outside' = dark (I < I_canal - soft_drop) connected regions of at least
    --min_out_mm3 that touch the search box (soft tissue around the mandible, not marrow pockets).
    Returns the path (voxels, global crop coords) and the index where it first enters the outside."""
    from skimage.graph import MCP_Geometric
    R = int(np.ceil(args.box_mm / sp))
    c = np.round(p0).astype(int)
    lo_ = np.maximum(c - R, 0); hi_ = np.minimum(c + R + 1, v.shape)
    sub = vs[tuple(slice(a, b) for a, b in zip(lo_, hi_))]
    g = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lo_, hi_)], indexing="ij"), -1)
    fwd = ((g - p0) @ t_out) * sp >= -args.back_mm                    # only the half ahead of the start
    seed = tuple(c - lo_)
    rr = max(int(round(0.6 / sp)), 1)
    I_c = float(np.median(sub[tuple(slice(max(s - rr, 0), s + rr + 1) for s in seed)]))
    cost = 1 + np.exp(np.clip((sub - I_c) / args.tau, -20, 20))
    cost[~fwd] = np.inf
    if args.outside == "bone":                                       # outside the closed, filled bone
        from skimage.morphology import ball
        bone = sub > args.bone_thr
        r = max(int(round(args.close_mm / sp)), 1)
        hull = ndimage.binary_closing(np.pad(bone, r), ball(r))[r:-r, r:-r, r:-r]
        hull = ndimage.binary_fill_holes(hull)
        dark = ~hull & fwd
    else:
        dark = (sub < I_c - args.soft_drop) & fwd
    lab, nl = ndimage.label(dark)
    if nl == 0:
        return None, None
    sizes = ndimage.sum(np.ones_like(lab), lab, np.arange(1, nl + 1)) * sp ** 3
    border = set(np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                           lab[:, :, 0].ravel(), lab[:, :, -1].ravel()])))
    keep = [i + 1 for i in range(nl) if sizes[i] >= args.min_out_mm3 and (i + 1) in border]
    outside = np.isin(lab, keep)
    if not outside.any():
        return None, None
    mcp = MCP_Geometric(np.where(np.isfinite(cost), cost, 1e12))
    acc, _ = mcp.find_costs([seed])
    tgt = np.unravel_index(np.argmin(np.where(outside, acc, np.inf)), acc.shape)
    path = np.array(mcp.traceback(tgt))
    j = int(np.argmax(outside[tuple(path.T)]))
    return path + lo_, j


def detect(enc, args):
    m = np.convolve(enc, np.ones(args.win) / args.win, mode="full")[args.win - 1:len(enc) + args.win - 1]
    hit = np.where(m < args.thr)[0]
    return int(hit[0]) if len(hit) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rays", required=True)
    ap.add_argument("--labels", required=True, help="original label path, {} = case id")
    ap.add_argument("--skip", nargs="*", default=load_cases())
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cut_mm", type=float, nargs="+", default=[0, 5, 10])
    ap.add_argument("--dir_mm", type=float, default=4.0)
    ap.add_argument("--max_mm", type=float, default=25.0, help="最多往外追多遠(從起點算)")
    ap.add_argument("--recentre_mm", type=float, default=0.4)
    ap.add_argument("--rise", type=float, default=150)
    ap.add_argument("--r_max", type=float, default=2.5)
    ap.add_argument("--thr", type=float, default=0.55)
    ap.add_argument("--win", type=int, default=3)
    ap.add_argument("--method", choices=["path", "follow"], default="path",
                    help="path = 從管內走最便宜的路穿出骨頭; follow = 沿管腔追,骨頭包住比例掉下去")
    ap.add_argument("--box_mm", type=float, default=15.0, help="path: 起點周圍找多大範圍")
    ap.add_argument("--back_mm", type=float, default=1.0, help="path: 只往前找(容許往後這麼多)")
    ap.add_argument("--tau", type=float, default=100.0, help="path: 越亮越貴的尺度(HU)")
    ap.add_argument("--soft_drop", type=float, default=150.0, help="path: 比管內暗這麼多算軟組織")
    ap.add_argument("--min_out_mm3", type=float, default=50.0, help="path: 軟組織區最小體積")
    ap.add_argument("--outside", choices=["bone", "dark"], default="bone",
                    help="bone = 骨頭補實後的外面; dark = 比管內暗(會把管末段當外面)")
    ap.add_argument("--bone_thr", type=float, default=700.0, help="骨頭門檻(HU)")
    ap.add_argument("--close_mm", type=float, default=2.0, help="骨頭補實時封住多大的洞(管、孔)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    files = sorted(glob.glob(args.rays))
    files = files[:args.limit] if args.limit else files
    for npz in files:
        cid, sp, sides = G.load_case(npz)
        if cid in args.skip:
            continue
        _, _, crop, lo, v, sides = G.prep_volume(cid, sides, sp)
        vs = ndimage.gaussian_filter(v, 0.4 / sp)
        lab = np.asanyarray(nib.load(args.labels.format(cid)).dataobj)[crop] > 0
        cc, nc = ndimage.label(lab)
        nd = int(round(args.dir_mm / 0.4))
        for s, d in sides.items():
            n = len(d["c"]); cov = np.isfinite(d["r_agree"]).mean(1)
            # label voxels of this side: components touching this side's track
            ids = {int(cc[tuple(np.clip(np.round(p).astype(int), 0, np.array(lab.shape) - 1))]) for p in d["click"]}
            side_lab = np.isin(cc, [i for i in ids if i > 0]) if ids - {0} else lab
            ends = []
            for end, sg in ((0, -1), (n - 1, 1)):
                t_out = d["c"][end] - d["c"][end - sg * nd]; t_out /= np.linalg.norm(t_out)
                ends.append((end, sg, t_out, label_end(side_lab, d["click"][end], t_out, sp)))
            upper = int(np.argmax([e[3][2] for e in ends]))
            for ei, (end, sg, t_out, ans) in enumerate(ends):
                name = "mandibular" if ei == upper else "mental"
                known = np.where(cov >= 0.25)[0]
                last_known = (known.min() if sg < 0 else known.max()) if len(known) else None
                starts = [(f"cut{c:g}", end - sg * int(round(c / 0.4))) for c in args.cut_mm]
                if last_known is not None:
                    starts.append(("known", int(last_known)))
                for tag, k0 in starts:
                    if not 0 <= k0 - sg * nd < n or not 0 <= k0 < n:
                        continue
                    d0 = d["c"][k0] - d["c"][k0 - sg * nd]
                    if args.method == "path":
                        pth, jj = exit_path(v, vs, d["c"][k0], d0 / np.linalg.norm(d0), sp, args)
                        pts, j = (pth, jj) if pth is not None else (None, None)
                    else:
                        pts, enc = follow(v, vs, d["c"][k0], d0, sp, args)
                        j = detect(enc, args)
                    start_gap = float(np.linalg.norm((d["c"][k0] - ans) * sp))
                    if j is None:
                        err = along = None
                    else:
                        err = float(np.linalg.norm((pts[j] - ans) * sp))
                        along = float((pts[j] - ans) @ t_out * sp)       # + = past the label end
                    rows.append({"case": cid, "side": int(s), "end": name, "start": tag, "start_to_end_mm": start_gap,
                                 "found": j is not None, "err_mm": err, "along_mm": along})
        print(cid[-4:], len(rows), flush=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=0))
    for name in ("mandibular", "mental"):
        for tag in [f"cut{c:g}" for c in args.cut_mm] + ["known"]:
            x = [r for r in rows if r["end"] == name and r["start"] == tag]
            if not x:
                continue
            f = [r for r in x if r["found"]]
            e = np.array([r["err_mm"] for r in f]); a = np.array([r["along_mm"] for r in f])
            sg = np.median([r["start_to_end_mm"] for r in x])
            print(f"{name:10s} start {tag:6s} n={len(x):4d} (start {sg:4.1f}mm from end) found {len(f) / len(x):.0%}  "
                  f"err median {np.median(e):.2f}mm  <=2mm {np.mean(e <= 2):.0%}  <=4mm {np.mean(e <= 4):.0%}  "
                  f"along median {np.median(a):+.2f}mm (+ = past the label end)" if len(f) else f"{name} {tag} none found")


if __name__ == "__main__":
    main()
