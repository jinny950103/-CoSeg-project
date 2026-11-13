#!/usr/bin/env python3
"""Self-training labels (Dataset507): the sparse labels plus what 506 drew in the empty sections, filtered.

Like the ToothFairy 2023 winner (connectivity-based selective re-training), 506's cross-validated
prediction (each patient predicted by a fold that never saw them) is taken as a pseudo label only
where the sparse label has nothing (ignore), and only where it passes all of:
  - its 3D component touches the canal the strict labeller found (no isolated blobs);
  - in that section of the doctor track it is drawn within --near_mm of the track centre;
  - its equal-area diameter there is within --dmin..--dmax mm (literature: max 3.8mm), which drops
    the sections where 506 painted the marrow;
  - (--agree_dice > 0) the section lies in a gap of <= 16mm between clear sections and 506's shape
    there overlaps the Bayesian track through that gap (canal_track_hmm.py, same patient, works
    from the measured ends and the faint wall) with Dice >= --agree_dice: two unrelated estimates
    have to agree;
then canal = the model's canal there, background = a --wall_mm shell around it; everything else
stays ignore. The strict labels are never changed.

    python scripts/canal_pseudo507.py --out outputs/pseudo507
"""
import argparse
from case_lists import load_cases
import glob
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import canal_edge_gt as E  # noqa: E402
import canal_gap_fill as G  # noqa: E402
import canal_track_hmm as H  # noqa: E402
from canal_pred_review import section  # noqa: E402

IMG = "data/hospital_nifti_clean/images/{}.nii.gz"
DOC = "data/hospital_nifti_clean/labels/{}.nii.gz"
SPARSE = "outputs/sparse_label_v5_train34/chunk_*/{}_agree.nii.gz"
PRED = "outputs/pred506/Dataset506_CanalSparse_cv/HOSP_{}.nii.gz"


RAYS = "outputs/sparse_label_v6_rays/chunk_0*/{}_rays.npz"


def track_masks(cid, n_px, px):
    """{(side, k): (n_px, n_px) bool} track contour in the section planes of every empty section in a gap."""
    _, sp, sides = G.load_case(glob.glob(RAYS.format(cid))[0])
    _, _, _, lo, v, sides = G.prep_volume(cid, sides, sp)
    g = (np.arange(n_px) - n_px // 2) * px
    A, B = np.meshgrid(g, g, indexing="ij")
    out = {}
    for s, d in sides.items():
        R = d["r_agree"]; good = np.where(np.isfinite(R).mean(1) >= 0.5)[0]
        for a, b in zip(good[:-1], good[1:]):
            if not 1 <= b - a - 1 <= 40:
                continue
            cen, ra, rb, _ = H.ends(d, a + 1, b, sp)
            Ev = np.stack([H.evidence(v, cen[j], d["u"][k], d["w"][k], sp, 60) for j, k in enumerate(range(a + 1, b))])
            mean, _ = H.track(Ev, ra, rb, 0.08)
            for j, k in enumerate(range(a + 1, b)):
                oa = (cen[j] - d["click"][k]) @ d["u"][k]; ob = (cen[j] - d["click"][k]) @ d["w"][k]
                th = np.mod(np.arctan2(B - ob, A - oa), 2 * np.pi)
                rk = mean[j][np.round(th / (2 * np.pi) * len(H.ANG)).astype(int) % len(H.ANG)] / sp
                out[(s, k)] = np.hypot(A - oa, B - ob) <= rk
    return out


def build(cid, args):
    im = nib.load(glob.glob(SPARSE.format(cid))[0])
    base = np.asanyarray(im.dataobj).astype(np.uint8)
    sp = float(im.header.get_zooms()[0]); mm = lambda x: x / sp
    pred = np.asanyarray(nib.load(PRED.format(cid)).dataobj) > 0
    doc = np.asanyarray(nib.load(DOC.format(cid)).dataobj) > 0
    cc, _ = ndimage.label(pred)
    touch = np.unique(cc[(base == 1) & pred]); touch = touch[touch > 0]
    keep = np.isin(cc, touch)                                   # connected to the strict canal
    half, px = mm(4.0), mm(0.1)
    n = len(np.arange(-half, half + 1e-6, px)); mid = n // 2
    g = (np.arange(n) - mid) * 0.1
    near = np.hypot(*np.meshgrid(g, g, indexing="ij")) <= args.near_mm
    centres, ok_sec, stats = [], [], {"empty": 0, "drawn": 0, "size_ok": 0, "accepted": 0}
    tm = track_masks(cid, n, px) if args.agree_dice > 0 else {}
    for side, pts in E.click_tracks(doc).items():
        cl, tan = E.resample_track(pts, mm(0.4), 1.2, sp)
        prev_u = None
        for i in range(len(cl)):
            u, w = E.plane_basis(tan[i], prev_u); prev_u = u
            sl = section(base, cl[i], u, w, half, px, 0)
            empty = not (sl[near] == 1).any()
            acc = False
            if empty:
                stats["empty"] += 1
                sq = section(keep.astype(np.uint8), cl[i], u, w, half, px, 0) > 0
                if sq[near].any():
                    stats["drawn"] += 1
                    lab, _ = ndimage.label(sq)
                    comp = np.isin(lab, np.unique(lab[near & (lab > 0)]))
                    diam = 2 * np.sqrt(comp.sum() * 0.01 / np.pi)
                    acc = args.dmin <= diam <= args.dmax
                    stats["size_ok"] += int(acc)
                    if acc and args.agree_dice > 0:
                        t = tm.get((side, i))
                        acc = t is not None and 2 * (t & comp).sum() / (t.sum() + comp.sum()) >= args.agree_dice
                    stats["accepted"] += int(acc)
            centres.append(cl[i]); ok_sec.append(acc)
    centres, ok_sec = np.array(centres), np.array(ok_sec)
    # every voxel near the track belongs to its nearest section
    box = np.argwhere(ndimage.binary_dilation(doc, iterations=int(mm(args.track_mm))) | keep)
    dist, idx = cKDTree(centres).query(box)
    sel = box[(dist <= mm(args.track_mm)) & ok_sec[idx]]
    region = np.zeros(base.shape, bool); region[tuple(sel.T)] = True
    region &= base == 2
    canal = region & keep
    shell = region & ndimage.binary_dilation(canal, iterations=max(int(round(mm(args.wall_mm))), 1)) & ~pred
    out = base.copy(); out[canal] = 1; out[shell] = 0
    stats.update(case=cid, added_canal_vox=int(canal.sum()), added_bg_vox=int(shell.sum()),
                 base_canal_vox=int((base == 1).sum()))
    return out, im, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/pseudo507")
    ap.add_argument("--skip_pseudo", nargs="*", default=load_cases(), help="標註錯位,只複製原本稀疏標註")
    ap.add_argument("--near_mm", type=float, default=1.5)
    ap.add_argument("--dmin", type=float, default=1.0)
    ap.add_argument("--dmax", type=float, default=3.8)
    ap.add_argument("--track_mm", type=float, default=3.0)
    ap.add_argument("--wall_mm", type=float, default=0.4)
    ap.add_argument("--agree_dice", type=float, default=0.7, help="和貝氏追蹤的 Dice 下限(0=不檢查)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cases = sorted(Path(f).name[:10] for f in glob.glob(SPARSE.format("*")))
    allst = []
    for cid in cases:
        if cid in args.skip_pseudo:
            im = nib.load(glob.glob(SPARSE.format(cid))[0])
            nib.save(nib.Nifti1Image(np.asanyarray(im.dataobj).astype(np.uint8), im.affine, im.header), out / f"HOSP_{cid}.nii.gz")
            continue
        lab, im, st = build(cid, args)
        nib.save(nib.Nifti1Image(lab, im.affine, im.header), out / f"HOSP_{cid}.nii.gz")
        allst.append(st)
        print(f"{cid[-4:]} empty {st['empty']} drawn {st['drawn']} size ok {st['size_ok']} accepted {st['accepted']}  "
              f"+canal {st['added_canal_vox'] / max(st['base_canal_vox'], 1):.2f}x of strict", flush=True)
    (out / "stats.json").write_text(json.dumps(allst, indent=1))
    e = sum(s["empty"] for s in allst); a = sum(s["accepted"] for s in allst); dr = sum(s["drawn"] for s in allst)
    print(f"total empty {e}  drawn {dr} ({dr / e:.2f})  accepted {a} ({a / e:.2f})")


if __name__ == "__main__":
    main()
