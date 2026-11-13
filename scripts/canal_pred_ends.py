#!/usr/bin/env python3
"""Where does a segmentation model's predicted canal end, compared with the annotated canal ends?

To-do 2 (foramen), 2026-11-13, method A: no new training. The 508 model (5-fold CV, so each case
is predicted by a model that never saw it) already segments the whole canal. Its tip on each side
is taken as the foramen estimate and compared with the end of the original MCSTU annotation, the
same answer canal_foramen.py uses (label_end: centroid of the label voxels within tip_mm of the
farthest one, outward along the track).

Per side: prediction voxels = connected components of the prediction that touch this side's
label (dilated by --touch_mm). Tip = same label_end rule on those voxels, searched within
--near_mm of the answer. err = 3D distance; along = signed distance along the outward direction
(+ = the prediction goes past the annotated end, - = it stops short).

    CANAL_IMG="outputs/mcstu_edge/in/images/{}.nii.gz" ~/venvs/dentalseg/Scripts/python scripts/canal_pred_ends.py \
        --rays "outputs/mcstu_edge/sparse/chunk_*/*_rays.npz" --labels "outputs/mcstu_edge/in/labels/{}.nii.gz" \
        --pred "download508/Dataset508_MCSTUEdge_cv/{}.nii.gz" --skip --out outputs/pred_ends_508
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
from canal_foramen import label_end  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rays", required=True)
    ap.add_argument("--labels", required=True, help="original label path, {} = case id")
    ap.add_argument("--pred", required=True, help="model prediction path, {} = case id")
    ap.add_argument("--skip", nargs="*", default=load_cases())
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dir_mm", type=float, default=4.0)
    ap.add_argument("--touch_mm", type=float, default=1.0)
    ap.add_argument("--near_mm", type=float, default=20.0)
    ap.add_argument("--tip_mm", type=float, default=0.8)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    files = sorted(glob.glob(args.rays))
    for npz in files[:args.limit] if args.limit else files:
        cid, sp, sides = G.load_case(npz)
        if cid in args.skip:
            continue
        _, _, crop, _, _, sides = G.prep_volume(cid, sides, sp)
        lab = np.asanyarray(nib.load(args.labels.format(cid)).dataobj)[crop] > 0
        P = np.asanyarray(nib.load(args.pred.format(cid)).dataobj)[crop] == 1
        cc, _ = ndimage.label(lab)
        pc, _ = ndimage.label(P)
        nd = int(round(args.dir_mm / 0.4))
        it = max(1, int(round(args.touch_mm / sp)))
        for s, d in sides.items():
            n = len(d["c"])
            ids = {int(cc[tuple(np.clip(np.round(p).astype(int), 0, np.array(lab.shape) - 1))]) for p in d["click"]}
            side_lab = np.isin(cc, [i for i in ids if i > 0]) if ids - {0} else lab
            touch = ndimage.binary_dilation(side_lab, iterations=it)
            pids = np.unique(pc[touch & P]); pids = pids[pids > 0]
            side_pred = np.isin(pc, pids)
            ends = []
            for end, sg in ((0, -1), (n - 1, 1)):
                t_out = d["c"][end] - d["c"][end - sg * nd]; t_out /= np.linalg.norm(t_out)
                ends.append((t_out, label_end(side_lab, d["click"][end], t_out, sp)))
            upper = int(np.argmax([e[1][2] for e in ends]))
            for ei, (t_out, ans) in enumerate(ends):
                name = "mandibular" if ei == upper else "mental"
                row = {"case": cid, "side": int(s), "end": name, "found": bool(side_pred.any())}
                if side_pred.any():
                    tip = label_end(side_pred, ans, t_out, sp, near_mm=args.near_mm, tip_mm=args.tip_mm)
                    if np.allclose(tip, ans):                     # no prediction near this end
                        row["found"] = False
                    else:
                        row["err_mm"] = float(np.linalg.norm((tip - ans) * sp))
                        row["along_mm"] = float((tip - ans) @ t_out * sp)
                rows.append(row)
        print(cid[-4:], len(rows), flush=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=0))
    print("model tip vs annotated canal end (3D distance; along: + = past the end, - = short)")
    for name in ("mandibular", "mental"):
        x = [r for r in rows if r["end"] == name]
        f = [r for r in x if r["found"]]
        if not f:
            continue
        e = np.array([r["err_mm"] for r in f]); a = np.array([r["along_mm"] for r in f])
        print(f"  {name:10s} n={len(x):4d} found {len(f) / len(x):.0%}  err median {np.median(e):.2f}mm  "
              f"mean {e.mean():.2f}  <=2mm {np.mean(e <= 2):.0%}  <=4mm {np.mean(e <= 4):.0%}  "
              f"along median {np.median(a):+.2f}mm  p10/p90 {np.percentile(a, 10):+.1f}/{np.percentile(a, 90):+.1f}")


if __name__ == "__main__":
    main()
