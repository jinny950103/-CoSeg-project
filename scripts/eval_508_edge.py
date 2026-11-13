#!/usr/bin/env python3
"""Score predictions on the 18 MCSTU independent-test cases against their edge-detection labels.

The test labels come from the same pipeline as the Dataset508 training labels
(outputs/mcstu_edge/test_extrap/<cid>_extrap.nii.gz, values 0..6):
    0 measured bg, 1 measured canal, 2 ignore, 3 predicted canal, 4 predicted bg shell,
    5 extrapolated canal, 6 extrapolated bg shell.
Ignore voxels (2) are never scored. Three numbers per case:
    dice_all       all canal (1,3,5) vs all background (0,4,6)   <- main score, same mapping as training
    dice_measured  only voxels the edge detector actually measured (0,1); 3..6 also ignored
    recall_pred    fraction of predicted canal (3) voxels the model marks as canal
    recall_extrap  same for extrapolated canal (5)

    python scripts/eval_508_edge.py --pred download508/Dataset508_MCSTUEdge_ens5
    # reference: how well the ORIGINAL MCSTU labels agree with the edge labels
    python scripts/eval_508_edge.py --pred data/nnUNet_raw/Dataset504_MixedCanalClean/labelsTs_eval
"""
import argparse
import glob
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def dice(p, g):
    s = p.sum() + g.sum()
    return float(2 * (p & g).sum() / s) if s else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="folder with <cid>.nii.gz predictions (0/1)")
    ap.add_argument("--gt", default="outputs/mcstu_edge/test_extrap")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for gp in sorted(glob.glob(f"{args.gt}/MCSTU_TEST_*_extrap.nii.gz")):
        cid = Path(gp).name[: -len("_extrap.nii.gz")]
        pp = Path(args.pred) / f"{cid}.nii.gz"
        if not pp.exists():
            print(f"  {cid}: no prediction, skipped")
            continue
        g = np.asanyarray(nib.load(gp).dataobj).astype(np.uint8)
        p = np.asanyarray(nib.load(pp).dataobj) > 0
        assert p.shape == g.shape, f"{cid}: shape {p.shape} vs {g.shape}"
        canal = np.isin(g, (1, 3, 5))
        scored = g != 2
        meas = np.isin(g, (0, 1))
        r = {"case": cid,
             "dice_all": dice(p & scored, canal),
             "dice_measured": dice(p & meas, g == 1),
             "recall_pred": float(p[g == 3].mean()) if (g == 3).any() else float("nan"),
             "recall_extrap": float(p[g == 5].mean()) if (g == 5).any() else float("nan")}
        rows.append(r)
        print(f"  {cid:20s} all {r['dice_all']:.3f}  measured {r['dice_measured']:.3f}  "
              f"pred-recall {r['recall_pred']:.3f}  extrap-recall {r['recall_extrap']:.3f}")

    keys = ["dice_all", "dice_measured", "recall_pred", "recall_extrap"]
    summ = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    summ["n"] = len(rows)
    print("\nmean (n=%d): " % len(rows) + "  ".join(f"{k} {summ[k]:.4f}" for k in keys))
    out = Path(args.out) if args.out else Path(args.pred) / "eval_508_edge.json"
    json.dump({"summary": summ, "cases": rows}, open(out, "w"), indent=2)
    print(f"written {out}")


if __name__ == "__main__":
    main()
