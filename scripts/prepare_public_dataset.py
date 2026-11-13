#!/usr/bin/env python3
"""Turn a public dataset with full canal labels into the input this pipeline expects.

The pipeline was built on hospital CBCT where a doctor clicked the canal once per slice along
volume axis 1 (front-back). Those clicks only say roughly where the canal is; the wall itself is
measured from the image. A full public label is used the same way: per canal (3D connected
component) and per slice along axis 1, the label centroid becomes one "click", written as a ball
so that canal_edge_gt.click_tracks recovers the same centre line. The label shape (smooth, even
width) is not used for anything else.

Volumes whose axis 1 is not front-back (A/P) are reoriented to the closest canonical (RAS)
orientation; image and label are always written with the same affine.

Output:
  <out>/images/<cid>.nii.gz   image (unchanged values)
  <out>/labels/<cid>.nii.gz   click-style label (uint8, 0/1)

    python scripts/prepare_public_dataset.py --images "D/imagesTr/*_0000.nii.gz" \
        --labels D/labelsTr --out data/public_clicks
    then: canal_sparse_label.py --image data/public_clicks/images/*.nii.gz \
                                --label data/public_clicks/labels/*.nii.gz ...
"""
import argparse
import glob
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

BALL_R2 = 16  # voxels^2; must exceed the r2=8 erosion in canal_edge_gt.click_tracks


def case_id(path):
    name = Path(path).name
    name = re.sub(r"\.nii(\.gz)?$", "", name)
    return re.sub(r"_0000$", "", name)


def to_ap_axis1(img):
    if nib.aff2axcodes(img.affine)[1] in ("A", "P"):
        return img
    return nib.as_closest_canonical(img)


def clicks_from_full(lab, min_vox=50):
    """Per connected component, the centroid of each slice along axis 1."""
    comp, n = ndimage.label(lab > 0, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(comp.ravel())[1:]
    pts = []
    for k in np.argsort(sizes)[::-1][:2]:          # left and right canal
        if sizes[k] < min_vox:
            continue
        idx = np.argwhere(comp == k + 1)
        for j in np.unique(idx[:, 1]):
            pts.append(idx[idx[:, 1] == j].mean(0))
    return np.array(pts), n


def draw_balls(shape, pts, r2=BALL_R2):
    out = np.zeros(shape, np.uint8)
    rv = int(np.ceil(np.sqrt(r2)))
    zz, yy, xx = np.mgrid[-rv:rv + 1, -rv:rv + 1, -rv:rv + 1]
    ball = zz ** 2 + yy ** 2 + xx ** 2 <= r2
    off = np.argwhere(ball) - rv
    for p in np.round(pts).astype(int):
        q = off + p
        q = q[((q >= 0) & (q < np.array(shape))).all(1)]
        out[q[:, 0], q[:, 1], q[:, 2]] = 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="glob of image files")
    ap.add_argument("--labels", required=True, help="folder of full labels, named <cid>.nii.gz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(args.images))
    if not files:
        raise SystemExit(f"no images match {args.images}")
    for ip in files:
        cid = case_id(ip)
        lp = Path(args.labels) / f"{cid}.nii.gz"
        if not lp.exists():
            print(f"{cid}: no label {lp}, skipped")
            continue
        im, lb = to_ap_axis1(nib.load(ip)), to_ap_axis1(nib.load(lp))
        if im.shape != lb.shape or not np.allclose(im.affine, lb.affine, atol=1e-3):
            print(f"{cid}: image/label geometry differs, skipped")
            continue
        lab = np.asanyarray(lb.dataobj)
        pts, ncomp = clicks_from_full(lab)
        if len(pts) == 0:
            print(f"{cid}: empty label, skipped")
            continue
        clicks = draw_balls(lab.shape, pts)
        nib.save(nib.Nifti1Image(np.asanyarray(im.dataobj), im.affine, im.header), out / "images" / f"{cid}.nii.gz")
        h = lb.header.copy(); h.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(clicks, lb.affine, h), out / "labels" / f"{cid}.nii.gz")
        sp = im.header.get_zooms()[:3]
        print(f"{cid}: {nib.aff2axcodes(im.affine)} spacing {tuple(round(float(s), 2) for s in sp)} "
              f"components {ncomp} clicks {len(pts)}")


if __name__ == "__main__":
    main()
