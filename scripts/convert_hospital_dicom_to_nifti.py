#!/usr/bin/env python3
"""Convert hospital CBCT DICOM series to NIfTI, paired with the sphere_v2 NIfTI labels.

Source layout:
    <src>/<PID>/<PID>/no_label/<PID>_nolabel_###.dcm    original image slices
    <src>/<PID>/<PID>/label/<PID>_label_###.dcm         (unused -- NIfTI labels are preferred)
    <labels>/<PID>/<PID>_sphere_v2.nii.gz               label volume

Output layout (split-agnostic staging, mirrors MCSTU development_dataset):
    <out>/images/<PID>.nii.gz
    <out>/labels/<PID>.nii.gz
    <out>/conversion_report.json

Geometry notes
--------------
pydicom's ``pixel_array`` is indexed (Rows, Columns) == (y, x), while NIfTI wants
(x, y, z), hence the ``transpose(1, 0, 2)`` when stacking. The affine is built from
ImageOrientationPatient / ImagePositionPatient in DICOM's LPS frame and converted to
NIfTI's RAS frame by negating the first two rows. Each converted image is checked
against its label's shape and affine so a silent misalignment cannot slip through.
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom


def read_series(series_dir):
    """Read a DICOM series into (volume_xyz, affine_ras). Volume is HU as int16."""
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"no .dcm files in {series_dir}")

    headers = []
    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        headers.append((f, ds))

    # Sort along the slice axis using the patient-space z coordinate, falling back to
    # InstanceNumber when ImagePositionPatient is absent.
    def sort_key(item):
        _, ds = item
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None:
            return float(ipp[2])
        return float(getattr(ds, "InstanceNumber", 0))

    headers.sort(key=sort_key)

    ref = headers[0][1]
    rows, cols = int(ref.Rows), int(ref.Columns)
    row_spacing, col_spacing = (float(v) for v in ref.PixelSpacing)

    iop = np.asarray([float(v) for v in ref.ImageOrientationPatient], dtype=float)
    row_dir, col_dir = iop[0:3], iop[3:6]  # row_dir advances with the column index

    origin = np.asarray([float(v) for v in ref.ImagePositionPatient], dtype=float)

    # Slice direction and spacing straight from the endpoint positions: this stays
    # correct for gantry-tilted or non-contiguous series, unlike SliceThickness.
    n = len(headers)
    last_ipp = np.asarray([float(v) for v in headers[-1][1].ImagePositionPatient], dtype=float)
    if n > 1:
        step = (last_ipp - origin) / (n - 1)
        slice_spacing = float(np.linalg.norm(step))
        slice_dir = step / slice_spacing if slice_spacing > 0 else np.cross(row_dir, col_dir)
    else:
        slice_spacing = float(getattr(ref, "SliceThickness", 1.0))
        slice_dir = np.cross(row_dir, col_dir)

    slope = float(getattr(ref, "RescaleSlope", 1.0))
    intercept = float(getattr(ref, "RescaleIntercept", 0.0))

    # Stack as (rows, cols, z) == (y, x, z), then reorder to (x, y, z).
    vol = np.empty((rows, cols, n), dtype=np.float32)
    for k, (f, _) in enumerate(headers):
        ds = pydicom.dcmread(f)
        vol[:, :, k] = ds.pixel_array.astype(np.float32)
    vol = vol * slope + intercept
    vol = np.transpose(vol, (1, 0, 2))
    vol = np.rint(vol).astype(np.int16)

    # LPS affine: column index -> row_dir * col_spacing, row index -> col_dir * row_spacing.
    affine_lps = np.eye(4)
    affine_lps[:3, 0] = row_dir * col_spacing
    affine_lps[:3, 1] = col_dir * row_spacing
    affine_lps[:3, 2] = slice_dir * slice_spacing
    affine_lps[:3, 3] = origin

    # LPS -> RAS
    affine_ras = affine_lps.copy()
    affine_ras[0, :] *= -1
    affine_ras[1, :] *= -1

    meta = {
        "n_slices": n,
        "shape": [int(x) for x in vol.shape],
        "spacing": [col_spacing, row_spacing, slice_spacing],
        "rescale_slope": slope,
        "rescale_intercept": intercept,
        "hu_min": int(vol.min()),
        "hu_max": int(vol.max()),
    }
    return vol, affine_ras, meta


def convert_patient(pid, src_root, label_root, out_root, strict=True):
    series_dir = src_root / pid / pid / "no_label"
    label_path = label_root / pid / f"{pid}_sphere_v2.nii.gz"

    if not series_dir.is_dir():
        return {"patient": pid, "status": "skipped", "reason": f"no image series at {series_dir}"}
    if not label_path.exists():
        return {"patient": pid, "status": "skipped", "reason": f"no v2 label at {label_path}"}

    vol, affine, meta = read_series(series_dir)
    lbl = nib.load(label_path)

    # The label was authored on the image grid; verify rather than assume.
    shape_ok = tuple(lbl.shape) == tuple(vol.shape)
    affine_ok = bool(np.allclose(lbl.affine, affine, atol=1e-3))
    if strict and not shape_ok:
        return {"patient": pid, "status": "failed", "reason": "shape mismatch",
                "image_shape": meta["shape"], "label_shape": [int(x) for x in lbl.shape]}

    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)

    img_nii = nib.Nifti1Image(vol, affine)
    img_nii.header.set_zooms(tuple(float(s) for s in meta["spacing"]))
    nib.save(img_nii, out_root / "images" / f"{pid}.nii.gz")

    lbl_data = np.asanyarray(lbl.dataobj).astype(np.uint8)
    lbl_out = nib.Nifti1Image(lbl_data, affine)
    lbl_out.header.set_zooms(tuple(float(s) for s in meta["spacing"]))
    nib.save(lbl_out, out_root / "labels" / f"{pid}.nii.gz")

    return {
        "patient": pid,
        "status": "ok",
        "shape": meta["shape"],
        "spacing": [round(s, 4) for s in meta["spacing"]],
        "n_slices": meta["n_slices"],
        "hu_range": [meta["hu_min"], meta["hu_max"]],
        "label_fg_voxels": int((lbl_data > 0).sum()),
        "label_shape_matched": shape_ok,
        "label_affine_matched": affine_ok,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default=r"D:\0911\hospital_dataset",
                        help="root containing <PID>/<PID>/no_label/*.dcm")
    parser.add_argument("--labels",
                        default=r"C:\Users\WIN11\Downloads\0923\sphere_output_v2\sphere_output_v2",
                        help="root containing <PID>/<PID>_sphere_v2.nii.gz")
    parser.add_argument("--out", default="./data/hospital_nifti")
    parser.add_argument("--patients", nargs="*", default=None,
                        help="patient IDs to convert (default: every ID found under --src)")
    parser.add_argument("--allow-shape-mismatch", action="store_true",
                        help="write output even when label and image grids disagree")
    args = parser.parse_args()

    src_root = Path(args.src)
    label_root = Path(args.labels)
    out_root = Path(args.out)

    pids = args.patients or sorted(p.name for p in src_root.iterdir() if p.is_dir())

    report = []
    for pid in pids:
        try:
            r = convert_patient(pid, src_root, label_root, out_root,
                                strict=not args.allow_shape_mismatch)
        except Exception as e:  # keep going so one bad series cannot stall the batch
            r = {"patient": pid, "status": "error", "reason": f"{type(e).__name__}: {e}"}
        report.append(r)
        print(f"[{r['status']:7s}] {pid}" + ("" if r["status"] == "ok" else f"  <- {r.get('reason','')}"))

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "conversion_report.json", "w") as f:
        json.dump(report, f, indent=2)

    counts = {}
    for r in report:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\nsummary:", counts)
    print("report ->", out_root / "conversion_report.json")


if __name__ == "__main__":
    main()
