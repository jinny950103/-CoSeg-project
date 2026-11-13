#!/usr/bin/env python3
"""Run the public DentalSegmentator nnU-Net model (Zenodo 10829675, CC-BY-4.0) and keep only the
mandibular canal (its label 5) as a 0/1 mask.

Used only as an independent "rough location" check for the edge-detection labels: the model was
trained on smooth public-style annotations, so its boundary is not taken as ground truth.

The nnU-Net CLI fails on Windows (worker processes cannot see the GPU), so this calls the
sequential predictor in one process.

    <venv>/python scripts/run_dentalsegmentator.py --model <.../nnUNetTrainer__nnUNetPlans__3d_fullres>
        --image a.nii.gz b.nii.gz ... --out outputs/dentalseg
"""
import argparse
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

CANAL = 5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tta", action="store_true", help="test-time augmentation(慢約 8 倍)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=args.tta,
                           device=torch.device("cuda"), verbose=False, allow_tqdm=False)
    pred.initialize_from_trained_model_folder(args.model, use_folds=(0,), checkpoint_name="checkpoint_final.pth")
    for ip in args.image:
        cid = Path(ip).name.replace(".nii.gz", "")
        dst = out / f"{cid}_dentalseg_canal.nii.gz"
        if dst.exists():
            continue
        t = time.time()
        seg = pred.predict_from_files_sequential([[ip]], None, save_probabilities=False, overwrite=True)[0]
        # nnU-Net returns the array in the input's own axis order
        im = nib.load(ip)
        canal = (np.asarray(seg) == CANAL).astype(np.uint8)
        if canal.shape != im.shape:
            canal = canal.transpose(2, 1, 0)
        assert canal.shape == im.shape, (canal.shape, im.shape)
        nib.save(nib.Nifti1Image(canal, im.affine), dst)
        print(f"{cid} canal_vox={int(canal.sum())} {time.time() - t:.0f}s "
              f"gpu_peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)


if __name__ == "__main__":
    main()
