#!/usr/bin/env python3
"""Convert MCSTU NIfTI dataset to nnU-Net v2 raw dataset format (Dataset501_MandibularCanal).

Source layout (as downloaded from Kaggle/Zenodo):
    development_dataset/img_nii/<uid>.nii.gz
    development_dataset/labels/<uid>.nii.gz
    independent_test_dataset/VOL_<n>/image.nii.gz
    independent_test_dataset/VOL_<n>/label.nii.gz

Output layout (nnU-Net v2 raw dataset):
    <out>/Dataset501_MandibularCanal/imagesTr/MCSTU_###_0000.nii.gz
    <out>/Dataset501_MandibularCanal/labelsTr/MCSTU_###.nii.gz
    <out>/Dataset501_MandibularCanal/imagesTs/MCSTU_TEST_VOL_<n>_0000.nii.gz
    <out>/Dataset501_MandibularCanal/labelsTs_eval/MCSTU_TEST_VOL_<n>.nii.gz   (kept for evaluation; not an nnU-Net convention dir)
    <out>/Dataset501_MandibularCanal/dataset.json
    <out>/Dataset501_MandibularCanal/case_id_mapping.json
"""
import argparse
import json
import random
import shutil
from pathlib import Path

DATASET_DIRNAME = "Dataset501_MandibularCanal"


def convert_development(src_root, out_dataset_dir, limit=None, seed=42):
    img_dir = src_root / "development_dataset" / "img_nii"
    lbl_dir = src_root / "development_dataset" / "labels"
    uids = sorted(p.name for p in img_dir.glob("*.nii.gz"))

    if limit is not None and limit < len(uids):
        rng = random.Random(seed)
        uids = sorted(rng.sample(uids, limit))

    imagesTr = out_dataset_dir / "imagesTr"
    labelsTr = out_dataset_dir / "labelsTr"
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)

    mapping = {}
    for i, uid in enumerate(uids, start=1):
        case_id = f"MCSTU_{i:03d}"
        mapping[case_id] = uid

        src_img = img_dir / uid
        src_lbl = lbl_dir / uid
        if not src_lbl.exists():
            raise FileNotFoundError(f"missing label for {uid}")

        shutil.copy2(src_img, imagesTr / f"{case_id}_0000.nii.gz")
        shutil.copy2(src_lbl, labelsTr / f"{case_id}.nii.gz")

    return mapping


def convert_independent_test(src_root, out_dataset_dir):
    test_root = src_root / "independent_test_dataset"
    imagesTs = out_dataset_dir / "imagesTs"
    labelsTs_eval = out_dataset_dir / "labelsTs_eval"
    imagesTs.mkdir(parents=True, exist_ok=True)
    labelsTs_eval.mkdir(parents=True, exist_ok=True)

    mapping = {}
    for vol_dir in sorted(test_root.iterdir()):
        if not vol_dir.is_dir():
            continue
        src_img = vol_dir / "image.nii.gz"
        src_lbl = vol_dir / "label.nii.gz"
        if not (src_img.exists() and src_lbl.exists()):
            raise FileNotFoundError(f"missing image/label in {vol_dir}")

        case_id = f"MCSTU_TEST_{vol_dir.name}"
        mapping[case_id] = vol_dir.name

        shutil.copy2(src_img, imagesTs / f"{case_id}_0000.nii.gz")
        shutil.copy2(src_lbl, labelsTs_eval / f"{case_id}.nii.gz")

    return mapping


def write_dataset_json(out_dataset_dir, num_training):
    dataset_json = {
        "channel_names": {"0": "CBCT"},
        "labels": {
            "background": 0,
            "mandibular_canal": 1,
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz",
    }
    with open(out_dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=r"C:\Users\WIN11\mcstu_data\MCSTU",
                         help="MCSTU dataset root (contains development_dataset/ and independent_test_dataset/)")
    parser.add_argument("--out", default="./data/nnUNet_raw",
                         help="output nnUNet_raw directory")
    parser.add_argument("--pilot", type=int, default=None,
                         help="limit number of training cases (for a quick pilot run)")
    parser.add_argument("--include-test", action="store_true",
                         help="also convert independent_test_dataset into imagesTs/labelsTs_eval")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src_root = Path(args.src)
    out_dataset_dir = Path(args.out) / DATASET_DIRNAME
    out_dataset_dir.mkdir(parents=True, exist_ok=True)

    train_mapping = convert_development(src_root, out_dataset_dir, limit=args.pilot, seed=args.seed)
    write_dataset_json(out_dataset_dir, len(train_mapping))

    case_mapping = {"training": train_mapping}

    if args.include_test:
        test_mapping = convert_independent_test(src_root, out_dataset_dir)
        case_mapping["test"] = test_mapping

    with open(out_dataset_dir / "case_id_mapping.json", "w") as f:
        json.dump(case_mapping, f, indent=2, ensure_ascii=False)

    print(f"converted {len(train_mapping)} training cases -> {out_dataset_dir / 'imagesTr'}")
    if args.include_test:
        print(f"converted {len(case_mapping['test'])} test cases -> {out_dataset_dir / 'imagesTs'}")
    print(f"dataset.json written -> {out_dataset_dir / 'dataset.json'}")


if __name__ == "__main__":
    main()
