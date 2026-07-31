"""
compare_dental_segmentator.py
=============================
用 DentalSegmentator (nnU-Net) 預測下顎神經管，
然後與你的 GT label 和 CoSeg 預測做 3D Dice 比較。

使用方式：
    python compare_dental_segmentator.py \
        --dental_pred_dir /path/to/dental_segmentator_output \
        --gt_dir /path/to/MCSTU/independent_test_dataset \
        --coseg_pred_dir /path/to/coseg_predictions \
        --gt_canal_label 1 \
        --output_csv results_comparison.csv

如果只跑 DentalSegmentator vs GT（還沒有 CoSeg 預測），省略 --coseg_pred_dir 即可。
"""

import os
import argparse
import numpy as np
import nibabel as nib
from glob import glob
import csv


def load_nifti(path):
    """載入 NIfTI 檔案，回傳 numpy array 和 affine"""
    nii = nib.load(path)
    return nii.get_fdata(), nii.affine, nii.header


def compute_3d_dice(pred, gt, smooth=1e-5):
    pred = (pred > 0).astype(np.float32)
    gt = (gt > 0).astype(np.float32)
    intersection = np.sum(pred * gt)
    return (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)


def compute_3d_iou(pred, gt, smooth=1e-5):
    pred = (pred > 0).astype(np.float32)
    gt = (gt > 0).astype(np.float32)
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    return (intersection + smooth) / (union + smooth)


def compute_hausdorff_95(pred, gt):
    """計算 95th percentile Hausdorff Distance（如果有 scipy）"""
    try:
        from scipy.ndimage import distance_transform_edt
        pred_bin = (pred > 0).astype(np.uint8)
        gt_bin = (gt > 0).astype(np.uint8)

        if pred_bin.sum() == 0 or gt_bin.sum() == 0:
            return float('inf')

        # 計算表面點
        from scipy.ndimage import binary_erosion
        pred_surface = pred_bin ^ binary_erosion(pred_bin)
        gt_surface = gt_bin ^ binary_erosion(gt_bin)

        # 距離場
        dt_pred = distance_transform_edt(~pred_bin)
        dt_gt = distance_transform_edt(~gt_bin)

        # 表面到表面的距離
        dist_pred_to_gt = dt_gt[pred_surface > 0]
        dist_gt_to_pred = dt_pred[gt_surface > 0]

        all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
        return np.percentile(all_distances, 95)
    except Exception:
        return float('nan')


def main():
    parser = argparse.ArgumentParser(description="比較 DentalSegmentator vs CoSeg vs GT")
    parser.add_argument("--dental_pred_dir", required=True,
                        help="DentalSegmentator 的預測輸出資料夾 (含 .nii.gz)")
    parser.add_argument("--gt_dir", required=True,
                        help="GT 資料夾 (如 MCSTU/independent_test_dataset)，"
                             "每個子資料夾含 label.nii.gz")
    parser.add_argument("--coseg_pred_dir", default=None,
                        help="(選填) CoSeg 的 3D 預測資料夾，每個子資料夾含預測 .nii.gz")
    parser.add_argument("--gt_canal_label", type=int, default=1,
                        help="GT label.nii.gz 中代表下顎神經管的 label 值 (預設=1)")
    parser.add_argument("--dental_canal_label", type=int, default=5,
                        help="DentalSegmentator 輸出中 mandibular canal 的 label 值 (預設=5)")
    parser.add_argument("--output_csv", default="comparison_results.csv",
                        help="結果輸出 CSV 路徑")
    args = parser.parse_args()

    # ── 找到所有病人 ──
    vol_dirs = sorted(glob(os.path.join(args.gt_dir, "VOL_*")))
    if not vol_dirs:
        # 嘗試不同命名
        vol_dirs = sorted(glob(os.path.join(args.gt_dir, "Patient_*")))
    if not vol_dirs:
        print("❌ 找不到病人資料夾，請確認 --gt_dir 路徑")
        return

    results = []

    print("=" * 70)
    print("🦷 DentalSegmentator vs CoSeg — 3D Dice 比較")
    print("=" * 70)
    print(f"  GT 資料夾:              {args.gt_dir}")
    print(f"  DentalSeg 預測資料夾:   {args.dental_pred_dir}")
    print(f"  CoSeg 預測資料夾:       {args.coseg_pred_dir or '(未提供)'}")
    print(f"  GT canal label:         {args.gt_canal_label}")
    print(f"  DentalSeg canal label:  {args.dental_canal_label}")
    print(f"  找到 {len(vol_dirs)} 個病人")
    print("=" * 70)

    for vol_dir in vol_dirs:
        vol_name = os.path.basename(vol_dir)
        print(f"\n📂 處理 {vol_name}...")

        # ── 1. 讀取 GT ──
        gt_path = os.path.join(vol_dir, "label.nii.gz")
        if not os.path.exists(gt_path):
            print(f"  ⚠️ 找不到 GT: {gt_path}，跳過")
            continue

        gt_data, gt_affine, gt_header = load_nifti(gt_path)
        gt_canal = (gt_data == args.gt_canal_label).astype(np.float32)

        if gt_canal.sum() == 0:
            print(f"  ⚠️ GT 中沒有 label={args.gt_canal_label} 的體素，跳過")
            # 印出 GT 中有哪些 label 值供偵錯
            unique_labels = np.unique(gt_data)
            print(f"     GT 中的 label 值: {unique_labels}")
            continue

        gt_voxels = int(gt_canal.sum())
        print(f"  GT canal voxels: {gt_voxels}")

        row = {"patient": vol_name, "gt_voxels": gt_voxels}

        # ── 2. 讀取 DentalSegmentator 預測 ──
        # nnU-Net 的輸出檔名通常跟輸入一樣，或是 image.nii.gz → image.nii.gz
        dental_pred_path = None
        for candidate in [
            os.path.join(args.dental_pred_dir, f"{vol_name}.nii.gz"),
            os.path.join(args.dental_pred_dir, f"{vol_name}_image.nii.gz"),
            os.path.join(args.dental_pred_dir, f"image.nii.gz"),
            os.path.join(args.dental_pred_dir, vol_name, "image.nii.gz"),
        ]:
            if os.path.exists(candidate):
                dental_pred_path = candidate
                break

        # 也搜尋目錄中所有包含 vol_name 的檔案
        if dental_pred_path is None:
            for f in os.listdir(args.dental_pred_dir):
                if vol_name.lower() in f.lower() and f.endswith(".nii.gz"):
                    dental_pred_path = os.path.join(args.dental_pred_dir, f)
                    break

        if dental_pred_path and os.path.exists(dental_pred_path):
            dental_data, _, _ = load_nifti(dental_pred_path)
            dental_canal = (dental_data == args.dental_canal_label).astype(np.float32)

            # 確認尺寸一致
            if dental_canal.shape != gt_canal.shape:
                print(f"  ⚠️ DentalSeg 尺寸 {dental_canal.shape} != GT {gt_canal.shape}")
                print(f"     嘗試 resample...")
                # 簡單的最近鄰插值對齊
                from scipy.ndimage import zoom
                zoom_factors = [g / d for g, d in zip(gt_canal.shape, dental_canal.shape)]
                dental_canal = zoom(dental_canal, zoom_factors, order=0)

            dice_dental = compute_3d_dice(dental_canal, gt_canal)
            iou_dental = compute_3d_iou(dental_canal, gt_canal)
            hd95_dental = compute_hausdorff_95(dental_canal, gt_canal)

            row["dental_dice"] = dice_dental
            row["dental_iou"] = iou_dental
            row["dental_hd95"] = hd95_dental
            row["dental_voxels"] = int(dental_canal.sum())

            print(f"  🦷 DentalSeg → Dice: {dice_dental:.4f} | IoU: {iou_dental:.4f} | HD95: {hd95_dental:.2f}")
        else:
            print(f"  ⚠️ 找不到 DentalSeg 預測: {vol_name}")
            row["dental_dice"] = None
            row["dental_iou"] = None
            row["dental_hd95"] = None

        # ── 3. 讀取 CoSeg 預測（如有提供）──
        if args.coseg_pred_dir:
            coseg_pred_path = None
            for candidate in [
                os.path.join(args.coseg_pred_dir, f"{vol_name}.nii.gz"),
                os.path.join(args.coseg_pred_dir, vol_name, "prediction.nii.gz"),
                os.path.join(args.coseg_pred_dir, vol_name, "pred.nii.gz"),
            ]:
                if os.path.exists(candidate):
                    coseg_pred_path = candidate
                    break

            if coseg_pred_path:
                coseg_data, _, _ = load_nifti(coseg_pred_path)
                coseg_canal = (coseg_data > 0).astype(np.float32)

                if coseg_canal.shape != gt_canal.shape:
                    from scipy.ndimage import zoom
                    zoom_factors = [g / d for g, d in zip(gt_canal.shape, coseg_canal.shape)]
                    coseg_canal = zoom(coseg_canal, zoom_factors, order=0)

                dice_coseg = compute_3d_dice(coseg_canal, gt_canal)
                iou_coseg = compute_3d_iou(coseg_canal, gt_canal)
                hd95_coseg = compute_hausdorff_95(coseg_canal, gt_canal)

                row["coseg_dice"] = dice_coseg
                row["coseg_iou"] = iou_coseg
                row["coseg_hd95"] = hd95_coseg

                print(f"  🧠 CoSeg    → Dice: {dice_coseg:.4f} | IoU: {iou_coseg:.4f} | HD95: {hd95_coseg:.2f}")
            else:
                print(f"  ⚠️ 找不到 CoSeg 預測: {vol_name}")

        results.append(row)

    # ── 彙總 ──
    print("\n" + "=" * 70)
    print("📊 總結")
    print("=" * 70)

    dental_dices = [r["dental_dice"] for r in results if r.get("dental_dice") is not None]
    if dental_dices:
        print(f"\n🦷 DentalSegmentator (mandibular canal only):")
        print(f"   平均 3D Dice: {np.mean(dental_dices):.4f} ± {np.std(dental_dices):.4f}")
        dental_ious = [r["dental_iou"] for r in results if r.get("dental_iou") is not None]
        print(f"   平均 3D IoU:  {np.mean(dental_ious):.4f} ± {np.std(dental_ious):.4f}")

    coseg_dices = [r.get("coseg_dice") for r in results if r.get("coseg_dice") is not None]
    if coseg_dices:
        print(f"\n🧠 CoSeg:")
        print(f"   平均 3D Dice: {np.mean(coseg_dices):.4f} ± {np.std(coseg_dices):.4f}")
        coseg_ious = [r.get("coseg_iou") for r in results if r.get("coseg_iou") is not None]
        print(f"   平均 3D IoU:  {np.mean(coseg_ious):.4f} ± {np.std(coseg_ious):.4f}")

    # ── 輸出 CSV ──
    if results:
        fieldnames = list(results[0].keys())
        # 確保所有 key 都有
        for r in results:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(args.output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n📁 詳細結果已存至: {args.output_csv}")

    print("=" * 70)


if __name__ == "__main__":
    main()
