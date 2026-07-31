"""
compare_dental_segmentator_v2.py
================================
DentalSegmentator vs CoSeg 完整比較（含管狀結構指標）

新增指標：
  - clDice（拓撲連續性）
  - Connectivity Rate（最大連通元件比例）
  - Breakpoint Count（Z 軸斷裂次數）
  - Centerline Distance（中心線平均距離）

用法：
  python compare_dental_segmentator_v2.py \
    --dental_pred_dir dental_segmentator_comparison/nnunet_output \
    --gt_dir data/public_data/nifti_original \
    --gt_canal_label 1 \
    --dental_canal_label 5 \
    --output_csv comparison_results_v2.csv
"""

import os
import argparse
import numpy as np
import nibabel as nib
from glob import glob
import csv
from tubular_metrics import compute_all_metrics


def load_nifti(path):
    nii = nib.load(path)
    return nii.get_fdata(), nii.affine, nii.header


def main():
    parser = argparse.ArgumentParser(description="DentalSegmentator vs CoSeg — 管狀結構指標比較")
    parser.add_argument("--dental_pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--coseg_pred_dir", default=None)
    parser.add_argument("--gt_canal_label", type=int, default=1)
    parser.add_argument("--dental_canal_label", type=int, default=5)
    parser.add_argument("--output_csv", default="comparison_results_v2.csv")
    args = parser.parse_args()

    vol_dirs = sorted(glob(os.path.join(args.gt_dir, "VOL_*")))
    if not vol_dirs:
        vol_dirs = sorted(glob(os.path.join(args.gt_dir, "Patient_*")))
    if not vol_dirs:
        print("❌ 找不到病人資料夾")
        return

    results = []

    print("=" * 80)
    print("🦷 DentalSegmentator vs CoSeg — 管狀結構指標完整比較")
    print("=" * 80)
    print(f"  指標: Dice, IoU, HD95, clDice, Connectivity Rate, Breakpoints, Centerline Dist")
    print(f"  GT 資料夾:    {args.gt_dir}")
    print(f"  DentalSeg:    {args.dental_pred_dir}")
    print(f"  CoSeg:        {args.coseg_pred_dir or '(未提供)'}")
    print(f"  找到 {len(vol_dirs)} 個病人")
    print("=" * 80)

    for vol_dir in vol_dirs:
        vol_name = os.path.basename(vol_dir)
        print(f"\n{'='*60}")
        print(f"📂 {vol_name}")
        print(f"{'='*60}")

        # ── GT ──
        gt_path = os.path.join(vol_dir, "label.nii.gz")
        if not os.path.exists(gt_path):
            print(f"  ⚠️ 找不到 GT，跳過")
            continue

        gt_data, _, _ = load_nifti(gt_path)
        gt_canal = (gt_data == args.gt_canal_label).astype(np.float32)

        if gt_canal.sum() == 0:
            unique_labels = np.unique(gt_data)
            print(f"  ⚠️ GT 中無 label={args.gt_canal_label}（有: {unique_labels}），跳過")
            continue

        row = {"patient": vol_name, "gt_voxels": int(gt_canal.sum())}

        # ── GT 本身的拓撲資訊 ──
        from tubular_metrics import compute_connectivity_rate, compute_num_components
        gt_conn = compute_connectivity_rate(gt_canal)
        gt_ncomp = compute_num_components(gt_canal)
        print(f"  GT: {int(gt_canal.sum())} voxels, {gt_ncomp} components, connectivity {gt_conn:.3f}")

        # ── DentalSegmentator ──
        dental_pred_path = None
        for candidate in [
            os.path.join(args.dental_pred_dir, f"{vol_name}.nii.gz"),
            os.path.join(args.dental_pred_dir, f"{vol_name}_image.nii.gz"),
            os.path.join(args.dental_pred_dir, vol_name, "image.nii.gz"),
        ]:
            if os.path.exists(candidate):
                dental_pred_path = candidate
                break

        if dental_pred_path is None:
            for f in os.listdir(args.dental_pred_dir):
                if vol_name.lower() in f.lower() and f.endswith(".nii.gz"):
                    dental_pred_path = os.path.join(args.dental_pred_dir, f)
                    break

        if dental_pred_path:
            dental_data, _, _ = load_nifti(dental_pred_path)
            dental_canal = (dental_data == args.dental_canal_label).astype(np.float32)

            if dental_canal.shape != gt_canal.shape:
                from scipy.ndimage import zoom
                zoom_factors = [g / d for g, d in zip(gt_canal.shape, dental_canal.shape)]
                dental_canal = zoom(dental_canal, zoom_factors, order=0)

            print(f"\n  🦷 DentalSegmentator:")
            dental_metrics = compute_all_metrics(dental_canal, gt_canal)
            for k, v in dental_metrics.items():
                row[f"dental_{k}"] = v
                if isinstance(v, float):
                    print(f"     {k:25s}: {v:.4f}")
                else:
                    print(f"     {k:25s}: {v}")
        else:
            print(f"  ⚠️ 找不到 DentalSeg 預測")

        # ── CoSeg ──
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

                print(f"\n  🧠 CoSeg:")
                coseg_metrics = compute_all_metrics(coseg_canal, gt_canal)
                for k, v in coseg_metrics.items():
                    row[f"coseg_{k}"] = v
                    if isinstance(v, float):
                        print(f"     {k:25s}: {v:.4f}")
                    else:
                        print(f"     {k:25s}: {v}")

        results.append(row)

    # ── 彙總 ──
    print("\n" + "=" * 80)
    print("📊 總結比較")
    print("=" * 80)

    metric_keys = ["dice", "iou", "hd95", "cldice", "connectivity_rate", "breakpoints", "centerline_distance"]
    metric_names = {
        "dice": "3D Dice",
        "iou": "3D IoU",
        "hd95": "HD95",
        "cldice": "clDice ⭐",
        "connectivity_rate": "Connectivity Rate",
        "breakpoints": "Breakpoints (fewer=better)",
        "centerline_distance": "Centerline Dist (lower=better)",
    }

    for prefix, label in [("dental", "🦷 DentalSegmentator"), ("coseg", "🧠 CoSeg")]:
        values = {}
        for mk in metric_keys:
            key = f"{prefix}_{mk}"
            vals = [r[key] for r in results if key in r and r[key] is not None and r[key] != float('inf')]
            if vals:
                values[mk] = vals

        if not values:
            continue

        print(f"\n{label}:")
        print(f"  {'Metric':<35s} {'Mean':>10s} {'Std':>10s}")
        print(f"  {'-'*55}")
        for mk in metric_keys:
            if mk in values:
                vals = values[mk]
                name = metric_names.get(mk, mk)
                if mk == "breakpoints":
                    print(f"  {name:<35s} {np.mean(vals):>10.1f} {np.std(vals):>10.1f}")
                else:
                    print(f"  {name:<35s} {np.mean(vals):>10.4f} {np.std(vals):>10.4f}")

    # ── 如果兩邊都有，印出 head-to-head ──
    dental_has = any(f"dental_dice" in r for r in results)
    coseg_has = any(f"coseg_dice" in r for r in results)

    if dental_has and coseg_has:
        print(f"\n{'='*80}")
        print(f"⚔️  Head-to-Head (逐病人比較)")
        print(f"{'='*80}")
        print(f"  {'Patient':<12s} {'Metric':<15s} {'DentalSeg':>12s} {'CoSeg':>12s} {'Winner':>10s}")
        print(f"  {'-'*65}")
        for r in results:
            for mk in ["dice", "cldice", "breakpoints"]:
                dk = f"dental_{mk}"
                ck = f"coseg_{mk}"
                if dk in r and ck in r:
                    dv = r[dk]
                    cv = r[ck]
                    if mk == "breakpoints":
                        winner = "CoSeg" if cv <= dv else "Dental"
                        print(f"  {r['patient']:<12s} {mk:<15s} {dv:>12.0f} {cv:>12.0f} {winner:>10s}")
                    else:
                        winner = "CoSeg" if cv >= dv else "Dental"
                        print(f"  {r['patient']:<12s} {mk:<15s} {dv:>12.4f} {cv:>12.4f} {winner:>10s}")

    # ── CSV ──
    if results:
        all_keys = []
        for r in results:
            for k in r.keys():
                if k not in all_keys:
                    all_keys.append(k)

        with open(args.output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n📁 CSV 已存至: {args.output_csv}")

    print("=" * 80)


if __name__ == "__main__":
    main()
