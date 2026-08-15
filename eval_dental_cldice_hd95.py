"""
eval_dental_cldice_hd95.py — DentalSegmentator 的 clDice + HD95
================================================================
用跟 CoSeg eval_cldice_hd95.py 完全相同的方法和尺寸計算，
確保兩邊公平比較。

用法：
  python eval_dental_cldice_hd95.py \
    --dental_pred_dir dental_segmentator_comparison/nnunet_output \
    --gt_dir data/public_data/nifti_original \
    --gt_canal_label 1 \
    --dental_canal_label 5
"""
import os, gc, argparse
import numpy as np
import nibabel as nib
import cv2
from glob import glob
from scipy.ndimage import distance_transform_edt, binary_erosion

SKEL_SIZE = 128   # 跟 CoSeg eval 一樣
HD95_SIZE = 256   # 跟 CoSeg eval 一樣


def skeletonize_3d_safe(volume):
    """跟 CoSeg eval 完全相同的骨架化方法"""
    try:
        from skimage.morphology import skeletonize_3d
        return skeletonize_3d(volume.astype(np.uint8)).astype(np.float32)
    except Exception:
        from skimage.morphology import skeletonize
        skel = np.zeros_like(volume, dtype=np.float32)
        for z in range(volume.shape[0]):
            if volume[z].sum() > 0:
                skel[z] = skeletonize(volume[z].astype(bool)).astype(np.float32)
        return skel


def compute_cldice(pred, gt, smooth=1e-5):
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 1.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return 0.0

    print("     骨架化 pred...", end="", flush=True)
    skel_pred = skeletonize_3d_safe(pred_bin)
    print(f" {int(skel_pred.sum())} voxels")

    print("     骨架化 gt...", end="", flush=True)
    skel_gt = skeletonize_3d_safe(gt_bin)
    print(f" {int(skel_gt.sum())} voxels")

    if skel_pred.sum() == 0 or skel_gt.sum() == 0:
        return 0.0

    tprec = (skel_pred * gt_bin).sum() / (skel_pred.sum() + smooth)
    tsens = (skel_gt * pred_bin).sum() / (skel_gt.sum() + smooth)
    cldice = 2.0 * tprec * tsens / (tprec + tsens + smooth)

    del skel_pred, skel_gt
    gc.collect()

    return float(cldice)


def compute_hd95(pred, gt):
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float('inf')

    pred_surface = pred_bin ^ binary_erosion(pred_bin).astype(np.uint8)
    gt_surface = gt_bin ^ binary_erosion(gt_bin).astype(np.uint8)

    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return float('inf')

    print("     計算距離場...", end="", flush=True)
    dt_pred = distance_transform_edt(~pred_bin.astype(bool))
    dt_gt = distance_transform_edt(~gt_bin.astype(bool))
    print(" done")

    dist_pred_to_gt = dt_gt[pred_surface > 0]
    dist_gt_to_pred = dt_pred[gt_surface > 0]

    all_dist = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])

    del dt_pred, dt_gt, pred_surface, gt_surface
    gc.collect()

    return float(np.percentile(all_dist, 95))


def compute_dice(p, g, s=1e-5):
    i = np.sum(p * g)
    return (2.*i+s)/(np.sum(p)+np.sum(g)+s)


def resize_volume(vol, target_size):
    """將 3D volume 的 XY 平面 resize"""
    resized = np.zeros((vol.shape[0], target_size, target_size), dtype=vol.dtype)
    for z in range(vol.shape[0]):
        resized[z] = cv2.resize(vol[z].astype(np.float32), (target_size, target_size),
                                interpolation=cv2.INTER_NEAREST)
    return resized


def main():
    parser = argparse.ArgumentParser(description="DentalSegmentator clDice + HD95")
    parser.add_argument("--dental_pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--gt_canal_label", type=int, default=1)
    parser.add_argument("--dental_canal_label", type=int, default=5)
    args = parser.parse_args()

    vol_dirs = sorted(glob(os.path.join(args.gt_dir, "VOL_*")))
    if not vol_dirs:
        print("❌ 找不到病人資料夾")
        return

    print("=" * 60)
    print("🦷 DentalSegmentator — clDice + HD95 評估")
    print(f"   Skeleton size: {SKEL_SIZE}x{SKEL_SIZE} (跟 CoSeg eval 一致)")
    print(f"   HD95 size: {HD95_SIZE}x{HD95_SIZE}")
    print(f"   找到 {len(vol_dirs)} 個病人")
    print("=" * 60)

    all_results = []

    for vol_dir in vol_dirs:
        vol_name = os.path.basename(vol_dir)

        # 讀 GT
        gt_path = os.path.join(vol_dir, "label.nii.gz")
        if not os.path.exists(gt_path):
            print(f"  ⚠️ {vol_name}: 找不到 GT，跳過")
            continue

        gt_data = nib.load(gt_path).get_fdata()
        gt_canal = (gt_data == args.gt_canal_label).astype(np.float32)
        del gt_data
        gc.collect()

        if gt_canal.sum() == 0:
            print(f"  ⚠️ {vol_name}: GT 中無 label={args.gt_canal_label}，跳過")
            continue

        # 讀 DentalSeg 預測
        dental_pred_path = None
        for candidate in [
            os.path.join(args.dental_pred_dir, f"{vol_name}.nii.gz"),
            os.path.join(args.dental_pred_dir, f"{vol_name}_image.nii.gz"),
        ]:
            if os.path.exists(candidate):
                dental_pred_path = candidate
                break

        if dental_pred_path is None:
            for f in os.listdir(args.dental_pred_dir):
                if vol_name.lower() in f.lower() and f.endswith(".nii.gz"):
                    dental_pred_path = os.path.join(args.dental_pred_dir, f)
                    break

        if not dental_pred_path:
            print(f"  ⚠️ {vol_name}: 找不到 DentalSeg 預測，跳過")
            continue

        dental_data = nib.load(dental_pred_path).get_fdata()
        dental_canal = (dental_data == args.dental_canal_label).astype(np.float32)
        del dental_data
        gc.collect()

        # 尺寸對齊
        if dental_canal.shape != gt_canal.shape:
            from scipy.ndimage import zoom
            zoom_factors = [g / d for g, d in zip(gt_canal.shape, dental_canal.shape)]
            dental_canal = zoom(dental_canal, zoom_factors, order=0)

        print(f"\n📊 {vol_name}:")
        print(f"   GT voxels: {int(gt_canal.sum())}")
        print(f"   Pred voxels: {int(dental_canal.sum())}")

        # === Resize 到小尺寸 ===
        # 注意：nii.gz 的軸順序可能是 (X, Y, Z)，需要逐 Z 切片 resize
        # 先確認哪個軸是 Z（通常是最後一個或最大的）
        print(f"   Volume shape: {gt_canal.shape}")

        # 轉成 (Z, H, W) 格式，取最後一軸當 Z
        gt_zhw = np.transpose(gt_canal, (2, 0, 1))
        pred_zhw = np.transpose(dental_canal, (2, 0, 1))

        del gt_canal, dental_canal
        gc.collect()

        # 全解析度 Dice
        dice_full = compute_dice(pred_zhw, gt_zhw)
        print(f"   Dice (full res): {dice_full:.4f}")

        # === clDice（用 SKEL_SIZE）===
        print(f"   計算 clDice (size={SKEL_SIZE}):")
        gt_skel = resize_volume((gt_zhw > 0).astype(np.uint8), SKEL_SIZE)
        pred_skel = resize_volume((pred_zhw > 0).astype(np.uint8), SKEL_SIZE)

        cldice = compute_cldice(pred_skel, gt_skel)

        del gt_skel, pred_skel
        gc.collect()

        # === HD95（用 HD95_SIZE）===
        print(f"   計算 HD95 (size={HD95_SIZE}):")
        gt_hd = resize_volume((gt_zhw > 0).astype(np.uint8), HD95_SIZE).astype(np.float32)
        pred_hd = resize_volume((pred_zhw > 0).astype(np.uint8), HD95_SIZE).astype(np.float32)

        hd95 = compute_hd95(pred_hd, gt_hd)
        dice_hd = compute_dice(pred_hd, gt_hd)

        del gt_hd, pred_hd, gt_zhw, pred_zhw
        gc.collect()

        result = {
            "patient": vol_name,
            "dice_full": dice_full,
            "dice_resized": dice_hd,
            "cldice": cldice,
            "hd95": hd95,
        }
        all_results.append(result)

        print(f"\n   ✅ {vol_name} 結果:")
        print(f"      Dice (full):    {dice_full:.4f}")
        print(f"      Dice (resized): {dice_hd:.4f}")
        print(f"      clDice:         {cldice:.4f}")
        print(f"      HD95:           {hd95:.2f}")

    # 彙總
    print(f"\n{'='*60}")
    print("📊 DentalSegmentator 總結")
    print(f"{'='*60}")

    dices = [r["dice_full"] for r in all_results]
    cldices = [r["cldice"] for r in all_results]
    hd95s = [r["hd95"] for r in all_results if r["hd95"] != float('inf')]

    print(f"  {'指標':<20s} {'Mean':>10s} {'Std':>10s}")
    print(f"  {'-'*40}")
    print(f"  {'Dice (full res)':<20s} {np.mean(dices):>10.4f} {np.std(dices):>10.4f}")
    print(f"  {'clDice':<20s} {np.mean(cldices):>10.4f} {np.std(cldices):>10.4f}")
    if hd95s:
        print(f"  {'HD95':<20s} {np.mean(hd95s):>10.2f} {np.std(hd95s):>10.2f}")

    print(f"\n  Per-patient:")
    for r in all_results:
        print(f"    {r['patient']}: Dice={r['dice_full']:.4f}  clDice={r['cldice']:.4f}  HD95={r['hd95']:.2f}")

    print(f"\n  ⚠️ 注意：clDice 和 HD95 在 {SKEL_SIZE}/{HD95_SIZE} 尺寸下計算，")
    print(f"     跟 CoSeg eval_cldice_hd95.py 使用完全相同的尺寸和方法。")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
