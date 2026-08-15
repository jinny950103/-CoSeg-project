"""
eval_cldice_hd95.py — 專門計算 clDice 和 HD95
================================================
記憶體優化：
  - Volume 縮到 128x128 做 skeletonize（最吃 RAM 的部分）
  - 逐病人處理，處理完立刻釋放
  - HD95 用 256x256 計算
"""
import os, cv2, torch, gc
import numpy as np
import hydra
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, binary_erosion, label as scipy_label

from model import CoSeg
from sam2.build_sam import build_sam2

PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
EVAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data/public_data/eval")
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "outputs/coseg_v4_best.pth")
SAM2_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
CONTEXT_SLICES = 1

# 用小尺寸算，省 RAM
SKEL_SIZE = 128   # skeletonize 用 128（最吃 RAM）
HD95_SIZE = 256   # HD95 用 256


def skeletonize_3d_safe(volume):
    """安全的 3D 骨架化，失敗就用 2D 逐切片"""
    try:
        from skimage.morphology import skeletonize_3d
        return skeletonize_3d(volume.astype(np.uint8)).astype(np.float32)
    except Exception:
        # Fallback: 逐切片 2D 骨架化
        from skimage.morphology import skeletonize
        skel = np.zeros_like(volume, dtype=np.float32)
        for z in range(volume.shape[0]):
            if volume[z].sum() > 0:
                skel[z] = skeletonize(volume[z].astype(bool)).astype(np.float32)
        return skel


def compute_cldice(pred, gt, smooth=1e-5):
    """clDice"""
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
    """HD95"""
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float('inf')

    # 表面點
    pred_surface = pred_bin ^ binary_erosion(pred_bin).astype(np.uint8)
    gt_surface = gt_bin ^ binary_erosion(gt_bin).astype(np.uint8)

    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return float('inf')

    # 距離場
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


def main():
    device = "cuda"
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')

    print(f"📦 載入 SAM2: {SAM2_CHECKPOINT}")
    model = CoSeg(build_sam2("sam2_hiera_l.yaml", SAM2_CHECKPOINT, mode="train")).to(device)

    print(f"📦 載入 CoSeg: {MODEL_WEIGHTS_PATH}")
    sd = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
    model.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}, strict=True)
    model.eval()

    pm = np.array([123.675, 116.280, 103.530], dtype=np.float32)
    ps = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    patients = sorted([p for p in os.listdir(EVAL_DATA_DIR) if p.startswith("Patient_")])

    print("=" * 60)
    print("🧠 clDice + HD95 專用評估")
    print(f"   Skeleton size: {SKEL_SIZE}x{SKEL_SIZE}")
    print(f"   HD95 size: {HD95_SIZE}x{HD95_SIZE}")
    print("=" * 60)

    all_results = []

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for pid in patients:
            idir = os.path.join(EVAL_DATA_DIR, pid, "image_1024")
            mdir = os.path.join(EVAL_DATA_DIR, pid, "mask_sem_1024")
            if not os.path.isdir(idir):
                continue

            ifs = sorted([f for f in os.listdir(idir) if f.endswith(".png")])
            if not ifs:
                continue

            # 預載灰度
            ag = {}
            for f in ifs:
                g = cv2.imread(os.path.join(idir, f), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                ag[int(f.replace(".png", "").rsplit("_", 1)[-1])] = g

            # 推論，存兩種尺寸
            prob_skel, gt_skel = [], []
            prob_hd, gt_hd = [], []

            for f in tqdm(ifs, desc=f"推論 {pid}"):
                idx = int(f.replace(".png", "").rsplit("_", 1)[-1])
                c = ag[idx]
                p_ = ag.get(idx - 1, c)
                n_ = ag.get(idx + 1, c)
                img = np.stack([p_, c, n_], axis=-1)
                gt = np.load(os.path.join(mdir, f.replace(".png", ".npy"))).astype(np.float32)

                t = torch.tensor((img - pm) / (ps + 1e-8)).permute(2, 0, 1).unsqueeze(0).float().to(device)
                _, s, _, _ = model(x=t)
                s = F.interpolate(s, size=(1024, 1024), mode='bilinear', align_corners=False)
                prob = torch.sigmoid(s[:, 0, :, :])[0].cpu().numpy()

                # 縮小到兩種尺寸
                prob_skel.append((cv2.resize(prob, (SKEL_SIZE, SKEL_SIZE), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8))
                gt_skel.append((cv2.resize(gt, (SKEL_SIZE, SKEL_SIZE), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8))

                prob_hd.append((cv2.resize(prob, (HD95_SIZE, HD95_SIZE), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8))
                gt_hd.append((cv2.resize(gt, (HD95_SIZE, HD95_SIZE), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8))

                del t, s

            torch.cuda.empty_cache()
            del ag
            gc.collect()

            # === clDice（用 128 尺寸）===
            print(f"\n📊 {pid} — 計算 clDice (size={SKEL_SIZE}):")
            pred_skel_vol = np.stack(prob_skel, axis=0).astype(np.float32)
            gt_skel_vol = np.stack(gt_skel, axis=0).astype(np.float32)
            del prob_skel, gt_skel

            cldice = compute_cldice(pred_skel_vol, gt_skel_vol)
            dice_skel = compute_dice(pred_skel_vol, gt_skel_vol)

            del pred_skel_vol, gt_skel_vol
            gc.collect()

            # === HD95（用 256 尺寸）===
            print(f"   計算 HD95 (size={HD95_SIZE}):")
            pred_hd_vol = np.stack(prob_hd, axis=0).astype(np.float32)
            gt_hd_vol = np.stack(gt_hd, axis=0).astype(np.float32)
            del prob_hd, gt_hd

            hd95 = compute_hd95(pred_hd_vol, gt_hd_vol)
            dice_hd = compute_dice(pred_hd_vol, gt_hd_vol)

            del pred_hd_vol, gt_hd_vol
            gc.collect()

            result = {
                "patient": pid,
                "dice": dice_hd,
                "cldice": cldice,
                "hd95": hd95,
            }
            all_results.append(result)

            print(f"\n   ✅ {pid} 結果:")
            print(f"      Dice:    {dice_hd:.4f}")
            print(f"      clDice:  {cldice:.4f}")
            print(f"      HD95:    {hd95:.2f}")

    # 彙總
    print(f"\n{'='*60}")
    print("📊 總結")
    print(f"{'='*60}")

    dices = [r["dice"] for r in all_results]
    cldices = [r["cldice"] for r in all_results]
    hd95s = [r["hd95"] for r in all_results if r["hd95"] != float('inf')]

    print(f"  {'指標':<15s} {'Mean':>10s} {'Std':>10s}")
    print(f"  {'-'*35}")
    print(f"  {'Dice':<15s} {np.mean(dices):>10.4f} {np.std(dices):>10.4f}")
    print(f"  {'clDice':<15s} {np.mean(cldices):>10.4f} {np.std(cldices):>10.4f}")
    if hd95s:
        print(f"  {'HD95':<15s} {np.mean(hd95s):>10.2f} {np.std(hd95s):>10.2f}")

    print(f"\n  Per-patient:")
    for r in all_results:
        print(f"    {r['patient']}: Dice={r['dice']:.4f}  clDice={r['cldice']:.4f}  HD95={r['hd95']:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
