"""
eval_v2_3d.py — 下顎神經管 3D 評估腳本 (改良版)
=================================================
改良重點：
  1. 2.5D 輸入：與訓練對齊，使用相鄰切片作為 RGB 通道
  2. Z 軸機率平滑：相鄰切片的預測機率做加權平均，消除斷裂
  3. 連通區域過濾：移除孤立小碎片，只保留主要結構
  4. 形態學閉合：填補神經管內部小洞
  5. 三合一對比圖：原圖 / Ground Truth / 預測 並列輸出
"""

import os
import cv2
import torch
import numpy as np
import hydra
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d, label as scipy_label
from scipy.ndimage import binary_closing, binary_fill_holes

from model import CoSeg
from sam2.build_sam import build_sam2

# ==========================================
# 📂 路徑設定
# ==========================================
PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
EVAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data/public_data/eval")
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "outputs/coseg_v2_best.pth")
DEBUG_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "debug_v2_output")
CONTEXT_SLICES = 1  # 與訓練對齊

# ==========================================
# 🎛️ 後處理超參數
# ==========================================
BINARY_THRESHOLD = 0.5
Z_SMOOTH_SIGMA = 1.5        # Z 軸高斯平滑 sigma (越大越平滑)
MIN_COMPONENT_VOXELS = 30   # 最小連通區域大小 (3D voxel 數)
MORPH_CLOSE_RADIUS = 3      # 形態學閉合半徑
SAVE_DEBUG_IMAGES = True     # 是否儲存三合一對比圖


# ==========================================
# 📐 3D 指標計算
# ==========================================
def compute_3d_metrics(pred_volume, gt_volume):
    smooth = 1e-5
    intersection = np.sum(pred_volume * gt_volume)
    sum_pred = np.sum(pred_volume)
    sum_gt = np.sum(gt_volume)
    union = sum_pred + sum_gt - intersection

    if sum_pred == 0 and sum_gt == 0:
        return None, None

    dice = (2.0 * intersection + smooth) / (sum_pred + sum_gt + smooth)
    iou = (intersection + smooth) / (union + smooth)
    return dice, iou


# ==========================================
# 🔧 後處理函式
# ==========================================
def postprocess_3d(prob_volume, binary_thresh=BINARY_THRESHOLD,
                   z_sigma=Z_SMOOTH_SIGMA, min_voxels=MIN_COMPONENT_VOXELS,
                   close_radius=MORPH_CLOSE_RADIUS):
    """
    3D 後處理 pipeline：

    Step 1: Z 軸機率平滑
      ─ 在二值化「之前」沿 Z 軸做高斯平滑
      ─ 效果：讓相鄰切片的預測互相「牽引」，大幅減少斷裂

    Step 2: 二值化

    Step 3: 形態學閉合 (逐切片)
      ─ 填補神經管內部小洞

    Step 4: 3D 連通區域過濾
      ─ 移除孤立的小碎片雜訊
    """
    # Step 1: Z 軸平滑（在機率空間做）
    if z_sigma > 0 and prob_volume.shape[0] > 3:
        prob_volume = gaussian_filter1d(prob_volume, sigma=z_sigma, axis=0)

    # Step 2: 二值化
    binary = (prob_volume > binary_thresh).astype(np.uint8)

    # Step 3: 形態學閉合（逐切片）
    if close_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_radius * 2 + 1, close_radius * 2 + 1)
        )
        for z in range(binary.shape[0]):
            binary[z] = cv2.morphologyEx(binary[z], cv2.MORPH_CLOSE, kernel)

    # Step 4: 3D 連通區域過濾
    if min_voxels > 0:
        labeled, n_components = scipy_label(binary)
        if n_components > 0:
            # 計算每個連通區域的大小
            component_sizes = np.bincount(labeled.ravel())
            # component_sizes[0] 是背景，從 1 開始
            for comp_id in range(1, n_components + 1):
                if component_sizes[comp_id] < min_voxels:
                    binary[labeled == comp_id] = 0

    return binary.astype(np.float32)


# ==========================================
# 🚀 主評估邏輯
# ==========================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

    # ── 建立模型 ──
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')
    model = CoSeg(build_sam2("sam2_hiera_l.yaml", None, mode="train"))

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # ── 載入權重 ──
    print(f"📦 載入模型權重: {MODEL_WEIGHTS_PATH}")
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)

    # 自動處理 DataParallel 前綴
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(new_state_dict, strict=True)
    else:
        model.load_state_dict(new_state_dict, strict=True)

    model.eval()

    # ── SAM2 正規化參數 ──
    pixel_mean = np.array([123.675, 116.280, 103.530], dtype=np.float32)
    pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    # ── 找到所有病人 ──
    patients = sorted([p for p in os.listdir(EVAL_DATA_DIR) if p.startswith("Patient_")])
    all_3d_dices, all_3d_ious = [], []
    all_3d_dices_raw, all_3d_ious_raw = [], []  # 無後處理對比

    print("=" * 60)
    print("🧠 下顎神經管 3D 評估 — 改良版 v2")
    print("=" * 60)
    print(f"  2.5D Context:       ±{CONTEXT_SLICES} slices")
    print(f"  Z-axis smoothing:   σ={Z_SMOOTH_SIGMA}")
    print(f"  Min component size: {MIN_COMPONENT_VOXELS} voxels")
    print(f"  Morph close radius: {MORPH_CLOSE_RADIUS}")
    print("=" * 60)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for patient_id in patients:
            img_dir = os.path.join(EVAL_DATA_DIR, patient_id, "image_1024")
            mask_dir = os.path.join(EVAL_DATA_DIR, patient_id, "mask_sem_1024")

            if not os.path.isdir(img_dir):
                continue

            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
            if not img_files:
                continue

            # ── 預載所有灰度切片（支援 2.5D）──
            all_gray = {}
            for img_f in img_files:
                gray = cv2.imread(
                    os.path.join(img_dir, img_f), cv2.IMREAD_GRAYSCALE
                ).astype(np.float32)
                # 從檔名取得 slice index
                slice_idx = int(img_f.replace(".png", "").rsplit("_", 1)[-1])
                all_gray[slice_idx] = gray

            sorted_indices = sorted(all_gray.keys())

            # ── 逐切片預測 ──
            prob_slices = {}
            gt_slices = {}

            for img_f in tqdm(img_files, desc=f"推論 {patient_id}"):
                slice_idx = int(img_f.replace(".png", "").rsplit("_", 1)[-1])
                center = all_gray[slice_idx]

                # 2.5D: 取得相鄰切片
                prev_idx = slice_idx - CONTEXT_SLICES
                next_idx = slice_idx + CONTEXT_SLICES
                prev_slice = all_gray.get(prev_idx, center)
                next_slice = all_gray.get(next_idx, center)

                img_3ch = np.stack([prev_slice, center, next_slice], axis=-1)

                # 讀取 GT
                gt_path = os.path.join(mask_dir, img_f.replace(".png", ".npy"))
                gt_mask = np.load(gt_path).astype(np.float32)

                # 正規化 + Tensor
                img_norm = (img_3ch - pixel_mean) / (pixel_std + 1e-8)
                img_tensor = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

                # 推論
                _, mask_pred_sem, _, _ = model(x=img_tensor)
                mask_pred_sem = F.interpolate(
                    mask_pred_sem, size=(1024, 1024),
                    mode='bilinear', align_corners=False
                )

                pred_prob = torch.sigmoid(mask_pred_sem[:, 0, :, :])[0].cpu().numpy()
                prob_slices[slice_idx] = pred_prob
                gt_slices[slice_idx] = (gt_mask > 0).astype(np.float32)

            # ── 組裝 3D Volume ──
            prob_volume = np.stack([prob_slices[i] for i in sorted_indices], axis=0)
            gt_volume = np.stack([gt_slices[i] for i in sorted_indices], axis=0)

            # ── 無後處理的基準分數 ──
            raw_pred = (prob_volume > BINARY_THRESHOLD).astype(np.float32)
            dice_raw, iou_raw = compute_3d_metrics(raw_pred, gt_volume)

            # ── 後處理 ──
            post_pred = postprocess_3d(prob_volume.copy())
            dice_post, iou_post = compute_3d_metrics(post_pred, gt_volume)

            if dice_raw is not None:
                all_3d_dices_raw.append(dice_raw)
                all_3d_ious_raw.append(iou_raw)
            if dice_post is not None:
                all_3d_dices.append(dice_post)
                all_3d_ious.append(iou_post)

            print(f"\n📊 {patient_id}:")
            print(f"   Raw  → 3D Dice: {dice_raw:.4f} | IoU: {iou_raw:.4f}")
            print(f"   Post → 3D Dice: {dice_post:.4f} | IoU: {iou_post:.4f}")

            # ── 儲存三合一對比圖 ──
            if SAVE_DEBUG_IMAGES:
                patient_debug_dir = os.path.join(DEBUG_OUTPUT_DIR, patient_id)
                os.makedirs(patient_debug_dir, exist_ok=True)

                for z_order, slice_idx in enumerate(sorted_indices):
                    gt_2d = gt_slices[slice_idx]
                    if gt_2d.sum() == 0:
                        continue  # 只輸出有 GT 的切片

                    center_gray = all_gray[slice_idx].astype(np.uint8)
                    img_vis = np.stack([center_gray] * 3, axis=-1)

                    gt_vis = (gt_2d * 255).astype(np.uint8)
                    gt_vis_3c = np.stack([gt_vis] * 3, axis=-1)

                    pred_2d = post_pred[z_order]
                    pred_vis = (pred_2d * 255).astype(np.uint8)
                    pred_vis_3c = np.stack([pred_vis] * 3, axis=-1)

                    # 疊加：在原圖上用顏色標示 GT(綠) 和 Pred(紅)
                    overlay = img_vis.copy()
                    overlay[gt_2d > 0, 1] = np.clip(
                        overlay[gt_2d > 0, 1].astype(np.int16) + 100, 0, 255
                    ).astype(np.uint8)
                    overlay[pred_2d > 0, 2] = np.clip(
                        overlay[pred_2d > 0, 2].astype(np.int16) + 100, 0, 255
                    ).astype(np.uint8)

                    combined = np.hstack((img_vis, gt_vis_3c, pred_vis_3c, overlay))

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(combined, 'Original', (20, 50), font, 1.5, (0, 255, 0), 3)
                    cv2.putText(combined, 'Ground Truth', (1024 + 20, 50), font, 1.5, (0, 255, 0), 3)
                    cv2.putText(combined, 'Prediction', (2048 + 20, 50), font, 1.5, (0, 255, 0), 3)
                    cv2.putText(combined, 'Overlay (G=GT, R=Pred)', (3072 + 20, 50), font, 1.2, (0, 255, 0), 3)

                    out_name = f"slice_{slice_idx:04d}_compare.png"
                    cv2.imwrite(os.path.join(patient_debug_dir, out_name), combined)

    # ── 彙總 ──
    print("\n" + "=" * 60)
    if all_3d_dices:
        print(f"📊 全體平均 (Raw,  無後處理): 3D Dice: {np.mean(all_3d_dices_raw):.4f} | IoU: {np.mean(all_3d_ious_raw):.4f}")
        print(f"🏆 全體平均 (Post, 含後處理): 3D Dice: {np.mean(all_3d_dices):.4f} | IoU: {np.mean(all_3d_ious):.4f}")
        print(f"\n   Per-patient breakdown:")
        for i, pid in enumerate(patients):
            if i < len(all_3d_dices):
                print(f"   {pid}: Dice={all_3d_dices[i]:.4f} IoU={all_3d_ious[i]:.4f}")
    else:
        print("⚠️ 沒有有效結果")
    print("=" * 60)


if __name__ == "__main__":
    main()
