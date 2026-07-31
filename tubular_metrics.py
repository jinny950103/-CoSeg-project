"""
tubular_metrics.py — 管狀結構專用評估指標
==========================================
包含：
  1. clDice (Centerline Dice) — 拓撲連續性的黃金指標
  2. Connectivity Rate — 最大連通元件佔比
  3. Breakpoint Count — Z 軸斷裂次數
  4. Centerline Distance — 中心線平均距離
  5. 標準 3D Dice / IoU / HD95

用法：
  from tubular_metrics import compute_all_metrics
  results = compute_all_metrics(pred_volume, gt_volume)
"""

import numpy as np
from scipy.ndimage import label as scipy_label
from scipy.ndimage import distance_transform_edt, binary_erosion


def skeletonize_3d_volume(volume):
    """
    3D 骨架化 — 嘗試用 skimage，失敗則用 scipy 的簡易版本
    """
    try:
        from skimage.morphology import skeletonize_3d
        return skeletonize_3d(volume.astype(np.uint8)).astype(np.float32)
    except ImportError:
        # Fallback: 迭代侵蝕直到只剩骨架
        from scipy.ndimage import binary_erosion, generate_binary_structure
        struct = generate_binary_structure(3, 1)
        skel = np.zeros_like(volume, dtype=np.float32)
        temp = volume.copy().astype(bool)

        while temp.any():
            eroded = binary_erosion(temp, structure=struct)
            boundary = temp & ~eroded
            skel[boundary] = 1
            temp = eroded

        return skel


def compute_cldice(pred, gt, smooth=1e-5):
    """
    clDice (Centerline Dice)
    ========================
    比普通 Dice 更適合管狀結構，因為它衡量的是「拓撲連續性」：
    - 即使體積重疊率高 (Dice 高)，如果中心線斷裂，clDice 會很低
    - 即使體積重疊率普通 (Dice 中等)，如果中心線完整，clDice 會很高

    公式：
      S_pred = skeleton(pred), S_gt = skeleton(gt)
      Tprec = |S_pred ∩ V_gt| / |S_pred|   (預測骨架有多少在 GT 體積內)
      Tsens = |S_gt ∩ V_pred| / |S_gt|      (GT 骨架有多少在預測體積內)
      clDice = 2 × Tprec × Tsens / (Tprec + Tsens)
    """
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 1.0  # 都是空的 → 完美匹配
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return 0.0  # 一個有一個沒有 → 完全不匹配

    # 骨架化
    skel_pred = skeletonize_3d_volume(pred_bin)
    skel_gt = skeletonize_3d_volume(gt_bin)

    if skel_pred.sum() == 0 or skel_gt.sum() == 0:
        return 0.0

    # Topology Precision: 預測骨架有多少落在 GT 體積內
    tprec = (skel_pred * gt_bin).sum() / (skel_pred.sum() + smooth)

    # Topology Sensitivity: GT 骨架有多少落在預測體積內
    tsens = (skel_gt * pred_bin).sum() / (skel_gt.sum() + smooth)

    # clDice
    cldice = 2.0 * tprec * tsens / (tprec + tsens + smooth)

    return float(cldice)


def compute_connectivity_rate(pred):
    """
    Connectivity Rate（最大連通元件比例）
    =====================================
    = 最大連通元件的 voxel 數 / 所有預測 voxel 數

    值越接近 1.0 代表預測越連續（理想情況只有一個連通元件）
    值越低代表預測越破碎（很多分散的小碎片）
    """
    pred_bin = (pred > 0).astype(np.uint8)

    if pred_bin.sum() == 0:
        return 0.0

    labeled, n_components = scipy_label(pred_bin)

    if n_components == 0:
        return 0.0

    # 找最大連通元件
    component_sizes = np.bincount(labeled.ravel())
    # component_sizes[0] 是背景
    if len(component_sizes) <= 1:
        return 0.0

    largest_component = component_sizes[1:].max()
    total_pred = pred_bin.sum()

    return float(largest_component / total_pred)


def compute_num_components(pred):
    """回傳 3D 連通元件數量"""
    pred_bin = (pred > 0).astype(np.uint8)
    if pred_bin.sum() == 0:
        return 0
    _, n = scipy_label(pred_bin)
    return int(n)


def compute_breakpoints(pred, gt=None):
    """
    Breakpoint Count（Z 軸斷裂次數）
    =================================
    沿 Z 軸檢查：如果某切片有預測、下一切片沒有、再下一切片又有，
    就算一次斷裂。

    如果提供了 GT，只在 GT 有神經管的 Z 範圍內計算。
    """
    pred_bin = (pred > 0).astype(np.uint8)

    # 如果有 GT，只看 GT 有出現的 Z 範圍
    if gt is not None:
        gt_bin = (gt > 0).astype(np.uint8)
        gt_per_slice = gt_bin.sum(axis=(1, 2))
        gt_slices = np.where(gt_per_slice > 0)[0]
        if len(gt_slices) == 0:
            return 0
        z_start = gt_slices.min()
        z_end = gt_slices.max()
    else:
        z_start = 0
        z_end = pred_bin.shape[0] - 1

    # 每個切片是否有預測
    has_pred = np.array([
        pred_bin[z].sum() > 0
        for z in range(z_start, z_end + 1)
    ])

    if len(has_pred) < 3:
        return 0

    # 計算斷裂：True → False 的轉換次數（在 GT 範圍內）
    breakpoints = 0
    in_canal = False

    for i, has in enumerate(has_pred):
        if has and not in_canal:
            if i > 0:  # 不是第一次出現，代表之前斷了
                breakpoints += 1
            in_canal = True
        elif not has and in_canal:
            in_canal = False

    return breakpoints


def compute_centerline_distance(pred, gt, voxel_spacing=None):
    """
    Centerline Distance（中心線平均距離）
    ======================================
    計算預測骨架到 GT 骨架的平均距離。
    值越小代表預測的路徑越準確。
    """
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float('inf')

    skel_pred = skeletonize_3d_volume(pred_bin)
    skel_gt = skeletonize_3d_volume(gt_bin)

    if skel_pred.sum() == 0 or skel_gt.sum() == 0:
        return float('inf')

    # 計算 GT 骨架的距離場
    dt_gt = distance_transform_edt(~skel_gt.astype(bool), sampling=voxel_spacing)
    dt_pred = distance_transform_edt(~skel_pred.astype(bool), sampling=voxel_spacing)

    # 預測骨架到 GT 骨架的平均距離
    dist_pred_to_gt = dt_gt[skel_pred > 0].mean()
    # GT 骨架到預測骨架的平均距離
    dist_gt_to_pred = dt_pred[skel_gt > 0].mean()

    # 雙向平均
    return float((dist_pred_to_gt + dist_gt_to_pred) / 2.0)


def compute_3d_dice(pred, gt, smooth=1e-5):
    pred_bin = (pred > 0).astype(np.float32)
    gt_bin = (gt > 0).astype(np.float32)
    intersection = np.sum(pred_bin * gt_bin)
    return float((2.0 * intersection + smooth) / (np.sum(pred_bin) + np.sum(gt_bin) + smooth))


def compute_3d_iou(pred, gt, smooth=1e-5):
    pred_bin = (pred > 0).astype(np.float32)
    gt_bin = (gt > 0).astype(np.float32)
    intersection = np.sum(pred_bin * gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin) - intersection
    return float((intersection + smooth) / (union + smooth))


def compute_hd95(pred, gt):
    """95th percentile Hausdorff Distance"""
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float('inf')

    pred_surface = pred_bin ^ binary_erosion(pred_bin).astype(np.uint8)
    gt_surface = gt_bin ^ binary_erosion(gt_bin).astype(np.uint8)

    dt_pred = distance_transform_edt(~pred_bin.astype(bool))
    dt_gt = distance_transform_edt(~gt_bin.astype(bool))

    dist_pred_to_gt = dt_gt[pred_surface > 0]
    dist_gt_to_pred = dt_pred[gt_surface > 0]

    if len(dist_pred_to_gt) == 0 or len(dist_gt_to_pred) == 0:
        return float('inf')

    all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    return float(np.percentile(all_distances, 95))


def compute_all_metrics(pred, gt):
    """
    一次計算所有指標，回傳 dict
    """
    results = {
        "dice": compute_3d_dice(pred, gt),
        "iou": compute_3d_iou(pred, gt),
        "hd95": compute_hd95(pred, gt),
        "cldice": compute_cldice(pred, gt),
        "connectivity_rate": compute_connectivity_rate(pred),
        "num_components": compute_num_components(pred),
        "breakpoints": compute_breakpoints(pred, gt),
        "centerline_distance": compute_centerline_distance(pred, gt),
        "pred_voxels": int((pred > 0).sum()),
        "gt_voxels": int((gt > 0).sum()),
    }
    return results
