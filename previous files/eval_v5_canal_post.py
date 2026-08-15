"""
eval_v5_canal_post.py — Canal-Aware 後處理 + 全指標評估
========================================================
結合 TTA + 管狀結構後處理 + Dice/clDice/HD95 一次跑完

後處理策略（針對管狀結構優化）：
  1. 保留 top-K 最大 connected components（神經管就是左右各一條）
  2. 每個 component 沿 z 軸追蹤 centerline（每切片的重心）
  3. 在斷裂的 z 切片插值 centerline，補上小圓盤填補斷裂
  4. 移除離 centerline 太遠的 outlier voxel（壓 HD95）
  5. 輕度 morphological closing 平滑邊界

用法：
  python eval_v5_canal_post.py
  python eval_v5_canal_post.py --weights outputs/coseg_v4_best.pth
  python eval_v5_canal_post.py --no-tta
"""
import os, cv2, torch, gc, argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import (
    gaussian_filter1d, label as scipy_label,
    distance_transform_edt, binary_erosion
)

import hydra
from model import CoSeg
from sam2.build_sam import build_sam2

# ==========================================
# 📂 路徑設定
# ==========================================
PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
EVAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data/public_data/eval")
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "outputs/coseg_v5_best.pth")
SAM2_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"

# ==========================================
# 🎛️ 評估參數
# ==========================================
EVAL_SIZE = 256          # 指標計算用的解析度
SKEL_SIZE = 128          # clDice 骨架化用的解析度（省 RAM）
FULL_RES = 1024          # 模型輸出解析度

# ==========================================
# 🔧 Canal-Aware 後處理參數
# ==========================================
TOP_K_COMPONENTS = 4         # 保留最大的 K 個 component
MIN_COMPONENT_VOXELS = 20   # 小於此的 component 直接丟掉
MORPH_CLOSE_RADIUS = 1       # 最後的 morphological closing 半徑
Z_SMOOTH_SIGMA = 0.5         # 非常輕微的 z 軸平滑（只在 prob 階段）


# ==========================================
# 📐 基礎指標
# ==========================================
def compute_dice(p, g, s=1e-5):
    i = np.sum(p * g)
    return (2. * i + s) / (np.sum(p) + np.sum(g) + s)


def compute_iou(p, g, s=1e-5):
    i = np.sum(p * g)
    return (i + s) / (np.sum(p) + np.sum(g) - i + s)


def skeletonize_3d_safe(volume):
    """安全的 3D 骨架化"""
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
    """clDice — 管狀結構專用指標"""
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 1.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return 0.0

    skel_pred = skeletonize_3d_safe(pred_bin)
    skel_gt = skeletonize_3d_safe(gt_bin)

    if skel_pred.sum() == 0 or skel_gt.sum() == 0:
        return 0.0

    tprec = (skel_pred * gt_bin).sum() / (skel_pred.sum() + smooth)
    tsens = (skel_gt * pred_bin).sum() / (skel_gt.sum() + smooth)
    cldice = 2.0 * tprec * tsens / (tprec + tsens + smooth)

    del skel_pred, skel_gt
    gc.collect()
    return float(cldice)


def compute_hd95(pred, gt):
    """HD95 — Hausdorff Distance 95th percentile"""
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float('inf')

    pred_surface = pred_bin ^ binary_erosion(pred_bin).astype(np.uint8)
    gt_surface = gt_bin ^ binary_erosion(gt_bin).astype(np.uint8)

    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return float('inf')

    dt_pred = distance_transform_edt(~pred_bin.astype(bool))
    dt_gt = distance_transform_edt(~gt_bin.astype(bool))

    dist_pred_to_gt = dt_gt[pred_surface > 0]
    dist_gt_to_pred = dt_pred[gt_surface > 0]
    all_dist = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])

    del dt_pred, dt_gt, pred_surface, gt_surface
    gc.collect()
    return float(np.percentile(all_dist, 95))


def count_breakpoints(pred, gt):
    """計算 z 方向的斷裂點數"""
    gz = gt.sum(axis=(1, 2))
    gr = np.where(gz > 0)[0]
    bp = 0
    if len(gr) > 0:
        pz = pred[gr[0]:gr[-1] + 1].sum(axis=(1, 2)) > 0
        in_canal = False
        for i, has in enumerate(pz):
            if has and not in_canal:
                if i > 0:
                    bp += 1
                in_canal = True
            elif not has and in_canal:
                in_canal = False
    return bp


# ==========================================
# 🧠 Canal-Aware 後處理
# ==========================================
def canal_aware_postprocess(prob_volume, gt_volume=None):
    """
    管狀結構後處理（保守版）

    策略：只做減法（去碎片），不做加法（不填補斷裂）。
    原因：散落碎片的重心不構成有意義的 centerline，插值會畫出大量假陽性。

    Steps:
      1. 極輕微 z 軸平滑（σ=0.5，只平滑機率）
      2. 二值化
      3. 3D connected components → 只保留 top-K 最大的
      4. 輕度 morphological closing

    Args:
        prob_volume: (Z, H, W) 機率圖，值域 [0, 1]
        gt_volume:   (Z, H, W) GT（未使用，保留介面相容性）

    Returns:
        binary: (Z, H, W) 後處理後的二值預測
    """
    Z, H, W = prob_volume.shape

    # --- Step 1: 極輕微 z 軸平滑 ---
    if Z_SMOOTH_SIGMA > 0 and Z > 3:
        prob_volume = gaussian_filter1d(prob_volume.copy(), sigma=Z_SMOOTH_SIGMA, axis=0)

    # --- Step 2: 二值化 ---
    binary = (prob_volume > 0.5).astype(np.uint8)

    if binary.sum() == 0:
        return binary.astype(np.float32)

    # --- Step 3: 只保留 top-K 最大 connected components ---
    labeled, n_comp = scipy_label(binary)
    if n_comp == 0:
        return binary.astype(np.float32)

    comp_sizes = np.bincount(labeled.ravel())[1:]  # 排除 background
    top_k = min(TOP_K_COMPONENTS, n_comp)
    top_indices = np.argsort(comp_sizes)[::-1][:top_k]
    top_labels = top_indices + 1

    clean = np.zeros_like(binary)
    for lbl in top_labels:
        if comp_sizes[lbl - 1] >= MIN_COMPONENT_VOXELS:
            clean[labeled == lbl] = 1

    if clean.sum() == 0:
        return clean.astype(np.float32)

    # --- Step 4: 輕度 morphological closing ---
    if MORPH_CLOSE_RADIUS > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (MORPH_CLOSE_RADIUS * 2 + 1, MORPH_CLOSE_RADIUS * 2 + 1)
        )
        for z in range(Z):
            if clean[z].sum() > 0:
                clean[z] = cv2.morphologyEx(clean[z], cv2.MORPH_CLOSE, kernel)

    return clean.astype(np.float32)


# ==========================================
# 🔮 推論
# ==========================================
def predict_slice(model, img_3ch, pm, ps, device):
    """單張推論，回傳 1024×1024 機率圖"""
    img_norm = (img_3ch - pm) / (ps + 1e-8)
    t = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
    _, s, _, _ = model(x=t)
    s = F.interpolate(s, size=(FULL_RES, FULL_RES), mode='bilinear', align_corners=False)
    prob = torch.sigmoid(s[:, 0, :, :])[0].cpu().numpy()
    del t, s
    return prob


def predict_patient(model, idir, mdir, ifs, ag, pm, ps, device, use_tta=True):
    """推論一個病人的所有切片，回傳機率和 GT volume"""
    prob_list, gt_list = [], []

    for f in tqdm(ifs, desc=f"推論 (TTA={'ON' if use_tta else 'OFF'})"):
        idx = int(f.replace(".png", "").rsplit("_", 1)[-1])
        c = ag[idx]
        p_ = ag.get(idx - 1, c)
        n_ = ag.get(idx + 1, c)
        img = np.stack([p_, c, n_], axis=-1)
        gt = np.load(os.path.join(mdir, f.replace(".png", ".npy"))).astype(np.float32)

        prob_orig = predict_slice(model, img, pm, ps, device)

        if use_tta:
            img_flip = np.flip(img, axis=1).copy()
            prob_flip = predict_slice(model, img_flip, pm, ps, device)
            prob_flip = np.flip(prob_flip, axis=1).copy()
            prob = (prob_orig + prob_flip) / 2.0
        else:
            prob = prob_orig

        # resize 到 EVAL_SIZE
        prob_list.append(cv2.resize(prob, (EVAL_SIZE, EVAL_SIZE),
                                    interpolation=cv2.INTER_LINEAR))
        gt_list.append((cv2.resize(gt, (EVAL_SIZE, EVAL_SIZE),
                                   interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8))

    prob_vol = np.stack(prob_list, axis=0)
    gt_vol = np.stack(gt_list, axis=0).astype(np.float32)
    del prob_list, gt_list
    return prob_vol, gt_vol


# ==========================================
# 📊 全指標評估
# ==========================================
def evaluate_volume(pred_binary, gt_vol, label=""):
    """計算所有指標"""
    dice = compute_dice(pred_binary, gt_vol)
    iou = compute_iou(pred_binary, gt_vol)

    # Connected components
    lb, nc = scipy_label(pred_binary.astype(np.uint8))
    if pred_binary.sum() > 0 and nc > 0:
        conn = np.bincount(lb.ravel())[1:].max() / pred_binary.sum()
    else:
        conn = 0.0
    bp = count_breakpoints(pred_binary, gt_vol)

    # clDice（用更小的尺寸）
    print(f"     計算 clDice (size={SKEL_SIZE})...", end="", flush=True)
    pred_skel = np.zeros((pred_binary.shape[0], SKEL_SIZE, SKEL_SIZE), dtype=np.uint8)
    gt_skel = np.zeros((gt_vol.shape[0], SKEL_SIZE, SKEL_SIZE), dtype=np.uint8)
    for z in range(pred_binary.shape[0]):
        pred_skel[z] = cv2.resize(pred_binary[z].astype(np.float32),
                                  (SKEL_SIZE, SKEL_SIZE),
                                  interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        gt_skel[z] = cv2.resize(gt_vol[z].astype(np.float32),
                                (SKEL_SIZE, SKEL_SIZE),
                                interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    cldice = compute_cldice(pred_skel, gt_skel)
    print(f" {cldice:.4f}")
    del pred_skel, gt_skel

    # HD95（在 EVAL_SIZE 直接算）
    print(f"     計算 HD95 (size={EVAL_SIZE})...", end="", flush=True)
    hd95 = compute_hd95(pred_binary, gt_vol)
    print(f" {hd95:.2f}")

    gc.collect()
    return {
        "dice": dice, "iou": iou, "cldice": cldice, "hd95": hd95,
        "comp": nc, "conn": conn, "bp": bp, "label": label
    }


# ==========================================
# 🚀 主程式
# ==========================================
def main():
    global TOP_K_COMPONENTS

    parser = argparse.ArgumentParser(description="CoSeg Canal-Aware 後處理評估")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS,
                        help="模型權重路徑")
    parser.add_argument("--no-tta", action="store_true",
                        help="不使用 TTA")
    parser.add_argument("--top-k", type=int, default=TOP_K_COMPONENTS,
                        help="保留最大的 K 個 component")
    args = parser.parse_args()

    TOP_K_COMPONENTS = args.top_k

    device = "cuda"
    use_tta = not args.no_tta

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')

    print(f"📦 載入 SAM2: {SAM2_CHECKPOINT}")
    model = CoSeg(build_sam2("sam2_hiera_l.yaml", SAM2_CHECKPOINT, mode="train")).to(device)

    print(f"📦 載入 CoSeg: {args.weights}")
    sd = torch.load(args.weights, map_location=device)
    model.load_state_dict(
        {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()},
        strict=True
    )
    model.eval()

    pm = np.array([123.675, 116.280, 103.530], dtype=np.float32)
    ps = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    patients = sorted([p for p in os.listdir(EVAL_DATA_DIR) if p.startswith("Patient_")])

    print("=" * 60)
    print("🧠 CoSeg 評估 — Canal-Aware 後處理 + 全指標")
    print(f"   權重: {os.path.basename(args.weights)}")
    print(f"   TTA: {'ON (原圖 + 水平翻轉)' if use_tta else 'OFF'}")
    print(f"   後處理: Top-{TOP_K_COMPONENTS} comp, Min={MIN_COMPONENT_VOXELS}vox, "
          f"Close r={MORPH_CLOSE_RADIUS}, Z-smooth σ={Z_SMOOTH_SIGMA}")
    print(f"   Eval size: {EVAL_SIZE}, Skeleton size: {SKEL_SIZE}")
    print("=" * 60)

    results_raw = []
    results_canal = []

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for pid in patients:
            idir = os.path.join(EVAL_DATA_DIR, pid, "image_1024")
            mdir = os.path.join(EVAL_DATA_DIR, pid, "mask_sem_1024")
            if not os.path.isdir(idir):
                continue

            ifs = sorted([f for f in os.listdir(idir) if f.endswith(".png")])
            if not ifs:
                continue

            # 預載灰度圖
            ag = {}
            for f in ifs:
                g = cv2.imread(os.path.join(idir, f), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                ag[int(f.replace(".png", "").rsplit("_", 1)[-1])] = g

            print(f"\n{'='*50}")
            print(f"📊 {pid} ({len(ifs)} slices)")
            print(f"{'='*50}")

            # 推論
            prob_vol, gt_vol = predict_patient(
                model, idir, mdir, ifs, ag, pm, ps, device, use_tta=use_tta
            )
            torch.cuda.empty_cache()
            del ag
            gc.collect()

            # === Raw（直接二值化，無後處理）===
            print(f"\n  🔹 Raw（無後處理）:")
            raw_binary = (prob_vol > 0.5).astype(np.float32)
            r_raw = evaluate_volume(raw_binary, gt_vol, "Raw")
            results_raw.append(r_raw)

            print(f"     Dice={r_raw['dice']:.4f}  clDice={r_raw['cldice']:.4f}  "
                  f"HD95={r_raw['hd95']:.2f}  Comp={r_raw['comp']}  BP={r_raw['bp']}")

            # === Canal-Aware 後處理 ===
            print(f"\n  🔸 Canal-Aware 後處理:")
            canal_binary = canal_aware_postprocess(prob_vol.copy(), gt_vol)
            r_canal = evaluate_volume(canal_binary, gt_vol, "Canal-Post")
            results_canal.append(r_canal)

            print(f"     Dice={r_canal['dice']:.4f}  clDice={r_canal['cldice']:.4f}  "
                  f"HD95={r_canal['hd95']:.2f}  Comp={r_canal['comp']}  BP={r_canal['bp']}")

            # 改善量
            d_dice = r_canal['dice'] - r_raw['dice']
            d_cldice = r_canal['cldice'] - r_raw['cldice']
            d_hd95 = r_canal['hd95'] - r_raw['hd95']
            print(f"\n     📈 改善: Dice {d_dice:+.4f}  clDice {d_cldice:+.4f}  HD95 {d_hd95:+.2f}")

            del prob_vol, gt_vol, raw_binary, canal_binary
            gc.collect()

    # ==========================================
    # 📊 彙總
    # ==========================================
    print(f"\n{'='*70}")
    print("📊 最終總結")
    print(f"{'='*70}")

    def print_summary(label, results):
        dices = [r['dice'] for r in results]
        cldices = [r['cldice'] for r in results]
        hd95s = [r['hd95'] for r in results if r['hd95'] != float('inf')]
        comps = [r['comp'] for r in results]
        bps = [r['bp'] for r in results]
        conns = [r['conn'] for r in results]

        print(f"\n  📌 {label}:")
        print(f"     {'指標':<12s} {'Mean':>8s} {'Std':>8s}")
        print(f"     {'-'*30}")
        print(f"     {'Dice':<12s} {np.mean(dices):>8.4f} {np.std(dices):>8.4f}")
        print(f"     {'clDice':<12s} {np.mean(cldices):>8.4f} {np.std(cldices):>8.4f}")
        if hd95s:
            print(f"     {'HD95':<12s} {np.mean(hd95s):>8.2f} {np.std(hd95s):>8.2f}")
        print(f"     {'Components':<12s} {np.mean(comps):>8.1f}")
        print(f"     {'Conn':<12s} {np.mean(conns):>8.4f}")
        print(f"     {'Breakpoints':<12s} {np.mean(bps):>8.1f}")

        print(f"\n     Per-patient:")
        for i, pid in enumerate(sorted(
            [p for p in os.listdir(EVAL_DATA_DIR) if p.startswith("Patient_")]
        )):
            if i < len(results):
                r = results[i]
                print(f"       {pid}: Dice={r['dice']:.4f}  clDice={r['cldice']:.4f}  "
                      f"HD95={r['hd95']:.2f}  Comp={r['comp']}  BP={r['bp']}")

    tta_label = "TTA" if use_tta else "No TTA"
    print_summary(f"{tta_label} + Raw", results_raw)
    print_summary(f"{tta_label} + Canal-Aware Post", results_canal)

    # 跟 DentalSeg 比較
    print(f"\n  {'─'*50}")
    print(f"  📋 vs DentalSegmentator (參考):")
    print(f"     DentalSeg:   Dice=0.7589  clDice=0.8577  HD95=2.16")
    canal_dices = [r['dice'] for r in results_canal]
    canal_cldices = [r['cldice'] for r in results_canal]
    canal_hd95s = [r['hd95'] for r in results_canal if r['hd95'] != float('inf')]
    print(f"     CoSeg (ours): Dice={np.mean(canal_dices):.4f}  "
          f"clDice={np.mean(canal_cldices):.4f}  "
          f"HD95={np.mean(canal_hd95s):.2f}" if canal_hd95s else "")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()