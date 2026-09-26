"""
eval_public_v2.py — 公開 Test 評估 + Z 軸後處理
=================================================
新增：
  --smooth SIGMA   在 3D probability volume 的 z 軸做 Gaussian smoothing（預設不啟用）
  --zspan N        只保留 z-span >= N 的 connected component（預設不啟用）
  --threshold T    二值化閾值（預設 0.5）

用法：
  # 原版（無後處理，跟 eval_public.py 一樣）
  python eval_public_v2.py --weights outputs/coseg_v8_best.pth

  # Z 軸 smoothing（sigma=1.5）
  python eval_public_v2.py --weights outputs/coseg_v8_best.pth --smooth 1.5

  # Z 軸 smoothing + 只保留 z-span>=10 的 component
  python eval_public_v2.py --weights outputs/coseg_v8_best.pth --smooth 1.5 --zspan 10

  # 掃一組 sigma 值
  python eval_public_v2.py --weights outputs/coseg_v8_best.pth --sweep

  # 也支援醫院 eval
  python eval_public_v2.py --weights outputs/coseg_v8_best.pth --eval-dir data/hospital_data/eval --smooth 1.5
"""
import os, cv2, torch, gc, argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, binary_erosion, label as scipy_label
from scipy.ndimage import gaussian_filter1d
import hydra
from model_v6 import CoSegV6
from sam2.build_sam import build_sam2

PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
PUBLIC_EVAL_DIR = os.path.join(PROJECT_ROOT, "data/public_data/eval")
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "outputs/coseg_v8_best.pth")
SAM2_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
EVAL_SIZE, SKEL_SIZE, FULL_RES = 256, 128, 1024


# ==========================================
# 指標計算
# ==========================================
def compute_dice(p, g, s=1e-5):
    i = np.sum(p * g); return (2 * i + s) / (np.sum(p) + np.sum(g) + s)

def skeletonize_3d_safe(vol):
    try:
        from skimage.morphology import skeletonize_3d
        return skeletonize_3d(vol.astype(np.uint8)).astype(np.float32)
    except:
        from skimage.morphology import skeletonize
        s = np.zeros_like(vol, dtype=np.float32)
        for z in range(vol.shape[0]):
            if vol[z].sum() > 0: s[z] = skeletonize(vol[z].astype(bool)).astype(np.float32)
        return s

def compute_cldice(pred, gt, smooth=1e-5):
    pb, gb = (pred > 0).astype(np.uint8), (gt > 0).astype(np.uint8)
    if pb.sum() == 0 and gb.sum() == 0: return 1.0
    if pb.sum() == 0 or gb.sum() == 0: return 0.0
    sp, sg = skeletonize_3d_safe(pb), skeletonize_3d_safe(gb)
    if sp.sum() == 0 or sg.sum() == 0: return 0.0
    tp = (sp * gb).sum() / (sp.sum() + smooth); ts = (sg * pb).sum() / (sg.sum() + smooth)
    r = float(2 * tp * ts / (tp + ts + smooth)); del sp, sg; gc.collect(); return r

def compute_hd95(pred, gt):
    pb, gb = (pred > 0).astype(np.uint8), (gt > 0).astype(np.uint8)
    if pb.sum() == 0 or gb.sum() == 0: return float('inf')
    ps = pb ^ binary_erosion(pb).astype(np.uint8)
    gs = gb ^ binary_erosion(gb).astype(np.uint8)
    if ps.sum() == 0 or gs.sum() == 0: return float('inf')
    dp = distance_transform_edt(~pb.astype(bool))
    dg = distance_transform_edt(~gb.astype(bool))
    ad = np.concatenate([dg[ps > 0], dp[gs > 0]]); del dp, dg; gc.collect()
    return float(np.percentile(ad, 95))

def count_bp(pred, gt):
    gr = np.where(gt.sum(axis=(1, 2)) > 0)[0]; bp = 0
    if len(gr) > 0:
        pz = pred[gr[0]:gr[-1] + 1].sum(axis=(1, 2)) > 0; ic = False
        for i, h in enumerate(pz):
            if h and not ic:
                if i > 0: bp += 1
                ic = True
            elif not h and ic: ic = False
    return bp


# ==========================================
# 推論
# ==========================================
def predict_one(model, img_3ch, pm, ps, device):
    t = torch.tensor((img_3ch - pm) / (ps + 1e-8)).permute(2, 0, 1).unsqueeze(0).float().to(device)
    o = model(x=t); s = F.interpolate(o[1], (FULL_RES, FULL_RES), mode='bilinear', align_corners=False)
    return torch.sigmoid(s[:, 0, :, :])[0].cpu().numpy()

def predict_tta4(model, img, pm, ps, device):
    p0 = predict_one(model, img, pm, ps, device)
    p1 = np.flip(predict_one(model, np.flip(img, 1).copy(), pm, ps, device), 1).copy()
    p2 = np.flip(predict_one(model, np.flip(img, 0).copy(), pm, ps, device), 0).copy()
    p3 = np.flip(np.flip(predict_one(model, np.flip(np.flip(img, 1), 0).copy(), pm, ps, device), 0), 1).copy()
    return (p0 + p1 + p2 + p3) / 4.0

def predict_tta2(model, img, pm, ps, device):
    p0 = predict_one(model, img, pm, ps, device)
    p1 = np.flip(predict_one(model, np.flip(img, 1).copy(), pm, ps, device), 1).copy()
    return (p0 + p1) / 2.0


# ==========================================
# Z 軸後處理
# ==========================================
def postprocess_3d(prob_volume, sigma=0.0, zspan_min=0, threshold=0.5):
    """
    在 3D probability volume 上做後處理

    Args:
        prob_volume: (Z, H, W) float, 0~1 的機率值
        sigma: z 軸 Gaussian smoothing 的 sigma（0=不做）
        zspan_min: 只保留 z-span >= 這個值的 component（0=不過濾）
        threshold: 二值化閾值

    Returns:
        binary: (Z, H, W) float, 0/1
        info: dict with stats
    """
    info = {"sigma": sigma, "zspan_min": zspan_min, "threshold": threshold}

    # Step 1: Z 軸 Gaussian smoothing（在 probability 上）
    if sigma > 0:
        # 只在 z 軸（axis=0）做 smoothing
        smoothed = gaussian_filter1d(prob_volume, sigma=sigma, axis=0)
        # 確保值域還在 [0, 1]
        smoothed = np.clip(smoothed, 0, 1)
        info["smooth_diff"] = float(np.abs(smoothed - prob_volume).mean())
    else:
        smoothed = prob_volume

    # Step 2: 二值化
    binary = (smoothed > threshold).astype(np.float32)

    # Step 3: Z-span component filtering
    if zspan_min > 0 and binary.sum() > 0:
        labeled, n_comp = scipy_label(binary.astype(np.uint8))
        info["comp_before"] = n_comp

        kept = np.zeros_like(binary)
        n_kept = 0
        for c in range(1, n_comp + 1):
            comp_mask = (labeled == c)
            z_indices = np.where(comp_mask.any(axis=(1, 2)))[0]
            z_span = z_indices[-1] - z_indices[0] + 1 if len(z_indices) > 0 else 0
            if z_span >= zspan_min:
                kept[comp_mask] = 1.0
                n_kept += 1

        binary = kept
        info["comp_after"] = n_kept
    else:
        labeled, n_comp = scipy_label(binary.astype(np.uint8))
        info["comp_before"] = n_comp
        info["comp_after"] = n_comp

    return binary, info


# ==========================================
# 評估單一病人
# ==========================================
def eval_patient(model, pid, idir, mdir, pm, ps_norm, device, tta_mode, eval_size,
                 smooth_sigma, zspan_min, threshold):
    ifs = sorted([f for f in os.listdir(idir) if f.endswith(".png")])
    if not ifs: return None

    ag = {}
    for f in ifs:
        g = cv2.imread(os.path.join(idir, f), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        ag[int(f.replace(".png", "").rsplit("_", 1)[-1])] = g

    probs, gts = [], []
    for f in tqdm(ifs, desc=f"推論 {pid}"):
        idx = int(f.replace(".png", "").rsplit("_", 1)[-1])
        c = ag[idx]; p_ = ag.get(idx - 1, c); n_ = ag.get(idx + 1, c)
        img = np.stack([p_, c, n_], axis=-1)
        gt = np.load(os.path.join(mdir, f.replace(".png", ".npy"))).astype(np.float32)

        if tta_mode == "none":
            prob = predict_one(model, img, pm, ps_norm, device)
        elif tta_mode == "2":
            prob = predict_tta2(model, img, pm, ps_norm, device)
        else:
            prob = predict_tta4(model, img, pm, ps_norm, device)

        probs.append(cv2.resize(prob, (eval_size, eval_size), interpolation=cv2.INTER_LINEAR))
        gts.append((cv2.resize(gt, (eval_size, eval_size), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8))

    torch.cuda.empty_cache(); del ag

    pv = np.stack(probs, 0)  # (Z, H, W) probability
    gv = np.stack(gts, 0).astype(np.float32)

    # === 無後處理的 baseline ===
    pb_raw = (pv > 0.5).astype(np.float32)
    dice_raw = compute_dice(pb_raw, gv)

    # === 後處理 ===
    pb, pp_info = postprocess_3d(pv, sigma=smooth_sigma, zspan_min=zspan_min, threshold=threshold)

    # 指標
    dice = compute_dice(pb, gv)
    lb, nc = scipy_label(pb.astype(np.uint8))
    bp = count_bp(pb, gv)

    print(f"  clDice...", end="", flush=True)
    sk = min(SKEL_SIZE, eval_size)
    ps2 = np.zeros((pb.shape[0], sk, sk), dtype=np.uint8)
    gs2 = np.zeros((gv.shape[0], sk, sk), dtype=np.uint8)
    for z in range(pb.shape[0]):
        ps2[z] = cv2.resize(pb[z], (sk, sk), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        gs2[z] = cv2.resize(gv[z], (sk, sk), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    cld = compute_cldice(ps2, gs2); print(f" {cld:.4f}"); del ps2, gs2

    print(f"  HD95...", end="", flush=True)
    hd = compute_hd95(pb, gv); print(f" {hd:.2f}")

    del pv, gv, pb, pb_raw; gc.collect()

    return {
        "p": pid, "dice": dice, "dice_raw": dice_raw,
        "cldice": cld, "hd95": hd, "comp": nc, "bp": bp,
        "comp_before": pp_info.get("comp_before", nc),
    }


# ==========================================
# 主程式
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--eval-dir", default=PUBLIC_EVAL_DIR)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--tta2", action="store_true")
    parser.add_argument("--eval-size", type=int, default=EVAL_SIZE)
    parser.add_argument("--smooth", type=float, default=0.0,
                        help="Z 軸 Gaussian smoothing sigma（0=不做）")
    parser.add_argument("--zspan", type=int, default=0,
                        help="只保留 z-span >= N 的 component（0=不過濾）")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="二值化閾值")
    parser.add_argument("--sweep", action="store_true",
                        help="自動掃一組 sigma + zspan 組合")
    args = parser.parse_args()

    # 載入模型
    device = "cuda"
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')
    model = CoSegV6(build_sam2("sam2_hiera_l.yaml", SAM2_CHECKPOINT, mode="train")).to(device)
    sd = torch.load(args.weights, map_location=device)
    model.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}, strict=True)
    model.eval(); del sd; gc.collect()

    pm = np.array([123.675, 116.280, 103.530], dtype=np.float32)
    ps_norm = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    tta_mode = "none" if args.no_tta else ("2" if args.tta2 else "4")
    tta_label = {"none": "No TTA", "2": "2-view", "4": "4-view"}[tta_mode]
    weight_name = os.path.basename(args.weights).replace(".pth", "")

    # 掃描病人
    patients = sorted([p for p in os.listdir(args.eval_dir)
                       if os.path.isdir(os.path.join(args.eval_dir, p, "image_1024"))])

    if not patients:
        print(f"❌ 找不到病人: {args.eval_dir}")
        return

    # === SWEEP 模式 ===
    if args.sweep:
        print("=" * 70)
        print(f"🔍 Sweep 模式 — {weight_name} ({tta_label} TTA)")
        print(f"   Patients: {', '.join(patients)}")
        print("=" * 70)

        # 先跑一次推論，存 probability volumes
        print("\n📋 推論中（只做一次）...")
        patient_data = {}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for pid in patients:
                idir = os.path.join(args.eval_dir, pid, "image_1024")
                mdir = os.path.join(args.eval_dir, pid, "mask_sem_1024")
                ifs = sorted([f for f in os.listdir(idir) if f.endswith(".png")])

                ag = {}
                for f in ifs:
                    g = cv2.imread(os.path.join(idir, f), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                    ag[int(f.replace(".png", "").rsplit("_", 1)[-1])] = g

                probs, gts = [], []
                for f in tqdm(ifs, desc=f"推論 {pid}"):
                    idx = int(f.replace(".png", "").rsplit("_", 1)[-1])
                    c = ag[idx]; p_ = ag.get(idx - 1, c); n_ = ag.get(idx + 1, c)
                    img = np.stack([p_, c, n_], axis=-1)
                    gt = np.load(os.path.join(mdir, f.replace(".png", ".npy"))).astype(np.float32)

                    if tta_mode == "none":
                        prob = predict_one(model, img, pm, ps_norm, device)
                    elif tta_mode == "2":
                        prob = predict_tta2(model, img, pm, ps_norm, device)
                    else:
                        prob = predict_tta4(model, img, pm, ps_norm, device)

                    probs.append(cv2.resize(prob, (args.eval_size, args.eval_size), interpolation=cv2.INTER_LINEAR))
                    gts.append((cv2.resize(gt, (args.eval_size, args.eval_size), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8))

                torch.cuda.empty_cache(); del ag
                patient_data[pid] = {
                    "probs": np.stack(probs, 0),
                    "gts": np.stack(gts, 0).astype(np.float32),
                }

        # 掃參數組合
        sigmas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        zspans = [0, 5, 10, 15, 20]
        thresholds = [0.5]

        print(f"\n{'='*70}")
        print(f"📊 Sweep 結果")
        print(f"{'='*70}")
        print(f"{'sigma':>6s} {'zspan':>6s} {'thr':>5s} | {'Dice':>8s} {'clDice':>8s} {'HD95':>8s} {'Comp':>6s} {'BP':>5s} | {'vs raw':>7s}")
        print(f"{'-'*75}")

        best_dice = 0
        best_params = None

        for sigma in sigmas:
            for zspan in zspans:
                for thr in thresholds:
                    all_dice, all_cld, all_hd, all_comp, all_bp, all_raw = [], [], [], [], [], []

                    for pid, data in patient_data.items():
                        pv, gv = data["probs"], data["gts"]

                        raw_dice = compute_dice((pv > 0.5).astype(np.float32), gv)
                        all_raw.append(raw_dice)

                        pb, info = postprocess_3d(pv, sigma=sigma, zspan_min=zspan, threshold=thr)
                        all_dice.append(compute_dice(pb, gv))

                        lb, nc = scipy_label(pb.astype(np.uint8))
                        all_comp.append(nc)
                        all_bp.append(count_bp(pb, gv))

                        # clDice（快速版，小尺寸）
                        sk = min(SKEL_SIZE, args.eval_size)
                        ps2 = np.zeros((pb.shape[0], sk, sk), dtype=np.uint8)
                        gs2 = np.zeros((gv.shape[0], sk, sk), dtype=np.uint8)
                        for z in range(pb.shape[0]):
                            ps2[z] = cv2.resize(pb[z], (sk, sk), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                            gs2[z] = cv2.resize(gv[z], (sk, sk), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                        all_cld.append(compute_cldice(ps2, gs2))

                        all_hd.append(compute_hd95(pb, gv))

                    md = np.mean(all_dice)
                    mc = np.mean(all_cld)
                    mh = np.mean([h for h in all_hd if h != float('inf')]) if any(h != float('inf') for h in all_hd) else float('inf')
                    mr = np.mean(all_raw)
                    diff = md - mr

                    marker = " ⭐" if md > best_dice else ""
                    print(f"{sigma:>6.1f} {zspan:>6d} {thr:>5.2f} | {md:>8.4f} {mc:>8.4f} {mh:>8.2f} {np.mean(all_comp):>6.1f} {np.mean(all_bp):>5.1f} | {diff:>+7.4f}{marker}")

                    if md > best_dice:
                        best_dice = md
                        best_params = (sigma, zspan, thr)

        print(f"\n{'='*70}")
        print(f"🏆 最佳: sigma={best_params[0]}, zspan={best_params[1]}, threshold={best_params[2]}")
        print(f"   Dice: {best_dice:.4f} (raw: {mr:.4f}, 提升: {best_dice - mr:+.4f})")
        print(f"{'='*70}")

        # 用最佳參數印 per-patient
        print(f"\n📊 最佳參數 per-patient:")
        for pid, data in patient_data.items():
            pv, gv = data["probs"], data["gts"]
            raw_dice = compute_dice((pv > 0.5).astype(np.float32), gv)
            pb, _ = postprocess_3d(pv, sigma=best_params[0], zspan_min=best_params[1], threshold=best_params[2])
            pp_dice = compute_dice(pb, gv)
            lb, nc = scipy_label(pb.astype(np.uint8))
            bp = count_bp(pb, gv)
            print(f"  {pid}: Dice {raw_dice:.4f} → {pp_dice:.4f} ({pp_dice - raw_dice:+.4f}) Comp={nc} BP={bp}")

        del patient_data; gc.collect()
        return

    # === 單一參數模式 ===
    pp_label = ""
    if args.smooth > 0 or args.zspan > 0:
        parts = []
        if args.smooth > 0: parts.append(f"smooth={args.smooth}")
        if args.zspan > 0: parts.append(f"zspan≥{args.zspan}")
        pp_label = f" + {' + '.join(parts)}"

    print("=" * 60)
    print(f"📊 eval — {weight_name} ({tta_label} TTA{pp_label})")
    print(f"   Patients: {', '.join(patients)}")
    print("=" * 60)

    results = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for pid in patients:
            idir = os.path.join(args.eval_dir, pid, "image_1024")
            mdir = os.path.join(args.eval_dir, pid, "mask_sem_1024")

            r = eval_patient(model, pid, idir, mdir, pm, ps_norm, device,
                             tta_mode, args.eval_size, args.smooth, args.zspan, args.threshold)
            if r is None: continue
            results.append(r)

            delta = r['dice'] - r['dice_raw']
            print(f"\n  📊 {pid}: Dice={r['dice']:.4f} (raw={r['dice_raw']:.4f}, {delta:+.4f}) "
                  f"clDice={r['cldice']:.4f} HD95={r['hd95']:.2f} "
                  f"Comp={r['comp_before']}→{r['comp']} BP={r['bp']}")

    # 總結
    print(f"\n{'='*60}")
    print(f"📊 總結 [{weight_name}] ({tta_label} TTA{pp_label})")
    print(f"{'='*60}")
    ds = [r['dice'] for r in results]; dr = [r['dice_raw'] for r in results]
    cs = [r['cldice'] for r in results]
    hs = [r['hd95'] for r in results if r['hd95'] != float('inf')]
    print(f"  {'指標':<12s} {'Mean':>8s} {'Std':>8s} {'vs raw':>8s}")
    print(f"  {'-'*40}")
    print(f"  {'Dice':<12s} {np.mean(ds):>8.4f} {np.std(ds):>8.4f} {np.mean(ds)-np.mean(dr):>+8.4f}")
    print(f"  {'Dice(raw)':<12s} {np.mean(dr):>8.4f} {np.std(dr):>8.4f}")
    print(f"  {'clDice':<12s} {np.mean(cs):>8.4f} {np.std(cs):>8.4f}")
    if hs: print(f"  {'HD95':<12s} {np.mean(hs):>8.2f} {np.std(hs):>8.2f}")
    print(f"  {'Components':<12s} {np.mean([r['comp'] for r in results]):>8.1f}")
    print(f"  {'Breakpoints':<12s} {np.mean([r['bp'] for r in results]):>8.1f}")
    print(f"\n  Per-patient:")
    for r in results:
        d = r['dice'] - r['dice_raw']
        print(f"    {r['p']}: Dice={r['dice']:.4f}({d:+.4f}) clDice={r['cldice']:.4f} HD95={r['hd95']:.2f} Comp={r['comp']} BP={r['bp']}")
    print(f"\n  📋 比較:")
    print(f"     v7:         Dice=0.7874  clDice=0.8287  HD95=1.91")
    print(f"     v8舊:       Dice=0.7861  clDice=0.8319  HD95=1.91")
    print(f"     DentalSeg:  Dice=0.7589  clDice=0.8577  HD95=2.16")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
