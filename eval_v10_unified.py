"""
eval_v10_unified.py — V10 統一評估（含 Test-Time BN Adaptation + 3D 後處理）
=============================================================================
策略 7: Test-Time BN Adaptation
  推論前先用該病人的全部 slice 跑一遍 forward（不更新權重），
  只更新 Domain-Specific BN 的 running mean/var → 穩定 1-3% 提升

用法：
  # 公開 test
  python eval_v10_unified.py --weights outputs/coseg_v10_best.pth --domain public

  # 醫院 test
  python eval_v10_unified.py --weights outputs/coseg_v10_best.pth --domain hospital

  # 關掉 TT-BN
  python eval_v10_unified.py --weights outputs/coseg_v10_best.pth --domain public --no-ttbn

  # 掃後處理參數
  python eval_v10_unified.py --weights outputs/coseg_v10_best.pth --domain public --sweep
"""
import os, cv2, torch, gc, argparse, copy
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import (distance_transform_edt, binary_erosion,
                            label as scipy_label, gaussian_filter1d)
import hydra
from model_v10 import CoSegV10, DomainBatchNorm2d
from sam2.build_sam import build_sam2

PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
PUBLIC_EVAL_DIR = os.path.join(PROJECT_ROOT, "data/public_data/eval")
HOSPITAL_EVAL_DIR = os.path.join(PROJECT_ROOT, "data/hospital_data/eval")
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "outputs/coseg_v10_best.pth")
SAM2_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
EVAL_SIZE, SKEL_SIZE, FULL_RES = 256, 128, 1024


# ==========================================
# 指標
# ==========================================
def compute_dice(p, g, s=1e-5):
    i = np.sum(p * g)
    return (2 * i + s) / (np.sum(p) + np.sum(g) + s)

def skeletonize_3d_safe(vol):
    try:
        from skimage.morphology import skeletonize_3d
        return skeletonize_3d(vol.astype(np.uint8)).astype(np.float32)
    except Exception:
        from skimage.morphology import skeletonize
        s = np.zeros_like(vol, dtype=np.float32)
        for z in range(vol.shape[0]):
            if vol[z].sum() > 0:
                s[z] = skeletonize(vol[z].astype(bool)).astype(np.float32)
        return s

def compute_cldice(pred, gt, smooth=1e-5):
    pb = (pred > 0).astype(np.uint8)
    gb = (gt > 0).astype(np.uint8)
    if pb.sum() == 0 and gb.sum() == 0: return 1.0
    if pb.sum() == 0 or gb.sum() == 0: return 0.0
    sp, sg = skeletonize_3d_safe(pb), skeletonize_3d_safe(gb)
    if sp.sum() == 0 or sg.sum() == 0: return 0.0
    tp = (sp * gb).sum() / (sp.sum() + smooth)
    ts = (sg * pb).sum() / (sg.sum() + smooth)
    r = float(2 * tp * ts / (tp + ts + smooth))
    del sp, sg; gc.collect()
    return r

def compute_hd95(pred, gt):
    pb = (pred > 0).astype(np.uint8)
    gb = (gt > 0).astype(np.uint8)
    if pb.sum() == 0 or gb.sum() == 0: return float('inf')
    ps = pb ^ binary_erosion(pb).astype(np.uint8)
    gs = gb ^ binary_erosion(gb).astype(np.uint8)
    if ps.sum() == 0 or gs.sum() == 0: return float('inf')
    dp = distance_transform_edt(~pb.astype(bool))
    dg = distance_transform_edt(~gb.astype(bool))
    ad = np.concatenate([dg[ps > 0], dp[gs > 0]])
    del dp, dg; gc.collect()
    return float(np.percentile(ad, 95))


# ==========================================
# Test-Time BN Adaptation
# ==========================================
def test_time_bn_adapt(model, all_images, domain, device, num_passes=1):
    """
    用該病人的全部 slice 更新 Domain-Specific BN 的 running stats。
    不更新任何權重，只更新 BN 的 running_mean / running_var。

    Args:
        model: CoSegV10
        all_images: list of (3, H, W) tensors — 該病人所有 slice
        domain: 0 or 1
        device: cuda
        num_passes: 跑幾遍（1 通常就夠）
    """
    # 保存原始 BN 狀態
    bn_states = {}
    for name, m in model.named_modules():
        if isinstance(m, DomainBatchNorm2d):
            bn = m.bns[domain]
            bn_states[name] = {
                'running_mean': bn.running_mean.clone(),
                'running_var': bn.running_var.clone(),
                'num_batches_tracked': bn.num_batches_tracked.clone(),
            }
            # 重置 running stats
            bn.running_mean.zero_()
            bn.running_var.fill_(1.0)
            bn.num_batches_tracked.zero_()
            bn.momentum = 0.1  # 用較大 momentum 快速適應

    model.set_domain(domain)
    model.train()  # BN 在 train 模式才更新 running stats

    # 但凍結所有參數（不要 backward）
    with torch.no_grad():
        for _ in range(num_passes):
            # 小 batch 跑所有 slices
            batch_size = 8
            for i in range(0, len(all_images), batch_size):
                batch = torch.stack(all_images[i:i+batch_size]).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _ = model(x=batch)
                del batch
                torch.cuda.empty_cache()

    model.eval()


def restore_bn_states(model, bn_states, domain):
    """恢復 BN 到 adaptation 前的狀態"""
    for name, m in model.named_modules():
        if isinstance(m, DomainBatchNorm2d) and name in bn_states:
            bn = m.bns[domain]
            bn.running_mean.copy_(bn_states[name]['running_mean'])
            bn.running_var.copy_(bn_states[name]['running_var'])
            bn.num_batches_tracked.copy_(bn_states[name]['num_batches_tracked'])


# ==========================================
# 推論
# ==========================================
def predict_one(model, img_3ch, pm, ps, device):
    t = torch.tensor((img_3ch - pm) / (ps + 1e-8)).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        o = model(x=t)
        s = F.interpolate(o[1], (FULL_RES, FULL_RES), mode='bilinear', align_corners=False)
    return torch.sigmoid(s[:, 0, :, :])[0].cpu().numpy()

def predict_tta4(model, img, pm, ps, device):
    p0 = predict_one(model, img, pm, ps, device)
    p1 = np.flip(predict_one(model, np.flip(img, 1).copy(), pm, ps, device), 1).copy()
    p2 = np.flip(predict_one(model, np.flip(img, 0).copy(), pm, ps, device), 0).copy()
    p3 = np.flip(np.flip(predict_one(model, np.flip(np.flip(img, 1), 0).copy(),
                                      pm, ps, device), 0), 1).copy()
    return (p0 + p1 + p2 + p3) / 4.0


# ==========================================
# 3D 後處理
# ==========================================
def postprocess_3d(prob_volume, sigma=0.0, zspan_min=0, threshold=0.5):
    if sigma > 0:
        smoothed = np.clip(gaussian_filter1d(prob_volume, sigma=sigma, axis=0), 0, 1)
    else:
        smoothed = prob_volume
    binary = (smoothed > threshold).astype(np.float32)
    if zspan_min > 0 and binary.sum() > 0:
        labeled, n_comp = scipy_label(binary.astype(np.uint8))
        kept = np.zeros_like(binary)
        for c in range(1, n_comp + 1):
            z_indices = np.where((labeled == c).any(axis=(1, 2)))[0]
            if len(z_indices) > 0 and (z_indices[-1] - z_indices[0] + 1) >= zspan_min:
                kept[labeled == c] = 1.0
        binary = kept
    return binary


# ==========================================
# 評估單一病人
# ==========================================
def eval_patient(model, pid, idir, mdir, pm, ps_norm, device,
                 use_tta, eval_size, sigma, zspan, threshold,
                 use_ttbn, domain):
    ifs = sorted([f for f in os.listdir(idir) if f.endswith(".png")])
    if not ifs:
        return None

    # 預載灰度圖
    ag = {}
    for f in ifs:
        g = cv2.imread(os.path.join(idir, f), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        ag[int(f.replace(".png", "").rsplit("_", 1)[-1])] = g

    # Test-Time BN Adaptation
    if use_ttbn:
        print(f"  🔄 TT-BN adaptation ({len(ifs)} slices)...")
        all_imgs_for_ttbn = []
        for f in ifs:
            idx = int(f.replace(".png", "").rsplit("_", 1)[-1])
            c = ag[idx]
            p_ = ag.get(idx - 1, c)
            n_ = ag.get(idx + 1, c)
            img = np.stack([p_, c, n_], axis=-1)
            img_norm = (img - pm) / (ps_norm + 1e-8)
            all_imgs_for_ttbn.append(torch.tensor(img_norm).permute(2, 0, 1).float())
        test_time_bn_adapt(model, all_imgs_for_ttbn, domain, device)
        del all_imgs_for_ttbn
        torch.cuda.empty_cache()

    model.set_domain(domain)
    model.eval()

    probs, gts = [], []
    for f in tqdm(ifs, desc=f"推論 {pid}"):
        idx = int(f.replace(".png", "").rsplit("_", 1)[-1])
        c = ag[idx]
        p_ = ag.get(idx - 1, c)
        n_ = ag.get(idx + 1, c)
        img = np.stack([p_ if p_ is not None else c, c,
                        n_ if n_ is not None else c], axis=-1)

        mask_path = os.path.join(mdir, f.replace(".png", ".npy"))
        if not os.path.exists(mask_path):
            continue
        gt = np.load(mask_path).astype(np.float32)

        if use_tta:
            prob = predict_tta4(model, img, pm, ps_norm, device)
        else:
            prob = predict_one(model, img, pm, ps_norm, device)

        probs.append(cv2.resize(prob, (eval_size, eval_size), interpolation=cv2.INTER_LINEAR))
        gts.append((cv2.resize(gt, (eval_size, eval_size),
                                interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8))

    torch.cuda.empty_cache()
    del ag

    if not probs:
        return None

    pv = np.stack(probs, 0)
    gv = np.stack(gts, 0).astype(np.float32)

    # Raw dice
    dice_raw = compute_dice((pv > 0.5).astype(np.float32), gv)

    # 後處理
    pb = postprocess_3d(pv, sigma=sigma, zspan_min=zspan, threshold=threshold)
    dice = compute_dice(pb, gv)

    # clDice
    sk = min(SKEL_SIZE, eval_size)
    ps2 = np.zeros((pb.shape[0], sk, sk), dtype=np.uint8)
    gs2 = np.zeros((gv.shape[0], sk, sk), dtype=np.uint8)
    for z in range(pb.shape[0]):
        ps2[z] = cv2.resize(pb[z], (sk, sk), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        gs2[z] = cv2.resize(gv[z], (sk, sk), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    cld = compute_cldice(ps2, gs2)
    del ps2, gs2

    hd = compute_hd95(pb, gv)

    lb, nc = scipy_label(pb.astype(np.uint8))

    return {
        'p': pid, 'dice': dice, 'dice_raw': dice_raw,
        'cldice': cld, 'hd95': hd, 'comp': nc,
    }


# ==========================================
# 主程式
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--domain", choices=["public", "hospital", "both"], default="both")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--no-ttbn", action="store_true", help="關掉 Test-Time BN Adapt")
    parser.add_argument("--eval-size", type=int, default=EVAL_SIZE)
    parser.add_argument("--smooth", type=float, default=0.0)
    parser.add_argument("--zspan", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sweep", action="store_true")
    # Model config（要跟訓練時一致）
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--no-domain-bn", action="store_true")
    parser.add_argument("--no-cross-slice", action="store_true")
    args = parser.parse_args()

    device = "cuda"
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')

    use_lora = not args.no_lora
    use_domain_bn = not args.no_domain_bn
    use_cross_slice = not args.no_cross_slice

    print(f"📦 載入模型: {args.weights}")
    model = CoSegV10(
        build_sam2("sam2_hiera_l.yaml", SAM2_CHECKPOINT, mode="train"),
        use_lora=use_lora,
        use_domain_bn=use_domain_bn,
        use_cross_slice=use_cross_slice,
    )

    sd = torch.load(args.weights, map_location=device)
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    model.load_state_dict(
        {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()},
        strict=True)
    model = model.to(device)
    model.eval()
    del sd; gc.collect()

    pm = np.array([123.675, 116.280, 103.530], dtype=np.float32)
    ps_norm = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    use_tta = not args.no_tta
    use_ttbn = use_domain_bn and not args.no_ttbn
    weight_name = os.path.basename(args.weights).replace(".pth", "")

    eval_configs = []
    if args.domain in ("public", "both"):
        eval_configs.append(("public", PUBLIC_EVAL_DIR, 0))
    if args.domain in ("hospital", "both"):
        eval_configs.append(("hospital", HOSPITAL_EVAL_DIR, 1))

    for domain_name, eval_dir, domain_id in eval_configs:
        patients = sorted([p for p in os.listdir(eval_dir)
                           if os.path.isdir(os.path.join(eval_dir, p, "image_1024"))])
        if not patients:
            print(f"❌ 找不到病人: {eval_dir}")
            continue

        tta_label = "4-view TTA" if use_tta else "No TTA"
        ttbn_label = " + TT-BN" if use_ttbn else ""

        print(f"\n{'='*60}")
        print(f"📊 {domain_name.upper()} eval — {weight_name} ({tta_label}{ttbn_label})")
        print(f"   Patients: {', '.join(patients)}")
        print(f"{'='*60}")

        results = []
        with torch.no_grad():
            for pid in patients:
                idir = os.path.join(eval_dir, pid, "image_1024")
                mdir = os.path.join(eval_dir, pid, "mask_sem_1024")

                # TT-BN 會改 model 狀態，每個病人前先 deep copy BN states
                if use_ttbn:
                    # 保存原始 BN
                    bn_backup = {}
                    for name, m in model.named_modules():
                        if isinstance(m, DomainBatchNorm2d):
                            bn = m.bns[domain_id]
                            bn_backup[name] = {
                                'rm': bn.running_mean.clone(),
                                'rv': bn.running_var.clone(),
                                'nb': bn.num_batches_tracked.clone(),
                            }

                r = eval_patient(model, pid, idir, mdir, pm, ps_norm, device,
                                 use_tta, args.eval_size, args.smooth, args.zspan,
                                 args.threshold, use_ttbn, domain_id)

                # 恢復 BN
                if use_ttbn:
                    for name, m in model.named_modules():
                        if isinstance(m, DomainBatchNorm2d) and name in bn_backup:
                            bn = m.bns[domain_id]
                            bn.running_mean.copy_(bn_backup[name]['rm'])
                            bn.running_var.copy_(bn_backup[name]['rv'])
                            bn.num_batches_tracked.copy_(bn_backup[name]['nb'])

                if r is None:
                    continue
                results.append(r)
                delta = r['dice'] - r['dice_raw']
                print(f"\n  📊 {pid}: Dice={r['dice']:.4f} (raw={r['dice_raw']:.4f}, {delta:+.4f}) "
                      f"clDice={r['cldice']:.4f} HD95={r['hd95']:.2f} Comp={r['comp']}")

        if not results:
            print("❌ 沒有結果")
            continue

        # 總結
        ds = [r['dice'] for r in results]
        cs = [r['cldice'] for r in results]
        hs = [r['hd95'] for r in results if r['hd95'] != float('inf')]

        print(f"\n{'='*60}")
        print(f"📊 {domain_name.upper()} 總結 [{weight_name}] ({tta_label}{ttbn_label})")
        print(f"{'='*60}")
        print(f"  {'指標':<12s} {'Mean':>8s} {'Std':>8s}")
        print(f"  {'-'*30}")
        print(f"  {'Dice':<12s} {np.mean(ds):>8.4f} {np.std(ds):>8.4f}")
        print(f"  {'clDice':<12s} {np.mean(cs):>8.4f} {np.std(cs):>8.4f}")
        if hs:
            print(f"  {'HD95':<12s} {np.mean(hs):>8.2f} {np.std(hs):>8.2f}")

        print(f"\n  Per-patient:")
        for r in results:
            d = r['dice'] - r['dice_raw']
            print(f"    {r['p']}: Dice={r['dice']:.4f}({d:+.4f}) "
                  f"clDice={r['cldice']:.4f} HD95={r['hd95']:.2f} Comp={r['comp']}")

        if domain_name == "public":
            print(f"\n  📋 比較:")
            print(f"     v7:         Dice=0.7874  clDice=0.8287  HD95=1.91")
            print(f"     DentalSeg:  Dice=0.7589  clDice=0.8577  HD95=2.16")
        elif domain_name == "hospital":
            print(f"\n  📋 比較:")
            print(f"     hospital best: Dice=0.6518  clDice=0.6794  HD95=2.20")
            print(f"     ⚠️  醫院 label 是 0.75mm sphere 膨脹，指標只能跟自己比")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
