"""
eval_v6.py — CoSeg v6 評估（TTA + 全指標）
=============================================
用法：
  python eval_v6.py
  python eval_v6.py --weights outputs/coseg_v6_best.pth --no-tta
"""
import os, cv2, torch, gc, argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, binary_erosion, label as scipy_label
import hydra

from model_v6 import CoSegV6
from sam2.build_sam import build_sam2

PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
EVAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data/public_data/eval")
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "outputs/coseg_v6_best.pth")
SAM2_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"

EVAL_SIZE = 256
SKEL_SIZE = 128
FULL_RES = 1024


def compute_dice(p, g, s=1e-5):
    i = np.sum(p * g)
    return (2. * i + s) / (np.sum(p) + np.sum(g) + s)

def compute_iou(p, g, s=1e-5):
    i = np.sum(p * g)
    return (i + s) / (np.sum(p) + np.sum(g) - i + s)

def skeletonize_3d_safe(volume):
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
    gz = gt.sum(axis=(1, 2))
    gr = np.where(gz > 0)[0]
    bp = 0
    if len(gr) > 0:
        pz = pred[gr[0]:gr[-1]+1].sum(axis=(1, 2)) > 0
        ic = False
        for i, h in enumerate(pz):
            if h and not ic:
                if i > 0: bp += 1
                ic = True
            elif not h and ic: ic = False
    return bp

def predict_slice(model, img_3ch, pm, ps, device):
    img_norm = (img_3ch - pm) / (ps + 1e-8)
    t = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
    outputs = model(x=t)
    s = outputs[1]  # semantic mask (index 1)
    s = F.interpolate(s, size=(FULL_RES, FULL_RES), mode='bilinear', align_corners=False)
    prob = torch.sigmoid(s[:, 0, :, :])[0].cpu().numpy()
    del t, s
    return prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    device = "cuda"
    use_tta = not args.no_tta

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')

    print(f"📦 載入 SAM2: {SAM2_CHECKPOINT}")
    model = CoSegV6(build_sam2("sam2_hiera_l.yaml", SAM2_CHECKPOINT, mode="train")).to(device)

    print(f"📦 載入 CoSegV6: {args.weights}")
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
    print(f"🧠 CoSeg v6 評估 (AG + DS)")
    print(f"   TTA: {'ON' if use_tta else 'OFF'}")
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

            ag = {}
            for f in ifs:
                g = cv2.imread(os.path.join(idir, f), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                ag[int(f.replace(".png", "").rsplit("_", 1)[-1])] = g

            prob_list, gt_list = [], []

            for f in tqdm(ifs, desc=f"推論 {pid}"):
                idx = int(f.replace(".png", "").rsplit("_", 1)[-1])
                c = ag[idx]
                p_ = ag.get(idx - 1, c)
                n_ = ag.get(idx + 1, c)
                img = np.stack([p_, c, n_], axis=-1)
                gt = np.load(os.path.join(mdir, f.replace(".png", ".npy"))).astype(np.float32)

                prob = predict_slice(model, img, pm, ps, device)
                if use_tta:
                    img_flip = np.flip(img, axis=1).copy()
                    prob_flip = predict_slice(model, img_flip, pm, ps, device)
                    prob_flip = np.flip(prob_flip, axis=1).copy()
                    prob = (prob + prob_flip) / 2.0

                prob_list.append(cv2.resize(prob, (EVAL_SIZE, EVAL_SIZE), interpolation=cv2.INTER_LINEAR))
                gt_list.append((cv2.resize(gt, (EVAL_SIZE, EVAL_SIZE), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8))

            torch.cuda.empty_cache()
            del ag

            prob_vol = np.stack(prob_list, axis=0)
            gt_vol = np.stack(gt_list, axis=0).astype(np.float32)
            del prob_list, gt_list

            pred_binary = (prob_vol > 0.5).astype(np.float32)

            dice = compute_dice(pred_binary, gt_vol)
            iou = compute_iou(pred_binary, gt_vol)

            lb, nc = scipy_label(pred_binary.astype(np.uint8))
            conn = (np.bincount(lb.ravel())[1:].max() / pred_binary.sum()) if pred_binary.sum() > 0 and nc > 0 else 0
            bp = count_breakpoints(pred_binary, gt_vol)

            # clDice
            print(f"  計算 clDice...", end="", flush=True)
            pred_skel = np.zeros((pred_binary.shape[0], SKEL_SIZE, SKEL_SIZE), dtype=np.uint8)
            gt_skel = np.zeros((gt_vol.shape[0], SKEL_SIZE, SKEL_SIZE), dtype=np.uint8)
            for z in range(pred_binary.shape[0]):
                pred_skel[z] = cv2.resize(pred_binary[z], (SKEL_SIZE, SKEL_SIZE), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                gt_skel[z] = cv2.resize(gt_vol[z], (SKEL_SIZE, SKEL_SIZE), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            cldice = compute_cldice(pred_skel, gt_skel)
            print(f" {cldice:.4f}")
            del pred_skel, gt_skel

            # HD95
            print(f"  計算 HD95...", end="", flush=True)
            hd95 = compute_hd95(pred_binary, gt_vol)
            print(f" {hd95:.2f}")

            result = {"patient": pid, "dice": dice, "iou": iou, "cldice": cldice,
                      "hd95": hd95, "comp": nc, "conn": conn, "bp": bp}
            all_results.append(result)

            print(f"\n  📊 {pid}: Dice={dice:.4f}  clDice={cldice:.4f}  HD95={hd95:.2f}  Comp={nc}  BP={bp}")

            del prob_vol, gt_vol, pred_binary
            gc.collect()

    # 彙總
    print(f"\n{'='*60}")
    print("📊 總結")
    print(f"{'='*60}")

    dices = [r['dice'] for r in all_results]
    cldices = [r['cldice'] for r in all_results]
    hd95s = [r['hd95'] for r in all_results if r['hd95'] != float('inf')]
    comps = [r['comp'] for r in all_results]
    bps = [r['bp'] for r in all_results]

    print(f"  {'指標':<12s} {'Mean':>8s} {'Std':>8s}")
    print(f"  {'-'*30}")
    print(f"  {'Dice':<12s} {np.mean(dices):>8.4f} {np.std(dices):>8.4f}")
    print(f"  {'clDice':<12s} {np.mean(cldices):>8.4f} {np.std(cldices):>8.4f}")
    if hd95s:
        print(f"  {'HD95':<12s} {np.mean(hd95s):>8.2f} {np.std(hd95s):>8.2f}")
    print(f"  {'Components':<12s} {np.mean(comps):>8.1f}")
    print(f"  {'Breakpoints':<12s} {np.mean(bps):>8.1f}")

    print(f"\n  Per-patient:")
    for r in all_results:
        print(f"    {r['patient']}: Dice={r['dice']:.4f}  clDice={r['cldice']:.4f}  HD95={r['hd95']:.2f}  Comp={r['comp']}  BP={r['bp']}")

    print(f"\n  📋 vs DentalSegmentator:")
    print(f"     DentalSeg:    Dice=0.7589  clDice=0.8577  HD95=2.16")
    print(f"     CoSeg v6:     Dice={np.mean(dices):.4f}  clDice={np.mean(cldices):.4f}  HD95={np.mean(hd95s):.2f}" if hd95s else "")
    print(f"     CoSeg v5 TTA: Dice=0.7608  clDice=0.7930  HD95=3.97")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
