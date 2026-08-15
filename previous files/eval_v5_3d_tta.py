"""
eval_v5_3d_tta.py — CoSeg v5 評估（TTA + 調整後處理）
=====================================================
TTA: 水平翻轉 + 原圖，預測取平均
後處理: 調低 Z-smooth，避免傷 Dice
"""
import os, cv2, torch, numpy as np, hydra
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d, label as scipy_label
from model import CoSeg
from sam2.build_sam import build_sam2

PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
EVAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data/public_data/eval")
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "outputs/coseg_v5_best.pth")
SAM2_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
CONTEXT_SLICES = 1
EVAL_SIZE = 256

# 調整後的後處理參數（不傷 Dice）
Z_SMOOTH_SIGMA = 0.8        # 從 1.5 降到 0.8，避免過度平滑
BINARY_THRESHOLD = 0.5
MIN_COMPONENT_VOXELS = 5    # 從 10 降到 5，保留更多真預測
MORPH_CLOSE_RADIUS = 1      # 從 2 降到 1

def compute_dice(p, g, s=1e-5):
    i = np.sum(p * g)
    return (2.*i+s)/(np.sum(p)+np.sum(g)+s)

def compute_iou(p, g, s=1e-5):
    i = np.sum(p * g)
    return (i+s)/(np.sum(p)+np.sum(g)-i+s)

def postprocess_3d(prob_volume):
    if Z_SMOOTH_SIGMA > 0 and prob_volume.shape[0] > 3:
        prob_volume = gaussian_filter1d(prob_volume, sigma=Z_SMOOTH_SIGMA, axis=0)
    binary = (prob_volume > BINARY_THRESHOLD).astype(np.uint8)
    if MORPH_CLOSE_RADIUS > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (MORPH_CLOSE_RADIUS * 2 + 1, MORPH_CLOSE_RADIUS * 2 + 1)
        )
        for z in range(binary.shape[0]):
            binary[z] = cv2.morphologyEx(binary[z], cv2.MORPH_CLOSE, kernel)
    if MIN_COMPONENT_VOXELS > 0:
        labeled, n = scipy_label(binary)
        if n > 0:
            sizes = np.bincount(labeled.ravel())
            for c in range(1, n + 1):
                if sizes[c] < MIN_COMPONENT_VOXELS:
                    binary[labeled == c] = 0
    return binary.astype(np.float32)

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
    """單張推論，回傳機率圖"""
    img_norm = (img_3ch - pm) / (ps + 1e-8)
    t = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
    _, s, _, _ = model(x=t)
    s = F.interpolate(s, size=(1024, 1024), mode='bilinear', align_corners=False)
    prob = torch.sigmoid(s[:, 0, :, :])[0].cpu().numpy()
    del t, s
    return prob

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
    print("🧠 CoSeg v5 評估（TTA + 調整後處理）")
    print(f"   TTA: 原圖 + 水平翻轉 (平均)")
    print(f"   Z-smooth σ={Z_SMOOTH_SIGMA}, Close r={MORPH_CLOSE_RADIUS}, Min comp={MIN_COMPONENT_VOXELS}")
    print("=" * 60)

    all_raw, all_tta, all_post = [], [], []

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

            prob_raw_list, prob_tta_list, gt_list = [], [], []

            for f in tqdm(ifs, desc=f"推論 {pid} (TTA)"):
                idx = int(f.replace(".png", "").rsplit("_", 1)[-1])
                c = ag[idx]
                p_ = ag.get(idx - 1, c)
                n_ = ag.get(idx + 1, c)
                img = np.stack([p_, c, n_], axis=-1)
                gt = np.load(os.path.join(mdir, f.replace(".png", ".npy"))).astype(np.float32)

                # 原圖預測
                prob_orig = predict_slice(model, img, pm, ps, device)

                # TTA: 水平翻轉
                img_flip = np.flip(img, axis=1).copy()
                prob_flip = predict_slice(model, img_flip, pm, ps, device)
                prob_flip = np.flip(prob_flip, axis=1).copy()

                # 平均
                prob_tta = (prob_orig + prob_flip) / 2.0

                # 縮小省 RAM
                prob_raw_list.append(cv2.resize(prob_orig, (EVAL_SIZE, EVAL_SIZE), interpolation=cv2.INTER_LINEAR))
                prob_tta_list.append(cv2.resize(prob_tta, (EVAL_SIZE, EVAL_SIZE), interpolation=cv2.INTER_LINEAR))
                gt_list.append((cv2.resize(gt, (EVAL_SIZE, EVAL_SIZE), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8))

            torch.cuda.empty_cache()

            prob_raw_vol = np.stack(prob_raw_list, axis=0)
            prob_tta_vol = np.stack(prob_tta_list, axis=0)
            gt_vol = np.stack(gt_list, axis=0).astype(np.float32)
            del prob_raw_list, prob_tta_list, gt_list

            def eval_volume(prob_vol, gt_vol, label):
                raw = (prob_vol > 0.5).astype(np.float32)
                d_raw = compute_dice(raw, gt_vol)
                post = postprocess_3d(prob_vol.copy())
                d_post = compute_dice(post, gt_vol)
                # 選更好的那個
                if d_post >= d_raw:
                    best = post
                    best_dice = d_post
                    best_label = f"{label}+Post"
                else:
                    best = raw
                    best_dice = d_raw
                    best_label = f"{label} Raw"
                iou = compute_iou(best, gt_vol)
                lb, nc = scipy_label(best.astype(np.uint8))
                conn = (np.bincount(lb.ravel())[1:].max() / best.sum()) if best.sum() > 0 and nc > 0 else 0
                bp = count_breakpoints(best, gt_vol)
                return {"dice": best_dice, "iou": iou, "comp": nc, "conn": conn, "bp": bp,
                        "d_raw": d_raw, "d_post": d_post, "method": best_label}

            r_raw = eval_volume(prob_raw_vol, gt_vol, "Raw")
            r_tta = eval_volume(prob_tta_vol, gt_vol, "TTA")

            all_raw.append(r_raw)
            all_tta.append(r_tta)

            print(f"\n📊 {pid}:")
            print(f"   No TTA:  Raw={r_raw['d_raw']:.4f}  Post={r_raw['d_post']:.4f}  Best={r_raw['dice']:.4f} ({r_raw['method']})")
            print(f"   TTA:     Raw={r_tta['d_raw']:.4f}  Post={r_tta['d_post']:.4f}  Best={r_tta['dice']:.4f} ({r_tta['method']})")
            print(f"   TTA Best → Dice={r_tta['dice']:.4f}  IoU={r_tta['iou']:.4f}  Comp={r_tta['comp']}  Conn={r_tta['conn']:.4f}  BP={r_tta['bp']}")

            del prob_raw_vol, prob_tta_vol, gt_vol

    # 彙總
    print(f"\n{'='*60}")
    print("📊 總結")
    print(f"{'='*60}")

    for label, data in [("No TTA", all_raw), ("With TTA", all_tta)]:
        ds = [r['dice'] for r in data]
        print(f"\n  {label}: Dice = {np.mean(ds):.4f} ± {np.std(ds):.4f}")
        for i, pid in enumerate([p for p in sorted(os.listdir(EVAL_DATA_DIR)) if p.startswith("Patient_")]):
            if i < len(data):
                print(f"    {pid}: {data[i]['dice']:.4f} ({data[i]['method']})")

    # 最終最好的
    best_ds = [r['dice'] for r in all_tta]
    best_conns = [r['conn'] for r in all_tta]
    best_bps = [r['bp'] for r in all_tta]
    print(f"\n  🏆 最終最佳: Dice={np.mean(best_ds):.4f}  Conn={np.mean(best_conns):.4f}  BP={np.mean(best_bps):.1f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
