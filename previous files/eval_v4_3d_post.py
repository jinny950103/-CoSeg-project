"""
eval_v4_3d_post.py — CoSeg v4 評估（含後處理）
"""
import os, cv2, torch, numpy as np, hydra
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d, label as scipy_label
from model import CoSeg
from sam2.build_sam import build_sam2

PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
EVAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data/public_data/eval")
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "outputs/coseg_v4_best.pth")
SAM2_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
CONTEXT_SLICES = 1
EVAL_SIZE = 256

# 後處理參數
Z_SMOOTH_SIGMA = 1.5
BINARY_THRESHOLD = 0.5
MIN_COMPONENT_VOXELS = 10   # 在 256 尺度下，相當於原始 ~160 voxels
MORPH_CLOSE_RADIUS = 2

def compute_dice(p, g, s=1e-5):
    i = np.sum(p * g)
    return (2.*i+s)/(np.sum(p)+np.sum(g)+s)

def compute_iou(p, g, s=1e-5):
    i = np.sum(p * g)
    return (i+s)/(np.sum(p)+np.sum(g)-i+s)

def postprocess_3d(prob_volume):
    # Step 1: Z 軸機率平滑（在二值化之前）
    if Z_SMOOTH_SIGMA > 0 and prob_volume.shape[0] > 3:
        prob_volume = gaussian_filter1d(prob_volume, sigma=Z_SMOOTH_SIGMA, axis=0)

    # Step 2: 二值化
    binary = (prob_volume > BINARY_THRESHOLD).astype(np.uint8)

    # Step 3: 形態學閉合（逐切片填補小洞）
    if MORPH_CLOSE_RADIUS > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (MORPH_CLOSE_RADIUS * 2 + 1, MORPH_CLOSE_RADIUS * 2 + 1)
        )
        for z in range(binary.shape[0]):
            binary[z] = cv2.morphologyEx(binary[z], cv2.MORPH_CLOSE, kernel)

    # Step 4: 3D 連通區域過濾（移除小碎片）
    if MIN_COMPONENT_VOXELS > 0:
        labeled, n = scipy_label(binary)
        if n > 0:
            sizes = np.bincount(labeled.ravel())
            for comp_id in range(1, n + 1):
                if sizes[comp_id] < MIN_COMPONENT_VOXELS:
                    binary[labeled == comp_id] = 0

    return binary.astype(np.float32)

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
    all_raw, all_post = [], []

    print("=" * 60)
    print("🧠 CoSeg v4 3D 評估（含後處理）")
    print(f"   Z-smooth σ={Z_SMOOTH_SIGMA}, Close r={MORPH_CLOSE_RADIUS}, Min comp={MIN_COMPONENT_VOXELS}")
    print("=" * 60)

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

            prob_list, gt_list = [], []

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

                # 縮小省 RAM
                prob_list.append(cv2.resize(prob, (EVAL_SIZE, EVAL_SIZE), interpolation=cv2.INTER_LINEAR))
                gt_list.append((cv2.resize(gt, (EVAL_SIZE, EVAL_SIZE), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8))

                del t, s

            torch.cuda.empty_cache()

            prob_vol = np.stack(prob_list, axis=0)
            gt_vol = np.stack(gt_list, axis=0).astype(np.float32)
            del prob_list, gt_list

            # === Raw（無後處理）===
            raw_pred = (prob_vol > 0.5).astype(np.float32)
            raw_dice = compute_dice(raw_pred, gt_vol)
            raw_iou = compute_iou(raw_pred, gt_vol)
            raw_labeled, raw_nc = scipy_label(raw_pred.astype(np.uint8))
            raw_conn = (np.bincount(raw_labeled.ravel())[1:].max() / raw_pred.sum()) if raw_pred.sum() > 0 and raw_nc > 0 else 0

            # 斷裂計算
            def count_bp(pred, gt):
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

            raw_bp = count_bp(raw_pred, gt_vol)

            # === Post（含後處理）===
            post_pred = postprocess_3d(prob_vol.copy())
            post_dice = compute_dice(post_pred, gt_vol)
            post_iou = compute_iou(post_pred, gt_vol)
            post_labeled, post_nc = scipy_label(post_pred.astype(np.uint8))
            post_conn = (np.bincount(post_labeled.ravel())[1:].max() / post_pred.sum()) if post_pred.sum() > 0 and post_nc > 0 else 0
            post_bp = count_bp(post_pred, gt_vol)

            all_raw.append({"p": pid, "dice": raw_dice, "iou": raw_iou, "comp": raw_nc, "conn": raw_conn, "bp": raw_bp})
            all_post.append({"p": pid, "dice": post_dice, "iou": post_iou, "comp": post_nc, "conn": post_conn, "bp": post_bp})

            print(f"\n📊 {pid}:")
            print(f"   Raw:  Dice={raw_dice:.4f}  IoU={raw_iou:.4f}  Comp={raw_nc:>4d}  Conn={raw_conn:.4f}  BP={raw_bp}")
            print(f"   Post: Dice={post_dice:.4f}  IoU={post_iou:.4f}  Comp={post_nc:>4d}  Conn={post_conn:.4f}  BP={post_bp}")

            del prob_vol, gt_vol, raw_pred, post_pred, raw_labeled, post_labeled

    # 彙總
    print(f"\n{'='*60}")
    print("📊 總結比較")
    print(f"{'='*60}")

    for label, data in [("Raw（無後處理）", all_raw), ("Post（含後處理）", all_post)]:
        ds = [r['dice'] for r in data]
        ios = [r['iou'] for r in data]
        cs = [r['comp'] for r in data]
        cns = [r['conn'] for r in data]
        bs = [r['bp'] for r in data]
        print(f"\n  {label}:")
        print(f"    Dice:         {np.mean(ds):.4f} ± {np.std(ds):.4f}")
        print(f"    IoU:          {np.mean(ios):.4f} ± {np.std(ios):.4f}")
        print(f"    Components:   {np.mean(cs):.1f}")
        print(f"    Connectivity: {np.mean(cns):.4f}")
        print(f"    Breakpoints:  {np.mean(bs):.1f}")

    print(f"\n  後處理提升:")
    for i in range(len(all_raw)):
        diff = all_post[i]['dice'] - all_raw[i]['dice']
        print(f"    {all_raw[i]['p']}: Dice {all_raw[i]['dice']:.4f} → {all_post[i]['dice']:.4f} ({'+' if diff>=0 else ''}{diff:.4f})")

    print(f"{'='*60}")

if __name__ == "__main__":
    main()
