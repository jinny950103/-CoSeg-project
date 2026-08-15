"""
eval_v4_3d.py — CoSeg v4 評估腳本（載入 SAM2 預訓練權重版本）
"""
import os, cv2, torch, numpy as np, hydra
import torch.nn.functional as F
from tqdm import tqdm
from model import CoSeg
from sam2.build_sam import build_sam2

PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
EVAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data/public_data/eval")
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "outputs/coseg_v4_best.pth")
SAM2_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
CONTEXT_SLICES = 1
EVAL_SIZE = 256  # 縮小省 RAM

def compute_dice(pred, gt, smooth=1e-5):
    intersection = np.sum(pred * gt)
    return (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)

def compute_iou(pred, gt, smooth=1e-5):
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    return (intersection + smooth) / (union + smooth)

def main():
    device = "cuda"
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')

    print(f"📦 載入 SAM2 預訓練權重: {SAM2_CHECKPOINT}")
    model = CoSeg(build_sam2("sam2_hiera_l.yaml", SAM2_CHECKPOINT, mode="train"))
    model = model.to(device)

    print(f"📦 載入 CoSeg 訓練權重: {MODEL_WEIGHTS_PATH}")
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
    new_sd = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    model.load_state_dict(new_sd, strict=True)
    model.eval()

    pixel_mean = np.array([123.675, 116.280, 103.530], dtype=np.float32)
    pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    patients = sorted([p for p in os.listdir(EVAL_DATA_DIR) if p.startswith("Patient_")])
    all_results = []

    print("=" * 60)
    print("🧠 CoSeg v4 3D 評估")
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

            # 預載灰度
            all_gray = {}
            for img_f in img_files:
                gray = cv2.imread(os.path.join(img_dir, img_f), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                idx = int(img_f.replace(".png", "").rsplit("_", 1)[-1])
                all_gray[idx] = gray
            sorted_idx = sorted(all_gray.keys())

            pred_list, gt_list = [], []

            for img_f in tqdm(img_files, desc=f"推論 {patient_id}"):
                idx = int(img_f.replace(".png", "").rsplit("_", 1)[-1])
                center = all_gray[idx]
                prev_s = all_gray.get(idx - CONTEXT_SLICES, center)
                next_s = all_gray.get(idx + CONTEXT_SLICES, center)
                img_3ch = np.stack([prev_s, center, next_s], axis=-1)

                gt = np.load(os.path.join(mask_dir, img_f.replace(".png", ".npy"))).astype(np.float32)

                img_norm = (img_3ch - pixel_mean) / (pixel_std + 1e-8)
                img_t = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

                _, pred_sem, _, _ = model(x=img_t)
                pred_sem = F.interpolate(pred_sem, size=(1024, 1024), mode='bilinear', align_corners=False)
                prob = torch.sigmoid(pred_sem[:, 0, :, :])[0].cpu().numpy()

                # 縮小省 RAM
                prob_small = cv2.resize(prob, (EVAL_SIZE, EVAL_SIZE), interpolation=cv2.INTER_LINEAR)
                gt_small = cv2.resize(gt, (EVAL_SIZE, EVAL_SIZE), interpolation=cv2.INTER_NEAREST)

                pred_list.append((prob_small > 0.5).astype(np.uint8))
                gt_list.append((gt_small > 0).astype(np.uint8))

                del img_t, pred_sem

            torch.cuda.empty_cache()

            pred_vol = np.stack(pred_list, axis=0)
            gt_vol = np.stack(gt_list, axis=0)
            del pred_list, gt_list

            dice = compute_dice(pred_vol, gt_vol)
            iou = compute_iou(pred_vol, gt_vol)

            # 連通元件
            from scipy.ndimage import label as scipy_label
            labeled, n_comp = scipy_label(pred_vol)
            if pred_vol.sum() > 0:
                sizes = np.bincount(labeled.ravel())
                conn_rate = sizes[1:].max() / pred_vol.sum() if len(sizes) > 1 else 0
            else:
                conn_rate = 0
                n_comp = 0

            # 斷裂
            gt_per_z = gt_vol.sum(axis=(1, 2))
            gt_range = np.where(gt_per_z > 0)[0]
            breakpoints = 0
            if len(gt_range) > 0:
                pred_per_z = pred_vol[gt_range[0]:gt_range[-1]+1].sum(axis=(1, 2)) > 0
                in_canal = False
                for i, has in enumerate(pred_per_z):
                    if has and not in_canal:
                        if i > 0:
                            breakpoints += 1
                        in_canal = True
                    elif not has and in_canal:
                        in_canal = False

            all_results.append({
                "patient": patient_id,
                "dice": dice,
                "iou": iou,
                "components": n_comp,
                "connectivity": conn_rate,
                "breakpoints": breakpoints,
                "pred_voxels": int(pred_vol.sum()),
                "gt_voxels": int(gt_vol.sum()),
            })

            print(f"\n📊 {patient_id}:")
            print(f"   Dice:             {dice:.4f}")
            print(f"   IoU:              {iou:.4f}")
            print(f"   Components:       {n_comp}")
            print(f"   Connectivity:     {conn_rate:.4f}")
            print(f"   Breakpoints:      {breakpoints}")
            print(f"   Pred voxels:      {int(pred_vol.sum())}")
            print(f"   GT voxels:        {int(gt_vol.sum())}")

            del pred_vol, gt_vol, labeled

    # 彙總
    print(f"\n{'='*60}")
    print("📊 總結")
    print(f"{'='*60}")
    dices = [r["dice"] for r in all_results]
    ious = [r["iou"] for r in all_results]
    conns = [r["connectivity"] for r in all_results]
    bps = [r["breakpoints"] for r in all_results]
    comps = [r["components"] for r in all_results]

    print(f"  平均 Dice:         {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"  平均 IoU:          {np.mean(ious):.4f} ± {np.std(ious):.4f}")
    print(f"  平均 Connectivity: {np.mean(conns):.4f}")
    print(f"  平均 Components:   {np.mean(comps):.1f}")
    print(f"  平均 Breakpoints:  {np.mean(bps):.1f}")

    print(f"\n  Per-patient:")
    for r in all_results:
        print(f"    {r['patient']}: Dice={r['dice']:.4f} IoU={r['iou']:.4f} "
              f"Conn={r['connectivity']:.4f} BP={r['breakpoints']} Comp={r['components']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
