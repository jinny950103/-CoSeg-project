"""
visualize_predictions.py — 生成 Image / GT / Prediction 對照圖
================================================================
從 test set 隨機抽取切片，生成並排比較圖

用法：
  python visualize_predictions.py
  python visualize_predictions.py --weights outputs/coseg_v7_best.pth --num 10
  python visualize_predictions.py --weights outputs/coseg_v6_best.pth --patient Patient_4
"""
import os, cv2, torch, argparse, random
import numpy as np
import torch.nn.functional as F
import hydra
import matplotlib
matplotlib.use('Agg')  # 無 GUI 環境
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from model_v6 import CoSegV6
from sam2.build_sam import build_sam2

PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
EVAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data/public_data/eval")
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "outputs/coseg_v7_best.pth")
SAM2_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs/visualizations")


def predict_slice(model, img_3ch, pm, ps, device):
    """推論一張切片，回傳機率圖"""
    img_norm = (img_3ch - pm) / (ps + 1e-8)
    t = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(x=t)
        s = outputs[1]
        s = F.interpolate(s, size=(1024, 1024), mode='bilinear', align_corners=False)
        prob = torch.sigmoid(s[:, 0, :, :])[0].cpu().numpy()
    del t, s
    return prob


def create_comparison_figure(img_gray, gt_mask, pred_prob, patient_id, slice_idx,
                              dice_score, save_path):
    """
    生成一張 4 欄對照圖：
      1. 原始影像
      2. GT overlay
      3. Prediction overlay
      4. GT vs Pred 比較（綠=正確, 紅=假陽性, 藍=漏掉）
    """
    pred_mask = (pred_prob > 0.5).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(f'{patient_id} — Slice {slice_idx}  (Dice: {dice_score:.4f})',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. 原始影像
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title('CT Image', fontsize=13)
    axes[0].axis('off')

    # 2. GT overlay（綠色）
    img_rgb = np.stack([img_gray]*3, axis=-1).astype(np.float32)
    img_rgb = img_rgb / img_rgb.max() if img_rgb.max() > 0 else img_rgb
    gt_overlay = img_rgb.copy()
    gt_overlay[gt_mask > 0] = [0, 1, 0]  # 綠色
    gt_blend = 0.6 * img_rgb + 0.4 * gt_overlay
    axes[1].imshow(np.clip(gt_blend, 0, 1))
    axes[1].set_title('Ground Truth', fontsize=13, color='green')
    axes[1].axis('off')

    # 3. Prediction overlay（黃色）
    pred_overlay = img_rgb.copy()
    pred_overlay[pred_mask > 0] = [1, 0.8, 0]  # 黃色
    pred_blend = 0.6 * img_rgb + 0.4 * pred_overlay
    axes[2].imshow(np.clip(pred_blend, 0, 1))
    axes[2].set_title('Prediction', fontsize=13, color='orange')
    axes[2].axis('off')

    # 4. 比較圖
    # 綠 = True Positive, 紅 = False Positive, 藍 = False Negative
    compare = img_rgb.copy() * 0.4
    tp = (pred_mask > 0) & (gt_mask > 0)
    fp = (pred_mask > 0) & (gt_mask == 0)
    fn = (pred_mask == 0) & (gt_mask > 0)
    compare[tp] = [0, 1, 0]    # 綠：正確
    compare[fp] = [1, 0, 0]    # 紅：假陽性
    compare[fn] = [0, 0.4, 1]  # 藍：漏掉
    axes[3].imshow(np.clip(compare, 0, 1))
    axes[3].set_title('TP(綠) / FP(紅) / FN(藍)', fontsize=13)
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  💾 {save_path}")


def compute_slice_dice(pred, gt, smooth=1e-5):
    p = (pred > 0.5).astype(np.float32)
    inter = np.sum(p * gt)
    return (2 * inter + smooth) / (np.sum(p) + np.sum(gt) + smooth)


def main():
    parser = argparse.ArgumentParser(description="生成預測對照圖")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="模型權重")
    parser.add_argument("--num", type=int, default=5, help="每個病人生成幾張圖")
    parser.add_argument("--patient", type=str, default=None, help="指定病人（如 Patient_1）")
    parser.add_argument("--output", default=OUTPUT_DIR, help="輸出資料夾")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')

    print(f"📦 載入模型: {args.weights}")
    model = CoSegV6(build_sam2("sam2_hiera_l.yaml", SAM2_CHECKPOINT, mode="train")).to(device)
    sd = torch.load(args.weights, map_location=device)
    model.load_state_dict(
        {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()},
        strict=True
    )
    model.eval()

    pm = np.array([123.675, 116.280, 103.530], dtype=np.float32)
    ps = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    # 找病人
    if args.patient:
        patients = [args.patient]
    else:
        patients = sorted([p for p in os.listdir(EVAL_DATA_DIR) if p.startswith("Patient_")])

    print(f"\n🎨 生成對照圖（每個病人 {args.num} 張）")
    print(f"   輸出: {args.output}")
    print("=" * 50)

    for pid in patients:
        idir = os.path.join(EVAL_DATA_DIR, pid, "image_1024")
        mdir = os.path.join(EVAL_DATA_DIR, pid, "mask_sem_1024")
        if not os.path.isdir(idir):
            print(f"  ⚠️ {pid} 找不到，跳過")
            continue

        ifs = sorted([f for f in os.listdir(idir) if f.endswith(".png")])
        if not ifs:
            continue

        # 預載灰度圖
        ag = {}
        for f in ifs:
            g = cv2.imread(os.path.join(idir, f), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            ag[int(f.replace(".png", "").rsplit("_", 1)[-1])] = g

        # 找含神經管的切片
        positive_slices = []
        for f in ifs:
            idx = int(f.replace(".png", "").rsplit("_", 1)[-1])
            gt = np.load(os.path.join(mdir, f.replace(".png", ".npy")))
            if gt.sum() > 0:
                positive_slices.append((f, idx))

        if not positive_slices:
            print(f"  {pid}: 沒有正樣本切片")
            continue

        # 隨機選 N 張（優先選正樣本）
        selected = random.sample(positive_slices, min(args.num, len(positive_slices)))

        print(f"\n📊 {pid} ({len(positive_slices)} 正樣本切片，選 {len(selected)} 張)")

        for f, idx in selected:
            c = ag[idx]
            p_ = ag.get(idx - 1, c)
            n_ = ag.get(idx + 1, c)
            img_3ch = np.stack([p_, c, n_], axis=-1)
            gt = np.load(os.path.join(mdir, f.replace(".png", ".npy"))).astype(np.float32)
            gt = (gt > 0).astype(np.float32)

            prob = predict_slice(model, img_3ch, pm, ps, device)
            dice = compute_slice_dice(prob, gt)

            save_name = f"{pid}_slice{idx}_dice{dice:.3f}.png"
            save_path = os.path.join(args.output, save_name)

            create_comparison_figure(c, gt, prob, pid, idx, dice, save_path)

    torch.cuda.empty_cache()
    print(f"\n✅ 完成！所有圖片在: {args.output}")


if __name__ == "__main__":
    main()
