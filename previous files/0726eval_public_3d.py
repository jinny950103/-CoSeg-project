import os
import cv2
import torch
import numpy as np
import hydra
import torch.nn.functional as F
from tqdm import tqdm
from model import CoSeg
from sam2.build_sam import build_sam2

PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
EVAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data/public_data/eval")
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "outputs/coseg_public_mandibular_canal_best.pth")

def compute_3d_metrics(pred_volume, gt_volume):
    smooth = 1e-5
    intersection = np.sum(pred_volume * gt_volume)
    sum_pred = np.sum(pred_volume)
    sum_gt = np.sum(gt_volume)
    union = sum_pred + sum_gt - intersection
    
    if sum_pred == 0 and sum_gt == 0: return None, None
    return (2. * intersection + smooth) / (sum_pred + sum_gt + smooth), (intersection + smooth) / (union + smooth)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')
    
    model = CoSeg(build_sam2("sam2_hiera_l.yaml", None, mode="train"))
    model = torch.nn.DataParallel(model).to(device) if torch.cuda.device_count() > 1 else model.to(device)
    
    print(f"📦 正在載入模型權重: {MODEL_WEIGHTS_PATH}")
    
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(new_state_dict, strict=True)
    else:
        model.load_state_dict(new_state_dict, strict=True)
        
    model.eval()

    patients = sorted([p for p in os.listdir(EVAL_DATA_DIR) if p.startswith("Patient_")])
    all_3d_dices, all_3d_ious = [], []

    print("🚀 開始高速 3D 影像評估 (對比圖將輸出至 0726_compare 資料夾) ...")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for patient_id in patients:
            img_dir = os.path.join(EVAL_DATA_DIR, patient_id, "image_1024")
            mask_dir = os.path.join(EVAL_DATA_DIR, patient_id, "mask_sem_1024")
            
            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
            if not img_files: continue
                
            pred_slices, gt_slices = [], []
            for img_f in tqdm(img_files, desc=f"評估 {patient_id}"):
                img_original = cv2.imread(os.path.join(img_dir, img_f), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                img = np.stack([img_original]*3, axis=-1) if len(img_original.shape) == 2 else img_original
                
                gt_mask = np.load(os.path.join(mask_dir, img_f.replace(".png", ".npy"))).astype(np.float32)
                
                img_norm = (img - np.array([123.675, 116.280, 103.530])) / (np.array([58.395, 57.12, 57.375]) + 1e-8)
                img_tensor = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
                
                _, mask_pred_sem, _, _ = model(x=img_tensor)
                mask_pred_sem = torch.nn.functional.interpolate(mask_pred_sem, size=(1024, 1024), mode='bilinear', align_corners=False)
                
                pred_prob = torch.sigmoid(mask_pred_sem[:, 0, :, :])[0].cpu().numpy()
                pred_slices.append((pred_prob > 0.5).astype(np.float32))
                gt_slices.append((gt_mask > 0).astype(np.float32))

                if (gt_mask > 0).any():
                    # 🌟 已經將輸出資料夾改為 0726_compare
                    debug_dir = os.path.join(PROJECT_ROOT, "0726_compare")
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    img_vis = img.astype(np.uint8)
                    
                    gt_vis_1c = (gt_mask * 255).astype(np.uint8)
                    gt_vis_3c = np.stack([gt_vis_1c]*3, axis=-1)
                    
                    pred_vis_1c = ((pred_prob > 0.5) * 255).astype(np.uint8)
                    pred_vis_3c = np.stack([pred_vis_1c]*3, axis=-1)
                    
                    combined_vis = np.hstack((img_vis, gt_vis_3c, pred_vis_3c))
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(combined_vis, 'Original', (20, 60), font, 2, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(combined_vis, 'Ground Truth', (1024 + 20, 60), font, 2, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(combined_vis, 'Prediction', (2048 + 20, 60), font, 2, (0, 255, 0), 4, cv2.LINE_AA)
                    
                    base_name = img_f.replace('.png', '')
                    cv2.imwrite(os.path.join(debug_dir, f"{patient_id}_{base_name}_compare.png"), combined_vis)

            dice, iou = compute_3d_metrics(np.stack(pred_slices), np.stack(gt_slices))
            if dice is not None:
                all_3d_dices.append(dice)
                all_3d_ious.append(iou)
                print(f"📊 {patient_id} 結算 ── 3D Dice: {dice:.4f} | IoU: {iou:.4f}")
            else:
                print(f"⚠️ {patient_id} 查無特徵。")

    if all_3d_dices:
        print(f"\n🏆 測試集全體病人平均 ── 3D Dice: {np.mean(all_3d_dices):.4f} | IoU: {np.mean(all_3d_ious):.4f}")

if __name__ == "__main__":
    main()