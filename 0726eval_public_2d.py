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

def compute_2d_metrics(pred_mask, gt_mask):
    smooth = 1e-5
    intersection = np.sum(pred_mask * gt_mask)
    sum_pred = np.sum(pred_mask)
    sum_gt = np.sum(gt_mask)
    union = sum_pred + sum_gt - intersection
    
    # 加上 smooth 可以確保當預測和答案都是全黑時，Dice 和 IoU 會完美給出 1.0 分
    dice = (2. * intersection + smooth) / (sum_pred + sum_gt + smooth)
    iou = (intersection + smooth) / (union + smooth)
    return dice, iou

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
    all_2d_dices, all_2d_ious = [], []

    print("🚀 開始 2D 影像精準評估 (計算所有切片，無任何後處理過濾) ...")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for patient_id in patients:
            img_dir = os.path.join(EVAL_DATA_DIR, patient_id, "image_1024")
            mask_dir = os.path.join(EVAL_DATA_DIR, patient_id, "mask_sem_1024")
            
            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
            if not img_files: continue
            
            patient_dices, patient_ious = [], []
            
            for img_f in tqdm(img_files, desc=f"評估 {patient_id}"):
                gt_mask = np.load(os.path.join(mask_dir, img_f.replace(".png", ".npy"))).astype(np.float32)
                gt_mask_binary = (gt_mask > 0).astype(np.float32)
                
                img_original = cv2.imread(os.path.join(img_dir, img_f), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                img = np.stack([img_original]*3, axis=-1) if len(img_original.shape) == 2 else img_original
                
                img_norm = (img - np.array([123.675, 116.280, 103.530])) / (np.array([58.395, 57.12, 57.375]) + 1e-8)
                img_tensor = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
                
                _, mask_pred_sem, _, _ = model(x=img_tensor)
                mask_pred_sem = torch.nn.functional.interpolate(mask_pred_sem, size=(1024, 1024), mode='bilinear', align_corners=False)
                
                pred_prob = torch.sigmoid(mask_pred_sem[:, 0, :, :])[0].cpu().numpy()
                pred_mask_binary = (pred_prob > 0.5).astype(np.float32)
                
                dice, iou = compute_2d_metrics(pred_mask_binary, gt_mask_binary)
                patient_dices.append(dice)
                patient_ious.append(iou)

            if patient_dices:
                avg_p_dice = np.mean(patient_dices)
                avg_p_iou = np.mean(patient_ious)
                all_2d_dices.extend(patient_dices)
                all_2d_ious.extend(patient_ious)
                print(f"📊 {patient_id} 結算 ── 全切片 2D 平均 Dice: {avg_p_dice:.4f} | IoU: {avg_p_iou:.4f}")
            else:
                print(f"⚠️ {patient_id} 查無資料。")

    if all_2d_dices:
        print(f"\n🏆 測試集全體 2D 切片平均 ── Dice: {np.mean(all_2d_dices):.4f} | IoU: {np.mean(all_2d_ious):.4f}")

if __name__ == "__main__":
    main()