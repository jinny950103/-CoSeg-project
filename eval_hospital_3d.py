import os
import cv2
import torch
import pydicom
import numpy as np
import hydra
from tqdm import tqdm
from model import CoSeg
from sam2.build_sam import build_sam2

PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
HOSPITAL_DATA_DIR = os.path.join(PROJECT_ROOT, "hospital_dataset")
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "coseg_hospital_iac_best.pth")
SAM2_CFG = "sam2_hiera_l.yaml"

# 🎯 確定 AI 預測神經管的通道是 1
TARGET_PRED_CLASS = 1    

def compute_3d_metrics(pred_volume, gt_volume):
    """計算 3D 的 Dice 和 IoU 分數"""
    smooth = 1e-5
    
    intersection = np.sum(pred_volume * gt_volume)
    sum_pred = np.sum(pred_volume)
    sum_gt = np.sum(gt_volume)
    union = sum_pred + sum_gt - intersection
    
    if sum_pred == 0 and sum_gt == 0:
        return None, None
        
    dice_score = (2. * intersection + smooth) / (sum_pred + sum_gt + smooth)
    iou_score = (intersection + smooth) / (union + smooth)
    
    return dice_score, iou_score

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')
    
    sam_model = build_sam2(SAM2_CFG, None, mode=None) 
    model = CoSeg(sam_model)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device), strict=True)
    model = model.to(device)
    model.eval()

    patients = [p for p in os.listdir(HOSPITAL_DATA_DIR) if os.path.isdir(os.path.join(HOSPITAL_DATA_DIR, p))]
    patients.sort()
    
    all_3d_dices = []
    all_3d_ious = []

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for patient_id in patients:
            print(f"\n正在評估病人: {patient_id} ...")
            
            base_dir = os.path.join(HOSPITAL_DATA_DIR, patient_id)
            if os.path.isdir(os.path.join(base_dir, patient_id)):
                base_dir = os.path.join(base_dir, patient_id)
                
            img_dir = os.path.join(base_dir, "no_label")
            # 🌟 修改 1: 將答案卷路徑指向我們剛做好的 npy 資料夾
            mask_dir = os.path.join(base_dir, "0724mask") 
            
            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".dcm")])
            
            if len(img_files) == 0:
                continue
                
            patient_pred_slices = []
            patient_gt_slices = []
            
            for idx, img_f in enumerate(tqdm(img_files, total=len(img_files), desc="處理 3D 切片中")):
                img_path = os.path.join(img_dir, img_f)
                
                # 🌟 修改 2: 將 .dcm 檔名換成 .npy 檔名來找對應的答案卷
                lbl_f = img_f.replace("nolabel", "label").replace("no_label", "label").replace(".dcm", ".npy")
                lbl_path = os.path.join(mask_dir, lbl_f)
                
                # --- A. 影像前處理 ---
                dcm_img = pydicom.dcmread(img_path)
                img = dcm_img.pixel_array.astype(np.float32)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0
                if len(img.shape) == 2:
                    img = np.stack([img]*3, axis=-1)
                img = cv2.resize(img, (1024, 1024))
                pixel_mean = np.array([123.675, 116.280, 103.530], dtype=np.float32)
                pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
                img = (img - pixel_mean) / (pixel_std + 1e-8)
                img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # --- B. 讀取預先做好的 Ground Truth (.npy) ---
                # 🌟 修改 3: 改用 np.load 讀取 .npy，並確保尺寸對齊
                if os.path.exists(lbl_path):
                    gt_mask = np.load(lbl_path).astype(np.float32)
                    # 如果之前存的 npy 是 256x256，要在這裡放大回 1024x1024 與預測圖比對
                    if gt_mask.shape != (1024, 1024):
                        gt_mask = cv2.resize(gt_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                else:
                    # 萬一該切片沒有對應的 npy 檔，就給全黑的空遮罩
                    gt_mask = np.zeros((1024, 1024), dtype=np.float32)
                
                # --- C. 模型預測 ---
                mask_pred_ins, mask_pred_sem, _, _ = model(x=img_tensor)
                mask_pred_ins, mask_pred_sem = model(x=img_tensor, prob_ins=mask_pred_ins, prob_sem=mask_pred_sem)
                
                # 🌟 配合訓練時的設定，把預測結果放大回 1024x1024，並取第 0 個通道
                pred_mask_sem_1024 = torch.nn.functional.interpolate(mask_pred_sem, size=(1024, 1024), mode='bilinear', align_corners=False)
                
                # 套用 sigmoid 將數值轉換成 0~1 的機率，大於 0.5 的就當作是神經管
                pred_prob = torch.sigmoid(pred_mask_sem_1024[:, 0, :, :])[0].cpu().numpy()
                pred_mask = (pred_prob > 0.5).astype(np.float32)
                
                patient_pred_slices.append(pred_mask)
                patient_gt_slices.append(gt_mask)                
            # 計算 3D Dice 與 IoU
            pred_volume = np.stack(patient_pred_slices, axis=0)
            gt_volume = np.stack(patient_gt_slices, axis=0)
            
            patient_3d_dice, patient_3d_iou = compute_3d_metrics(pred_volume, gt_volume)
            
            if patient_3d_dice is not None:
                all_3d_dices.append(patient_3d_dice)
                all_3d_ious.append(patient_3d_iou)
                print(f"🌟 病人 {patient_id} 結算：")
                print(f"   ➤ 3D Volume Dice: {patient_3d_dice:.4f}")
                print(f"   ➤ 3D Volume IoU:  {patient_3d_iou:.4f}")
                print(f"   ➤ AI 預測像素總數: {int(np.sum(pred_volume))}")
                print(f"   ➤ 醫生標註像素總數: {int(np.sum(gt_volume))}")
            else:
                print(f"⚠️ 病人 {patient_id} 查無特徵，跳過計分。")
            
    if len(all_3d_dices) > 0:
        mean_3d_dice = np.mean(all_3d_dices)
        mean_3d_iou = np.mean(all_3d_ious)
        print(f"🏆 最終結算：醫院資料集平均 3D Dice Score: {mean_3d_dice:.4f}")
        print(f"🏆 最終結算：醫院資料集平均 3D IoU Score:  {mean_3d_iou:.4f}")
    else:
        print("❌ 無法計算平均分數。")

if __name__ == "__main__":
    main()