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
# 針對剛才的病人進行抽樣測試
PATIENT_ID = "0017026056" 
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "coseg_hospital_iac_best.pth")

# 建立一個資料夾來存圖片
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "debug_2d_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def window_image(img, window_center=400, window_width=1500):
    """標準的 CT 窗寬窗位轉換 (這裡是 Bone Window 的常用設定)"""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min) * 255.0
    return img

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')
    sam_model = build_sam2("sam2_hiera_l.yaml", None, mode=None) 
    model = CoSeg(sam_model).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device), strict=True)
    model.eval()

    img_dir = os.path.join(PROJECT_ROOT, f"hospital_dataset/{PATIENT_ID}/{PATIENT_ID}/no_label")
    mask_dir = os.path.join(PROJECT_ROOT, f"hospital_dataset/{PATIENT_ID}/{PATIENT_ID}/0725mask")
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".dcm")])
    
    print(f"🚀 開始進行 2D 切片對位檢查，將跳過無標註的空切片...")
    
    with torch.no_grad():
        for fname in tqdm(img_files):
            lbl_fname = fname.replace("nolabel", "label").replace("no_label", "label").replace(".dcm", ".npy")
            lbl_path = os.path.join(mask_dir, lbl_fname)
            
            # --- 1. 讀取 GT 答案卷 ---
            if not os.path.exists(lbl_path):
                continue
                
            gt_mask = np.load(lbl_path).astype(np.float32)
            if gt_mask.shape != (1024, 1024):
                gt_mask = cv2.resize(gt_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            gt_mask = (gt_mask > 0).astype(np.float32)
            
            # 🌟 核心防線：如果這張圖醫生沒畫重點，我們直接跳過，不測負樣本！
            if gt_mask.sum() == 0:
                continue

            # --- 2. 讀取並處理 DICOM ---
            dcm_path = os.path.join(img_dir, fname)
            dcm = pydicom.dcmread(dcm_path)
            raw_img = dcm.pixel_array.astype(np.float32)
            
            # 使用 CT Window 截斷極端值，避免被金屬或空氣干擾對比度
            img_windowed = window_image(raw_img, window_center=400, window_width=1500)
            
            img_resized = cv2.resize(img_windowed, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            img_rgb = np.stack([img_resized]*3, axis=-1)
            
            pixel_mean = np.array([123.675, 116.280, 103.530], dtype=np.float32)
            pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
            img_norm = (img_rgb - pixel_mean) / (pixel_std + 1e-8)
            img_tensor = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # --- 3. AI 預測 ---
            _, mask_sem, _, _ = model(x=img_tensor)
            mask_sem = torch.nn.functional.interpolate(mask_sem, size=(1024, 1024), mode='bilinear', align_corners=False)
            pred_prob = torch.sigmoid(mask_sem)[0, 0].cpu().numpy()
            pred_mask = (pred_prob > 0.5).astype(np.float32)
            
            # 計算 2D Dice
            intersection = np.sum(pred_mask * gt_mask)
            dice = (2. * intersection) / (np.sum(pred_mask) + np.sum(gt_mask) + 1e-5)
            
            print(f"\n切片 {fname} | 2D Dice: {dice:.4f} | GT 點數: {gt_mask.sum()} | AI 預測點數: {pred_mask.sum()}")
            
            # --- 4. 畫圖與存檔 ---
            # 背景轉為灰階圖
            overlay = img_rgb.copy().astype(np.uint8)
            
            # GT 畫成綠色
            overlay[gt_mask == 1] = [0, 255, 0] 
            
            # 預測畫成紅色
            # (如果重疊的地方會變成黃色，或者保留綠色)
            overlay[pred_mask == 1] = [0, 0, 255] 
            
            out_path = os.path.join(OUTPUT_DIR, f"debug_{fname.replace('.dcm', '.png')}")
            cv2.imwrite(out_path, overlay)

if __name__ == "__main__":
    main()