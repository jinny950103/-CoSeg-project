import os
import cv2
import torch
import pydicom
import numpy as np
import hydra
import matplotlib.pyplot as plt
from model import CoSeg
from sam2.build_sam import build_sam2

PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
HOSPITAL_DATA_DIR = os.path.join(PROJECT_ROOT, "hospital_dataset")
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "coseg_puma_latest.pth")
SAM2_CFG = "sam2_hiera_l.yaml"

TARGET_PRED_CLASS = 1

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
    
    # 我們只抓第一個病人來畫圖除錯
    patient_id = patients[0]
    print(f"開始繪製病人 {patient_id} 的除錯圖像...")
    
    base_dir = os.path.join(HOSPITAL_DATA_DIR, patient_id)
    if os.path.isdir(os.path.join(base_dir, patient_id)):
        base_dir = os.path.join(base_dir, patient_id)
        
    img_dir = os.path.join(base_dir, "no_label")
    label_dir = os.path.join(base_dir, "label")
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".dcm")])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".dcm")])
    
    # 抓取最中間的那張切片
    mid_idx = len(img_files) // 2
    
    img_path = os.path.join(img_dir, img_files[mid_idx])
    lbl_path = os.path.join(label_dir, label_files[mid_idx])
    
    # 1. 處理輸入影像
    dcm_img = pydicom.dcmread(img_path)
    img_raw = dcm_img.pixel_array.astype(np.float32)
    img_norm = (img_raw - img_raw.min()) / (img_raw.max() - img_raw.min() + 1e-8) * 255.0
    if len(img_norm.shape) == 2:
        img_norm = np.stack([img_norm]*3, axis=-1)
    img_resized = cv2.resize(img_norm, (1024, 1024))
    
    pixel_mean = np.array([123.675, 116.280, 103.530], dtype=np.float32)
    pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    img_input = (img_resized - pixel_mean) / (pixel_std + 1e-8)
    img_tensor = torch.tensor(img_input).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 2. 處理 Ground Truth
    dcm_lbl = pydicom.dcmread(lbl_path)
    gt_raw = dcm_lbl.pixel_array.astype(np.float32)
    gt_resized = cv2.resize(gt_raw, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    
    gt_max_val = gt_resized.max()
    print(f"GT 影像最大值: {gt_max_val}")
    if gt_max_val > 500: 
        gt_mask = (gt_resized > (gt_max_val - 200)).astype(np.float32) 
    else:
        gt_mask = np.zeros_like(gt_resized)
        
    # 3. AI 預測
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        mask_pred_ins, mask_pred_sem, _, _ = model(x=img_tensor)
        mask_pred_ins, mask_pred_sem = model(x=img_tensor, prob_ins=mask_pred_ins, prob_sem=mask_pred_sem)
        pred_sem_prob = torch.softmax(mask_pred_sem, dim=1)
        pred_sem_mask_class = torch.argmax(pred_sem_prob, dim=1)[0].cpu().numpy()
        pred_mask = (pred_sem_mask_class == TARGET_PRED_CLASS).astype(np.float32)
    
    # 4. 畫圖並存檔
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original CT Image")
    plt.imshow(img_resized[:,:,0], cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Extracted Ground Truth")
    plt.imshow(gt_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("AI Prediction (Class 1)")
    plt.imshow(pred_mask, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(PROJECT_ROOT, "debug_visual_output.png")
    plt.savefig(save_path)
    print(f"✅ 除錯圖片已儲存至：{save_path}")
    print("請使用 VS Code 開啟這個圖片檔案，看看兩邊到底錯開在哪裡！")

if __name__ == "__main__":
    main()