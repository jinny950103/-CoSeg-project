import os
import torch
import pydicom
import numpy as np
import hydra
from model import CoSeg
from sam2.build_sam import build_sam2

PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "coseg_hospital_iac_best.pth")
PATIENT_ID = "0017026056"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 載入模型
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')
    sam_model = build_sam2("sam2_hiera_l.yaml", None, mode=None) 
    model = CoSeg(sam_model).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device), strict=True)
    model.eval()

    img_dir = os.path.join(PROJECT_ROOT, f"hospital_dataset/{PATIENT_ID}/{PATIENT_ID}/no_label")
    mask_dir = os.path.join(PROJECT_ROOT, f"hospital_dataset/{PATIENT_ID}/{PATIENT_ID}/0724mask")
    
    files = sorted([f for f in os.listdir(img_dir) if f.endswith(".dcm")])
    
    print(f"{'切片檔名':<30} | {'GT 像素數':<10} | {'AI 預測像素數':<15} | {'交集 (Intersection)'}")
    print("-" * 75)

    # 抽查中間可能有標記的切片 (例如第 80 到 95 張)
    for fname in files[80:95]:
        dcm_path = os.path.join(img_dir, fname)
        lbl_fname = fname.replace("nolabel", "label").replace("no_label", "label").replace(".dcm", ".npy")
        npy_path = os.path.join(mask_dir, lbl_fname)
        
        # 讀取 DICOM 並預處理
        dcm = pydicom.dcmread(dcm_path)
        img = dcm.pixel_array.astype(np.float32)
        img = cv2_resize_safe(img) # 確保尺寸正確
        
        # 讀取 GT 並確保尺寸為 1024x1024
        gt_mask = np.zeros((1024, 1024), dtype=np.float32)
        if os.path.exists(npy_path):
            raw_gt = np.load(npy_path).astype(np.float32)
            if raw_gt.shape != (1024, 1024):
                raw_gt = cv2.resize(raw_gt, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            gt_mask = (raw_gt > 0).astype(np.float32)            
        # AI 預測
        with torch.no_grad():
            img_tensor = prepare_tensor(img, device)
            _, mask_sem, _, _ = model(x=img_tensor)
            mask_sem = torch.nn.functional.interpolate(mask_sem, size=(1024, 1024), mode='bilinear', align_corners=False)
            pred_prob = torch.sigmoid(mask_sem)[0, 0].cpu().numpy()            
            pred_mask = (pred_prob > 0.5).astype(np.float32)
            
        gt_sum = int(gt_mask.sum())
        pred_sum = int(pred_mask.sum())
        intersection = int(np.sum(pred_mask * gt_mask))
        
        print(f"{fname:<30} | {gt_sum:<10} | {pred_sum:<15} | {intersection}")

def cv2_resize_safe(img):
    import cv2
    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0
    img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=-1)
    return img

def prepare_tensor(img, device):
    pixel_mean = np.array([123.675, 116.280, 103.530], dtype=np.float32)
    pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    img = (img - pixel_mean) / (pixel_std + 1e-8)
    return torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)

if __name__ == "__main__":
    import cv2
    main()