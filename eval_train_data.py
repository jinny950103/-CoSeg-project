import os
import cv2
import torch
import numpy as np
import hydra
from tqdm import tqdm
from model import CoSeg
from sam2.build_sam import build_sam2

PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "coseg_hospital_iac_best.pth")

# 直接指向你當初的訓練/驗證資料夾
IMG_DIR = os.path.join(PROJECT_ROOT, "data/image_1024")
MASK_DIR = os.path.join(PROJECT_ROOT, "data/mask_sem_1024")

def compute_metrics(pred, gt):
    smooth = 1e-5
    intersection = np.sum(pred * gt)
    sum_pred = np.sum(pred)
    sum_gt = np.sum(gt)
    
    if sum_pred == 0 and sum_gt == 0:
        return 1.0 # 兩邊都是空的，算完全答對
    
    dice = (2. * intersection + smooth) / (sum_pred + sum_gt + smooth)
    return dice

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 載入模型
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')
    sam_model = build_sam2("sam2_hiera_l.yaml", None, mode=None) 
    model = CoSeg(sam_model).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device), strict=True)
    model.eval()

    # 2. 抓取所有訓練圖片
    img_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".png")])
    if len(img_files) == 0:
        print("找不到訓練圖片！")
        return

    print(f"🚀 開始對 {len(img_files)} 張訓練/驗證影像進行 Sanity Check...")
    
    dices = []
    
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for img_name in tqdm(img_files):
            img_path = os.path.join(IMG_DIR, img_name)
            npy_name = img_name.replace(".png", ".npy")
            npy_path = os.path.join(MASK_DIR, npy_name)
            
            if not os.path.exists(npy_path):
                continue
                
            # --- A. 讀取影像 (PNG 格式) ---
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            if img.shape != (1024, 1024):
                img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            if len(img.shape) == 2:
                img = np.stack([img]*3, axis=-1)
                
            pixel_mean = np.array([123.675, 116.280, 103.530], dtype=np.float32)
            pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
            img = (img - pixel_mean) / (pixel_std + 1e-8)
            img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # --- B. 讀取 GT 答案卷 ---
            gt_mask = np.load(npy_path).astype(np.float32)
            
            # 🌟 加入尺寸防呆：如果不是 1024，就用最近鄰插值拉大
            if gt_mask.shape != (1024, 1024):
                gt_mask = cv2.resize(gt_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                
            gt_mask = (gt_mask > 0).astype(np.float32)
            
            # --- C. 模型預測 ---
            _, mask_sem, _, _ = model(x=img_tensor)
            mask_sem = torch.nn.functional.interpolate(mask_sem, size=(1024, 1024), mode='bilinear', align_corners=False)
            
            # 使用正確的單通道 Sigmoid解碼
            pred_prob = torch.sigmoid(mask_sem)[0, 0].cpu().numpy()
            pred_mask = (pred_prob > 0.5).astype(np.float32)
            
            # 計算單張 Dice
            dice = compute_metrics(pred_mask, gt_mask)
            dices.append(dice)
            
    # 3. 結算成績
    mean_dice = np.mean(dices)
    print(f"\n🏆 訓練資料集最終平均 Dice Score: {mean_dice:.4f}")
    
    if mean_dice > 0.7:
        print("✅ 恭喜！模型在訓練資料上表現優異。這證明了「模型沒壞」、「評估程式寫法正確」。")
        print("👉 接下來只要把醫院的 3D 答案卷用 extract_iac_mask_1024.py 對齊，分數就會出來了！")
    else:
        print("⚠️ 警告：模型連在訓練資料上都考不好，我們需要回去檢查預測的機率輸出。")

if __name__ == "__main__":
    main()