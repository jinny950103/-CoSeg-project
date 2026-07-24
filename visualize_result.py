import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import hydra
from model import CoSeg
from sam2.build_sam import build_sam2

PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "coseg_hospital_iac_best.pth")

# 🌟 關鍵修改：直接拿訓練集/驗證集裡面的 PNG 圖片來測試
TEST_IMG = os.path.join(PROJECT_ROOT, "data/image_1024/VOL_14_slice_121.png")
TEST_NPY = os.path.join(PROJECT_ROOT, "data/mask_sem_1024/VOL_14_slice_121.npy")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 載入模型
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')
    sam_model = build_sam2("sam2_hiera_l.yaml", None, mode=None) 
    model = CoSeg(sam_model)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device), strict=True)
    model = model.to(device)
    model.eval()

    # 2. 準備輸入影像
    img = cv2.imread(TEST_IMG, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    
    # 🌟 關鍵修正：強制將圖片 Resize 到模型規定的 1024x1024
    img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    img_show = img.copy()
    
    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=-1)
        
    pixel_mean = np.array([123.675, 116.280, 103.530], dtype=np.float32)
    pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    
    img = (img - pixel_mean) / (pixel_std + 1e-8)
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)
    # 3. 讀取 Ground Truth
    gt_mask = np.load(TEST_NPY).astype(np.float32)
    gt_mask = (gt_mask > 0).astype(np.float32) # 二值化防呆

    # 4. AI 預測
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        mask_pred_ins, mask_pred_sem, _, _ = model(x=img_tensor)
        mask_pred_ins, mask_pred_sem = model(x=img_tensor, prob_ins=mask_pred_ins, prob_sem=mask_pred_sem)
        
        pred_mask_sem_1024 = torch.nn.functional.interpolate(mask_pred_sem, size=(1024, 1024), mode='bilinear', align_corners=False)
        pred_prob = torch.sigmoid(pred_mask_sem_1024[:, 0, :, :])[0].cpu().numpy()
        ai_pred_mask = (pred_prob > 0.5).astype(np.float32)

    # 5. 畫圖對比並儲存
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("1. Original Image (PNG)")
    plt.imshow(img_show, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(f"2. Doctor GT (Pixels: {int(gt_mask.sum())})")
    plt.imshow(img_show, cmap='gray')
    plt.imshow(gt_mask, cmap='jet', alpha=0.5) 
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"3. AI Prediction (Pixels: {int(ai_pred_mask.sum())})")
    plt.imshow(img_show, cmap='gray')
    plt.imshow(ai_pred_mask, cmap='jet', alpha=0.5) 
    plt.axis('off')

    save_path = os.path.join(PROJECT_ROOT, "compare_result_final.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ 終極對比圖已儲存至: {save_path}，快去打開來看看！")

if __name__ == "__main__":
    main()