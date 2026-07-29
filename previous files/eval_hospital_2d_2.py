import os
import cv2
import torch
import pydicom
import numpy as np
import hydra
from tqdm import tqdm
from model import CoSeg
from sam2.build_sam import build_sam2

# ==========================================
# 參數設定區
# ==========================================
PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
HOSPITAL_DATA_DIR = os.path.join(PROJECT_ROOT, "hospital_dataset")
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "coseg_puma_latest.pth")
SAM2_CFG = "sam2_hiera_l.yaml"

# 🎯 確定 AI 預測神經管的通道是 1
TARGET_PRED_CLASS = 1    
# ==========================================

def compute_2d_metrics(pred_mask, gt_mask):
    """計算單張 2D 切片的 Dice 和 IoU 分數"""
    smooth = 1e-5
    
    intersection = np.sum(pred_mask * gt_mask)
    sum_pred = np.sum(pred_mask)
    sum_gt = np.sum(gt_mask)
    union = sum_pred + sum_gt - intersection
    
    # 如果這張切片 GT 和預測都沒有東西，視為完美的 True Negative
    if sum_pred == 0 and sum_gt == 0:
        return 1.0, 1.0
    
    # 如果只有其中一個有東西（完全沒交集），分數就是 0
    if sum_pred == 0 or sum_gt == 0:
        return 0.0, 0.0
        
    dice_score = (2. * intersection + smooth) / (sum_pred + sum_gt + smooth)
    iou_score = (intersection + smooth) / (union + smooth)
    
    return dice_score, iou_score

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用的運算設備: {device}")

    # 1. 載入模型
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')
    
    print("正在建立模型架構...")
    sam_model = build_sam2(SAM2_CFG, None, mode=None) 
    model = CoSeg(sam_model)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    print(f"載入權重: {MODEL_WEIGHTS_PATH}")
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device), strict=True)
    model = model.to(device)
    model.eval()
    print("✅ 成功載入 AI 模型權重！")
    print("-" * 50)

    # 2. 獲取病人列表
    patients = [p for p in os.listdir(HOSPITAL_DATA_DIR) if os.path.isdir(os.path.join(HOSPITAL_DATA_DIR, p))]
    patients.sort()
    
    overall_dices = []
    overall_ious = []

    # 3. 開始 2D 評估
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for patient_id in patients:
            print(f"\n正在評估病人: {patient_id} ...")
            
            base_dir = os.path.join(HOSPITAL_DATA_DIR, patient_id)
            if os.path.isdir(os.path.join(base_dir, patient_id)):
                base_dir = os.path.join(base_dir, patient_id)
                
            img_dir = os.path.join(base_dir, "no_label")
            label_dir = os.path.join(base_dir, "label")
            
            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".dcm")])
            label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".dcm")])
            
            if len(img_files) == 0:
                continue
                
            patient_dices = []
            patient_ious = []
            
            for idx, (img_f, lbl_f) in enumerate(tqdm(zip(img_files, label_files), total=len(img_files), desc="處理 2D 切片中")):
                img_path = os.path.join(img_dir, img_f)
                lbl_path = os.path.join(label_dir, lbl_f)
                
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
                
                # --- B. 讀取 Ground Truth 並萃取「最亮的點點」 ---
                dcm_lbl = pydicom.dcmread(lbl_path)
                gt = dcm_lbl.pixel_array.astype(np.float32)
                gt = cv2.resize(gt, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                
                gt_max_val = gt.max()
                # 只有當影像中有明顯亮點時才提取
                if gt_max_val > 500: 
                    gt_mask = (gt > (gt_max_val - 200)).astype(np.float32) 
                else:
                    gt_mask = np.zeros_like(gt)
                
                # --- C. 模型預測 ---
                mask_pred_ins, mask_pred_sem, _, _ = model(x=img_tensor)
                mask_pred_ins, mask_pred_sem = model(x=img_tensor, prob_ins=mask_pred_ins, prob_sem=mask_pred_sem)
                
                pred_sem_prob = torch.softmax(mask_pred_sem, dim=1)
                pred_sem_mask_class = torch.argmax(pred_sem_prob, dim=1)[0].cpu().numpy()
                
                # 🎯 鎖定 AI 預測神經管的通道
                pred_mask = (pred_sem_mask_class == TARGET_PRED_CLASS).astype(np.float32)
                
                # --- D. 計算單張切片的 Metric ---
                # 為了避免大量沒有標註也沒有預測的切片洗高分數 (True Negatives)，
                # 我們可以選擇只記錄有標註或有預測的切片的分數
                if np.sum(gt_mask) > 0 or np.sum(pred_mask) > 0:
                    dice, iou = compute_2d_metrics(pred_mask, gt_mask)
                    patient_dices.append(dice)
                    patient_ious.append(iou)
                    overall_dices.append(dice)
                    overall_ious.append(iou)

            
            # 結算單一病人的平均 2D 分數 (只算有特徵的切片)
            if len(patient_dices) > 0:
                print(f"🌟 病人 {patient_id} 結算 (共 {len(patient_dices)} 張有效切片)：")
                print(f"   ➤ 平均 2D Dice: {np.mean(patient_dices):.4f}")
                print(f"   ➤ 平均 2D IoU:  {np.mean(patient_ious):.4f}")
            else:
                print(f"⚠️ 病人 {patient_id} 查無有效特徵切片。")
            
            print("-" * 50)
            
    # 4. 印出最終醫院資料集的平均 2D 分數
    if len(overall_dices) > 0:
        print(f"🏆 最終結算：醫院資料集平均 2D Dice Score: {np.mean(overall_dices):.4f}")
        print(f"🏆 最終結算：醫院資料集平均 2D IoU Score:  {np.mean(overall_ious):.4f}")
    else:
        print("❌ 無法計算總平均分數。")

if __name__ == "__main__":
    main()