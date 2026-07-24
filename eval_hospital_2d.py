import os
import cv2
import torch
import pydicom
import numpy as np
import hydra
import torch.nn.functional as F
from tqdm import tqdm

# --- 載入專案中的模型架構 ---
from model import CoSeg
from sam2.build_sam import build_sam2

# ==========================================
# 參數設定區
# ==========================================
PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
HOSPITAL_DATA_DIR = os.path.join(PROJECT_ROOT, "hospital_dataset")
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "coseg_puma_latest.pth")

# SAM2 的預設設定檔
SAM2_CFG = "sam2_hiera_l.yaml"
# ==========================================

def compute_2d_metrics(pred_mask, gt_mask):
    """計算單張 2D 切片的 Dice Score 與 IoU Score"""
    smooth = 1e-5
    
    intersection = np.sum(pred_mask * gt_mask)
    sum_pred = np.sum(pred_mask)
    sum_gt = np.sum(gt_mask)
    
    # 如果預測和答案都是全黑（空白切片），直接回傳 None (代表不計分)
    if sum_pred == 0 and sum_gt == 0:
        return None, None
        
    union = sum_pred + sum_gt - intersection
    dice_score = (2. * intersection + smooth) / (sum_pred + sum_gt + smooth)
    iou_score = (intersection + smooth) / (union + smooth)

    return dice_score, iou_score

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用的運算設備: {device}")

    # ---------------------------------------------------------
    # 1. 載入訓練好的模型
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 2. 獲取所有病人的資料夾
    # ---------------------------------------------------------
    patients = [p for p in os.listdir(HOSPITAL_DATA_DIR) if os.path.isdir(os.path.join(HOSPITAL_DATA_DIR, p))]
    patients.sort()
    print(f"找到 {len(patients)} 位病人的測試資料：{patients}")
    print("-" * 50)

    # ---------------------------------------------------------
    # 3. 針對每個病人獨立進行預測與評估
    # ---------------------------------------------------------
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for patient_id in patients:
            print(f"\n正在評估病人: {patient_id} ...")
            
            # 對應路徑結構： hospital_dataset/0017026056/0017026056/no_label/
            base_dir = os.path.join(HOSPITAL_DATA_DIR, patient_id)
            if os.path.isdir(os.path.join(base_dir, patient_id)):
                base_dir = os.path.join(base_dir, patient_id)
                
            img_dir = os.path.join(base_dir, "no_label")
            label_dir = os.path.join(base_dir, "label")
            
            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".dcm")])
            label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".dcm")])
            
            if len(img_files) == 0:
                print(f"⚠️ 警告：在 {img_dir} 找不到任何 dcm 檔案，跳過此病人。")
                continue
                
            assert len(img_files) == len(label_files), f"❌ {patient_id} 的影像與標註數量不一致！"
            
            slice_dices = []
            slice_ious = []
            
            # 設定抓取這個病人「正中間」的那張切片來 Debug
            middle_idx = len(img_files) // 2
            
            for idx, (img_f, lbl_f) in enumerate(tqdm(zip(img_files, label_files), total=len(img_files), desc="處理切片中")):
                img_path = os.path.join(img_dir, img_f)
                lbl_path = os.path.join(label_dir, lbl_f)
                
                # --- A. 處理輸入影像 (模擬 dataloader.py 的 PNG 轉換與標準化) ---
                dcm_img = pydicom.dcmread(img_path)
                img = dcm_img.pixel_array.astype(np.float32)
                
                # 將 DICOM 原始數值拉伸並轉換為 0~255
                img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0
                
                if len(img.shape) == 2:
                    img = np.stack([img]*3, axis=-1)
                
                img = cv2.resize(img, (1024, 1024))
                
                # 嚴格套用 dataloader.py 中的 mean 與 std
                pixel_mean = np.array([123.675, 116.280, 103.530], dtype=np.float32)
                pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
                img = (img - pixel_mean) / (pixel_std + 1e-8)
                
                img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # --- B. 處理標準答案 (label) ---
                dcm_lbl = pydicom.dcmread(lbl_path)
                gt = dcm_lbl.pixel_array.astype(np.float32)
                gt = cv2.resize(gt, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                
                # 🎯 關鍵修改：只把標籤為 5 (下顎神經管) 的地方變成答案！
                gt = (gt == 5).astype(np.float32) 
                
                # --- C. AI 模型預測 ---
                mask_pred_ins, mask_pred_sem, _, _ = model(x=img_tensor)
                mask_pred_ins, mask_pred_sem = model(x=img_tensor, prob_ins=mask_pred_ins, prob_sem=mask_pred_sem)
                
                # 1. Ins 通道
                pred_ins_prob = torch.sigmoid(mask_pred_ins[:, 0:1, :, :])[0, 0].cpu().numpy()
                pred_ins_mask = (pred_ins_prob >= 0.1).astype(np.float32)
                
                # 2. Sem 通道
                pred_sem_prob = torch.softmax(mask_pred_sem, dim=1)
                pred_sem_mask = torch.argmax(pred_sem_prob, dim=1)[0].cpu().numpy().astype(np.float32)
                pred_sem_mask = (pred_sem_mask > 0).astype(np.float32)
                
                # 🚨 開天眼：擷取「正中間」的切片
                if patient_id == patients[0] and idx == middle_idx:
                    cv2.imwrite("debug_1_GT_label_middle.png", gt * 255)
                    cv2.imwrite("debug_2_Pred_Ins_middle.png", pred_ins_mask * 255)
                    cv2.imwrite("debug_3_Pred_Sem_middle.png", pred_sem_mask * 255)
                    print(f"\n📸 已輸出第 {middle_idx} 張切片的 debug 圖片！")

                # 我們改用 Sem 通道來算分數
                d_score, i_score = compute_2d_metrics(pred_sem_mask, gt)
                
                # 如果這張切片不是完全空白的，我們才把它加入平均值計算
                if d_score is not None:
                    slice_dices.append(d_score)
                    slice_ious.append(i_score)
                
            # 4. 結算該病人的平均分數
            if len(slice_dices) > 0:
                patient_avg_dice = np.mean(slice_dices)
                patient_avg_iou = np.mean(slice_ious)
            else:
                patient_avg_dice = 0.0
                patient_avg_iou = 0.0
            
            print(f"🌟 病人 {patient_id} 結算 (僅計算含有目標的切片)：")
            print(f"   ➤ 有效切片數量: {len(slice_dices)} / {len(img_files)}")
            print(f"   ➤ 2D 平均 Dice Score: {patient_avg_dice:.4f}")
            print(f"   ➤ 2D 平均 IoU Score:  {patient_avg_iou:.4f}")
            print("-" * 50)

if __name__ == "__main__":
    main()