import os
import cv2
import pydicom
import numpy as np
from tqdm import tqdm

# ==========================================
# 路徑設定
# ==========================================
PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
HOSPITAL_DIR = os.path.join(PROJECT_ROOT, "hospital_dataset") # 原始資料根目錄

TARGET_SIZE = (256, 256) # 配合訓練時的尺寸

def process_iac_dots(patient_id, patient_dir):
    # 【來源】：指向原始帶有醫生點的 dcm 資料夾
    label_dir = os.path.join(patient_dir, "label")
    
    # 【目的】：將轉換好的 npy 存在各病人的 0724mask 資料夾裡面
    mask_dir = os.path.join(patient_dir, "0724mask")
    
    if not os.path.exists(label_dir):
        return

    os.makedirs(mask_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(label_dir) if f.endswith(".dcm")])
    
    if len(files) == 0:
        return

    print(f"正在精準萃取病人 {patient_id} 的神經管標註點...")
    
    for fname in tqdm(files):
        dcm_path = os.path.join(label_dir, fname)
        
        # 1. 讀取 label 資料夾內的 DICOM
        try:
            dcm = pydicom.dcmread(dcm_path)
            img = dcm.pixel_array.astype(np.float32)
        except Exception as e:
            continue
            
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

        # 2. 尋找極端亮點
        max_val = img.max()
        mask = np.zeros(TARGET_SIZE, dtype=np.uint8)

        # 假設大於 1500 且極度接近最大值，才有可能是人工標記
        if max_val > 1500: 
            raw_mask = (img >= (max_val - 10)).astype(np.uint8)
            
            # 3. 濾除大面積雜訊 (例如假牙反光)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(raw_mask)
            
            for i in range(1, num_labels): 
                area = stats[i, cv2.CC_STAT_AREA]
                if area < 30: # 醫生的打點很小，面積大於 30 忽略
                    mask[labels == i] = 1

        # 4. 點膨脹 (Dilation)，讓點變明顯一點
        if mask.sum() > 0:
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        # 5. 存檔到新的 0724mask 資料夾，並改副檔名為 .npy
        out_name = fname.replace(".dcm", ".npy")
        np.save(os.path.join(mask_dir, out_name), mask)

def main():
    print("🚀 開始萃取神經管 (IAC) 專屬 Mask...")
    patients = sorted([d for d in os.listdir(HOSPITAL_DIR) if os.path.isdir(os.path.join(HOSPITAL_DIR, d))])
    
    for patient_id in patients:
        patient_dir = os.path.join(HOSPITAL_DIR, patient_id)
        # 處理可能的雙層資料夾 (例如 0017026056/0017026056/)
        if os.path.isdir(os.path.join(patient_dir, patient_id)):
            patient_dir = os.path.join(patient_dir, patient_id)
            
        process_iac_dots(patient_id, patient_dir)
        
    print(f"\n🎉 全部轉換完成！答案卷已儲存至各病人的 0724mask 資料夾中。")

if __name__ == "__main__":
    main()