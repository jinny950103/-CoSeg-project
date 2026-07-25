import os
import cv2
import pydicom
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
HOSPITAL_DIR = os.path.join(PROJECT_ROOT, "hospital_dataset")
TARGET_SIZE = (1024, 1024)

def process_iac_dots(patient_id, patient_dir):
    label_dir = os.path.join(patient_dir, "label")
    # 🌟 修改存檔路徑到全新的 0725mask 資料夾
    mask_dir = os.path.join(patient_dir, "0725mask")
    
    if not os.path.exists(label_dir):
        return

    os.makedirs(mask_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(label_dir) if f.endswith(".dcm")])
    if len(files) == 0:
        return

    print(f"正在以 1024x1024 精準萃取病人 {patient_id} 的神經管標註點 (存至 0725mask)...")
    
    for fname in tqdm(files):
        dcm_path = os.path.join(label_dir, fname)
        try:
            dcm = pydicom.dcmread(dcm_path)
            img = dcm.pixel_array.astype(np.float32)
        except Exception:
            continue
            
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

        max_val = img.max()
        mask = np.zeros(TARGET_SIZE, dtype=np.uint8)

        if max_val > 1500: 
            raw_mask = (img >= (max_val - 10)).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(raw_mask)
            
            for i in range(1, num_labels): 
                area = stats[i, cv2.CC_STAT_AREA]
                if area < 300: 
                    mask[labels == i] = 1

        if mask.sum() > 0:
            kernel = np.ones((9, 9), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        out_name = fname.replace(".dcm", ".npy")
        np.save(os.path.join(mask_dir, out_name), mask)

def main():
    print("🚀 開始萃取 1024x1024 神經管 Mask，並建立 0725mask 資料夾...")
    patients = sorted([d for d in os.listdir(HOSPITAL_DIR) if os.path.isdir(os.path.join(HOSPITAL_DIR, d))])
    
    for patient_id in patients:
        patient_dir = os.path.join(HOSPITAL_DIR, patient_id)
        if os.path.isdir(os.path.join(patient_dir, patient_id)):
            patient_dir = os.path.join(patient_dir, patient_id)
            process_iac_dots(patient_id, patient_dir)
            
    print(f"\n🎉 重新生成完畢！所有答案卷已安全存入 0725mask。")

if __name__ == "__main__":
    main()