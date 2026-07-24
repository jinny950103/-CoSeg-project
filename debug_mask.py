import os
import pydicom
import numpy as np

# 🌟 補上雙層資料夾的 0017026056
label_dir = "/home/u9444861/-CoSeg-project/hospital_dataset/0017026056/0017026056/label"
mask_dir = "/home/u9444861/-CoSeg-project/hospital_dataset/0017026056/0017026056/0724mask"

print(f"{'切片檔名':<15} | {'DICOM 最大亮度':<15} | {'NPY 標記像素數量'}")
print("-" * 55)

files = sorted([f for f in os.listdir(label_dir) if f.endswith(".dcm")])

# 抽查中間可能有神經管的切片 (第 80 到 90 張)
for fname in files[80:90]: 
    dcm_path = os.path.join(label_dir, fname)
    img = pydicom.dcmread(dcm_path).pixel_array
    
    npy_path = os.path.join(mask_dir, fname.replace(".dcm", ".npy"))
    if os.path.exists(npy_path):
        mask = np.load(npy_path)
        npy_sum = int(mask.sum())
    else:
        npy_sum = "檔案不存在"
        
    print(f"{fname:<15} | {img.max():<20} | {npy_sum}")