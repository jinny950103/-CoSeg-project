import os
import cv2
import torch
import numpy as np
import hydra
import torch.nn.functional as F
from tqdm import tqdm
from model import CoSeg
from sam2.build_sam import build_sam2

# --- 路徑設定 ---
PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
# 測試資料：公開資料集的 1, 3, 4 號病人
EVAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data/public_data/eval")
# 🌟 關鍵修改：使用「上一次訓練的醫院模型」權重
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "outputs/coseg_hospital_iac_best.pth")

def compute_3d_metrics(pred_volume, gt_volume):
    smooth = 1e-5
    intersection = np.sum(pred_volume * gt_volume)
    sum_pred = np.sum(pred_volume)
    sum_gt = np.sum(gt_volume)
    union = sum_pred + sum_gt - intersection
    
    if sum_pred == 0 and sum_gt == 0: return None, None
    return (2. * intersection + smooth) / (sum_pred + sum_gt + smooth), (intersection + smooth) / (union + smooth)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')
    
    # 初始化模型並載入舊權重
    model = CoSeg(build_sam2("sam2_hiera_l.yaml", None, mode=None))
    model = torch.nn.DataParallel(model).to(device) if torch.cuda.device_count() > 1 else model.to(device)
    
    print(f"📦 正在載入模型權重: {MODEL_WEIGHTS_PATH}")
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device), strict=True)
    model.eval()

    # 取得 eval 目錄下所有 Patient 資料夾
    patients = sorted([p for p in os.listdir(EVAL_DATA_DIR) if p.startswith("Patient_")])
    all_3d_dices, all_3d_ious = [], []

    print("🚀 開始 3D 影像外部驗證 (使用醫院模型評估公開資料) ...")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for patient_id in patients:
            img_dir = os.path.join(EVAL_DATA_DIR, patient_id, "image_1024")
            mask_dir = os.path.join(EVAL_DATA_DIR, patient_id, "mask_sem_1024")
            
            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
            if not img_files: continue
                
            pred_slices, gt_slices = [], []
            for img_f in tqdm(img_files, desc=f"評估 {patient_id}"):
                # 讀取 PNG 影像與 NPY Mask
                img = cv2.imread(os.path.join(img_dir, img_f), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                img = np.stack([img]*3, axis=-1) if len(img.shape) == 2 else img
                
                gt_mask = np.load(os.path.join(mask_dir, img_f.replace(".png", ".npy"))).astype(np.float32)
                
                # 影像標準化
                img = (img - np.array([123.675, 116.280, 103.530])) / (np.array([58.395, 57.12, 57.375]) + 1e-8)
                img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
                
                # 模型預測
                _, mask_pred_sem, _, _ = model(x=img_tensor)
                mask_pred_sem = torch.nn.functional.interpolate(mask_pred_sem, size=(1024, 1024), mode='bilinear', align_corners=False)
                
                # 轉機率與二值化 (擷取通道 0)
                pred_prob = torch.sigmoid(mask_pred_sem[:, 0, :, :])[0].cpu().numpy()
                pred_slices.append((pred_prob > 0.5).astype(np.float32))
                gt_slices.append((gt_mask > 0).astype(np.float32))

            # 計算這名病人的 3D Metrics
            dice, iou = compute_3d_metrics(np.stack(pred_slices), np.stack(gt_slices))
            if dice is not None:
                all_3d_dices.append(dice)
                all_3d_ious.append(iou)
                print(f"📊 {patient_id} 結算 ── 3D Dice: {dice:.4f} | IoU: {iou:.4f}")
            else:
                print(f"⚠️ {patient_id} 查無特徵。")

    if all_3d_dices:
        print(f"\n🏆 外部驗證全體平均 ── 3D Dice: {np.mean(all_3d_dices):.4f} | IoU: {np.mean(all_3d_ious):.4f}")

if __name__ == "__main__":
    main()