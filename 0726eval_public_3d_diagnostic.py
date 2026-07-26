import os
import cv2
import torch
import numpy as np
import hydra
import torch.nn.functional as F
from tqdm import tqdm
from model import CoSeg
from sam2.build_sam import build_sam2

PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
EVAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data/public_data/eval")
# 💡 這裡可以先用你原本的權重，或是最新的 2.5D 權重來測
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "outputs/coseg_public_mandibular_canal_best.pth")

def calculate_3d_metrics(pred, gt, threshold=0.5):
    """依照前輩建議修改的嚴謹版 3D 計算與診斷程式"""
    pred = np.asarray(pred)
    gt = np.asarray(gt)

    if pred.shape != gt.shape:
        raise ValueError(f"Shape 不一致: Prediction={pred.shape}, GT={gt.shape}")

    pred_binary = pred >= threshold
    gt_binary = gt > 0

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()

    pred_count = pred_binary.sum()
    gt_count = gt_binary.sum()

    dice = (2.0 * intersection) / (pred_count + gt_count + 1e-8)
    iou = intersection / (union + 1e-8)

    print("\n--- 🩺 3D 空間對齊診斷報告 ---")
    print(f"🔸 GT shape: {gt.shape}")
    print(f"🔸 Pred shape: {pred.shape}")
    print(f"🔸 Ground Truth voxel 數量: {gt_count}")
    print(f"🔸 Prediction voxel 數量: {pred_count}")
    print(f"🔸 Pred / GT 體積比 (Ratio): {(pred_count / (gt_count + 1e-8)):.4f}")
    print(f"🔸 Intersection voxel 數量: {intersection}")
    
    # 找出含有特徵的 Z 軸切片範圍 (Positive Slices)
    gt_positive_slices = np.where(gt_binary.reshape(gt_binary.shape[0], -1).sum(axis=1) > 0)[0]
    pred_positive_slices = np.where(pred_binary.reshape(pred_binary.shape[0], -1).sum(axis=1) > 0)[0]
    
    if len(gt_positive_slices) > 0:
        print(f"🔸 GT 包含神經管的切片範圍: 第 {gt_positive_slices[0]} 張 ~ 第 {gt_positive_slices[-1]} 張")
    else:
        print("🔸 GT 包含神經管的切片範圍: 無")
        
    if len(pred_positive_slices) > 0:
        print(f"🔸 Pred 預測出特徵的切片範圍: 第 {pred_positive_slices[0]} 張 ~ 第 {pred_positive_slices[-1]} 張")
    else:
        print("🔸 Pred 預測出特徵的切片範圍: 無")
    print("--------------------------------")

    return dice, iou

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')
    
    model = CoSeg(build_sam2("sam2_hiera_l.yaml", None, mode="train"))
    model = torch.nn.DataParallel(model).to(device) if torch.cuda.device_count() > 1 else model.to(device)
    
    print(f"📦 正在載入模型權重: {MODEL_WEIGHTS_PATH}")
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
    new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(new_state_dict, strict=True)
    else:
        model.load_state_dict(new_state_dict, strict=True)
        
    model.eval()
    patients = sorted([p for p in os.listdir(EVAL_DATA_DIR) if p.startswith("Patient_")])
    all_3d_dices, all_3d_ious = [], []

    print("🚀 開始高精度 3D 診斷評估 (已修復檔案排序問題) ...")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for patient_id in patients:
            img_dir = os.path.join(EVAL_DATA_DIR, patient_id, "image_1024")
            mask_dir = os.path.join(EVAL_DATA_DIR, patient_id, "mask_sem_1024")
            
            # ==========================================
            # 🌟 致命錯誤修復：改用數值精準排序，確保 Z 軸正確
            # ==========================================
            def get_slice_number(filename):
                # 從 "Patient_1_VOL_1_slice_313.png" 提取數字 313
                return int(filename.split('_slice_')[1].split('.png')[0])
                
            img_files = [f for f in os.listdir(img_dir) if f.endswith(".png")]
            img_files.sort(key=get_slice_number) 
            if not img_files: continue
            # ==========================================
                
            pred_slices, gt_slices = [], []
            for img_f in tqdm(img_files, desc=f"評估 {patient_id}"):
                img_original = cv2.imread(os.path.join(img_dir, img_f), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                img = np.stack([img_original]*3, axis=-1) if len(img_original.shape) == 2 else img_original
                
                gt_mask = np.load(os.path.join(mask_dir, img_f.replace(".png", ".npy"))).astype(np.float32)
                
                img_norm = (img - np.array([123.675, 116.280, 103.530])) / (np.array([58.395, 57.12, 57.375]) + 1e-8)
                img_tensor = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
                
                _, mask_pred_sem, _, _ = model(x=img_tensor)
                mask_pred_sem = torch.nn.functional.interpolate(mask_pred_sem, size=(1024, 1024), mode='bilinear', align_corners=False)
                
                pred_prob = torch.sigmoid(mask_pred_sem[:, 0, :, :])[0].cpu().numpy()
                # 這裡保留機率值，交給後面的 calculate_3d_metrics 進行 Threshold 二值化
                pred_slices.append(pred_prob)
                gt_slices.append(gt_mask)

            print(f"\n[{patient_id} 評估結果]")
            # 🌟 使用 0.5 作為預設門檻，呼叫前輩寫的診斷公式
            dice, iou = calculate_3d_metrics(np.stack(pred_slices), np.stack(gt_slices), threshold=0.5)
            
            all_3d_dices.append(dice)
            all_3d_ious.append(iou)
            print(f"📊 最終結算 ── 3D Dice: {dice:.4f} | IoU: {iou:.4f}\n")

    if all_3d_dices:
        print(f"🏆 測試集全體病人平均 ── 3D Dice: {np.mean(all_3d_dices):.4f} | IoU: {np.mean(all_3d_ious):.4f}")

if __name__ == "__main__":
    main()