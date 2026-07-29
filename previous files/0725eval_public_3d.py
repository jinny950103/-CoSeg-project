import os
import cv2
import torch
import numpy as np
import hydra
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from model import CoSeg
from sam2.build_sam import build_sam2

PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
EVAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data/public_data/eval")
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "outputs/coseg_public_mandibular_canal_best.pth")

class PatientDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_files):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = img_files
        self.pixel_mean = np.array([123.675, 116.280, 103.530], dtype=np.float32)
        self.pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_f = self.img_files[idx]
        
        # 讀取影像
        img_path = os.path.join(self.img_dir, img_f)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if len(img.shape) == 2: 
            img = np.stack([img]*3, axis=-1)
            
        # 讀取 Mask
        mask_path = os.path.join(self.mask_dir, img_f.replace(".png", ".npy"))
        gt_mask = np.load(mask_path).astype(np.float32)
        gt_mask = (gt_mask > 0).astype(np.float32)
        
        # 正規化與轉 Tensor
        img = (img - self.pixel_mean) / (self.pixel_std + 1e-8)
        img_tensor = torch.tensor(img).permute(2, 0, 1).float()
        
        return img_tensor, torch.tensor(gt_mask)

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
    
    # 建立模型與包裝
    model = CoSeg(build_sam2("sam2_hiera_l.yaml", None, mode="train"))
    model = torch.nn.DataParallel(model).to(device) if torch.cuda.device_count() > 1 else model.to(device)

    # 🌟 強制載入權重並自動清除可能多出來的 "module." 前綴
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(new_state_dict, strict=True)
    else:
        model.load_state_dict(new_state_dict, strict=True)
        
    model.eval()

    patients = sorted([p for p in os.listdir(EVAL_DATA_DIR) if p.startswith("Patient_")])
    all_3d_dices, all_3d_ious = [], []

    print("🚀 開始高速 3D 影像評估 (已修復權重載入與通道偵測) ...")
    
    target_channel = None 

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for patient_id in patients:
            img_dir = os.path.join(EVAL_DATA_DIR, patient_id, "image_1024")
            mask_dir = os.path.join(EVAL_DATA_DIR, patient_id, "mask_sem_1024")
            
            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
            if not img_files: continue
            
            dataset = PatientDataset(img_dir, mask_dir, img_files)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
            
            pred_slices, gt_slices = [], []
            
            for imgs, gt_masks in tqdm(dataloader, desc=f"評估 {patient_id}"):
                imgs = imgs.to(device)
                
                _, mask_pred_sem, _, _ = model(x=imgs)
                mask_pred_sem = F.interpolate(mask_pred_sem, size=(1024, 1024), mode='bilinear', align_corners=False)
                
                if target_channel is None:
                    max_probs = [torch.sigmoid(mask_pred_sem[:, c:c+1, :, :]).max().item() for c in range(mask_pred_sem.shape[1])]
                    target_channel = int(np.argmax(max_probs))
                    print(f"🌟 自動鎖定目標通道：Channel {target_channel} (各通道最高機率: {[f'{p:.4f}' for p in max_probs]})")

                pred_target = mask_pred_sem[:, target_channel:target_channel+1, :, :]
                pred_probs = torch.sigmoid(pred_target).squeeze(1).cpu().numpy()
                
                for i in range(pred_probs.shape[0]):
                    # 🌟 暫時印出測試集的真實機率極值，看看它到底猜了什麼
                    print(f"測試集預測的最大機率值: {pred_probs[i].max():.4f}")
                    pred_slices.append((pred_probs[i] > 0.1).astype(np.float32)) # 把門檻降到 0.1 試試看                    
                    gt_slices.append(gt_masks[i].numpy())

            dice, iou = compute_3d_metrics(np.stack(pred_slices), np.stack(gt_slices))
            if dice is not None:
                all_3d_dices.append(dice)
                all_3d_ious.append(iou)
                print(f"📊 {patient_id} 結算 ── 3D Dice: {dice:.4f} | IoU: {iou:.4f}")
            else:
                print(f"⚠️ {patient_id} 查無特徵。")

    if all_3d_dices:
        print(f"\n🏆 測試集全體病人平均 ── 3D Dice: {np.mean(all_3d_dices):.4f} | IoU: {np.mean(all_3d_ious):.4f}")

if __name__ == "__main__":
    main()