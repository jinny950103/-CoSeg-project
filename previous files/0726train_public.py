import os
import json
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import hydra
from monai.losses import DiceCELoss
from torch.optim import lr_scheduler  # 🌟 載入學習率衰減套件
import random  # 🌟 載入隨機套件，用於陷阱題機制

from model import CoSeg
from sam2.build_sam import build_sam2

# ==========================================
# 📂 路徑設定 
# ==========================================
PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
JSON_PATH = "data/public_data/train/data_split_2_13_14.json"
TRAIN_IMG_DIR = "data/public_data/train/image_1024"      
TRAIN_MASK_DIR = "data/public_data/train/mask_sem_1024"  
OUTPUT_WEIGHTS = "outputs/coseg_public_mandibular_canal_best.pth"

class PublicDataset(Dataset):
    def __init__(self, img_dir, mask_dir, file_list):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.file_list = file_list
        self.pixel_mean = np.array([123.675, 116.280, 103.530], dtype=np.float32)
        self.pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        base_name = self.file_list[idx]
        
        img_path = os.path.join(self.img_dir, base_name + ".png")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if len(img.shape) == 2: 
            img = np.stack([img]*3, axis=-1)
            
        mask_path = os.path.join(self.mask_dir, base_name + ".npy")
        gt_mask = np.load(mask_path).astype(np.float32)
        gt_mask = (gt_mask > 0).astype(np.float32)
        
        img = (img - self.pixel_mean) / (self.pixel_std + 1e-8)
        img_tensor = torch.tensor(img).permute(2, 0, 1).float()
        mask_tensor = torch.tensor(gt_mask).unsqueeze(0) 
        
        return img_tensor, mask_tensor

def compute_dice(pred, target, smooth=1e-5):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with open(JSON_PATH, 'r') as f:
        splits = json.load(f)
    
    train_dataset = PublicDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, splits['train'])
    val_dataset = PublicDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, splits['val'])
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=4, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)
    }

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')
    model = CoSeg(build_sam2("sam2_hiera_l.yaml", None, mode="train"))
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # ==========================================
    # 🌟 精準半凍結邏輯 (完美解放預測頭)
    # ==========================================
    for n, value in model.named_parameters():
        if "image_encoder" in n:
            if "edge" in n or "neck" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False
        else:
            # 預測頭等其他部分全部解凍，讓模型真正有辦法輸出預測
            value.requires_grad = True

    # 印出參數量檢查
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())
    print(f"🧠 可訓練參數量: {trainable_params/1e6:.2f}M / 總參數量: {total_params/1e6:.2f}M ({trainable_params/total_params*100:.2f}%)")

    # ==========================================
    # 🌟 優化器與學習率衰減設定
    # ==========================================
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    
    criterion = DiceCELoss(include_background=True, to_onehot_y=False, softmax=False, sigmoid=True)

    num_epochs = 80
    best_dice = 0.0

    print(f"🚀 開始訓練下顎管分割模型 (共 {num_epochs} Epochs，已啟用完美防禦機制)...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs-1}")
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_dice = 0.0
            valid_batches = 0
            
            pbar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()}")
            for imgs, masks in pbar:
                
                # ==========================================
                # 🌟 負樣本陷阱題機制
                # ==========================================
                if phase == 'train' and masks.sum() == 0:
                    # 空圖片有 85% 機率跳過，15% 放行
                    if random.random() > 0.15:
                        continue
                
                imgs = imgs.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        _, pred_mask_sem, _, _ = model(x=imgs)
                        pred_mask_sem = F.interpolate(pred_mask_sem,size=(1024,1024), mode='bilinear', align_corners=False)
                        
                        # 指定通道計算
                        pred_target = pred_mask_sem[:, 0:1, :, :]
                        loss = criterion(pred_target, masks)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                probs = torch.sigmoid(pred_target.float())
                batch_dice = compute_dice(probs, masks)
                
                running_loss += loss.item() * imgs.size(0)
                running_dice += batch_dice * imgs.size(0)
                valid_batches += imgs.size(0)
                
                pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Dice': f"{batch_dice:.4f}"})

            # 計算當前 Epoch 平均
            epoch_loss = running_loss / valid_batches if valid_batches > 0 else 0
            epoch_dice = running_dice / valid_batches if valid_batches > 0 else 0
            
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} | Dice: {epoch_dice:.4f}")
            
            # ==========================================
            # 🌟 聰明存檔法
            # ==========================================
            if phase == 'val':
                if epoch_dice > best_dice:
                    best_dice = epoch_dice
                    torch.save(model.state_dict(), OUTPUT_WEIGHTS)
                    print(f"🏆 破紀錄！最佳模型已更新並儲存 (Val Dice: {best_dice:.4f})")
                else:
                    print(f"  - 未破紀錄 (目前最佳 Val Dice: {best_dice:.4f})")
        
        # 每個 Epoch 結束後執行一次學習率衰減
        exp_lr_scheduler.step()
        print(f"📉 目前學習率 (Learning Rate): {exp_lr_scheduler.get_last_lr()[0]:.6f}")

if __name__ == "__main__":
    main()