import os
import json
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import hydra
import torch.nn.functional as F

# 🌟 匯入終極武器：MONAI 的混合損失函數
from monai.losses import DiceCELoss

from model import CoSeg
from sam2.build_sam import build_sam2

# ==========================================
# 📂 路徑設定 (請確認以下路徑與你的伺服器環境相符)
# ==========================================
PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
JSON_PATH = "data/public_data/train/data_split_2_13_14.json"
TRAIN_IMG_DIR = "data/public_data/train/image_1024"      # 訓練影像路徑
TRAIN_MASK_DIR = "data/public_data/train/mask_sem_1024"  # 訓練標籤路徑
OUTPUT_WEIGHTS = "outputs/coseg_public_mandibular_canal_best.pth"

class PublicDataset(Dataset):
    def __init__(self, img_dir, mask_dir, file_list):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.file_list = file_list
        # SAM2 預設的正規化參數
        self.pixel_mean = np.array([123.675, 116.280, 103.530], dtype=np.float32)
        self.pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        base_name = self.file_list[idx]
        
        # 讀取影像
        img_path = os.path.join(self.img_dir, base_name + ".png")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if len(img.shape) == 2: 
            img = np.stack([img]*3, axis=-1)
            
        # 讀取 Mask 答案卷
        mask_path = os.path.join(self.mask_dir, base_name + ".npy")
        gt_mask = np.load(mask_path).astype(np.float32)
        gt_mask = (gt_mask > 0).astype(np.float32)
        
        # 正規化並轉為 Tensor
        img = (img - self.pixel_mean) / (self.pixel_std + 1e-8)
        img_tensor = torch.tensor(img).permute(2, 0, 1).float()
        
        # MONAI DiceCELoss 需要的 Mask 維度為 (Channel, H, W)
        mask_tensor = torch.tensor(gt_mask).unsqueeze(0) 
        
        return img_tensor, mask_tensor

# 輔助計算 Dice 的函數 (用於顯示進度條分數)
def compute_dice(pred, target, smooth=1e-5):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 讀取 JSON 分割名單
    with open(JSON_PATH, 'r') as f:
        splits = json.load(f)
    
    # 2. 建立 DataLoader (啟動 num_workers 高速搬運)
    train_dataset = PublicDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, splits['train'])
    val_dataset = PublicDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, splits['val'])
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=4, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)
    }

    # 3. 初始化 SAM2 模型
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')
    model = CoSeg(build_sam2("sam2_hiera_l.yaml", None, mode="train"))
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # 4. 設定 Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)    
    # 🌟 5. 換上專剋微小目標的 Loss：BCE + Dice
    criterion = DiceCELoss(include_background=True, to_onehot_y=False, softmax=False, sigmoid=True)

    num_epochs = 30
    best_dice = 0.0

    print("🚀 開始訓練下顎管分割模型 (已啟用 DiceCELoss 與多執行緒加速)...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs-1}")
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_dice = 0.0
            
            pbar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()}")
            for imgs, masks in pbar:
                imgs = imgs.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        # 模型預測
                        _, pred_mask_sem, _, _ = model(x=imgs)
                        
                        # 🌟 新增這行：把 256x256 放大回 1024x1024 對齊答案卷
                        pred_mask_sem = F.interpolate(pred_mask_sem, size=(1024, 1024), mode='bilinear', align_corners=False)
                        
                        # 🌟 計算強化的 Loss
                        loss = criterion(pred_mask_sem, masks)
                    
                if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 計算 Dice 供畫面顯示
                probs = torch.sigmoid(pred_mask_sem.float())
                batch_dice = compute_dice(probs, masks)
                
                running_loss += loss.item() * imgs.size(0)
                running_dice += batch_dice * imgs.size(0)
                
                pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Dice': f"{batch_dice:.4f}"})

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_dice = running_dice / len(dataloaders[phase].dataset)
            
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} | Dice: {epoch_dice:.4f}")
            
            # 🌟 改成：只要跑完驗證，就直接強制儲存最新的權重
            if phase == 'val':
                torch.save(model.state_dict(), OUTPUT_WEIGHTS)
                print(f"💾 模型已強制更新並儲存至: {OUTPUT_WEIGHTS} (Epoch {epoch} 結算)")

if __name__ == "__main__":
    main()