import os
import cv2
import time
import json
import argparse
import numpy as np
from tqdm import tqdm
import pydicom

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

import albumentations as A
import hydra
from sam2.build_sam import build_sam2
from model import CoSeg
from monai.losses import DiceCELoss
from loss import DiceEval  # 確保你們的 loss.py 裡有這個評估函數

# ==========================================
# 1. 專屬 DataLoader: HospitalDataset
# ==========================================
class HospitalDataset(Dataset):
    def __init__(self, json_file, split_name, img_dir, mask_dir, transforms=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        
        # 讀取 JSON 名單 (train, valid, test)
        with open(json_file, 'r') as f:
            df = json.load(f)
        self.file_list = df[split_name]
        
        # SAM2 預設的正規化參數與尺寸
        self.sam_img_size = 1024
        self.pixel_mean = torch.Tensor([123.675, 116.280, 103.530]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        
        # 組裝路徑
        img_path = os.path.join(self.img_dir, file_name)
        # 檔名從 .png 換回 .npy (如果你的 json 裡面寫的是 .png)
        mask_name = file_name.replace(".png", ".npy")
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 讀取影像 (PNG 格式)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        # 如果是灰階，疊加成 3 通道
        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)
            
        # 讀取 Mask (.npy)
        if os.path.exists(mask_path):
            mask = np.load(mask_path).astype(np.float32)
        else:
            # 防呆：萬一名單有，但沒產出 mask，就給全黑
            mask = np.zeros(img.shape[:2], dtype=np.float32)
            
        # 二值化防呆
        mask = (mask > 0).astype(np.float32)

        # 資料擴增與縮放 (1024x1024)
        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # 轉換為 Tensor (C, H, W)
        img_tensor = torch.tensor(img).permute(2, 0, 1).float()
        mask_tensor = torch.tensor(mask).unsqueeze(0).float() # (1, H, W)

        # SAM 正規化
        img_tensor = (img_tensor - self.pixel_mean) / (self.pixel_std + 1e-8)

        # Pad (以防萬一尺寸不對)
        h, w = img_tensor.shape[-2:]
        padh = self.sam_img_size - h
        padw = self.sam_img_size - w
        if padh > 0 or padw > 0:
            img_tensor = F.pad(img_tensor, (0, padw, 0, padh))
            mask_tensor = F.pad(mask_tensor, (0, padw, 0, padh))

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "file_name": file_name
        }

# ==========================================
# 2. 訓練核心循環
# ==========================================
def train_model(model, dataloaders, optimizer, scheduler, num_epochs, save_name):
    since = time.time()
    best_loss = float('inf')
    
    # 簡化：只關注神經管的二元分類 Loss
    dice_ce_loss_b = DiceCELoss(include_background=True, to_onehot_y=False, softmax=False, sigmoid=True)
    accuracy_metric = DiceEval(include_background=True, to_onehot_y=False, softmax=False, sigmoid=True)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)  

            running_loss = []
            running_dice = []

            for data_dict in tqdm(dataloaders[phase]):      
                img = Variable(data_dict["image"].cuda())
                gt_mask = Variable(data_dict["mask"].cuda())

                optimizer.zero_grad()

                # 前向傳播
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if phase == 'train':
                        pred_mask_ins, pred_mask_sem, _, _ = model(x=img)
                        # 將模型預測調整至 1024x1024 (如果模型輸出較小)
                        pred_mask_sem_1024 = F.interpolate(pred_mask_sem, size=(1024, 1024), mode='bilinear', align_corners=False)
                        
                        # 這裡我們將語義分割的第一個通道 (或你需要指定哪個) 視為神經管預測
                        # 為了簡化，我們直接拿 pred_mask_sem 來對齊 gt_mask
                        # 注意：依據你 eval 的寫法，你可能預測多個類別，我們這裡只針對 TARGET_PRED_CLASS 進行優化
                        # 假設通道 1 是神經管
                        pred_target = pred_mask_sem_1024[:, 0:1, :, :] 
                        
                        loss = dice_ce_loss_b(pred_target, gt_mask)
                        
                        loss.backward()
                        optimizer.step()
                    else:
                        with torch.no_grad():
                            pred_mask_ins, pred_mask_sem, _, _ = model(x=img)
                            pred_mask_sem_1024 = F.interpolate(pred_mask_sem, size=(1024, 1024), mode='bilinear', align_corners=False)
                            pred_target = pred_mask_sem_1024[:, 0:1, :, :] 
                            loss = dice_ce_loss_b(pred_target, gt_mask)

                    # 評估 Dice
                    dice_score = accuracy_metric(pred_target, gt_mask)

                running_loss.append(loss.item())
                running_dice.append(dice_score.mean().item())

            epoch_loss = np.mean(running_loss)
            epoch_dice = np.mean(running_dice)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} | Dice: {epoch_dice:.4f}')

            # 儲存最佳模型 (不再覆蓋 latest，使用專屬名稱)
            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    save_path = f'outputs/{save_name}_best.pth'
                    torch.save(best_model_wts, save_path)
                    print(f"🌟 新紀錄！模型已儲存至: {save_path}")

                scheduler.step()
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

# ==========================================
# 3. 主程式入口
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hospital', help='資料集名稱')  
    parser.add_argument('--jsonfile', type=str, default='data/data_split.json', help='名單位置')
    parser.add_argument('--img_dir', type=str, default='data/image_1024', help='影像資料夾')
    parser.add_argument('--mask_dir', type=str, default='data/mask_sem_1024', help='答案卷資料夾')
    parser.add_argument('--batch', type=int, default=2, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=50, help='訓練回合數')
    # 給這個模型一個新的名字，避免覆蓋原本的
    parser.add_argument('--save_name', type=str, default='coseg_hospital_iac', help='儲存的模型名稱')
    args = parser.parse_args()

    os.makedirs('outputs/', exist_ok=True)

    # 建立 Dataset 與 DataLoader
    transforms = A.Compose([A.Resize(1024, 1024)])
    
    train_dataset = HospitalDataset(args.jsonfile, 'train', args.img_dir, args.mask_dir, transforms)
    val_dataset = HospitalDataset(args.jsonfile, 'valid', args.img_dir, args.mask_dir, transforms)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)
    
    dataloaders = {'train': train_loader, 'valid': val_loader}
    
    # 初始化 SAM2 模型
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')
    model = CoSeg(build_sam2("sam2_hiera_l.yaml", None, mode="train"))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.cuda()

    # 凍結不需要訓練的層，只開 edge/neck (依照前輩的邏輯)
    for n, value in model.module.model.image_encoder.named_parameters() if isinstance(model, nn.DataParallel) else model.model.image_encoder.named_parameters():
        if ("edge" in n) or ("neck" in n):
            value.requires_grad = True
        else:
            value.requires_grad = False

    # 優化器設定
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    
    print(f"🚀 開始針對 {args.dataset} 進行客製化訓練...")
    train_model(model, dataloaders, optimizer, exp_lr_scheduler, num_epochs=args.epoch, save_name=args.save_name)