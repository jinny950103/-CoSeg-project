"""
train_v4.py — 下顎神經管分割訓練腳本 (v4 — 載入 SAM2 預訓練權重)
=================================================================
v3 → v4 改良重點：
  1. ✅ 載入 SAM2 Hiera-Large 預訓練權重（最關鍵的修正）
  2. ✅ 修正 boundary loss autocast 問題
  3. ✅ 加速：BATCH_SIZE=16, NUM_WORKERS=8, 每 3 epoch 驗證
  4. ✅ 其餘保留 v3 全部改進
"""

import os
import json
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import torch.nn.functional as F
import hydra
import random
import math
import gc
from scipy.ndimage import gaussian_filter, map_coordinates

from model import CoSeg
from sam2.build_sam import build_sam2

# ==========================================
# 📂 路徑設定 — 修改這裡對齊新 JSON
# ==========================================
PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
JSON_PATH = "data/public_data/train/data_split_2_5_6_7_8_13_14.json"
TRAIN_IMG_DIR = "data/public_data/train/image_1024"
TRAIN_MASK_DIR = "data/public_data/train/mask_sem_1024"
OUTPUT_DIR = "outputs"
OUTPUT_WEIGHTS = os.path.join(OUTPUT_DIR, "coseg_v4_best.pth")
OUTPUT_LAST = os.path.join(OUTPUT_DIR, "coseg_v4_last.pth")
SAM2_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"

# ==========================================
# 🎛️ 超參數
# ==========================================
BATCH_SIZE = 16
ACCUM_STEPS = 1
NUM_EPOCHS = 150
BASE_LR = 1e-4
WARMUP_EPOCHS = 5
NUM_WORKERS = 8
POSITIVE_WEIGHT = 4.0     # 正樣本過採樣權重（提高）
CONTEXT_SLICES = 1
AUG_PROB = 0.6             # 增強機率（提高）


# ==========================================
# 🔧 資料增強 v3（大幅強化版）
# ==========================================
def elastic_deform(image, mask, alpha=1000, sigma=25):
    """更強的彈性變形"""
    shape = image.shape[:2]
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices_y = np.clip(y + dy, 0, shape[0] - 1).astype(np.float64)
    indices_x = np.clip(x + dx, 0, shape[1] - 1).astype(np.float64)

    if len(image.shape) == 3:
        result_img = np.zeros_like(image)
        for c in range(image.shape[2]):
            result_img[:, :, c] = map_coordinates(
                image[:, :, c], [indices_y, indices_x], order=1, mode='reflect'
            )
    else:
        result_img = map_coordinates(image, [indices_y, indices_x], order=1, mode='reflect')

    result_mask = map_coordinates(mask, [indices_y, indices_x], order=0, mode='reflect')
    return result_img, result_mask


def random_affine(image, mask, max_rotation=20, max_scale=0.15, max_shear=10):
    """隨機仿射變換：旋轉 + 縮放 + 剪切"""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    angle = random.uniform(-max_rotation, max_rotation)
    scale = random.uniform(1 - max_scale, 1 + max_scale)

    M = cv2.getRotationMatrix2D(center, angle, scale)

    # 加入剪切
    shear_x = math.tan(math.radians(random.uniform(-max_shear, max_shear)))
    shear_y = math.tan(math.radians(random.uniform(-max_shear, max_shear)))
    S = np.array([[1, shear_x, 0], [shear_y, 1, 0]], dtype=np.float64)
    M = M + S * 0.3  # 混合剪切

    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_REFLECT_101)
    return image, mask


def copy_paste_canal(image, mask, all_positive_crops):
    """
    Copy-Paste 增強：從其他正樣本切片複製神經管區域貼到當前切片。
    這讓模型看到更多位置的神經管。
    """
    if not all_positive_crops or random.random() > 0.3:
        return image, mask

    crop_img, crop_mask = random.choice(all_positive_crops)

    # 找到 crop_mask 中神經管的 bounding box
    ys, xs = np.where(crop_mask > 0)
    if len(ys) == 0:
        return image, mask

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # 加一些 padding
    pad = 20
    y_min = max(0, y_min - pad)
    y_max = min(crop_mask.shape[0] - 1, y_max + pad)
    x_min = max(0, x_min - pad)
    x_max = min(crop_mask.shape[1] - 1, x_max + pad)

    patch_mask = crop_mask[y_min:y_max+1, x_min:x_max+1]
    patch_img = crop_img[y_min:y_max+1, x_min:x_max+1]

    # 隨機偏移貼到目標位置
    max_offset_y = image.shape[0] - patch_mask.shape[0]
    max_offset_x = image.shape[1] - patch_mask.shape[1]
    if max_offset_y <= 0 or max_offset_x <= 0:
        return image, mask

    offset_y = random.randint(0, max_offset_y)
    offset_x = random.randint(0, max_offset_x)

    # 只貼有神經管的像素
    region_mask = patch_mask > 0
    ph, pw = patch_mask.shape

    if len(image.shape) == 3:
        for c in range(image.shape[2]):
            target = image[offset_y:offset_y+ph, offset_x:offset_x+pw, c]
            source = patch_img[..., c] if len(patch_img.shape) == 3 else patch_img
            target[region_mask] = source[region_mask]
    else:
        image[offset_y:offset_y+ph, offset_x:offset_x+pw][region_mask] = patch_img[region_mask]

    mask[offset_y:offset_y+ph, offset_x:offset_x+pw][region_mask] = 1.0

    return image, mask


def augment_v3(image, mask, positive_crops=None):
    """
    v3 增強 pipeline（更強）
    """
    # 1. 水平翻轉
    if random.random() < AUG_PROB:
        image = np.flip(image, axis=1).copy()
        mask = np.flip(mask, axis=1).copy()

    # 2. 垂直翻轉
    if random.random() < AUG_PROB * 0.5:
        image = np.flip(image, axis=0).copy()
        mask = np.flip(mask, axis=0).copy()

    # 3. 隨機仿射（旋轉+縮放+剪切）— 更強
    if random.random() < AUG_PROB:
        image, mask = random_affine(image, mask, max_rotation=20, max_scale=0.15, max_shear=10)

    # 4. 彈性變形（更強、更高機率）
    if random.random() < 0.15:
        image, mask = elastic_deform(image, mask, alpha=1000, sigma=25)

    # 5. Copy-Paste 增強（關鍵！讓模型看到更多位置的神經管）
    if positive_crops and mask.sum() > 0:
        image, mask = copy_paste_canal(image, mask, positive_crops)

    # 6. 亮度 / 對比度（更強範圍）
    if random.random() < AUG_PROB:
        alpha = random.uniform(0.7, 1.3)
        beta = random.uniform(-20, 20)
        image = np.clip(alpha * image + beta, 0, 255)

    # 7. Gamma
    if random.random() < AUG_PROB * 0.5:
        gamma = random.uniform(0.6, 1.5)
        image = np.clip(np.power(image / 255.0, gamma) * 255.0, 0, 255)

    # 8. 高斯雜訊（更強）
    if random.random() < AUG_PROB * 0.4:
        noise = np.random.normal(0, random.uniform(5, 15), image.shape).astype(np.float32)
        image = np.clip(image + noise, 0, 255)

    # 9. 高斯模糊
    if random.random() < AUG_PROB * 0.3:
        ksize = random.choice([3, 5, 7])
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)

    # 10. 隨機遮蔽（Cutout）— 讓模型更 robust
    if random.random() < 0.2:
        h, w = image.shape[:2]
        cut_h = random.randint(50, 150)
        cut_w = random.randint(50, 150)
        cy = random.randint(0, h - cut_h)
        cx = random.randint(0, w - cut_w)
        image[cy:cy+cut_h, cx:cx+cut_w] = 0

    return image.astype(np.float32), (mask > 0.5).astype(np.float32)


# ==========================================
# 📦 2.5D Dataset v3
# ==========================================
class MandibularCanal25DDatasetV3(Dataset):
    def __init__(self, img_dir, mask_dir, file_list, is_train=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        self.pixel_mean = np.array([123.675, 116.280, 103.530], dtype=np.float32)
        self.pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        self.file_list = file_list
        self.patient_slice_map = {}
        self.entries = []

        for fname in file_list:
            parts = fname.rsplit('_slice_', 1)
            patient = parts[0]
            slice_num = int(parts[1])
            if patient not in self.patient_slice_map:
                self.patient_slice_map[patient] = {}
            self.patient_slice_map[patient][slice_num] = fname
            self.entries.append((patient, slice_num, fname))

        # 掃描正/負樣本
        self.has_canal = []
        self.positive_crops = []  # 儲存正樣本的 crop 供 Copy-Paste 用
        print(f"📊 掃描正/負樣本分布 ({'train' if is_train else 'val'})...")
        for patient, slice_num, fname in tqdm(self.entries, desc="Scanning"):
            mask_path = os.path.join(self.mask_dir, fname + ".npy")
            if os.path.exists(mask_path):
                m = np.load(mask_path)
                is_pos = m.sum() > 0
                self.has_canal.append(is_pos)

                # 收集正樣本 crop（只在訓練時且限制數量）
                if is_train and is_pos and len(self.positive_crops) < 200:
                    img_path = os.path.join(self.img_dir, fname + ".png")
                    if os.path.exists(img_path):
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                        self.positive_crops.append((img, m.astype(np.float32)))
            else:
                self.has_canal.append(False)

        n_pos = sum(self.has_canal)
        n_neg = len(self.has_canal) - n_pos
        print(f"   ✅ 正樣本: {n_pos} | ❌ 負樣本: {n_neg} | Copy-Paste pool: {len(self.positive_crops)}")

    def get_sample_weights(self):
        weights = []
        for has in self.has_canal:
            weights.append(POSITIVE_WEIGHT if has else 1.0)
        return weights

    def _load_slice_gray(self, patient, slice_num):
        if patient in self.patient_slice_map and slice_num in self.patient_slice_map[patient]:
            fname = self.patient_slice_map[patient][slice_num]
        else:
            fname = f"{patient}_slice_{slice_num}"

        img_path = os.path.join(self.img_dir, fname + ".png")
        if os.path.exists(img_path):
            return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        return None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        patient, slice_num, fname = self.entries[idx]

        center = self._load_slice_gray(patient, slice_num)
        prev_slice = self._load_slice_gray(patient, slice_num - CONTEXT_SLICES)
        next_slice = self._load_slice_gray(patient, slice_num + CONTEXT_SLICES)

        if prev_slice is None:
            prev_slice = center.copy()
        if next_slice is None:
            next_slice = center.copy()

        img = np.stack([prev_slice, center, next_slice], axis=-1)

        mask_path = os.path.join(self.mask_dir, fname + ".npy")
        gt_mask = np.load(mask_path).astype(np.float32)
        gt_mask = (gt_mask > 0).astype(np.float32)

        if self.is_train:
            img, gt_mask = augment_v3(img, gt_mask, self.positive_crops)

        img = (img - self.pixel_mean) / (self.pixel_std + 1e-8)
        img_tensor = torch.tensor(img).permute(2, 0, 1).float()
        mask_tensor = torch.tensor(gt_mask).unsqueeze(0).float()

        return img_tensor, mask_tensor


# ==========================================
# 🎯 損失函式：Focal + Dice + Boundary Awareness
# ==========================================
class FocalDiceBoundaryLoss(nn.Module):
    """
    v3 Loss = Focal + Dice + Boundary Loss
    Boundary Loss 讓模型更注意邊界像素
    """
    def __init__(self, focal_alpha=0.75, focal_gamma=2.0,
                 dice_weight=1.0, focal_weight=1.0, boundary_weight=0.5):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight

    def focal_loss(self, pred, target):
        pred = pred.clamp(1e-6, 1 - 1e-6)
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_term = (1 - p_t) ** self.focal_gamma
        loss = -alpha_t * focal_term * torch.log(p_t)
        return loss.mean()

    def dice_loss(self, pred, target, smooth=1e-5):
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def boundary_loss(self, pred, target):
        """邊界 Loss：用 Laplacian 抽出邊界，加強邊界區域的監督"""
        # 用 max pooling 做簡易邊界偵測
        kernel_size = 3
        target_dilated = F.max_pool2d(target, kernel_size, stride=1, padding=kernel_size//2)
        target_eroded = -F.max_pool2d(-target, kernel_size, stride=1, padding=kernel_size//2)
        boundary = target_dilated - target_eroded

        if boundary.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        # 在邊界區域計算 BCE（關閉 autocast，因為 BCE 不支援 bfloat16）
        boundary_pred = pred * boundary
        boundary_target = target * boundary
        with torch.amp.autocast('cuda', enabled=False):
            bce = F.binary_cross_entropy(
                boundary_pred.float().clamp(1e-6, 1-1e-6),
                boundary_target.float(),
                reduction='sum'
            ) / (boundary.sum().float() + 1e-5)
        return bce

    def forward(self, logits, target):
        pred = torch.sigmoid(logits)
        loss = self.focal_weight * self.focal_loss(pred, target) + \
               self.dice_weight * self.dice_loss(pred, target) + \
               self.boundary_weight * self.boundary_loss(pred, target)
        return loss


# ==========================================
# 📐 修正的 Dice 計算（關鍵修正！）
# ==========================================
def compute_dice_correct(pred, target, smooth=1e-5):
    """
    v3 修正：空切片不計入平均！
    這樣訓練時的 Dice 才會反映真實的神經管分割品質。
    """
    pred = (pred > 0.5).float()
    batch_size = pred.shape[0]
    total_dice = 0.0
    valid_count = 0

    for i in range(batch_size):
        p = pred[i]
        t = target[i]

        # 如果 GT 是空的（沒有神經管），跳過不算
        if t.sum() == 0:
            continue

        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        dice = (2.0 * intersection + smooth) / (union + smooth)
        total_dice += dice.item()
        valid_count += 1

    if valid_count == 0:
        return None  # 回傳 None 代表這個 batch 全是空切片

    return total_dice / valid_count


# ==========================================
# 📈 Cosine Warmup Scheduler
# ==========================================
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * scale)

    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ==========================================
# 🚀 主訓練邏輯
# ==========================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(JSON_PATH, 'r') as f:
        splits = json.load(f)

    print("=" * 60)
    print("🧠 下顎神經管分割 — v4 (載入 SAM2 預訓練權重)")
    print("=" * 60)
    print(f"  訓練病人: {splits.get('train_patients', 'N/A')}")
    print(f"  驗證病人: {splits.get('val_patients', 'N/A')}")
    print(f"  2.5D Context:   ±{CONTEXT_SLICES} slices")
    print(f"  Batch size:     {BATCH_SIZE} x {ACCUM_STEPS} = {BATCH_SIZE * ACCUM_STEPS} effective")
    print(f"  Epochs:         {NUM_EPOCHS}")
    print(f"  Base LR:        {BASE_LR}")
    print(f"  Augmentation:   v3 (Copy-Paste + 仿射 + 彈性 + Cutout)")
    print(f"  Loss:           Focal + Dice + Boundary")
    print(f"  Dice 計算:      ✅ 已修正（排除空切片）")
    print("=" * 60)

    train_dataset = MandibularCanal25DDatasetV3(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR, splits['train'], is_train=True
    )
    val_dataset = MandibularCanal25DDatasetV3(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR, splits['val'], is_train=False
    )

    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_dataset), replacement=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        drop_last=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')
    print(f"📦 載入 SAM2 預訓練權重: {SAM2_CHECKPOINT}")
    model = CoSeg(build_sam2("sam2_hiera_l.yaml", SAM2_CHECKPOINT, mode="train"))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    for n, p in model.named_parameters():
        if "image_encoder" in n:
            p.requires_grad = ("edge" in n or "neck" in n)
        else:
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"🧠 可訓練: {trainable / 1e6:.2f}M / 總量: {total / 1e6:.2f}M ({trainable / total * 100:.1f}%)")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=BASE_LR, weight_decay=1e-4
    )
    scheduler = CosineWarmupScheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)
    criterion = FocalDiceBoundaryLoss(focal_alpha=0.75, focal_gamma=2.0, boundary_weight=0.5)

    best_dice = 0.0
    patience_counter = 0
    PATIENCE = 35

    print(f"\n🚀 開始訓練...")

    for epoch in range(NUM_EPOCHS):
        # ========== TRAIN ==========
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        dice_count = 0
        n_samples = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS-1} [Train]")
        for step, (imgs, masks) in enumerate(pbar):
            imgs = imgs.to(device)
            masks = masks.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, pred_mask_sem, _, _ = model(x=imgs)
                pred_mask_sem = F.interpolate(
                    pred_mask_sem, size=(1024, 1024),
                    mode='bilinear', align_corners=False
                )
                pred_target = pred_mask_sem[:, 0:1, :, :]
                loss = criterion(pred_target, masks) / ACCUM_STEPS

            loss.backward()

            if (step + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            # ✅ 修正的 Dice 計算
            probs = torch.sigmoid(pred_target.float().detach())
            batch_dice = compute_dice_correct(probs, masks)

            running_loss += loss.item() * ACCUM_STEPS * imgs.size(0)
            n_samples += imgs.size(0)

            if batch_dice is not None:
                running_dice += batch_dice
                dice_count += 1

            display_dice = running_dice / max(dice_count, 1)
            pbar.set_postfix({
                'Loss': f"{loss.item() * ACCUM_STEPS:.4f}",
                'Dice': f"{batch_dice:.4f}" if batch_dice else "N/A",
                'AvgDice': f"{display_dice:.4f}",
                'LR': f"{scheduler.get_lr()[0]:.2e}"
            })

        optimizer.step()
        optimizer.zero_grad()

        train_loss = running_loss / max(n_samples, 1)
        train_dice = running_dice / max(dice_count, 1)

        # ========== VALIDATION (每 3 個 epoch 跑一次) ==========
        if epoch % 3 != 0 and epoch != NUM_EPOCHS - 1:
            scheduler.step(epoch)
            print(f"\nEpoch {epoch} | Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} | Val: skipped | LR: {scheduler.get_lr()[0]:.2e}")
            save_dict = model.state_dict()
            torch.save(save_dict, OUTPUT_LAST)
            if epoch % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            continue

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_dice_count = 0
        val_n = 0

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS-1} [Val]"):
                imgs = imgs.to(device)
                masks = masks.to(device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, pred_mask_sem, _, _ = model(x=imgs)
                    pred_mask_sem = F.interpolate(
                        pred_mask_sem, size=(1024, 1024),
                        mode='bilinear', align_corners=False
                    )
                    pred_target = pred_mask_sem[:, 0:1, :, :]
                    loss = criterion(pred_target, masks)

                probs = torch.sigmoid(pred_target.float())
                batch_dice = compute_dice_correct(probs, masks)

                val_loss += loss.item() * imgs.size(0)
                val_n += imgs.size(0)

                if batch_dice is not None:
                    val_dice += batch_dice
                    val_dice_count += 1

        val_loss /= max(val_n, 1)
        val_dice_avg = val_dice / max(val_dice_count, 1)

        scheduler.step(epoch)

        print(f"\nEpoch {epoch} | "
              f"Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} ({dice_count} batches) | "
              f"Val Loss: {val_loss:.4f} Dice: {val_dice_avg:.4f} ({val_dice_count} batches) | "
              f"LR: {scheduler.get_lr()[0]:.2e}")

        save_dict = model.state_dict()
        torch.save(save_dict, OUTPUT_LAST)

        if val_dice_avg > best_dice and val_dice_count > 0:
            best_dice = val_dice_avg
            torch.save(save_dict, OUTPUT_WEIGHTS)
            print(f"🏆 新紀錄！Val Dice: {best_dice:.4f} → 已儲存")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  未破紀錄 (最佳: {best_dice:.4f}) | Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"⛔ Early stopping at epoch {epoch}")
            break

        # 定期清理記憶體
        if epoch % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\n✅ 訓練結束！最佳 Val Dice: {best_dice:.4f}")
    print(f"   模型路徑: {OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    main()
