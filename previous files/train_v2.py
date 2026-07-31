"""
train_v2.py — 下顎神經管分割訓練腳本 (大幅改良版)
=====================================================
改良重點：
  1. 2.5D Context：以相鄰切片 (z-1, z, z+1) 作為 RGB 三通道，給模型 Z 軸連續性資訊
  2. 資料增強 (Data Augmentation)：彈性變形、翻轉、旋轉、亮度對比、高斯雜訊
  3. 正樣本過採樣 (Positive Oversampling)：用 WeightedRandomSampler 讓含神經管的切片出現更多次
  4. Focal + Dice Loss：比 DiceCELoss 更能處理極端不平衡
  5. Cosine Annealing + Warmup：比 ExponentialLR 更穩定
  6. 梯度累積 (Gradient Accumulation)：等效更大 batch size
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import random
import math
from scipy.ndimage import gaussian_filter, map_coordinates

from model import CoSeg
from sam2.build_sam import build_sam2

# ==========================================
# 📂 路徑設定
# ==========================================
PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
JSON_PATH = "data/public_data/train/data_split_2_13_14.json"
TRAIN_IMG_DIR = "data/public_data/train/image_1024"
TRAIN_MASK_DIR = "data/public_data/train/mask_sem_1024"
OUTPUT_DIR = "outputs"
OUTPUT_WEIGHTS = os.path.join(OUTPUT_DIR, "coseg_v2_best.pth")
OUTPUT_LAST = os.path.join(OUTPUT_DIR, "coseg_v2_last.pth")

# ==========================================
# 🎛️ 超參數
# ==========================================
BATCH_SIZE = 4
ACCUM_STEPS = 2          # 梯度累積 → 等效 batch_size = 8
NUM_EPOCHS = 120
BASE_LR = 5e-5
WARMUP_EPOCHS = 5
NUM_WORKERS = 4
POSITIVE_WEIGHT = 3.0    # 正樣本過採樣權重
CONTEXT_SLICES = 1       # 2.5D: 前後各取幾張 (1 → 3通道)
AUG_PROB = 0.5           # 每種增強獨立機率


# ==========================================
# 🔧 資料增強函式 (使用 cv2 + scipy，不需額外安裝)
# ==========================================
def elastic_deform(image, mask, alpha=800, sigma=20):
    """彈性變形：對醫學影像分割極為有效"""
    shape = image.shape[:2]
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices_y = np.clip(y + dy, 0, shape[0] - 1).astype(np.float64)
    indices_x = np.clip(x + dx, 0, shape[1] - 1).astype(np.float64)

    # 對每個通道分別做
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


def augment(image, mask):
    """
    綜合資料增強 pipeline
    image: (H, W, 3) float32, mask: (H, W) float32
    """
    # 1. 水平翻轉
    if random.random() < AUG_PROB:
        image = np.flip(image, axis=1).copy()
        mask = np.flip(mask, axis=1).copy()

    # 2. 垂直翻轉
    if random.random() < AUG_PROB * 0.5:
        image = np.flip(image, axis=0).copy()
        mask = np.flip(mask, axis=0).copy()

    # 3. 隨機旋轉 (±15°)
    if random.random() < AUG_PROB:
        angle = random.uniform(-15, 15)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_REFLECT_101)

    # 4. 彈性變形 (低機率，計算較重)
    if random.random() < AUG_PROB * 0.4:
        image, mask = elastic_deform(image, mask)

    # 5. 亮度 / 對比度調整
    if random.random() < AUG_PROB:
        alpha = random.uniform(0.8, 1.2)  # 對比度
        beta = random.uniform(-15, 15)     # 亮度
        image = np.clip(alpha * image + beta, 0, 255)

    # 6. Gamma 調整
    if random.random() < AUG_PROB * 0.5:
        gamma = random.uniform(0.7, 1.4)
        image = np.clip(np.power(image / 255.0, gamma) * 255.0, 0, 255)

    # 7. 高斯雜訊
    if random.random() < AUG_PROB * 0.3:
        noise = np.random.normal(0, random.uniform(3, 10), image.shape).astype(np.float32)
        image = np.clip(image + noise, 0, 255)

    # 8. 高斯模糊
    if random.random() < AUG_PROB * 0.3:
        ksize = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)

    return image.astype(np.float32), (mask > 0.5).astype(np.float32)


# ==========================================
# 📦 2.5D Dataset（核心改進）
# ==========================================
class MandibularCanal25DDataset(Dataset):
    """
    核心改進：2.5D Context
    ─────────────────────
    原本：同一張灰度影像複製 3 次 → (gray, gray, gray)
    改良：載入相鄰切片     → (slice[z-1], slice[z], slice[z+1])

    這讓模型在不改架構的前提下獲得 Z 軸連續性資訊，
    是解決「碎玻璃」斷裂問題最關鍵的一步。
    """

    def __init__(self, img_dir, mask_dir, file_list, is_train=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        self.pixel_mean = np.array([123.675, 116.280, 103.530], dtype=np.float32)
        self.pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        # 解析 file_list，建立 per-patient 索引以支援 2.5D
        self.file_list = file_list
        self.patient_slice_map = {}  # patient -> {slice_num: filename}
        self.entries = []  # [(patient, slice_num, filename)]

        for fname in file_list:
            # 格式: VOL_13_slice_110
            parts = fname.rsplit('_slice_', 1)
            patient = parts[0]
            slice_num = int(parts[1])
            if patient not in self.patient_slice_map:
                self.patient_slice_map[patient] = {}
            self.patient_slice_map[patient][slice_num] = fname
            self.entries.append((patient, slice_num, fname))

        # 預先掃描哪些是正樣本 (含神經管)，供 WeightedRandomSampler 使用
        self.has_canal = []
        print(f"📊 掃描正/負樣本分布 ({'train' if is_train else 'val'})...")
        for patient, slice_num, fname in tqdm(self.entries, desc="Scanning"):
            mask_path = os.path.join(self.mask_dir, fname + ".npy")
            if os.path.exists(mask_path):
                m = np.load(mask_path)
                self.has_canal.append(m.sum() > 0)
            else:
                self.has_canal.append(False)

        n_pos = sum(self.has_canal)
        n_neg = len(self.has_canal) - n_pos
        print(f"   ✅ 正樣本 (含神經管): {n_pos} | ❌ 負樣本: {n_neg}")

    def get_sample_weights(self):
        """回傳每個樣本的權重，供 WeightedRandomSampler 使用"""
        weights = []
        for has in self.has_canal:
            weights.append(POSITIVE_WEIGHT if has else 1.0)
        return weights

    def _load_slice_gray(self, patient, slice_num):
        """載入某病人某層切片的灰度影像，不存在時回傳 None"""
        if patient in self.patient_slice_map and slice_num in self.patient_slice_map[patient]:
            fname = self.patient_slice_map[patient][slice_num]
        else:
            # 嘗試直接從硬碟讀取（此切片可能不在 file_list 中但實際存在）
            fname = f"{patient}_slice_{slice_num}"

        img_path = os.path.join(self.img_dir, fname + ".png")
        if os.path.exists(img_path):
            return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        return None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        patient, slice_num, fname = self.entries[idx]

        # ── 2.5D: 載入相鄰切片 ──
        center = self._load_slice_gray(patient, slice_num)
        prev_slice = self._load_slice_gray(patient, slice_num - CONTEXT_SLICES)
        next_slice = self._load_slice_gray(patient, slice_num + CONTEXT_SLICES)

        # 邊界處理：不存在就複製中心切片
        if prev_slice is None:
            prev_slice = center.copy()
        if next_slice is None:
            next_slice = center.copy()

        # 組合為 3 通道 (H, W, 3)
        img = np.stack([prev_slice, center, next_slice], axis=-1)

        # ── 載入 Mask ──
        mask_path = os.path.join(self.mask_dir, fname + ".npy")
        gt_mask = np.load(mask_path).astype(np.float32)
        gt_mask = (gt_mask > 0).astype(np.float32)

        # ── 資料增強（僅訓練階段）──
        if self.is_train:
            img, gt_mask = augment(img, gt_mask)

        # ── 正規化 (SAM2 標準) ──
        img = (img - self.pixel_mean) / (self.pixel_std + 1e-8)
        img_tensor = torch.tensor(img).permute(2, 0, 1).float()
        mask_tensor = torch.tensor(gt_mask).unsqueeze(0).float()

        return img_tensor, mask_tensor


# ==========================================
# 🎯 損失函式：Focal + Dice（處理極端不平衡）
# ==========================================
class FocalDiceLoss(nn.Module):
    """
    為什麼比 DiceCELoss 好？
    ─────────────────────────
    • Focal Loss: 自動降低「容易分類的背景」權重，讓模型專注在困難的邊界像素
    • Dice Loss: 直接優化 Dice 係數，對小目標敏感
    • 組合使用：Focal 負責分類品質，Dice 負責區域重疊率
    """

    def __init__(self, focal_alpha=0.75, focal_gamma=2.0, dice_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def focal_loss(self, pred, target):
        pred = pred.clamp(1e-6, 1 - 1e-6)
        # alpha-balanced focal loss
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

    def forward(self, logits, target):
        pred = torch.sigmoid(logits)
        return self.focal_weight * self.focal_loss(pred, target) + \
               self.dice_weight * self.dice_loss(pred, target)


# ==========================================
# 📐 評估指標
# ==========================================
def compute_dice(pred, target, smooth=1e-5):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


# ==========================================
# 📈 Cosine Warmup Scheduler
# ==========================================
class CosineWarmupScheduler:
    """Warmup + Cosine Annealing，比 ExponentialLR 更穩定"""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
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

    # ── 讀取 Data Split ──
    with open(JSON_PATH, 'r') as f:
        splits = json.load(f)

    print("=" * 60)
    print("🧠 下顎神經管分割 — 改良版訓練腳本 v2")
    print("=" * 60)
    print(f"  2.5D Context:   ±{CONTEXT_SLICES} slices")
    print(f"  Batch size:     {BATCH_SIZE} x {ACCUM_STEPS} accum = {BATCH_SIZE * ACCUM_STEPS} effective")
    print(f"  Epochs:         {NUM_EPOCHS}")
    print(f"  Base LR:        {BASE_LR}")
    print(f"  Warmup:         {WARMUP_EPOCHS} epochs")
    print(f"  Augmentation:   p={AUG_PROB}")
    print(f"  Positive weight: {POSITIVE_WEIGHT}x oversampling")
    print("=" * 60)

    # ── Dataset ──
    train_dataset = MandibularCanal25DDataset(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR, splits['train'], is_train=True
    )
    val_dataset = MandibularCanal25DDataset(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR, splits['val'], is_train=False
    )

    # ── 正樣本過採樣 Sampler ──
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        drop_last=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # ── 模型 ──
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')
    model = CoSeg(build_sam2("sam2_hiera_l.yaml", None, mode="train"))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # ── 半凍結：與原始腳本相同 ──
    for n, p in model.named_parameters():
        if "image_encoder" in n:
            p.requires_grad = ("edge" in n or "neck" in n)
        else:
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"🧠 可訓練: {trainable / 1e6:.2f}M / 總量: {total / 1e6:.2f}M ({trainable / total * 100:.1f}%)")

    # ── 優化器 + 排程器 ──
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=BASE_LR, weight_decay=1e-4
    )
    scheduler = CosineWarmupScheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)

    # ── 損失函式 ──
    criterion = FocalDiceLoss(focal_alpha=0.75, focal_gamma=2.0)

    # ── 訓練迴圈 ──
    best_dice = 0.0
    patience_counter = 0
    PATIENCE = 30  # Early stopping patience

    print(f"\n🚀 開始訓練...")

    for epoch in range(NUM_EPOCHS):
        # ========== TRAIN ==========
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        n_samples = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS - 1} [Train]")
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

            # 梯度累積
            if (step + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            probs = torch.sigmoid(pred_target.float().detach())
            batch_dice = compute_dice(probs, masks)

            running_loss += loss.item() * ACCUM_STEPS * imgs.size(0)
            running_dice += batch_dice * imgs.size(0)
            n_samples += imgs.size(0)

            pbar.set_postfix({
                'Loss': f"{loss.item() * ACCUM_STEPS:.4f}",
                'Dice': f"{batch_dice:.4f}",
                'LR': f"{scheduler.get_lr()[0]:.2e}"
            })

        # 處理剩餘梯度
        optimizer.step()
        optimizer.zero_grad()

        train_loss = running_loss / max(n_samples, 1)
        train_dice = running_dice / max(n_samples, 1)

        # ========== VALIDATION ==========
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_n = 0

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS - 1} [Val]"):
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
                batch_dice = compute_dice(probs, masks)

                val_loss += loss.item() * imgs.size(0)
                val_dice += batch_dice * imgs.size(0)
                val_n += imgs.size(0)

        val_loss /= max(val_n, 1)
        val_dice /= max(val_n, 1)

        # ── 學習率更新 ──
        scheduler.step(epoch)

        # ── 印出結果 ──
        print(f"\nEpoch {epoch} | Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f} | LR: {scheduler.get_lr()[0]:.2e}")

        # ── 存檔 ──
        save_dict = model.state_dict()
        torch.save(save_dict, OUTPUT_LAST)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(save_dict, OUTPUT_WEIGHTS)
            print(f"🏆 新紀錄！Val Dice: {best_dice:.4f} → 已儲存")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  未破紀錄 (最佳: {best_dice:.4f}) | Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"⛔ Early stopping triggered at epoch {epoch}")
            break

    print(f"\n✅ 訓練結束！最佳 Val Dice: {best_dice:.4f}")
    print(f"   模型路徑: {OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    main()
