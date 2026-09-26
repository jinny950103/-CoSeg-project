#!/usr/bin/env python3
"""
V11 訓練腳本 — ROI Cropping 二階段（適配 flat-slice 資料結構）
==============================================================

新增功能（相對 V10）：
  1. ROI Cropping：從 preprocess_v11 的 metadata.json 讀取 per-slice ROI bbox，
     裁切後 resize 到 1024×1024，讓 mandibular canal 佔更大比例
  2. 二階段訓練：
     Stage 1: 全圖訓練（粗定位）→ 產出 ROI 預測
     Stage 2: ROI 裁切訓練（精細分割）→ Dice 主要提升來源
  3. 資料已過 preprocess_v11_spacing.py（CLAHE + ROI 偵測）
  4. 保留 V10 所有功能：DoRA, DomainBN, 多種 loss, augmentation, auto-resume
  5. 可同時使用 preprocessed（V11）或原始 flat-slice（V10 格式）的資料

適配資料結構：
  A) V11 preprocessed (preprocess_v11_spacing.py 輸出):
     preprocessed_root/{patient_id}/images/0000.npy
     preprocessed_root/{patient_id}/masks/0000.npy
     preprocessed_root/{patient_id}/metadata.json
     + data_split_v11.json

  B) V10 原始 flat-slice (直接使用，不需前處理):
     data_dir/image_1024/Patient_5_slice_0.png
     data_dir/mask_sem_1024/Patient_5_slice_0.npy
     + data_split_v10_mixed.json

用法：
  # ─── V11 模式 (已前處理) ───
  # Stage 1: 全圖粗訓練
  python train_v11_roi.py \
      --stage 1 \
      --data-format v11 \
      --data-dir /path/to/preprocessed_v11 \
      --split-json /path/to/preprocessed_v11/data_split_v11.json \
      --epochs 60 --batch-size 8

  # Stage 2: ROI 精細訓練
  python train_v11_roi.py \
      --stage 2 \
      --data-format v11 \
      --data-dir /path/to/preprocessed_v11 \
      --split-json /path/to/preprocessed_v11/data_split_v11.json \
      --epochs 120 --batch-size 6 --roi-size 384 \
      --stage1-ckpt coseg_v11_stage1_best.pth

  # ─── V10 相容模式 (原始 flat slices, 無前處理, 無 ROI) ───
  python train_v11_roi.py \
      --stage 1 \
      --data-format v10 \
      --public-data-dir /path/to/public_data/train \
      --public-val-dir /path/to/public_data/val \
      --hospital-data-dir /path/to/hospital_data/train \
      --hospital-eval-dir /path/to/hospital_data/eval \
      --split-json /path/to/data_split_v10_mixed.json \
      --epochs 60

需要：model_v10.py（模型架構）在同一目錄下
"""

import os
import sys
import re
import json
import math
import time
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.amp import autocast, GradScaler

import cv2

# ─── CLAHE on-the-fly ───
def apply_clahe_2d(image, clip_limit=2.0, grid_size=8):
    """
    對 float32 [0,1] 灰度影像做 CLAHE。
    回傳 float32 [0,1]。
    """
    img_u8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    result = clahe.apply(img_u8)
    return result.astype(np.float32) / 255.0


# ─── 嘗試 import model_v10 ───
try:
    from model_v10 import CoSegV10
except ImportError:
    print("[警告] 找不到 model_v10.py，請確保在同一目錄下")
    CoSegV10 = None


# ═══════════════════════════════════════════════
# 超參數
# ═══════════════════════════════════════════════

SEED = 42
IMG_SIZE = 1024
NUM_WORKERS = 4

# Stage 1 預設
S1_DEFAULTS = dict(
    batch_size=8,
    grad_accum=4,
    epochs=60,
    base_lr=8e-5,
    roi_size=0,       # 0 = 全圖
    samples_per_domain=1200,
)

# Stage 2 預設
S2_DEFAULTS = dict(
    batch_size=6,
    grad_accum=5,
    epochs=120,
    base_lr=5e-5,
    roi_size=384,     # ROI crop size in pixels
    samples_per_domain=1500,
)

LOSS_WEIGHTS = dict(
    focal_dice=1.0,
    cldice=0.3,
    z_cont=0.1,
    sdf=0.2,
    boundary=0.15,
)

HOSPITAL_LOSS_WEIGHT = 0.7
BEST_MODEL_RATIO = (0.6, 0.4)  # pub, hosp


# ═══════════════════════════════════════════════
# Losses (繼承 V10)
# ═══════════════════════════════════════════════

class FocalDiceBoundaryLoss(nn.Module):
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.alpha = focal_alpha
        self.gamma = focal_gamma

    def forward(self, logits, targets, weight_map=None):
        probs = torch.sigmoid(logits)

        # Focal
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.where(targets > 0.5, probs, 1 - probs)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        if weight_map is not None:
            focal = focal * weight_map
        focal_loss = focal.mean()

        # Dice
        smooth = 1.0
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice.mean()

        return focal_loss + dice_loss


class SoftCLDiceLoss(nn.Module):
    def __init__(self, iter_=5):
        super().__init__()
        self.iter = iter_

    def soft_skel(self, img, iter_):
        kernel = torch.ones(1, 1, 3, 3, device=img.device, dtype=img.dtype)
        for _ in range(iter_):
            eroded = 1.0 - F.conv2d(1.0 - img, kernel, padding=1).clamp(0, 1)
            opened = F.conv2d(eroded, kernel, padding=1).clamp(0, 1)
            img = img - opened + eroded
        return img.clamp(0, 1)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        skel_p = self.soft_skel(probs, self.iter)
        skel_t = self.soft_skel(targets, self.iter)
        smooth = 1.0
        tprec = ((skel_p * targets).sum(dim=(2, 3)) + smooth) / (skel_p.sum(dim=(2, 3)) + smooth)
        tsens = ((skel_t * probs).sum(dim=(2, 3)) + smooth) / (skel_t.sum(dim=(2, 3)) + smooth)
        cldice = 2.0 * tprec * tsens / (tprec + tsens + 1e-7)
        return 1.0 - cldice.mean()


class ZContinuityLoss(nn.Module):
    def forward(self, logits):
        if logits.shape[0] < 2:
            return torch.tensor(0.0, device=logits.device)
        probs = torch.sigmoid(logits)
        diff = (probs[1:] - probs[:-1]).abs()
        return diff.mean()


class SDFAuxLoss(nn.Module):
    @staticmethod
    def compute_sdf_target(mask_np):
        from scipy.ndimage import distance_transform_edt
        if mask_np.sum() == 0:
            return np.zeros_like(mask_np, dtype=np.float32)
        if mask_np.sum() == mask_np.size:
            return np.ones_like(mask_np, dtype=np.float32)
        pos_dist = distance_transform_edt(mask_np)
        neg_dist = distance_transform_edt(1 - mask_np)
        sdf = pos_dist - neg_dist
        max_val = max(pos_dist.max(), neg_dist.max(), 1e-6)
        sdf = sdf / max_val
        return sdf.astype(np.float32)

    def forward(self, sdf_pred, sdf_target):
        return F.l1_loss(sdf_pred, sdf_target)


# ═══════════════════════════════════════════════
# GT-Aware Weight Map
# ═══════════════════════════════════════════════

def compute_gt_weight_map(mask_np, domain, boundary_weight=0.3, center_weight=1.0):
    if domain == 0:
        return np.ones_like(mask_np, dtype=np.float32)

    from scipy.ndimage import distance_transform_edt
    weight_map = np.ones_like(mask_np, dtype=np.float32)
    if mask_np.sum() > 0:
        dist = distance_transform_edt(mask_np)
        max_dist = dist.max()
        if max_dist > 0:
            normalized_dist = dist / max_dist
            weight_map = boundary_weight + (center_weight - boundary_weight) * normalized_dist
            weight_map[mask_np == 0] = 1.0
    return weight_map.astype(np.float32)


# ═══════════════════════════════════════════════
# Augmentation
# ═══════════════════════════════════════════════

def augment_roi(image, mask, weight_map, epoch, max_epochs, roi_mode=False):
    H, W = image.shape
    progress = min(epoch / max(max_epochs * 0.7, 1), 1.0)

    # Random flip
    if random.random() < 0.5:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()
        weight_map = np.fliplr(weight_map).copy()
    if random.random() < 0.5:
        image = np.flipud(image).copy()
        mask = np.flipud(mask).copy()
        weight_map = np.flipud(weight_map).copy()

    # Random rotation
    if random.random() < 0.4 * progress:
        angle = random.uniform(-15, 15) * progress
        M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (W, H), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask.astype(np.float32), M, (W, H),
                              flags=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.uint8)
        weight_map = cv2.warpAffine(weight_map, M, (W, H), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)

    # Elastic deformation
    if random.random() < 0.3 * progress:
        alpha = 15 * progress
        sigma = 4
        dx = cv2.GaussianBlur(np.random.randn(H, W).astype(np.float32) * alpha, (0, 0), sigma)
        dy = cv2.GaussianBlur(np.random.randn(H, W).astype(np.float32) * alpha, (0, 0), sigma)
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = cv2.remap(mask.astype(np.float32), map_x, map_y, cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.uint8)
        weight_map = cv2.remap(weight_map, map_x, map_y, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)

    # Intensity
    if random.random() < 0.5:
        shift = random.uniform(-0.08, 0.08) * progress
        image = np.clip(image + shift, 0, 1)
    if random.random() < 0.4:
        factor = random.uniform(0.85, 1.15)
        mean = image.mean()
        image = np.clip((image - mean) * factor + mean, 0, 1)
    if random.random() < 0.3:
        gamma = random.uniform(0.8, 1.2)
        image = np.clip(np.power(image + 1e-7, gamma), 0, 1)

    # Noise
    if random.random() < 0.3 * progress:
        noise = np.random.normal(0, 0.02 * progress, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0, 1)

    return image.astype(np.float32), mask, weight_map.astype(np.float32)


# ═══════════════════════════════════════════════
# 資料載入輔助
# ═══════════════════════════════════════════════

def parse_filename(fname):
    """解析 Patient_5_slice_0.png → ('Patient_5', 0)"""
    stem = Path(fname).stem
    m = re.match(r'^(.+)_slice_(\d+)$', stem)
    if m:
        return m.group(1), int(m.group(2))
    m = re.match(r'^slice_(\d+)$', stem)
    if m:
        return None, int(m.group(1))
    m = re.match(r'^(\d+)$', stem)
    if m:
        return None, int(m.group(1))
    return None, None


def load_image_file(path):
    """載入 image → float32 [0,1]"""
    p = str(path)
    if p.endswith('.npy'):
        img = np.load(p).astype(np.float32)
    else:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros((1024, 1024), dtype=np.float32)
        img = img.astype(np.float32) / 255.0

    if img.max() > 1.5:
        mn, mx = img.min(), img.max()
        if mx - mn > 1e-6:
            img = (img - mn) / (mx - mn)
    return img


def load_mask_file(path):
    """載入 mask → uint8 binary"""
    p = str(path)
    if p.endswith('.npy'):
        mask = np.load(p)
    else:
        mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros((1024, 1024), dtype=np.uint8)

    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    else:
        mask = (mask > 0.5).astype(np.uint8)
    return mask


# ═══════════════════════════════════════════════
# Dataset — V11 preprocessed format
# ═══════════════════════════════════════════════

class V11ROIDataset(Dataset):
    """
    V11 Dataset: 讀取 metadata.json 中記錄的原始檔案路徑。
    不需要複製圖片，直接從原始 PNG/NPY 讀取，on-the-fly 做 CLAHE。
    支援 ROI cropping (Stage 2)。
    2.5D: center slice ± 1 → 3 channels (RGB for SAM2)。
    """

    def __init__(self, patient_entries, stage=2, roi_size=384,
                 is_train=True, epoch=0, max_epochs=120, roi_jitter=0.15,
                 apply_clahe=True, clahe_clip=2.0):
        """
        Args:
            patient_entries: list of dicts with keys:
                'meta_dir': path to dir containing metadata.json
                'domain': 0 or 1
            stage: 1 (全圖) or 2 (ROI crop)
            roi_size: crop size in pixels (stage 2)
        """
        super().__init__()
        self.stage = stage
        self.roi_size = roi_size
        self.is_train = is_train
        self.epoch = epoch
        self.max_epochs = max_epochs
        self.roi_jitter = roi_jitter
        self.apply_clahe = apply_clahe
        self.clahe_clip = clahe_clip

        self.samples = []
        # 存每個病人的 slices 路徑列表，避免重複讀 metadata
        self.patient_data = {}  # patient_id → {slices: [...], ...}

        for entry in patient_entries:
            meta_dir = entry.get('meta_dir', entry.get('dir', ''))
            domain_id = entry['domain']

            meta_path = os.path.join(meta_dir, 'metadata.json')
            if not os.path.exists(meta_path):
                print(f"[警告] 找不到 metadata: {meta_path}")
                continue

            with open(meta_path) as f:
                meta = json.load(f)

            patient_id = meta['patient_id']
            num_slices = meta['num_slices']
            per_slice_roi = meta.get('per_slice_roi', [None] * num_slices)
            global_roi = meta.get('global_roi', None)
            fg_slices = set(meta.get('foreground_slices', range(num_slices)))
            slices_info = meta.get('slices', [])

            # 存到 patient_data
            self.patient_data[patient_id] = {
                'slices': slices_info,  # [{idx, img, mask}, ...]
                'domain': domain_id,
            }

            for i in range(1, num_slices - 1):  # skip first/last for 2.5D
                if self.is_train:
                    nearby_fg = any(j in fg_slices for j in range(max(0, i-2), min(num_slices, i+3)))
                    if not nearby_fg:
                        continue

                roi = per_slice_roi[i] if i < len(per_slice_roi) else None
                if roi is None and global_roi is not None:
                    roi = global_roi

                self.samples.append({
                    'patient_id': patient_id,
                    'slice_idx': i,
                    'domain_id': domain_id,
                    'roi': roi,
                    'num_slices': num_slices,
                })

        print(f"  V11Dataset: {len(self.samples)} samples from "
              f"{len(patient_entries)} patients (stage={stage}, roi={roi_size})")

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _load_slice(self, patient_id, idx, num_slices):
        """從原始檔案路徑讀取 slice，on-the-fly CLAHE"""
        idx = max(0, min(idx, num_slices - 1))
        slices_info = self.patient_data[patient_id]['slices']

        if idx < len(slices_info):
            img_path = slices_info[idx]['img']
            mask_path = slices_info[idx]['mask']
            image = load_image_file(img_path)
            mask = load_mask_file(mask_path)
        else:
            image = np.zeros((1024, 1024), dtype=np.float32)
            mask = np.zeros((1024, 1024), dtype=np.uint8)

        # On-the-fly CLAHE
        if self.apply_clahe and image.ndim == 2:
            image = apply_clahe_2d(image, self.clahe_clip)

        return image, mask

    def _apply_roi_crop(self, images, masks, weight_maps, roi, H, W):
        if roi is None or self.stage == 1:
            return images, masks, weight_maps

        y_min, y_max, x_min, x_max = roi

        if self.is_train and self.roi_jitter > 0:
            roi_h = y_max - y_min
            roi_w = x_max - x_min
            jitter_y = int(roi_h * self.roi_jitter)
            jitter_x = int(roi_w * self.roi_jitter)
            dy = random.randint(-jitter_y, jitter_y)
            dx = random.randint(-jitter_x, jitter_x)
            y_min = max(0, y_min + dy)
            y_max = min(H, y_max + dy)
            x_min = max(0, x_min + dx)
            x_max = min(W, x_max + dx)

        roi_h = y_max - y_min
        roi_w = x_max - x_min
        if roi_h < self.roi_size or roi_w < self.roi_size:
            cy = (y_min + y_max) // 2
            cx = (x_min + x_max) // 2
            half = self.roi_size // 2
            y_min = max(0, cy - half)
            y_max = min(H, cy + half)
            x_min = max(0, cx - half)
            x_max = min(W, cx + half)

        if self.is_train and random.random() < 0.3:
            scale = random.uniform(0.8, 1.3)
            cy = (y_min + y_max) // 2
            cx = (x_min + x_max) // 2
            new_h = int((y_max - y_min) * scale)
            new_w = int((x_max - x_min) * scale)
            y_min = max(0, cy - new_h // 2)
            y_max = min(H, cy + new_h // 2)
            x_min = max(0, cx - new_w // 2)
            x_max = min(W, cx + new_w // 2)

        cropped_imgs = [img[y_min:y_max, x_min:x_max] for img in images]
        cropped_masks = [m[y_min:y_max, x_min:x_max] for m in masks]
        cropped_wms = [wm[y_min:y_max, x_min:x_max] for wm in weight_maps]

        return cropped_imgs, cropped_masks, cropped_wms

    def __getitem__(self, idx):
        sample = self.samples[idx]
        patient_id = sample['patient_id']
        center = sample['slice_idx']
        domain_id = sample['domain_id']
        roi = sample['roi']
        num_slices = sample['num_slices']

        # 2.5D
        images, masks = [], []
        for si in [center - 1, center, center + 1]:
            img, msk = self._load_slice(patient_id, si, num_slices)
            images.append(img)
            masks.append(msk)

        H, W = images[0].shape

        weight_map_center = compute_gt_weight_map(masks[1], domain_id)
        weight_maps = [
            np.ones_like(masks[0], dtype=np.float32),
            weight_map_center,
            np.ones_like(masks[2], dtype=np.float32),
        ]

        if self.stage == 2:
            images, masks, weight_maps = self._apply_roi_crop(
                images, masks, weight_maps, roi, H, W)

        if self.is_train:
            for i in range(3):
                images[i], masks[i], weight_maps[i] = augment_roi(
                    images[i], masks[i], weight_maps[i],
                    self.epoch, self.max_epochs, roi_mode=(self.stage == 2))

        resized_imgs, resized_masks, resized_wms = [], [], []
        for i in range(3):
            img_r = cv2.resize(images[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            mask_r = cv2.resize(masks[i].astype(np.float32), (IMG_SIZE, IMG_SIZE),
                                interpolation=cv2.INTER_NEAREST)
            wm_r = cv2.resize(weight_maps[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            resized_imgs.append(img_r)
            resized_masks.append((mask_r > 0.5).astype(np.float32))
            resized_wms.append(wm_r)

        image_3ch = np.stack(resized_imgs, axis=0)
        mask_center = resized_masks[1][np.newaxis]
        wm_center = resized_wms[1][np.newaxis]
        sdf_target = SDFAuxLoss.compute_sdf_target(resized_masks[1])[np.newaxis]

        return (torch.from_numpy(image_3ch).float(),
                torch.from_numpy(mask_center).float(),
                torch.tensor(domain_id, dtype=torch.long),
                torch.from_numpy(wm_center).float(),
                torch.from_numpy(sdf_target).float())


# ═══════════════════════════════════════════════
# Dataset — V10 原始 flat-slice format (相容模式)
# ═══════════════════════════════════════════════

class V10FlatDataset(Dataset):
    """
    直接從原始 flat-slice 結構讀取 (不需前處理)。
    用於 V10 相容模式或 Stage 1 快速實驗。

    資料結構：
      data_dir/image_1024/Patient_5_slice_0.png
      data_dir/mask_sem_1024/Patient_5_slice_0.npy
    """

    def __init__(self, data_dirs, patient_ids_per_dir, domain_ids_per_dir,
                 is_train=True, epoch=0, max_epochs=120,
                 apply_clahe=True, clahe_clip=2.0):
        """
        Args:
            data_dirs: list of data directories (e.g., [public_train_dir, hospital_train_dir])
            patient_ids_per_dir: list of lists of patient IDs per directory
            domain_ids_per_dir: list of domain IDs per directory (0=public, 1=hospital)
            is_train: training mode
        """
        super().__init__()
        self.is_train = is_train
        self.epoch = epoch
        self.max_epochs = max_epochs
        self.apply_clahe = apply_clahe
        self.clahe_clip = clahe_clip

        # 掃描所有 slice，按病人分組
        self.patient_slices = {}  # {patient_id: {'slices': [(idx, img, mask)], 'domain': d}}
        self.samples = []

        for data_dir, pid_list, domain_id in zip(data_dirs, patient_ids_per_dir, domain_ids_per_dir):
            img_dir = Path(data_dir) / 'image_1024'
            mask_dir = Path(data_dir) / 'mask_sem_1024'

            if not img_dir.is_dir():
                # Per-patient structure (hospital eval)
                for pid in pid_list:
                    patient_dir = Path(data_dir) / pid
                    p_img_dir = patient_dir / 'image_1024'
                    p_mask_dir = patient_dir / 'mask_sem_1024'
                    if not p_img_dir.is_dir():
                        continue

                    slices = []
                    for img_file in sorted(p_img_dir.glob('*.png')):
                        _, slice_idx = parse_filename(img_file.name)
                        if slice_idx is None:
                            slice_idx = len(slices)
                        mask_file = p_mask_dir / f'{img_file.stem}.npy'
                        if not mask_file.exists():
                            mask_file = p_mask_dir / f'{img_file.stem}.png'
                        if mask_file.exists():
                            slices.append((slice_idx, str(img_file), str(mask_file)))

                    slices.sort(key=lambda x: x[0])
                    if slices:
                        self.patient_slices[pid] = {'slices': slices, 'domain': domain_id}
                continue

            # Flat structure: scan and group by patient
            pid_set = set(pid_list) if pid_list else None

            for img_file in sorted(img_dir.glob('*.png')):
                patient_id, slice_idx = parse_filename(img_file.name)
                if patient_id is None:
                    continue
                if pid_set is not None and patient_id not in pid_set:
                    continue

                mask_file = mask_dir / f'{img_file.stem}.npy'
                if not mask_file.exists():
                    mask_file = mask_dir / f'{img_file.stem}.png'
                if not mask_file.exists():
                    continue

                if patient_id not in self.patient_slices:
                    self.patient_slices[patient_id] = {'slices': [], 'domain': domain_id}

                self.patient_slices[patient_id]['slices'].append(
                    (slice_idx, str(img_file), str(mask_file)))

            # Sort slices per patient
            for pid in self.patient_slices:
                self.patient_slices[pid]['slices'].sort(key=lambda x: x[0])

        # 建立 sample list (2.5D: skip first/last)
        for pid, info in self.patient_slices.items():
            slices = info['slices']
            domain = info['domain']
            num = len(slices)
            for local_idx in range(1, num - 1):
                # Training: 只用有前景附近的 slices
                if is_train:
                    # 簡單策略: 檢查 center mask 是否有前景
                    # (lazy check, 在 __getitem__ 裡完整判斷)
                    pass
                self.samples.append({
                    'patient_id': pid,
                    'local_idx': local_idx,
                    'domain_id': domain,
                    'num_slices': num,
                })

        print(f"  V10FlatDataset: {len(self.samples)} samples from "
              f"{len(self.patient_slices)} patients")

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pid = sample['patient_id']
        local_idx = sample['local_idx']
        domain_id = sample['domain_id']
        slices = self.patient_slices[pid]['slices']
        num = len(slices)

        # 2.5D: center ± 1
        images, masks = [], []
        for offset in [-1, 0, 1]:
            si = max(0, min(local_idx + offset, num - 1))
            _, img_path, mask_path = slices[si]
            img = load_image_file(img_path)
            msk = load_mask_file(mask_path)

            # Optional CLAHE (since V10 data has no CLAHE)
            if self.apply_clahe and len(img.shape) == 2:
                img = apply_clahe_2d(img, clip_limit=self.clahe_clip)

            images.append(img)
            masks.append(msk)

        H, W = images[0].shape[:2]

        weight_map_center = compute_gt_weight_map(masks[1], domain_id)
        weight_maps = [
            np.ones((H, W), dtype=np.float32),
            weight_map_center,
            np.ones((H, W), dtype=np.float32),
        ]

        # Augmentation
        if self.is_train:
            for i in range(3):
                images[i], masks[i], weight_maps[i] = augment_roi(
                    images[i], masks[i], weight_maps[i],
                    self.epoch, self.max_epochs, roi_mode=False)

        # Resize to IMG_SIZE (data is already 1024×1024, but safety check)
        resized_imgs, resized_masks, resized_wms = [], [], []
        for i in range(3):
            if images[i].shape != (IMG_SIZE, IMG_SIZE):
                img_r = cv2.resize(images[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
                mask_r = cv2.resize(masks[i].astype(np.float32), (IMG_SIZE, IMG_SIZE),
                                    interpolation=cv2.INTER_NEAREST)
                wm_r = cv2.resize(weight_maps[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            else:
                img_r = images[i]
                mask_r = masks[i].astype(np.float32)
                wm_r = weight_maps[i]
            resized_imgs.append(img_r)
            resized_masks.append((mask_r > 0.5).astype(np.float32))
            resized_wms.append(wm_r)

        image_3ch = np.stack(resized_imgs, axis=0)
        mask_center = resized_masks[1][np.newaxis]
        wm_center = resized_wms[1][np.newaxis]
        sdf_target = SDFAuxLoss.compute_sdf_target(resized_masks[1])[np.newaxis]

        return (torch.from_numpy(image_3ch).float(),
                torch.from_numpy(mask_center).float(),
                torch.tensor(domain_id, dtype=torch.long),
                torch.from_numpy(wm_center).float(),
                torch.from_numpy(sdf_target).float())


# ═══════════════════════════════════════════════
# Domain-Alternating Sampler
# ═══════════════════════════════════════════════

class DomainAlternatingSampler(Sampler):
    def __init__(self, dataset, samples_per_domain=1500, seed=42):
        self.dataset = dataset
        self.samples_per_domain = samples_per_domain
        self.seed = seed

        self.domain_indices = defaultdict(list)
        for i, sample in enumerate(dataset.samples):
            self.domain_indices[sample['domain_id']].append(i)

        self.domains = sorted(self.domain_indices.keys())
        print(f"  Sampler: {len(self.domains)} domains, {samples_per_domain}/domain")
        for d in self.domains:
            print(f"    Domain {d}: {len(self.domain_indices[d])} samples")

    def __iter__(self):
        rng = random.Random(self.seed)
        indices = []
        for d in self.domains:
            pool = list(self.domain_indices[d])
            rng.shuffle(pool)
            while len(pool) < self.samples_per_domain:
                extra = list(self.domain_indices[d])
                rng.shuffle(extra)
                pool.extend(extra)
            indices.extend(pool[:self.samples_per_domain])
        rng.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.samples_per_domain * len(self.domains)


# ═══════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, dataloader, device, stage=2):
    model.eval()
    dice_sums = defaultdict(float)
    dice_counts = defaultdict(int)

    for batch in dataloader:
        images, masks, domains, weight_maps, sdfs = batch
        images = images.to(device)
        masks = masks.to(device)

        for d in domains.unique():
            d_val = d.item()
            d_mask = (domains == d_val)
            if d_mask.sum() == 0:
                continue

            d_images = images[d_mask]
            d_masks_gt = masks[d_mask]

            m = model.module if hasattr(model, 'module') else model
            if hasattr(m, 'set_domain'):
                m.set_domain(d_val)

            with autocast('cuda'):
                preds = model(d_images)
                if isinstance(preds, dict):
                    logits = preds['pred']
                elif isinstance(preds, (list, tuple)):
                    logits = preds[1]  # mask_sem
                else:
                    logits = preds

            if logits.shape[-2:] != d_masks_gt.shape[-2:]:
                logits = F.interpolate(logits, d_masks_gt.shape[-2:], mode='bilinear')

            probs = torch.sigmoid(logits)
            binary = (probs > 0.5).float()

            for j in range(binary.shape[0]):
                pred_j = binary[j, 0]
                gt_j = d_masks_gt[j, 0]
                intersection = (pred_j * gt_j).sum()
                union = pred_j.sum() + gt_j.sum()
                dice = (2.0 * intersection / union).item() if union > 0 else 1.0
                dice_sums[d_val] += dice
                dice_counts[d_val] += 1

    results = {d: dice_sums[d] / max(dice_counts[d], 1) for d in sorted(dice_sums.keys())}
    model.train()
    return results


# ═══════════════════════════════════════════════
# 資料載入 & Split 解析
# ═══════════════════════════════════════════════

def parse_v11_split(split_json, data_dir=None):
    """
    解析 V11 格式的 split JSON。

    V11 split 格式:
      {"train": [{"patient_id": ..., "meta_dir": ..., "domain": 0}, ...],
       "val": [...], "hospital_train": [...], "hospital_val": [...]}

    每個 entry 需要 'meta_dir' 或 'dir' 指向含 metadata.json 的目錄。
    """
    with open(split_json) as f:
        splits = json.load(f)

    # 確保每個 entry 都有 'meta_dir' key（V11ROIDataset 需要）
    def normalize_entries(entries):
        for e in entries:
            if 'meta_dir' not in e and 'dir' not in e:
                # 如果提供了 data_dir，嘗試拼接
                if data_dir:
                    pid = e.get('patient_id', '')
                    candidate = os.path.join(data_dir, pid)
                    if os.path.exists(os.path.join(candidate, 'metadata.json')):
                        e['meta_dir'] = candidate
        return entries

    train_entries = normalize_entries(
        splits.get('train', []) + splits.get('hospital_train', []))
    val_entries = normalize_entries(
        splits.get('val', []) + splits.get('hospital_val', []))

    return train_entries, val_entries


def parse_v10_split(split_json, public_data_dir, public_val_dir,
                    hospital_data_dir, hospital_eval_dir):
    """
    解析 V10 格式的 split JSON + 原始資料目錄，建立 V10FlatDataset 需要的參數。

    V10 split 格式:
      {"train": ["Patient_5_slice_0.png", ...],
       "val": ["Patient_13_slice_0.png", ...],
       "hospital_train": ["HOSP_xxx_slice_0.png", ...],
       "hospital_val": ["HOSP_yyy_slice_0.png", ...]}

    或：
      {"train": ["Patient_5", "Patient_6", ...], ...}
    """
    with open(split_json) as f:
        splits = json.load(f)

    def extract_patient_ids(file_list):
        """從 file list 提取 unique patient IDs。"""
        pids = set()
        for item in file_list:
            if isinstance(item, str):
                pid, _ = parse_filename(item)
                if pid is not None:
                    pids.add(pid)
                else:
                    # 可能就是 patient_id 本身 (e.g., "Patient_5")
                    pids.add(item)
            elif isinstance(item, dict):
                pids.add(item.get('patient_id', item.get('id', '')))
        return sorted(pids)

    train_pub_pids = extract_patient_ids(splits.get('train', []))
    val_pub_pids = extract_patient_ids(splits.get('val', []))
    train_hosp_pids = extract_patient_ids(splits.get('hospital_train', []))
    val_hosp_pids = extract_patient_ids(splits.get('hospital_val', []))

    # Training datasets
    train_dirs = []
    train_pid_lists = []
    train_domain_ids = []

    if public_data_dir and train_pub_pids:
        train_dirs.append(public_data_dir)
        train_pid_lists.append(train_pub_pids)
        train_domain_ids.append(0)

    if hospital_data_dir and train_hosp_pids:
        train_dirs.append(hospital_data_dir)
        train_pid_lists.append(train_hosp_pids)
        train_domain_ids.append(1)

    # Val datasets
    val_dirs = []
    val_pid_lists = []
    val_domain_ids = []

    if public_val_dir and val_pub_pids:
        val_dirs.append(public_val_dir)
        val_pid_lists.append(val_pub_pids)
        val_domain_ids.append(0)
    elif public_data_dir and val_pub_pids:
        # Val might be in same dir as train (flat structure)
        val_dirs.append(public_data_dir)
        val_pid_lists.append(val_pub_pids)
        val_domain_ids.append(0)

    if hospital_eval_dir and val_hosp_pids:
        val_dirs.append(hospital_eval_dir)
        val_pid_lists.append(val_hosp_pids)
        val_domain_ids.append(1)

    return (train_dirs, train_pid_lists, train_domain_ids,
            val_dirs, val_pid_lists, val_domain_ids)


# ═══════════════════════════════════════════════
# SAM2 模型載入
# ═══════════════════════════════════════════════

def load_sam2_model(cfg, ckpt):
    """載入 SAM2 pretrained model（需先初始化 Hydra）"""
    import hydra
    from sam2.build_sam import build_sam2

    # 清除舊的 Hydra 狀態，重新初始化 sam2_configs
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module("sam2_configs", version_base="1.2")

    print(f"  載入 SAM2: cfg={cfg}, ckpt={ckpt}")
    sam2_model = build_sam2(cfg, ckpt)
    return sam2_model


# ═══════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════

def train(args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    defaults = S2_DEFAULTS if args.stage == 2 else S1_DEFAULTS
    batch_size = args.batch_size or defaults['batch_size']
    grad_accum = args.grad_accum or defaults['grad_accum']
    epochs = args.epochs or defaults['epochs']
    base_lr = args.lr or defaults['base_lr']
    roi_size = args.roi_size if args.roi_size is not None else defaults['roi_size']
    samples_per_domain = args.samples_per_domain or defaults['samples_per_domain']

    print(f"\n{'='*60}")
    print(f"V11 Training — Stage {args.stage} — Format: {args.data_format}")
    print(f"{'='*60}")
    print(f"  Batch: {batch_size} × {grad_accum} = {batch_size * grad_accum} effective")
    print(f"  Epochs: {epochs}, LR: {base_lr}")
    print(f"  ROI: {roi_size} ({'全圖' if roi_size == 0 else f'{roi_size}px → {IMG_SIZE}px'})")
    print(f"  Samples/domain: {samples_per_domain}")

    # ─── Datasets ───
    print("\n建立 Dataset...")

    if args.data_format == 'v11':
        train_entries, val_entries = parse_v11_split(args.split_json, args.data_dir)
        print(f"  Split: {len(train_entries)} train, {len(val_entries)} val")

        train_dataset = V11ROIDataset(
            train_entries, stage=args.stage, roi_size=roi_size,
            is_train=True, epoch=0, max_epochs=epochs)
        val_dataset = V11ROIDataset(
            val_entries, stage=args.stage, roi_size=roi_size,
            is_train=False, epoch=0, max_epochs=epochs)

    elif args.data_format == 'v10':
        (train_dirs, train_pids, train_doms,
         val_dirs, val_pids, val_doms) = parse_v10_split(
            args.split_json,
            args.public_data_dir, args.public_val_dir,
            args.hospital_data_dir, args.hospital_eval_dir)

        print(f"  Train dirs: {train_dirs}")
        print(f"  Val dirs: {val_dirs}")

        train_dataset = V10FlatDataset(
            train_dirs, train_pids, train_doms,
            is_train=True, epoch=0, max_epochs=epochs)
        val_dataset = V10FlatDataset(
            val_dirs, val_pids, val_doms,
            is_train=False, epoch=0, max_epochs=epochs)
    else:
        print(f"[錯誤] 未知 data-format: {args.data_format}")
        sys.exit(1)

    sampler = DomainAlternatingSampler(train_dataset, samples_per_domain, seed=SEED)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)

    # ─── Model ───
    print("\n建立模型...")
    if CoSegV10 is None:
        print("[錯誤] 找不到 model_v10.py!")
        sys.exit(1)

    sam2_model = load_sam2_model(args.sam2_cfg, args.sam2_ckpt)

    model = CoSegV10(
        sam2_model,
        use_lora=True,
        use_domain_bn=True,
        use_cross_slice=not args.no_cross_slice,
        lora_rank=args.lora_rank,
    )

    if args.stage == 2 and args.stage1_ckpt:
        if os.path.exists(args.stage1_ckpt):
            print(f"  載入 Stage 1: {args.stage1_ckpt}")
            ckpt = torch.load(args.stage1_ckpt, map_location='cpu', weights_only=False)
            sd = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(sd, strict=False)
            print("  Stage 1 權重已載入")
        else:
            print(f"  [警告] Stage 1 ckpt 不存在: {args.stage1_ckpt}")

    model = model.to(device)
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total_p:,} total, {train_p:,} trainable")

    # ─── Losses ───
    focal_dice_loss = FocalDiceBoundaryLoss()
    cldice_loss = SoftCLDiceLoss(iter_=5)
    z_cont_loss = ZContinuityLoss()
    sdf_loss = SDFAuxLoss()

    # ─── Optimizer ───
    dora_params, new_params, decoder_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora' in name.lower() or 'dora' in name.lower() or 'magnitude' in name.lower():
            dora_params.append(param)
        elif any(k in name for k in ['domain_bn', 'cross_slice', 'ds_head', 'att_gate']):
            new_params.append(param)
        else:
            decoder_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': dora_params, 'lr': base_lr * 0.5, 'name': 'dora'},
        {'params': new_params, 'lr': base_lr, 'name': 'new'},
        {'params': decoder_params, 'lr': base_lr, 'name': 'decoder'},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr * 0.01)
    scaler = GradScaler('cuda')

    # ─── Auto-resume ───
    start_epoch = 0
    best_dice = 0.0
    ckpt_prefix = f'coseg_v11_stage{args.stage}'
    last_ckpt = f'{ckpt_prefix}_last.pth'
    best_ckpt = f'{ckpt_prefix}_best.pth'

    if not args.fresh and os.path.exists(last_ckpt):
        print(f"\n自動恢復: {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_dice = ckpt.get('best_dice', 0.0)
        for _ in range(start_epoch):
            scheduler.step()
        print(f"  從 epoch {start_epoch} 繼續, best={best_dice:.4f}")

    # ─── Training ───
    print(f"\n開始訓練 (epoch {start_epoch} → {epochs})")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_dataset.set_epoch(epoch)
        sampler.seed = SEED + epoch

        epoch_losses = defaultdict(float)
        epoch_steps = 0
        t0 = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            images, masks, domains, weight_maps, sdfs = batch
            images = images.to(device)
            masks = masks.to(device)
            weight_maps = weight_maps.to(device)
            sdfs = sdfs.to(device)

            total_loss = torch.tensor(0.0, device=device)

            for d in domains.unique():
                d_val = d.item()
                d_mask = (domains == d_val)
                if d_mask.sum() == 0:
                    continue

                d_images = images[d_mask]
                d_masks_gt = masks[d_mask]
                d_wms = weight_maps[d_mask]
                d_sdfs = sdfs[d_mask]

                m = model.module if hasattr(model, 'module') else model
                if hasattr(m, 'set_domain'):
                    m.set_domain(d_val)

                with autocast('cuda'):
                    outputs = model(d_images)
                    if isinstance(outputs, dict):
                        logits = outputs['pred']
                        sdf_pred = outputs.get('sdf', None)
                    elif isinstance(outputs, (list, tuple)):
                        # outputs = (mask_ins, mask_sem, prob_ins, prob_sem, ds_outputs)
                        # 用 mask_sem (index 1) 做 canal segmentation
                        logits = outputs[1]
                        sdf_pred = None  # SDF 需要獨立 head，目前不用
                    else:
                        logits = outputs
                        sdf_pred = None

                    if logits.shape[-2:] != d_masks_gt.shape[-2:]:
                        logits = F.interpolate(logits, d_masks_gt.shape[-2:], mode='bilinear')
                    if sdf_pred is not None and sdf_pred.shape[-2:] != d_sdfs.shape[-2:]:
                        sdf_pred = F.interpolate(sdf_pred, d_sdfs.shape[-2:], mode='bilinear')

                    loss_fd = focal_dice_loss(logits, d_masks_gt, d_wms) * LOSS_WEIGHTS['focal_dice']
                    loss_cl = cldice_loss(logits, d_masks_gt) * LOSS_WEIGHTS['cldice']
                    loss_zc = z_cont_loss(logits) * LOSS_WEIGHTS['z_cont']
                    d_loss = loss_fd + loss_cl + loss_zc

                    if sdf_pred is not None:
                        loss_sdf = sdf_loss(sdf_pred, d_sdfs) * LOSS_WEIGHTS['sdf']
                        d_loss = d_loss + loss_sdf
                    else:
                        loss_sdf = torch.tensor(0.0)

                    if d_val == 1:
                        d_loss = d_loss * HOSPITAL_LOSS_WEIGHT

                    total_loss = total_loss + d_loss

                epoch_losses['focal_dice'] += loss_fd.item()
                epoch_losses['cldice'] += loss_cl.item()
                epoch_losses['z_cont'] += loss_zc.item()
                if sdf_pred is not None:
                    epoch_losses['sdf'] += loss_sdf.item()

            total_loss = total_loss / grad_accum
            scaler.scale(total_loss).backward()
            epoch_losses['total'] += total_loss.item() * grad_accum
            epoch_steps += 1

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if (step + 1) % 100 == 0:
                avg_loss = epoch_losses['total'] / epoch_steps
                elapsed = time.time() - t0
                eta = elapsed / (step + 1) * (len(train_loader) - step - 1)
                print(f"  E{epoch} [{step+1}/{len(train_loader)}] "
                      f"loss={avg_loss:.4f} {elapsed:.0f}s eta={eta:.0f}s")

        scheduler.step()
        avg_losses = {k: v / max(epoch_steps, 1) for k, v in epoch_losses.items()}
        epoch_time = time.time() - t0

        # ─── Validation ───
        val_results = evaluate(model, val_loader, device, stage=args.stage)
        pub_dice = val_results.get(0, 0.0)
        hosp_dice = val_results.get(1, 0.0)
        combined = BEST_MODEL_RATIO[0] * pub_dice + BEST_MODEL_RATIO[1] * hosp_dice

        print(f"\nEpoch {epoch}/{epochs} ({epoch_time:.0f}s)")
        print(f"  Loss: {avg_losses['total']:.4f} "
              f"(fd={avg_losses.get('focal_dice',0):.4f} "
              f"cl={avg_losses.get('cldice',0):.4f} "
              f"zc={avg_losses.get('z_cont',0):.4f})")
        print(f"  Dice: pub={pub_dice:.4f} hosp={hosp_dice:.4f} combined={combined:.4f}")

        # ─── Save ───
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_dice': best_dice,
            'pub_dice': pub_dice,
            'hosp_dice': hosp_dice,
            'combined': combined,
            'stage': args.stage,
            'roi_size': roi_size,
            'args': vars(args),
        }

        torch.save(state, last_ckpt)

        if combined > best_dice:
            best_dice = combined
            state['best_dice'] = best_dice
            torch.save(state, best_ckpt)
            print(f"  ★ Best! {combined:.4f} → {best_ckpt}")

        if (epoch + 1) % 10 == 0:
            torch.save(state, f'{ckpt_prefix}_epoch{epoch}.pth')

    print(f"\n{'='*60}")
    print(f"訓練完成! Best combined Dice: {best_dice:.4f}")
    print(f"  {best_ckpt} / {last_ckpt}")


# ═══════════════════════════════════════════════
# Stage 1 → ROI 預測 (給 Stage 2)
# ═══════════════════════════════════════════════

@torch.no_grad()
def predict_rois(args):
    """
    用 Stage 1 模型對所有已前處理的資料跑推論，
    產生 predicted ROI bbox 並寫入 metadata.json。
    """
    print("用 Stage 1 模型預測 ROI...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sam2_model = load_sam2_model(args.sam2_cfg, args.sam2_ckpt)

    model = CoSegV10(
        sam2_model,
        use_lora=True,
        use_domain_bn=True,
        use_cross_slice=not args.no_cross_slice,
        lora_rank=args.lora_rank,
    )
    ckpt = torch.load(args.stage1_ckpt, map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    model.eval()

    from preprocess_v11_spacing import interpolate_rois

    data_dir = Path(args.data_dir)
    for meta_path in sorted(data_dir.rglob('metadata.json')):
        with open(meta_path) as f:
            meta = json.load(f)

        patient_id = meta['patient_id']
        domain = 0 if meta['domain'] == 'public' else 1
        num_slices = meta['num_slices']
        slices_info = meta.get('slices', [])

        print(f"  {patient_id} ({num_slices} slices)")

        m = model.module if hasattr(model, 'module') else model
        if hasattr(m, 'set_domain'):
            m.set_domain(domain)

        predicted_rois = []
        margin = 32

        for i in range(num_slices):
            imgs = []
            for si in [i - 1, i, i + 1]:
                si = max(0, min(si, num_slices - 1))
                if si < len(slices_info):
                    img_path = slices_info[si]['img']
                    img = load_image_file(img_path)
                    img = apply_clahe_2d(img)
                else:
                    img = np.zeros((1024, 1024), dtype=np.float32)
                imgs.append(img)

            H, W = imgs[0].shape
            imgs_r = [cv2.resize(im, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
                      for im in imgs]
            input_t = torch.from_numpy(np.stack(imgs_r)).unsqueeze(0).float().to(device)

            with autocast('cuda'):
                outputs = model(input_t)
                if isinstance(outputs, dict):
                    logits = outputs['pred']
                elif isinstance(outputs, (list, tuple)):
                    logits = outputs[1]  # mask_sem
                else:
                    logits = outputs

            pred = torch.sigmoid(logits[0, 0]).cpu().numpy()
            pred_binary = (pred > 0.3).astype(np.uint8)
            pred_orig = cv2.resize(pred_binary, (W, H), interpolation=cv2.INTER_NEAREST)

            ys, xs = np.where(pred_orig > 0)
            if len(ys) > 10:
                roi = [
                    max(0, int(ys.min()) - margin),
                    min(H, int(ys.max()) + margin),
                    max(0, int(xs.min()) - margin),
                    min(W, int(xs.max()) + margin),
                ]
            else:
                roi = None
            predicted_rois.append(roi)

        predicted_rois = interpolate_rois(predicted_rois, num_slices)

        meta['predicted_roi'] = [list(r) if r else None for r in predicted_rois]
        valid = [r for r in predicted_rois if r is not None]
        if valid:
            meta['predicted_global_roi'] = [
                min(r[0] for r in valid), max(r[1] for r in valid),
                min(r[2] for r in valid), max(r[3] for r in valid),
            ]

        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"    ROI slices: {sum(1 for r in predicted_rois if r is not None)}")

    print("ROI 預測完成!")


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='V11 Training: ROI Cropping Two-Stage',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--stage', type=int, required=True, choices=[1, 2])
    parser.add_argument('--data-format', type=str, default='v11', choices=['v10', 'v11'],
                        help='v11=preprocessed (per-patient dirs), v10=original flat slices')

    # V11 mode
    parser.add_argument('--data-dir', type=str, default=None,
                        help='V11: preprocessed root; V10: unused')
    parser.add_argument('--split-json', required=True)

    # V10 mode: 需要指定各資料目錄
    parser.add_argument('--public-data-dir', type=str, default=None,
                        help='V10: public train data dir')
    parser.add_argument('--public-val-dir', type=str, default=None,
                        help='V10: public val data dir (if separate)')
    parser.add_argument('--hospital-data-dir', type=str, default=None,
                        help='V10: hospital train data dir')
    parser.add_argument('--hospital-eval-dir', type=str, default=None,
                        help='V10: hospital eval data dir')

    # Training
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--grad-accum', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--samples-per-domain', type=int, default=None)

    # ROI
    parser.add_argument('--roi-size', type=int, default=None)
    parser.add_argument('--use-predicted-roi', action='store_true',
                        help='用 Stage 1 預測的 ROI (而非 GT)')

    # Model
    parser.add_argument('--no-cross-slice', action='store_true')
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--stage1-ckpt', type=str, default='coseg_v11_stage1_best.pth')

    # SAM2
    parser.add_argument('--sam2-cfg', type=str, default='sam2_hiera_l.yaml',
                        help='SAM2 config filename (在 sam2/configs/ 下)')
    parser.add_argument('--sam2-ckpt', type=str, default='checkpoints/sam2_hiera_large.pt',
                        help='SAM2 pretrained checkpoint path')

    # Resume
    parser.add_argument('--fresh', action='store_true')

    # Special
    parser.add_argument('--predict-rois', action='store_true',
                        help='只用 Stage 1 預測 ROI（需 V11 格式）')

    args = parser.parse_args()

    if args.predict_rois:
        predict_rois(args)
    else:
        train(args)


if __name__ == '__main__':
    main()