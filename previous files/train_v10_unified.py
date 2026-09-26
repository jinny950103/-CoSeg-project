"""
train_v10_unified.py — 單模型統一訓練，衝公開+醫院 Dice 0.85
================================================================
整合 7 大策略：
  1. LoRA（~40M 可訓練 vs 196M）
  2. Domain-Specific BN（公開/醫院各一組 BN）
  3. Cross-Slice Attention（z 軸特徵融合）
  4. GT-Aware Asymmetric Loss（醫院球形 GT 邊界降權）
  5. Soft-clDice + Z-Continuity Loss（拓撲連續性）
  6. Attention Gate + Deep Supervision
  7. Test-Time BN Adaptation（在 eval 腳本中）

訓練方式：
  - Domain-Alternating Batches：偶數 step 公開、奇數 step 醫院
  - 每 step 設 model.set_domain() → BN 統計分開算
  - Cross-Slice: 每個 sample 是 K=5 連續片的 window

用法：
  python train_v10_unified.py
  python train_v10_unified.py --resume outputs/coseg_v10_last.pth
  python train_v10_unified.py --no-cross-slice  # 關掉 cross-slice
"""
import os, json, cv2, torch, random, math, gc, argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import torch.nn.functional as F
import hydra
from scipy.ndimage import gaussian_filter, map_coordinates, distance_transform_edt

from model_v10 import CoSegV10
from sam2.build_sam import build_sam2

PROJECT_ROOT = "/home/u9444861/-CoSeg-project"

# === 資料路徑 ===
PUBLIC_IMG_DIR    = "data/public_data/train/image_1024"
PUBLIC_MASK_DIR   = "data/public_data/train/mask_sem_1024"
HOSPITAL_IMG_DIR  = "data/hospital_data/train/image_1024"
HOSPITAL_MASK_DIR = "data/hospital_data/train/mask_sem_1024"
JSON_PATH         = "data_split_v10_mixed.json"

OUTPUT_DIR        = "outputs"
OUTPUT_BEST       = os.path.join(OUTPUT_DIR, "coseg_v10_best.pth")
OUTPUT_LAST       = os.path.join(OUTPUT_DIR, "coseg_v10_last.pth")
SAM2_CHECKPOINT   = "checkpoints/sam2_hiera_large.pt"
WARMSTART_WEIGHTS = os.path.join(OUTPUT_DIR, "coseg_v8_best.pth")

# === 超參數 ===
BATCH_SIZE        = 3       # cross-slice K=5 → 每 step 15 張過 backbone
GRAD_ACCUM        = 5       # 有效 batch = 3×5 = 15
NUM_EPOCHS        = 120
BASE_LR           = 8e-5
WARMUP_EPOCHS     = 5
NUM_WORKERS       = 6
POSITIVE_WEIGHT   = 4.0
CROSS_SLICE_K     = 5       # 連續 5 片
AUG_PROB          = 0.5
DS_WEIGHTS        = [0.3, 0.2, 0.1]
CLDICE_WEIGHT     = 0.5
TOPO_WEIGHT       = 0.3     # Z-continuity loss 權重
PATIENCE          = 25
LORA_RANK         = 16

# === Feature Toggles ===
USE_LORA          = True
USE_DOMAIN_BN     = True
USE_CROSS_SLICE   = True
USE_GT_AWARE_LOSS = True
USE_TOPO_LOSS     = True
USE_CLDICE        = True


# ==========================================
# 🔧 Losses
# ==========================================
class FocalDiceBoundaryLoss(nn.Module):
    def __init__(self, fa=0.75, fg=2.0, dw=1.0, fw=1.0, bw=0.5):
        super().__init__()
        self.fa, self.fg, self.dw, self.fw, self.bw = fa, fg, dw, fw, bw

    def forward(self, logits, target, weight_map=None):
        """
        Args:
            logits: (B,1,H,W)
            target: (B,1,H,W)
            weight_map: (B,1,H,W) optional per-pixel weight（GT-aware）
        """
        pred = torch.sigmoid(logits)
        p = pred.clamp(1e-6, 1 - 1e-6)

        # Focal
        at = self.fa * target + (1 - self.fa) * (1 - target)
        pt = p * target + (1 - p) * (1 - target)
        focal = -(at * (1 - pt) ** self.fg * torch.log(pt))
        if weight_map is not None:
            focal = focal * weight_map
        focal = focal.mean()

        # Dice
        inter = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 1.0 - ((2 * inter + 1e-5) / (union + 1e-5)).mean()

        # Boundary
        td = F.max_pool2d(target, 3, 1, 1)
        te = -F.max_pool2d(-target, 3, 1, 1)
        b = td - te
        if b.sum() == 0:
            bnd = torch.tensor(0.0, device=logits.device)
        else:
            with torch.amp.autocast('cuda', enabled=False):
                bp = (pred * b).float().clamp(1e-6, 1 - 1e-6)
                bt = (target * b).float()
                bnd = F.binary_cross_entropy(bp, bt, reduction='sum') / (b.sum().float() + 1e-5)

        return self.fw * focal + self.dw * dice + self.bw * bnd


class SoftCLDiceLoss(nn.Module):
    def __init__(self, iterations=10, smooth=1e-5):
        super().__init__()
        self.iterations = iterations
        self.smooth = smooth

    def soft_erode(self, img):
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)

    def soft_dilate(self, img):
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)
        for _ in range(self.iterations):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def forward(self, pred, target):
        if target.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        pred = pred.clamp(1e-6, 1 - 1e-6)
        skel_pred = self.soft_skel(pred)
        skel_target = self.soft_skel(target)
        tp = (skel_pred * target).sum(dim=(2, 3))
        sp = skel_pred.sum(dim=(2, 3))
        ts = (skel_target * pred).sum(dim=(2, 3))
        sg = skel_target.sum(dim=(2, 3))
        tprec = (tp + self.smooth) / (sp + self.smooth)
        tsens = (ts + self.smooth) / (sg + self.smooth)
        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens + self.smooth)
        return cl_dice.mean()


class ZContinuityLoss(nn.Module):
    """
    Z 軸連續性 Loss（替代 Persistent Homology Topology Loss）
    懲罰同一個 window 內相鄰 slice 預測機率的劇烈變化。
    在 GT 有管的區域，相鄰 slice 的預測應該平滑過渡。
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, pred_probs: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            pred_probs: (B, K, 1, H, W) — K 片的 sigmoid 預測
            targets:    (B, K, 1, H, W) — K 片的 GT
        """
        if pred_probs.shape[1] < 2:
            return torch.tensor(0.0, device=pred_probs.device, requires_grad=True)

        B, K, C, H, W = pred_probs.shape

        # 相鄰 slice 差異
        diff = (pred_probs[:, 1:] - pred_probs[:, :-1]).abs()  # (B, K-1, 1, H, W)

        # 只在 GT 有管的區域計算（任一相鄰 slice 有管）
        gt_union = torch.max(targets[:, 1:], targets[:, :-1])  # (B, K-1, 1, H, W)

        # 超過 margin 的劇烈變化才懲罰
        penalty = F.relu(diff - self.margin)

        # 加權：只在管區域
        if gt_union.sum() == 0:
            return torch.tensor(0.0, device=pred_probs.device, requires_grad=True)

        loss = (penalty * gt_union).sum() / (gt_union.sum() + 1e-5)
        return loss


# ==========================================
# 🔧 GT-Aware Weight Map（醫院球形 GT 用）
# ==========================================
def compute_gt_weight_map(gt_mask: np.ndarray, boundary_width: int = 3) -> np.ndarray:
    """
    對醫院球形 GT，在邊界附近降低 loss 權重。
    中心可信度高 → weight=1.0
    邊界附近不確定 → weight=0.3
    遠離 GT 的背景 → weight=1.0（FP 確定是錯的）

    Returns: weight_map (H, W) float32, values in [0.3, 1.0]
    """
    if gt_mask.sum() == 0:
        return np.ones_like(gt_mask, dtype=np.float32)

    gt_bool = gt_mask > 0

    # GT 內部：離邊界越遠越可信
    inside_dist = distance_transform_edt(gt_bool)
    max_inside = inside_dist.max()
    if max_inside > 0:
        inside_weight = np.clip(inside_dist / max_inside, 0.3, 1.0)
    else:
        inside_weight = np.ones_like(gt_mask, dtype=np.float32) * 0.3

    # GT 外部：離邊界近的可能是真管（FP 不一定是錯的）
    outside_dist = distance_transform_edt(~gt_bool)
    outside_weight = np.clip(outside_dist / boundary_width, 0.3, 1.0)

    weight_map = np.where(gt_bool, inside_weight, outside_weight)
    return weight_map.astype(np.float32)


# ==========================================
# 🔧 Augmentation
# ==========================================
def elastic_deform(image, mask, alpha=200, sigma=20):
    shape = image.shape[:2]
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    iy = np.clip(y + dy, 0, shape[0] - 1).astype(np.float64)
    ix = np.clip(x + dx, 0, shape[1] - 1).astype(np.float64)
    if len(image.shape) == 3:
        ri = np.zeros_like(image)
        for c in range(image.shape[2]):
            ri[:, :, c] = map_coordinates(image[:, :, c], [iy, ix], order=1, mode='reflect')
    else:
        ri = map_coordinates(image, [iy, ix], order=1, mode='reflect')
    rm = map_coordinates(mask, [iy, ix], order=0, mode='reflect')
    return ri, rm


def augment(image, mask, prob=0.5):
    if random.random() < prob:
        image = np.flip(image, axis=1).copy()
        mask = np.flip(mask, axis=1).copy()
    if random.random() < prob * 0.5:
        image = np.flip(image, axis=0).copy()
        mask = np.flip(mask, axis=0).copy()
    if random.random() < prob:
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2),
                                     random.uniform(-20, 20),
                                     random.uniform(0.85, 1.15))
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_REFLECT_101)
    if random.random() < 0.15:
        image, mask = elastic_deform(image, mask)
    if random.random() < prob:
        image = np.clip(random.uniform(0.8, 1.2) * image + random.uniform(-15, 15), 0, 255)
    if random.random() < prob * 0.3:
        image = np.clip(image + np.random.normal(0, random.uniform(3, 10), image.shape), 0, 255)
    return image.astype(np.float32), (mask > 0.5).astype(np.float32)


# ==========================================
# 📦 Unified Window Dataset
# ==========================================
class UnifiedWindowDataset(Dataset):
    """
    公開 / 醫院統一 Dataset。
    每個 sample 是 K 片連續 window，回傳中心片 ± context。
    回傳: images(K,3,H,W), masks(K,1,H,W), domain(int), weight_maps(K,1,H,W)
    """
    def __init__(self, file_list, img_dir, mask_dir, domain_id,
                 is_train=True, window_k=5, use_gt_aware=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.domain_id = domain_id
        self.is_train = is_train
        self.window_k = window_k
        self.use_gt_aware = use_gt_aware and (domain_id == 1)  # 只有醫院用
        self.pixel_mean = np.array([123.675, 116.280, 103.530], dtype=np.float32)
        self.pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        # 按病人分組
        self.patient_slices = {}  # patient → {slice_num: filename}
        for fname in file_list:
            parts = fname.rsplit('_slice_', 1)
            patient, sn = parts[0], int(parts[1])
            self.patient_slices.setdefault(patient, {})[sn] = fname

        # 建立 window 列表（中心片必須有 mask 檔案）
        self.windows = []   # (patient, center_sn)
        self.has_canal = []
        half = window_k // 2

        print(f"📊 建立 windows (domain={domain_id}, {'train' if is_train else 'val'})...")
        for patient, slices in sorted(self.patient_slices.items()):
            sorted_sns = sorted(slices.keys())
            for sn in sorted_sns:
                # 確認中心片有 mask
                fname = slices[sn]
                mp = os.path.join(mask_dir, fname + ".npy")
                if not os.path.exists(mp):
                    continue
                self.windows.append((patient, sn))
                m = np.load(mp)
                self.has_canal.append(m.sum() > 0)

        pos = sum(self.has_canal)
        print(f"   ✅ {len(self.windows)} windows | 正: {pos} | 負: {len(self.windows)-pos}")

    def get_sample_weights(self):
        return [POSITIVE_WEIGHT if h else 1.0 for h in self.has_canal]

    def _load_gray(self, patient, sn):
        slices = self.patient_slices.get(patient, {})
        if sn in slices:
            p = os.path.join(self.img_dir, slices[sn] + ".png")
        else:
            p = os.path.join(self.img_dir, f"{patient}_slice_{sn}.png")
        if os.path.exists(p):
            return cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        return None

    def _make_2_5d(self, patient, sn):
        """製作 2.5D 三通道（前一片, 當前片, 後一片）"""
        c = self._load_gray(patient, sn)
        if c is None:
            c = np.zeros((1024, 1024), dtype=np.float32)
        p = self._load_gray(patient, sn - 1)
        n = self._load_gray(patient, sn + 1)
        return np.stack([p if p is not None else c,
                         c,
                         n if n is not None else c], axis=-1)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        patient, center_sn = self.windows[idx]
        half = self.window_k // 2

        images, masks, weight_maps = [], [], []

        for offset in range(-half, half + 1):
            sn = center_sn + offset
            img = self._make_2_5d(patient, sn)

            # Load mask
            slices = self.patient_slices.get(patient, {})
            if sn in slices:
                mp = os.path.join(self.mask_dir, slices[sn] + ".npy")
            else:
                mp = os.path.join(self.mask_dir, f"{patient}_slice_{sn}.npy")

            if os.path.exists(mp):
                gt = (np.load(mp) > 0).astype(np.float32)
            else:
                gt = np.zeros((1024, 1024), dtype=np.float32)

            # Augmentation（同一 window 內用相同的 aug 嗎？不，各片獨立 aug 比較簡單）
            if self.is_train:
                img, gt = augment(img, gt, AUG_PROB)

            # GT-Aware weight map
            if self.use_gt_aware:
                wm = compute_gt_weight_map(gt)
            else:
                wm = np.ones_like(gt, dtype=np.float32)

            # Normalize
            img = (img - self.pixel_mean) / (self.pixel_std + 1e-8)

            images.append(torch.tensor(img).permute(2, 0, 1).float())      # (3, H, W)
            masks.append(torch.tensor(gt).unsqueeze(0).float())             # (1, H, W)
            weight_maps.append(torch.tensor(wm).unsqueeze(0).float())      # (1, H, W)

        # Stack window
        images = torch.stack(images, 0)       # (K, 3, H, W)
        masks = torch.stack(masks, 0)         # (K, 1, H, W)
        weight_maps = torch.stack(weight_maps, 0)  # (K, 1, H, W)

        return images, masks, self.domain_id, weight_maps


# ==========================================
# 🔧 Helpers
# ==========================================
def compute_dice_correct(pred, target, smooth=1e-5):
    pred = (pred > 0.5).float()
    total, count = 0.0, 0
    for i in range(pred.shape[0]):
        if target[i].sum() == 0:
            continue
        inter = (pred[i] * target[i]).sum()
        total += (2 * inter + smooth) / (pred[i].sum() + target[i].sum() + smooth)
        count += 1
    return total / count if count > 0 else None


class CosineWarmupScheduler:
    def __init__(self, opt, warmup, total, min_lr=1e-7):
        self.opt, self.wu, self.tot, self.min_lr = opt, warmup, total, min_lr
        self.base_lrs = [pg['lr'] for pg in opt.param_groups]

    def step(self, epoch):
        if epoch < self.wu:
            s = (epoch + 1) / self.wu
        else:
            s = 0.5 * (1 + math.cos(math.pi * (epoch - self.wu) / (self.tot - self.wu)))
        for pg, blr in zip(self.opt.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, blr * s)

    def get_lr(self):
        return [pg['lr'] for pg in self.opt.param_groups]


# ==========================================
# 🚀 主程式
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--no-cross-slice", action="store_true")
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--no-domain-bn", action="store_true")
    parser.add_argument("--no-gt-aware", action="store_true")
    parser.add_argument("--no-topo", action="store_true")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    use_lora = USE_LORA and not args.no_lora
    use_domain_bn = USE_DOMAIN_BN and not args.no_domain_bn
    use_cross_slice = USE_CROSS_SLICE and not args.no_cross_slice
    use_gt_aware = USE_GT_AWARE_LOSS and not args.no_gt_aware
    use_topo = USE_TOPO_LOSS and not args.no_topo

    device = "cuda"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(JSON_PATH) as f:
        splits = json.load(f)

    print("=" * 70)
    print("🚀 CoSeg V10 — 統一訓練（7 大策略）")
    print("=" * 70)
    print(f"  LoRA:           {'✅' if use_lora else '❌'} (rank={LORA_RANK})")
    print(f"  Domain-Spec BN: {'✅' if use_domain_bn else '❌'}")
    print(f"  Cross-Slice:    {'✅' if use_cross_slice else '❌'} (K={CROSS_SLICE_K})")
    print(f"  GT-Aware Loss:  {'✅' if use_gt_aware else '❌'}")
    print(f"  Z-Continuity:   {'✅' if use_topo else '❌'} (w={TOPO_WEIGHT})")
    print(f"  Soft-clDice:    {'✅' if USE_CLDICE else '❌'} (w={CLDICE_WEIGHT})")
    print(f"  Batch: {args.batch_size} × accum {GRAD_ACCUM} = eff {args.batch_size * GRAD_ACCUM}")
    print(f"  Epochs: {args.epochs}, LR: {BASE_LR}")
    print("=" * 70)

    # --- 資料 ---
    pub_train = UnifiedWindowDataset(
        splits["train"], PUBLIC_IMG_DIR, PUBLIC_MASK_DIR,
        domain_id=0, is_train=True, window_k=CROSS_SLICE_K if use_cross_slice else 1,
        use_gt_aware=False)
    hosp_train = UnifiedWindowDataset(
        splits["hospital_train"], HOSPITAL_IMG_DIR, HOSPITAL_MASK_DIR,
        domain_id=1, is_train=True, window_k=CROSS_SLICE_K if use_cross_slice else 1,
        use_gt_aware=use_gt_aware)

    # Val：用公開 val（domain=0）
    pub_val = UnifiedWindowDataset(
        splits["val"], PUBLIC_IMG_DIR, PUBLIC_MASK_DIR,
        domain_id=0, is_train=False, window_k=CROSS_SLICE_K if use_cross_slice else 1,
        use_gt_aware=False)
    hosp_val = UnifiedWindowDataset(
        splits["hospital_val"], HOSPITAL_IMG_DIR, HOSPITAL_MASK_DIR,
        domain_id=1, is_train=False, window_k=CROSS_SLICE_K if use_cross_slice else 1,
        use_gt_aware=False)

    # Samplers（正樣本 4× 過採樣）
    pub_sampler = WeightedRandomSampler(pub_train.get_sample_weights(), 1500, True)
    hosp_sampler = WeightedRandomSampler(hosp_train.get_sample_weights(), 1500, True)

    pub_dl = DataLoader(pub_train, args.batch_size, sampler=pub_sampler,
                        drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)
    hosp_dl = DataLoader(hosp_train, args.batch_size, sampler=hosp_sampler,
                         drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)
    pub_val_dl = DataLoader(pub_val, 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    hosp_val_dl = DataLoader(hosp_val, 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # --- 模型 ---
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')
    model = CoSegV10(
        build_sam2("sam2_hiera_l.yaml", SAM2_CHECKPOINT, mode="train"),
        use_lora=use_lora, lora_rank=LORA_RANK,
        use_domain_bn=use_domain_bn, num_domains=2,
        use_cross_slice=use_cross_slice, cross_slice_k=CROSS_SLICE_K,
    )

    # Warm start（載入 v8 權重，strict=False 因為有新模組）
    start_epoch = 0
    best_dice = 0.0

    if args.resume and os.path.exists(args.resume):
        print(f"📦 Resume: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt['model'], strict=False)
        start_epoch = ckpt.get('epoch', 0) + 1
        best_dice = ckpt.get('best_dice', 0.0)
        print(f"   Epoch {start_epoch}, best_dice {best_dice:.4f}")
    elif os.path.exists(WARMSTART_WEIGHTS):
        print(f"📦 Warm start: {WARMSTART_WEIGHTS}")
        sd = torch.load(WARMSTART_WEIGHTS, map_location="cpu")
        if isinstance(sd, dict) and 'model' in sd:
            sd = sd['model']
        clean = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        miss, unexp = model.load_state_dict(clean, strict=False)
        print(f"   Missing: {len(miss)}, Unexpected: {len(unexp)}")
        if miss:
            print(f"   Missing 前 10: {miss[:10]}")
        del sd, clean

    model = model.to(device)

    # --- Optimizer（分組 LR）---
    lora_params, new_params, decoder_params = [], [], []
    new_names = {"ag_mid", "ag_high", "ds_head_high", "ds_head_mid", "ds_head_low",
                 "csa_high", "csa_mid", "csa_low"}

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'lora_' in n:
            lora_params.append(p)
        elif any(nm in n for nm in new_names):
            new_params.append(p)
        else:
            decoder_params.append(p)

    print(f"\n📊 參數分組:")
    print(f"   LoRA:    {sum(p.numel() for p in lora_params)/1e6:.2f}M (lr={BASE_LR*0.5:.1e})")
    print(f"   New:     {sum(p.numel() for p in new_params)/1e6:.2f}M (lr={BASE_LR:.1e})")
    print(f"   Decoder: {sum(p.numel() for p in decoder_params)/1e6:.2f}M (lr={BASE_LR:.1e})")

    optimizer = optim.AdamW([
        {"params": lora_params, "lr": BASE_LR * 0.5},
        {"params": new_params, "lr": BASE_LR},
        {"params": decoder_params, "lr": BASE_LR},
    ], weight_decay=1e-4)

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        if 'optimizer' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
                print("   Optimizer restored")
            except:
                print("   ⚠️ Optimizer restore failed, using fresh")

    scheduler = CosineWarmupScheduler(optimizer, WARMUP_EPOCHS, args.epochs)

    # --- Loss ---
    criterion = FocalDiceBoundaryLoss()
    cldice_loss = SoftCLDiceLoss(iterations=10) if USE_CLDICE else None
    topo_loss = ZContinuityLoss(margin=0.3) if use_topo else None

    # --- Training loop ---
    patience_counter = 0
    scaler = torch.amp.GradScaler('cuda')

    print(f"\n🚀 開始訓練... (from epoch {start_epoch})")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        rl, rd, dc, ns = {0: 0.0, 1: 0.0}, {0: 0.0, 1: 0.0}, {0: 0, 1: 0}, {0: 0, 1: 0}

        pub_iter = iter(pub_dl)
        hosp_iter = iter(hosp_dl)
        steps_per_epoch = max(len(pub_dl), len(hosp_dl))

        optimizer.zero_grad()
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{args.epochs-1}")

        for step in pbar:
            # Domain alternating
            domain = step % 2

            try:
                if domain == 0:
                    batch = next(pub_iter)
                else:
                    batch = next(hosp_iter)
            except StopIteration:
                # 短的那個 domain 重來
                if domain == 0:
                    pub_iter = iter(pub_dl)
                    batch = next(pub_iter)
                else:
                    hosp_iter = iter(hosp_dl)
                    batch = next(hosp_iter)

            imgs, masks_all, dom_id, wmap_all = batch
            imgs = imgs.to(device)           # (B, K, 3, H, W) or (B, 1, 3, H, W)
            masks_all = masks_all.to(device) # (B, K, 1, H, W)
            wmap_all = wmap_all.to(device)

            # 取中心片的 mask 和 weight_map
            center = imgs.shape[1] // 2
            masks = masks_all[:, center]     # (B, 1, H, W)
            wmap = wmap_all[:, center]       # (B, 1, H, W)

            model.set_domain(domain)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if use_cross_slice:
                    outputs = model(x=imgs)  # 5D input
                else:
                    outputs = model(x=imgs[:, center])  # 4D input

                pred_sem = outputs[1]
                ds_outs = outputs[-1]

                pred_sem = F.interpolate(pred_sem, (1024, 1024),
                                          mode='bilinear', align_corners=False)
                pred_t = pred_sem[:, 0:1, :, :]

                # Main loss (with GT-aware weight for hospital)
                if domain == 1 and use_gt_aware:
                    main_loss = criterion(pred_t, masks, weight_map=wmap)
                else:
                    main_loss = criterion(pred_t, masks)

                total_loss = main_loss

                # Soft-clDice
                if USE_CLDICE:
                    pred_small = F.interpolate(pred_t, (256, 256), mode="bilinear", align_corners=False)
                    mask_small = F.interpolate(masks, (256, 256), mode="nearest")
                    pred_prob = torch.sigmoid(pred_small.float()).clamp(1e-6, 1 - 1e-6)
                    cl = cldice_loss(pred_prob, mask_small)
                    total_loss = total_loss + CLDICE_WEIGHT * cl

                # Z-Continuity Loss
                if use_topo and use_cross_slice and imgs.shape[1] > 1:
                    # 需要所有 K 片的預測
                    with torch.no_grad():
                        all_preds = []
                        for k_idx in range(imgs.shape[1]):
                            o = model(x=imgs[:, k_idx])
                            ps = F.interpolate(o[1], (256, 256), mode='bilinear', align_corners=False)
                            all_preds.append(torch.sigmoid(ps[:, 0:1].float()))
                        all_preds = torch.stack(all_preds, 1)  # (B, K, 1, 256, 256)

                    # masks for all K slices
                    masks_k_small = F.interpolate(
                        masks_all.view(-1, 1, 1024, 1024), (256, 256), mode='nearest'
                    ).view(imgs.shape[0], imgs.shape[1], 1, 256, 256)

                    tl = topo_loss(all_preds, masks_k_small)
                    total_loss = total_loss + TOPO_WEIGHT * tl

                # Deep Supervision
                ds_loss = torch.tensor(0.0, device=device)
                for dp, dw in zip(ds_outs, DS_WEIGHTS):
                    dg = F.interpolate(masks, dp.shape[2:], mode='nearest')
                    ds_loss = ds_loss + dw * criterion(dp.float().clamp(-20, 20), dg)
                total_loss = total_loss + ds_loss

                total_loss = total_loss / GRAD_ACCUM

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                optimizer.zero_grad()
                continue

            total_loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            # Metrics
            with torch.no_grad():
                probs = torch.sigmoid(pred_t.float().detach())
                bd = compute_dice_correct(probs, masks)
                rl[domain] += total_loss.item() * GRAD_ACCUM * imgs.size(0)
                ns[domain] += imgs.size(0)
                if bd is not None:
                    rd[domain] += bd
                    dc[domain] += 1

            if step % 50 == 0:
                d0 = rd[0] / max(dc[0], 1)
                d1 = rd[1] / max(dc[1], 1)
                pbar.set_postfix(pub_dice=f"{d0:.3f}", hosp_dice=f"{d1:.3f}")

        # Epoch metrics
        td0 = rd[0] / max(dc[0], 1)
        td1 = rd[1] / max(dc[1], 1)

        # --- Validation（每 3 epoch + 最後一個）---
        if epoch % 3 != 0 and epoch != args.epochs - 1:
            scheduler.step(epoch)
            print(f"\nEpoch {epoch} | Pub: {td0:.4f} Hosp: {td1:.4f} | Val: skipped")
            # Save last
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_dice': best_dice,
            }, OUTPUT_LAST)
            continue

        # Full validation
        model.eval()
        val_results = {}
        for name, dl, dom in [("pub", pub_val_dl, 0), ("hosp", hosp_val_dl, 1)]:
            vd, vdc = 0.0, 0
            model.set_domain(dom)
            with torch.no_grad():
                for imgs, masks_all, _, _ in dl:
                    imgs, masks_all = imgs.to(device), masks_all.to(device)
                    center = imgs.shape[1] // 2
                    masks = masks_all[:, center]
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        if use_cross_slice:
                            outputs = model(x=imgs)
                        else:
                            outputs = model(x=imgs[:, center])
                        ps = F.interpolate(outputs[1], (1024, 1024),
                                            mode='bilinear', align_corners=False)
                    bd = compute_dice_correct(torch.sigmoid(ps[:, 0:1].float()), masks)
                    if bd is not None:
                        vd += bd
                        vdc += 1
            val_results[name] = vd / max(vdc, 1)

        scheduler.step(epoch)
        vd_pub = val_results["pub"]
        vd_hosp = val_results["hosp"]
        # 用兩個 domain 的加權平均決定最佳
        combined = 0.5 * vd_pub + 0.5 * vd_hosp

        print(f"\nEpoch {epoch} | Train: pub={td0:.4f} hosp={td1:.4f} | "
              f"Val: pub={vd_pub:.4f} hosp={vd_hosp:.4f} combined={combined:.4f} | "
              f"LR: {scheduler.get_lr()[0]:.2e}")

        # Save checkpoint
        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_dice': best_dice,
            'val_pub': vd_pub,
            'val_hosp': vd_hosp,
        }
        torch.save(ckpt, OUTPUT_LAST)

        if combined > best_dice:
            best_dice = combined
            torch.save(ckpt, OUTPUT_BEST)
            print(f"🏆 新紀錄！Combined: {best_dice:.4f} (pub={vd_pub:.4f} hosp={vd_hosp:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  未破紀錄 (最佳: {best_dice:.4f}) | Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"⛔ Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\n✅ 完成！最佳 Combined Dice: {best_dice:.4f}")
    print(f"   權重: {OUTPUT_BEST}")
    print(f"\n📋 下一步:")
    print(f"   python eval_v10_unified.py --weights {OUTPUT_BEST} --domain public")
    print(f"   python eval_v10_unified.py --weights {OUTPUT_BEST} --domain hospital")


if __name__ == "__main__":
    main()
