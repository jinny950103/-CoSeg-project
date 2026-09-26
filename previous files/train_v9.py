"""
train_v9.py — Distance-Aware Curriculum Training
====================================================
整合三個 2025-2026 最新技術，一次訓練搞定：

1. Pixel-level Distance-Aware Weighting (GSD-Net, Wang et al., MedIA 2026)
   → 醫院 sphere label 的中心 pixel loss 權重高，膨脹邊界權重低

2. Curriculum Sample Decay (Han et al., IEEE TMI 2025)
   → 醫院資料權重從 1.0 平滑衰減到 0.0

3. LR Curriculum
   → 前期 encoder LR 較高學 domain，後期壓低只修 decoder

vs v8:  固定權重混合 → 邊界被拉粗
vs v8-s2: 兩次訓練硬切 → 過渡不平滑
vs v9:  一次訓練，pixel-level 加權 + 平滑 curriculum
"""
import os, json, cv2, torch, random, math, gc
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import torch.nn.functional as F
import hydra
from scipy.ndimage import gaussian_filter, map_coordinates, distance_transform_edt

from model_v6 import CoSegV6
from sam2.build_sam import build_sam2

PROJECT_ROOT = "/home/u9444861/-CoSeg-project"

# === 資料路徑 ===
PUBLIC_IMG_DIR   = "data/public_data/train/image_1024"
PUBLIC_MASK_DIR  = "data/public_data/train/mask_sem_1024"
HOSPITAL_IMG_DIR = "data/hospital_data/train/image_1024"
HOSPITAL_MASK_DIR= "data/hospital_data/train/mask_sem_1024"

JSON_PATH        = "data_split_v8_mixed_v2.json"
OUTPUT_DIR       = "outputs"
OUTPUT_WEIGHTS   = os.path.join(OUTPUT_DIR, "coseg_v9_best.pth")
OUTPUT_LAST      = os.path.join(OUTPUT_DIR, "coseg_v9_last.pth")
SAM2_CHECKPOINT  = "checkpoints/sam2_hiera_large.pt"
WARMSTART_WEIGHTS= os.path.join(OUTPUT_DIR, "coseg_v7_best.pth")  # 從 v7 開始

BATCH_SIZE       = 12
NUM_EPOCHS       = 150
BASE_LR          = 1e-4
WARMUP_EPOCHS    = 5
NUM_WORKERS      = 8
POSITIVE_WEIGHT  = 4.0
CONTEXT_SLICES   = 1
AUG_PROB         = 0.7
DS_WEIGHTS       = [0.3, 0.2, 0.1]
CLDICE_WEIGHT    = 0.5
PATIENCE         = 35

# === v9 新增：Curriculum 參數 ===
CURRICULUM_START = 1.0      # 醫院初始權重
CURRICULUM_END   = 0.0      # 醫院最終權重
CURRICULUM_WARMUP = 10      # 前 N epochs 保持初始權重
CURRICULUM_DECAY_END = 130  # 在這個 epoch 衰減到最終權重

# === v9 新增：Distance-Aware 參數 ===
DIST_WEIGHT_MIN = 0.3       # 邊界 pixel 的最低權重
DIST_WEIGHT_MAX = 1.0       # 中心 pixel 的最高權重


# ==========================================
# 🔧 Curriculum Schedule
# ==========================================
def get_hospital_weight(epoch):
    """
    Curriculum decay: 醫院資料權重從 CURRICULUM_START 衰減到 CURRICULUM_END
    前 CURRICULUM_WARMUP epochs 保持不變，之後 cosine decay
    """
    if epoch < CURRICULUM_WARMUP:
        return CURRICULUM_START
    if epoch >= CURRICULUM_DECAY_END:
        return CURRICULUM_END

    progress = (epoch - CURRICULUM_WARMUP) / (CURRICULUM_DECAY_END - CURRICULUM_WARMUP)
    # Cosine decay（比 linear 更平滑）
    decay = 0.5 * (1 + math.cos(math.pi * progress))
    return CURRICULUM_END + (CURRICULUM_START - CURRICULUM_END) * decay


def get_lr_scale(epoch):
    """
    LR curriculum: 前期 encoder LR 正常，後期壓低
    回傳 (encoder_scale, head_scale)
    """
    if epoch < CURRICULUM_WARMUP:
        return 1.0, 1.0

    progress = min(1.0, (epoch - CURRICULUM_WARMUP) / (CURRICULUM_DECAY_END - CURRICULUM_WARMUP))
    # Encoder: 從 1.0 衰減到 0.1（後期幾乎不動 encoder）
    enc_scale = 0.1 + 0.9 * (1 - progress)
    # Head: 從 1.0 衰減到 0.5
    head_scale = 0.5 + 0.5 * (1 - progress)
    return enc_scale, head_scale


# ==========================================
# 🔧 Distance-Aware Weight Map
# ==========================================
def compute_distance_weight(mask):
    """
    對醫院 mask 計算 pixel-level 的 confidence weight
    中心 pixel → weight=1.0（可信）
    膨脹邊界 pixel → weight=0.3（不可信）

    公開資料不用這個（直接全 1.0）
    """
    mask_bin = (mask > 0.5).astype(np.uint8)
    if mask_bin.sum() == 0:
        return np.ones_like(mask, dtype=np.float32)

    # 計算每個前景 pixel 到最近邊界的距離
    dist = distance_transform_edt(mask_bin).astype(np.float32)
    max_dist = dist.max()

    if max_dist <= 0:
        return np.ones_like(mask, dtype=np.float32)

    # 距離歸一化到 [DIST_WEIGHT_MIN, DIST_WEIGHT_MAX]
    # 離中心越近 → 權重越大
    weight = np.ones_like(mask, dtype=np.float32)
    foreground = mask_bin > 0
    weight[foreground] = DIST_WEIGHT_MIN + (DIST_WEIGHT_MAX - DIST_WEIGHT_MIN) * (dist[foreground] / max_dist)

    return weight


# ==========================================
# 🔧 Soft-clDice Loss
# ==========================================
class SoftCLDiceLoss(nn.Module):
    def __init__(self, iterations=10, smooth=1e-5):
        super().__init__()
        self.iterations = iterations
        self.smooth = smooth

    def soft_erode(self, img):
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        p3 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        return torch.min(p2, p3)

    def soft_dilate(self, img):
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        img1 = self.soft_open(img)
        sk = F.relu(img - img1)
        for _ in range(self.iterations):
            img1 = self.soft_erode(img)
            img2 = self.soft_open(img1)
            delta = F.relu(img1 - img2)
            sk = sk + F.relu(delta - sk * delta)
            img = img1
        return sk

    def forward(self, pred, target):
        skel_pred = self.soft_skel(pred)
        skel_gt = self.soft_skel(target)
        tp = (skel_pred * target).sum(dim=(2, 3))
        sp = skel_pred.sum(dim=(2, 3))
        ts = (skel_gt * pred).sum(dim=(2, 3))
        sg = skel_gt.sum(dim=(2, 3))
        tprec = (tp + self.smooth) / (sp + self.smooth)
        tsens = (ts + self.smooth) / (sg + self.smooth)
        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens + self.smooth)
        return cl_dice.mean()


# ==========================================
# 🔧 Augmentation
# ==========================================
def elastic_deform(image, mask, alpha=200, sigma=20):
    shape = image.shape[:2]
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    iy = np.clip(y + dy, 0, shape[0]-1).astype(np.float64)
    ix = np.clip(x + dx, 0, shape[1]-1).astype(np.float64)
    if len(image.shape) == 3:
        ri = np.zeros_like(image)
        for c in range(image.shape[2]):
            ri[:,:,c] = map_coordinates(image[:,:,c], [iy, ix], order=1, mode='reflect')
    else:
        ri = map_coordinates(image, [iy, ix], order=1, mode='reflect')
    rm = map_coordinates(mask, [iy, ix], order=0, mode='reflect')
    return ri, rm


def random_affine(image, mask, max_rot=25, max_scale=0.2, max_shear=12):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), random.uniform(-max_rot, max_rot),
                                 random.uniform(1-max_scale, 1+max_scale))
    sx = math.tan(math.radians(random.uniform(-max_shear, max_shear)))
    sy = math.tan(math.radians(random.uniform(-max_shear, max_shear)))
    M = M + np.array([[1, sx, 0], [sy, 1, 0]], dtype=np.float64) * 0.3
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    return image, mask


def copy_paste_canal(image, mask, crops):
    if not crops or random.random() > 0.3:
        return image, mask
    ci, cm = random.choice(crops)
    ys, xs = np.where(cm > 0)
    if len(ys) == 0: return image, mask
    pad = 20
    y0, y1 = max(0, ys.min()-pad), min(cm.shape[0]-1, ys.max()+pad)
    x0, x1 = max(0, xs.min()-pad), min(cm.shape[1]-1, xs.max()+pad)
    pm, pi_ = cm[y0:y1+1, x0:x1+1], ci[y0:y1+1, x0:x1+1]
    moy, mox = image.shape[0]-pm.shape[0], image.shape[1]-pm.shape[1]
    if moy <= 0 or mox <= 0: return image, mask
    oy, ox = random.randint(0, moy), random.randint(0, mox)
    rm = pm > 0; ph, pw = pm.shape
    if len(image.shape) == 3:
        for c in range(image.shape[2]):
            src = pi_[..., c] if len(pi_.shape) == 3 else pi_
            image[oy:oy+ph, ox:ox+pw, c][rm] = src[rm]
    else:
        image[oy:oy+ph, ox:ox+pw][rm] = pi_[rm]
    mask[oy:oy+ph, ox:ox+pw][rm] = 1.0
    return image, mask


def augment_strong(image, mask, crops=None):
    if random.random() < AUG_PROB:
        image = np.flip(image, axis=1).copy(); mask = np.flip(mask, axis=1).copy()
    if random.random() < AUG_PROB * 0.6:
        image = np.flip(image, axis=0).copy(); mask = np.flip(mask, axis=0).copy()
    if random.random() < AUG_PROB:
        image, mask = random_affine(image, mask)
    if random.random() < 0.20:
        image, mask = elastic_deform(image, mask)
    if crops and mask.sum() > 0:
        image, mask = copy_paste_canal(image, mask, crops)
    if random.random() < AUG_PROB:
        image = np.clip(random.uniform(0.6, 1.4) * image + random.uniform(-25, 25), 0, 255)
    if random.random() < AUG_PROB * 0.5:
        image = np.clip(np.power(image / 255.0, random.uniform(0.5, 1.6)) * 255.0, 0, 255)
    if random.random() < AUG_PROB * 0.5:
        image = np.clip(image + np.random.normal(0, random.uniform(5, 20), image.shape), 0, 255)
    if random.random() < AUG_PROB * 0.4:
        image = cv2.GaussianBlur(image, (random.choice([3, 5, 7]),) * 2, 0)
    if random.random() < 0.25:
        h, w = image.shape[:2]
        ch, cw = random.randint(80, 200), random.randint(80, 200)
        cy, cx = random.randint(0, h-ch), random.randint(0, w-cw)
        image[cy:cy+ch, cx:cx+cw] = 0
    return image.astype(np.float32), (mask > 0.5).astype(np.float32)


# ==========================================
# 📦 Dataset（回傳 weight map）
# ==========================================
class CurriculumCanalDataset(Dataset):
    """
    跟 v8 的 MixedCanalDataset 一樣讀取公開+醫院，
    但額外回傳 pixel-level weight map：
      - 公開資料：weight = 全 1.0
      - 醫院資料：weight = distance-aware（中心高、邊界低）
    """
    def __init__(self, data_sources, is_train=True):
        self.is_train = is_train
        self.pixel_mean = np.array([123.675, 116.280, 103.530], dtype=np.float32)
        self.pixel_std  = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        self.patient_slice_map = {}
        self.entries = []

        for img_dir, mask_dir, file_list, source in data_sources:
            for fname in file_list:
                parts = fname.rsplit('_slice_', 1)
                patient, sn = parts[0], int(parts[1])
                key = (patient, source)
                self.patient_slice_map.setdefault(key, {})[sn] = fname
                self.entries.append((patient, sn, fname, img_dir, mask_dir, source))

        self.has_canal = []
        self.source_tags = []
        self.positive_crops = []

        # 預計算醫院資料的 distance weight maps
        self.dist_weights = {}  # fname -> weight_map

        print(f"📊 掃描 ({'train' if is_train else 'val'})...")
        for patient, sn, fname, img_dir, mask_dir, source in tqdm(self.entries, desc="Scanning"):
            mp = os.path.join(mask_dir, fname + ".npy")
            if os.path.exists(mp):
                m = np.load(mp)
                is_pos = m.sum() > 0
                self.has_canal.append(is_pos)

                # 醫院正樣本：預計算 distance weight
                if source == "hospital" and is_pos:
                    self.dist_weights[fname] = compute_distance_weight(m)

                if is_train and is_pos and len(self.positive_crops) < 200:
                    ip = os.path.join(img_dir, fname + ".png")
                    if os.path.exists(ip):
                        self.positive_crops.append(
                            (cv2.imread(ip, cv2.IMREAD_GRAYSCALE).astype(np.float32),
                             m.astype(np.float32)))
            else:
                self.has_canal.append(False)
            self.source_tags.append(source)

        n_pub = sum(1 for s in self.source_tags if s == "public")
        n_hos = sum(1 for s in self.source_tags if s == "hospital")
        n_pos = sum(self.has_canal)
        n_dist = len(self.dist_weights)
        print(f"   ✅ 公開: {n_pub} | 醫院: {n_hos} | 正: {n_pos} | 負: {len(self.has_canal)-n_pos}")
        print(f"   🎯 Distance weight maps: {n_dist}")

    def get_sample_weights(self, hospital_weight=1.0):
        """動態採樣權重（每個 epoch 可以更新 hospital_weight）"""
        weights = []
        for has_canal, source in zip(self.has_canal, self.source_tags):
            base = POSITIVE_WEIGHT if has_canal else 1.0
            if source == "hospital":
                base *= hospital_weight
            weights.append(base)
        return weights

    def _load_gray(self, patient, sn, img_dir, source):
        key = (patient, source)
        if key in self.patient_slice_map and sn in self.patient_slice_map[key]:
            fname = self.patient_slice_map[key][sn]
        else:
            return None
        p = os.path.join(img_dir, fname + ".png")
        return cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) if os.path.exists(p) else None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        patient, sn, fname, img_dir, mask_dir, source = self.entries[idx]
        c = self._load_gray(patient, sn, img_dir, source)
        p = self._load_gray(patient, sn - CONTEXT_SLICES, img_dir, source)
        n = self._load_gray(patient, sn + CONTEXT_SLICES, img_dir, source)
        img = np.stack([p if p is not None else c, c, n if n is not None else c], axis=-1)
        gt = (np.load(os.path.join(mask_dir, fname + ".npy")) > 0).astype(np.float32)

        # 取得 distance weight map
        if source == "hospital" and fname in self.dist_weights:
            wmap = self.dist_weights[fname].copy()
        else:
            wmap = np.ones_like(gt, dtype=np.float32)

        if self.is_train:
            img, gt = augment_strong(img, gt, self.positive_crops)
            # augmentation 可能改變 mask，但 weight map 跟 mask 一起做同樣的幾何變換太複雜
            # 簡化處理：augment 後重算 weight（只對醫院正樣本）
            if source == "hospital" and gt.sum() > 0:
                wmap = compute_distance_weight(gt)
            else:
                wmap = np.ones_like(gt, dtype=np.float32)

        img = (img - self.pixel_mean) / (self.pixel_std + 1e-8)
        return (torch.tensor(img).permute(2, 0, 1).float(),
                torch.tensor(gt).unsqueeze(0).float(),
                torch.tensor(wmap).unsqueeze(0).float())


# ==========================================
# 🎯 Weighted Loss（支援 pixel-level weight map）
# ==========================================
class WeightedFocalDiceBoundaryLoss(nn.Module):
    def __init__(self, fa=0.75, fg=2.0, dw=1.0, fw=1.0, bw=0.5):
        super().__init__()
        self.fa, self.fg, self.dw, self.fw, self.bw = fa, fg, dw, fw, bw

    def forward(self, logits, target, weight_map=None):
        pred = torch.sigmoid(logits)

        if weight_map is None:
            weight_map = torch.ones_like(target)

        # Weighted Focal Loss
        p = pred.clamp(1e-6, 1-1e-6)
        at = self.fa * target + (1-self.fa) * (1-target)
        pt = p * target + (1-p) * (1-target)
        focal_per_pixel = -at * (1-pt)**self.fg * torch.log(pt)
        focal = (focal_per_pixel * weight_map).mean()

        # Weighted Dice Loss
        inter = (pred * target * weight_map).sum(dim=(2, 3))
        union = (pred * weight_map).sum(dim=(2, 3)) + (target * weight_map).sum(dim=(2, 3))
        dice = 1.0 - ((2 * inter + 1e-5) / (union + 1e-5)).mean()

        # Boundary Loss（不加 weight，邊界本身就是重要的）
        td = F.max_pool2d(target, 3, 1, 1)
        te = -F.max_pool2d(-target, 3, 1, 1)
        b = td - te
        if b.sum() == 0:
            bnd = torch.tensor(0.0, device=logits.device)
        else:
            with torch.amp.autocast('cuda', enabled=False):
                bnd = F.binary_cross_entropy(
                    (pred * b).float().clamp(1e-6, 1-1e-6), (target * b).float(),
                    reduction='sum') / (b.sum().float() + 1e-5)

        return self.fw * focal + self.dw * dice + self.bw * bnd


def compute_dice_correct(pred, target, smooth=1e-5):
    pred = (pred > 0.5).float()
    total, count = 0.0, 0
    for i in range(pred.shape[0]):
        if target[i].sum() == 0: continue
        inter = (pred[i]*target[i]).sum()
        total += (2*inter+smooth)/(pred[i].sum()+target[i].sum()+smooth); count += 1
    return total/count if count > 0 else None


class CosineWarmupScheduler:
    def __init__(self, opt, warmup, total, min_lr=1e-7):
        self.opt, self.wu, self.tot, self.min_lr = opt, warmup, total, min_lr
        self.base_lrs = [pg['lr'] for pg in opt.param_groups]
    def step(self, epoch, enc_scale=1.0, head_scale=1.0):
        s = (epoch+1)/self.wu if epoch < self.wu else 0.5*(1+math.cos(math.pi*(epoch-self.wu)/(self.tot-self.wu)))
        scales = [enc_scale, enc_scale, head_scale]  # [encoder, AG/DS, head]
        for pg, blr, sc in zip(self.opt.param_groups, self.base_lrs, scales):
            pg['lr'] = max(self.min_lr, blr * s * sc)
    def get_lr(self):
        return [pg['lr'] for pg in self.opt.param_groups]


# ==========================================
# 🚀 主程式
# ==========================================
def main():
    device = "cuda"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(JSON_PATH) as f:
        splits = json.load(f)

    # 組合 data sources
    train_sources = [
        (PUBLIC_IMG_DIR, PUBLIC_MASK_DIR, splits['train'], "public"),
        (HOSPITAL_IMG_DIR, HOSPITAL_MASK_DIR, splits['hospital_train'], "hospital"),
    ]
    # Val 只用公開資料（公平比較）
    val_sources = [
        (PUBLIC_IMG_DIR, PUBLIC_MASK_DIR, splits['val'], "public"),
    ]

    print("="*60)
    print("🧠 v9: Distance-Aware Curriculum Training")
    print(f"  Public train:   {len(splits['train'])} slices")
    print(f"  Hospital train: {len(splits['hospital_train'])} slices")
    print(f"  Val:            {len(splits['val'])} slices (公開 only)")
    print(f"  Curriculum:     hospital weight {CURRICULUM_START} → {CURRICULUM_END}")
    print(f"                  warmup={CURRICULUM_WARMUP}, decay_end={CURRICULUM_DECAY_END}")
    print(f"  Distance-Aware: weight [{DIST_WEIGHT_MIN}, {DIST_WEIGHT_MAX}]")
    print(f"  Loss: Weighted(Focal+Dice+Boundary) + clDice(w={CLDICE_WEIGHT}) + DS({DS_WEIGHTS})")
    print(f"  Warm start: {WARMSTART_WEIGHTS}")
    print("="*60)

    train_ds = CurriculumCanalDataset(train_sources, is_train=True)
    val_ds   = CurriculumCanalDataset(val_sources, is_train=False)

    val_dl = DataLoader(val_ds, 4, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')
    print(f"📦 SAM2: {SAM2_CHECKPOINT}")
    model = CoSegV6(build_sam2("sam2_hiera_l.yaml", SAM2_CHECKPOINT, mode="train"))

    if os.path.exists(WARMSTART_WEIGHTS):
        print(f"📦 Warm start: {WARMSTART_WEIGHTS}")
        sd = torch.load(WARMSTART_WEIGHTS, map_location="cpu")
        clean = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        miss, unexp = model.load_state_dict(clean, strict=False)
        print(f"   Missing: {len(miss)}, Unexpected: {len(unexp)}")
        del sd, clean
    model = model.to(device)

    # 三組 LR
    new_names = {"ag_mid", "ag_high", "ds_head_high", "ds_head_mid", "ds_head_low"}
    enc_params, new_params, head_params = [], [], []
    for n, p in model.named_parameters():
        is_new = any(nm in n for nm in new_names)
        if is_new:
            p.requires_grad = True; new_params.append(p)
        elif "image_encoder" in n:
            if "edge" in n or "neck" in n:
                p.requires_grad = True; enc_params.append(p)
            elif "trunk.blocks" in n:
                bm = [s for s in n.split('.') if s.isdigit()]
                if bm and int(bm[0]) >= 16:
                    p.requires_grad = True; enc_params.append(p)
                else:
                    p.requires_grad = False
            else:
                p.requires_grad = False
        else:
            p.requires_grad = True; head_params.append(p)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"🧠 可訓練: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({trainable/total*100:.1f}%)")

    optimizer = optim.AdamW([
        {"params": enc_params,  "lr": BASE_LR * 0.1},    # 1e-5
        {"params": new_params,  "lr": BASE_LR * 0.5},    # 5e-5
        {"params": head_params, "lr": BASE_LR},           # 1e-4
    ], weight_decay=1e-4)
    scheduler = CosineWarmupScheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)
    criterion = WeightedFocalDiceBoundaryLoss()
    criterion_unweighted = WeightedFocalDiceBoundaryLoss()  # DS 用（不加 pixel weight）
    cldice_loss = SoftCLDiceLoss(iterations=10)

    best_dice, patience_counter = 0.0, 0

    print(f"\n🚀 開始 v9 訓練...")
    for epoch in range(NUM_EPOCHS):
        model.train()

        # === Curriculum: 更新醫院權重 ===
        hw = get_hospital_weight(epoch)
        enc_scale, head_scale = get_lr_scale(epoch)

        # 每個 epoch 重新建 sampler（因為醫院權重在變）
        sample_weights = train_ds.get_sample_weights(hospital_weight=hw)
        sampler = WeightedRandomSampler(sample_weights, len(train_ds), True)
        train_dl = DataLoader(train_ds, BATCH_SIZE, sampler=sampler, drop_last=True,
                              num_workers=NUM_WORKERS, pin_memory=True)

        if epoch % 10 == 0 or epoch < 5:
            print(f"\n  📅 Epoch {epoch}: hospital_weight={hw:.3f}, "
                  f"enc_lr_scale={enc_scale:.2f}, head_lr_scale={head_scale:.2f}")

        rl, rd, dc, ns = 0.0, 0.0, 0, 0
        optimizer.zero_grad()

        for step, (imgs, masks, wmaps) in enumerate(tqdm(train_dl, desc=f"Epoch {epoch}/{NUM_EPOCHS-1} [Train]")):
            imgs = imgs.to(device)
            masks = masks.to(device)
            wmaps = wmaps.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(x=imgs)
                pred_sem = outputs[1]
                ds_outs = outputs[-1]
                pred_sem = F.interpolate(pred_sem, (1024, 1024), mode='bilinear', align_corners=False)
                pred_t = pred_sem[:, 0:1, :, :]

                # Weighted main loss（用 distance-aware weight map）
                main_loss = criterion(pred_t, masks, wmaps)

                # Soft-clDice（在 256x256 計算）
                pred_256 = F.interpolate(pred_t, (256, 256), mode='bilinear', align_corners=False)
                mask_256 = F.interpolate(masks, (256, 256), mode='nearest')
                pred_prob = torch.sigmoid(pred_256.float()).clamp(1e-6, 1-1e-6)
                cl_loss = cldice_loss(pred_prob, mask_256)

                # DS（不加 pixel weight，避免 DS head 被 weight 干擾）
                ds_loss = torch.tensor(0.0, device=device)
                for dp, dw in zip(ds_outs, DS_WEIGHTS):
                    dg = F.interpolate(masks, dp.shape[2:], mode='nearest')
                    ds_loss = ds_loss + dw * criterion_unweighted(dp.float().clamp(-20, 20), dg)

                loss = main_loss + CLDICE_WEIGHT * cl_loss + ds_loss

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ Loss={loss.item()}, skip"); optimizer.zero_grad(); continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step(); optimizer.zero_grad()

            probs = torch.sigmoid(pred_t.float().detach())
            bd = compute_dice_correct(probs, masks)
            rl += loss.item() * imgs.size(0); ns += imgs.size(0)
            if bd is not None: rd += bd; dc += 1

        tl, td = rl/max(ns,1), rd/max(dc,1)

        if epoch % 3 != 0 and epoch != NUM_EPOCHS-1:
            scheduler.step(epoch, enc_scale, head_scale)
            print(f"\nEpoch {epoch} | Train Loss: {tl:.4f} Dice: {td:.4f} | Val: skipped | "
                  f"LR: {scheduler.get_lr()[0]:.2e} | HW: {hw:.3f}")
            torch.save(model.state_dict(), OUTPUT_LAST); continue

        # === Validation（只用公開資料） ===
        model.eval()
        vl, vd, vdc, vn = 0.0, 0.0, 0, 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Epoch {epoch}/{NUM_EPOCHS-1} [Val]"):
                imgs, masks, wmaps = batch
                imgs, masks = imgs.to(device), masks.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(x=imgs)
                    ps = F.interpolate(outputs[1], (1024,1024), mode='bilinear', align_corners=False)
                    pt = ps[:, 0:1, :, :]
                    lo = criterion_unweighted(pt, masks)  # Val 不用 weight
                bd = compute_dice_correct(torch.sigmoid(pt.float()), masks)
                vl += lo.item()*imgs.size(0); vn += imgs.size(0)
                if bd is not None: vd += bd; vdc += 1

        vda = vd/max(vdc,1); vl /= max(vn,1)
        scheduler.step(epoch, enc_scale, head_scale)
        print(f"\nEpoch {epoch} | Train: {tl:.4f}/{td:.4f} | Val: {vl:.4f}/{vda:.4f} | "
              f"LR: {scheduler.get_lr()[0]:.2e} | HW: {hw:.3f}")

        torch.save(model.state_dict(), OUTPUT_LAST)
        if vda > best_dice and vdc > 0:
            best_dice = vda
            torch.save(model.state_dict(), OUTPUT_WEIGHTS)
            print(f"🏆 新紀錄！Val Dice: {best_dice:.4f}"); patience_counter = 0
        else:
            patience_counter += 1
            print(f"  未破紀錄 (最佳: {best_dice:.4f}) | Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print(f"⛔ Early stopping at epoch {epoch}"); break
        if epoch % 10 == 0: gc.collect(); torch.cuda.empty_cache()

    print(f"\n✅ v9 完成！最佳 Val Dice: {best_dice:.4f}")
    print(f"   權重: {OUTPUT_WEIGHTS}")
    print(f"\n📋 Eval:")
    print(f"   python eval_public.py --weights {OUTPUT_WEIGHTS}")
    print(f"   python eval_hospital.py --weights {OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    main()
