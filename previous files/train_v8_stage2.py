"""
train_v8_stage2.py — Stage 2: 只用公開資料精修邊界
====================================================
Stage 1 (v8): 公開+醫院混合訓練 → 學 domain diversity，但邊界被 sphere label 拉粗
Stage 2 (這支): 只用公開資料 finetune → 精修邊界，保留 domain 知識

差異 vs v8:
  - 資料：只用公開 8 人 train + 2 人 val（不含醫院）
  - Warm start：從 v8_best（不是 v7_best）
  - LR：更小（encoder 1e-6, AG/DS 5e-6, head 5e-5）
  - Epochs：80（不用太多，只是微調）
  - Val：只用公開資料（Val Dice 可以直接跟 v7 比）
"""
import os, json, cv2, torch, random, math, gc
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import torch.nn.functional as F
import hydra
from scipy.ndimage import gaussian_filter, map_coordinates

from model_v6 import CoSegV6
from sam2.build_sam import build_sam2

PROJECT_ROOT = "/home/u9444861/-CoSeg-project"

# === 只用公開資料 ===
PUBLIC_IMG_DIR   = "data/public_data/train/image_1024"
PUBLIC_MASK_DIR  = "data/public_data/train/mask_sem_1024"
JSON_PATH        = "data_split_v8_mixed_v2.json"  # 只讀 train / val 欄位

OUTPUT_DIR       = "outputs"
OUTPUT_WEIGHTS   = os.path.join(OUTPUT_DIR, "coseg_v8s2_best.pth")  # s2 = stage2
OUTPUT_LAST      = os.path.join(OUTPUT_DIR, "coseg_v8s2_last.pth")
SAM2_CHECKPOINT  = "checkpoints/sam2_hiera_large.pt"
WARMSTART_WEIGHTS= os.path.join(OUTPUT_DIR, "coseg_v8_best.pth")    # 從 v8 開始

# === 超參數（比 v8 保守） ===
BATCH_SIZE       = 12
NUM_EPOCHS       = 80
BASE_LR          = 5e-5      # v8 是 1e-4，stage2 減半
WARMUP_EPOCHS    = 3         # v8 是 5，stage2 更短
NUM_WORKERS      = 8
POSITIVE_WEIGHT  = 4.0
CONTEXT_SLICES   = 1
AUG_PROB         = 0.5       # v8 是 0.7，stage2 減弱 augmentation（不要太擾動精修）
DS_WEIGHTS       = [0.3, 0.2, 0.1]
CLDICE_WEIGHT    = 0.5
PATIENCE         = 25        # v8 是 35，stage2 更快停


# ==========================================
# 🔧 Soft-clDice Loss（跟 v8 一樣）
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
        img1 = self.soft_erode(img)
        return self.soft_dilate(img1)

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
# 🔧 Augmentation（跟 v8 一樣，但用 AUG_PROB=0.5）
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
    if random.random() < 0.15:  # 比 v8 少
        image, mask = elastic_deform(image, mask)
    if crops and mask.sum() > 0:
        image, mask = copy_paste_canal(image, mask, crops)
    if random.random() < AUG_PROB:
        image = np.clip(random.uniform(0.7, 1.3) * image + random.uniform(-20, 20), 0, 255)
    if random.random() < AUG_PROB * 0.4:
        image = np.clip(np.power(image / 255.0, random.uniform(0.6, 1.4)) * 255.0, 0, 255)
    if random.random() < AUG_PROB * 0.4:
        image = np.clip(image + np.random.normal(0, random.uniform(5, 15), image.shape), 0, 255)
    if random.random() < AUG_PROB * 0.3:
        image = cv2.GaussianBlur(image, (random.choice([3, 5]),) * 2, 0)
    if random.random() < 0.2:
        h, w = image.shape[:2]
        ch, cw = random.randint(60, 160), random.randint(60, 160)
        cy, cx = random.randint(0, h-ch), random.randint(0, w-cw)
        image[cy:cy+ch, cx:cx+cw] = 0
    return image.astype(np.float32), (mask > 0.5).astype(np.float32)


# ==========================================
# 📦 Dataset（公開資料 only，簡化版）
# ==========================================
class PublicCanalDataset(Dataset):
    def __init__(self, file_list, is_train=True):
        self.is_train = is_train
        self.pixel_mean = np.array([123.675, 116.280, 103.530], dtype=np.float32)
        self.pixel_std  = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        self.patient_slice_map = {}
        self.entries = []

        for fname in file_list:
            parts = fname.rsplit('_slice_', 1)
            patient, sn = parts[0], int(parts[1])
            self.patient_slice_map.setdefault(patient, {})[sn] = fname
            self.entries.append((patient, sn, fname))

        self.has_canal = []
        self.positive_crops = []

        print(f"📊 掃描 ({'train' if is_train else 'val'})...")
        for patient, sn, fname in tqdm(self.entries, desc="Scanning"):
            mp = os.path.join(PUBLIC_MASK_DIR, fname + ".npy")
            if os.path.exists(mp):
                m = np.load(mp)
                is_pos = m.sum() > 0
                self.has_canal.append(is_pos)
                if is_train and is_pos and len(self.positive_crops) < 200:
                    ip = os.path.join(PUBLIC_IMG_DIR, fname + ".png")
                    if os.path.exists(ip):
                        self.positive_crops.append(
                            (cv2.imread(ip, cv2.IMREAD_GRAYSCALE).astype(np.float32),
                             m.astype(np.float32)))
            else:
                self.has_canal.append(False)

        n_pos = sum(self.has_canal)
        print(f"   ✅ 公開: {len(self.entries)} | 正: {n_pos} | 負: {len(self.has_canal)-n_pos}")

    def get_sample_weights(self):
        return [POSITIVE_WEIGHT if hc else 1.0 for hc in self.has_canal]

    def _load_gray(self, patient, sn):
        if patient in self.patient_slice_map and sn in self.patient_slice_map[patient]:
            fname = self.patient_slice_map[patient][sn]
        else:
            return None
        p = os.path.join(PUBLIC_IMG_DIR, fname + ".png")
        return cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) if os.path.exists(p) else None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        patient, sn, fname = self.entries[idx]
        c = self._load_gray(patient, sn)
        p = self._load_gray(patient, sn - CONTEXT_SLICES)
        n = self._load_gray(patient, sn + CONTEXT_SLICES)
        img = np.stack([p if p is not None else c, c, n if n is not None else c], axis=-1)
        gt = (np.load(os.path.join(PUBLIC_MASK_DIR, fname + ".npy")) > 0).astype(np.float32)
        if self.is_train:
            img, gt = augment_strong(img, gt, self.positive_crops)
        img = (img - self.pixel_mean) / (self.pixel_std + 1e-8)
        return torch.tensor(img).permute(2, 0, 1).float(), torch.tensor(gt).unsqueeze(0).float()


# ==========================================
# 🎯 Loss（跟 v8 一樣）
# ==========================================
class FocalDiceBoundaryLoss(nn.Module):
    def __init__(self, fa=0.75, fg=2.0, dw=1.0, fw=1.0, bw=0.5):
        super().__init__()
        self.fa, self.fg, self.dw, self.fw, self.bw = fa, fg, dw, fw, bw

    def forward(self, logits, target):
        pred = torch.sigmoid(logits)
        p = pred.clamp(1e-6, 1-1e-6)
        at = self.fa*target + (1-self.fa)*(1-target)
        pt = p*target + (1-p)*(1-target)
        focal = -(at * (1-pt)**self.fg * torch.log(pt)).mean()
        inter = (pred*target).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
        dice = 1.0 - ((2*inter+1e-5)/(union+1e-5)).mean()
        td = F.max_pool2d(target, 3, 1, 1)
        te = -F.max_pool2d(-target, 3, 1, 1)
        b = td - te
        if b.sum() == 0:
            bnd = torch.tensor(0.0, device=logits.device)
        else:
            with torch.amp.autocast('cuda', enabled=False):
                bnd = F.binary_cross_entropy(
                    (pred*b).float().clamp(1e-6, 1-1e-6), (target*b).float(),
                    reduction='sum') / (b.sum().float() + 1e-5)
        return self.fw*focal + self.dw*dice + self.bw*bnd


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
    def step(self, epoch):
        s = (epoch+1)/self.wu if epoch < self.wu else 0.5*(1+math.cos(math.pi*(epoch-self.wu)/(self.tot-self.wu)))
        for pg, blr in zip(self.opt.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, blr * s)
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

    # 只用公開資料
    train_list = splits['train']
    val_list = splits['val']

    print("="*60)
    print("🧠 Stage 2: 只用公開資料精修邊界")
    print(f"  Train: {len(train_list)} slices (公開 only)")
    print(f"  Val:   {len(val_list)} slices (公開 only)")
    print(f"  Warm start: {WARMSTART_WEIGHTS}")
    print(f"  LR: encoder={BASE_LR*0.02:.1e}, AG/DS={BASE_LR*0.1:.1e}, head={BASE_LR:.1e}")
    print(f"  Epochs: {NUM_EPOCHS}, Patience: {PATIENCE}")
    print(f"  AUG_PROB: {AUG_PROB} (比 v8 的 0.7 低)")
    print(f"  Output: {OUTPUT_WEIGHTS}")
    print("="*60)

    train_ds = PublicCanalDataset(train_list, is_train=True)
    val_ds   = PublicCanalDataset(val_list, is_train=False)

    sampler = WeightedRandomSampler(train_ds.get_sample_weights(), len(train_ds), True)
    train_dl = DataLoader(train_ds, BATCH_SIZE, sampler=sampler, drop_last=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds, 4, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')
    print(f"📦 SAM2: {SAM2_CHECKPOINT}")
    model = CoSegV6(build_sam2("sam2_hiera_l.yaml", SAM2_CHECKPOINT, mode="train"))

    # Warm start from v8_best
    if os.path.exists(WARMSTART_WEIGHTS):
        print(f"📦 Warm start: {WARMSTART_WEIGHTS}")
        sd = torch.load(WARMSTART_WEIGHTS, map_location="cpu")
        clean = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        miss, unexp = model.load_state_dict(clean, strict=False)
        print(f"   Missing: {len(miss)}, Unexpected: {len(unexp)}")
        del sd, clean
    else:
        print(f"⚠️ 找不到 {WARMSTART_WEIGHTS}，從 SAM2 scratch 開始！")
    model = model.to(device)

    # 三組 LR（比 v8 更小）
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
        {"params": enc_params,  "lr": BASE_LR * 0.02},   # 1e-6（v8 是 1e-5）
        {"params": new_params,  "lr": BASE_LR * 0.1},    # 5e-6（v8 是 5e-5）
        {"params": head_params, "lr": BASE_LR},           # 5e-5（v8 是 1e-4）
    ], weight_decay=1e-4)
    scheduler = CosineWarmupScheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)
    criterion = FocalDiceBoundaryLoss()
    cldice_loss = SoftCLDiceLoss(iterations=10)

    best_dice, patience_counter = 0.0, 0

    print(f"\n🚀 開始 Stage 2 精修...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        rl, rd, dc, ns = 0.0, 0.0, 0, 0
        optimizer.zero_grad()

        for step, (imgs, masks) in enumerate(tqdm(train_dl, desc=f"Epoch {epoch}/{NUM_EPOCHS-1} [Train]")):
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(x=imgs)
                pred_sem = outputs[1]
                ds_outs = outputs[-1]
                pred_sem = F.interpolate(pred_sem, (1024, 1024), mode='bilinear', align_corners=False)
                pred_t = pred_sem[:, 0:1, :, :]

                main_loss = criterion(pred_t, masks)

                # Soft-clDice（在 256x256 計算避免 OOM）
                pred_256 = F.interpolate(pred_t, (256, 256), mode='bilinear', align_corners=False)
                mask_256 = F.interpolate(masks, (256, 256), mode='nearest')
                pred_prob = torch.sigmoid(pred_256.float()).clamp(1e-6, 1-1e-6)
                cl_loss = cldice_loss(pred_prob, mask_256)

                # DS
                ds_loss = torch.tensor(0.0, device=device)
                for dp, dw in zip(ds_outs, DS_WEIGHTS):
                    dg = F.interpolate(masks, dp.shape[2:], mode='nearest')
                    ds_loss = ds_loss + dw * criterion(dp.float().clamp(-20, 20), dg)

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
            scheduler.step(epoch)
            print(f"\nEpoch {epoch} | Train Loss: {tl:.4f} Dice: {td:.4f} | Val: skipped | LR: {scheduler.get_lr()[0]:.2e}")
            torch.save(model.state_dict(), OUTPUT_LAST); continue

        model.eval()
        vl, vd, vdc, vn = 0.0, 0.0, 0, 0
        with torch.no_grad():
            for imgs, masks in tqdm(val_dl, desc=f"Epoch {epoch}/{NUM_EPOCHS-1} [Val]"):
                imgs, masks = imgs.to(device), masks.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(x=imgs)
                    ps = F.interpolate(outputs[1], (1024,1024), mode='bilinear', align_corners=False)
                    pt = ps[:, 0:1, :, :]
                    lo = criterion(pt, masks)
                bd = compute_dice_correct(torch.sigmoid(pt.float()), masks)
                vl += lo.item()*imgs.size(0); vn += imgs.size(0)
                if bd is not None: vd += bd; vdc += 1

        vda = vd/max(vdc,1); vl /= max(vn,1)
        scheduler.step(epoch)
        print(f"\nEpoch {epoch} | Train: {tl:.4f}/{td:.4f} | Val: {vl:.4f}/{vda:.4f} | LR: {scheduler.get_lr()[0]:.2e}")

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

    print(f"\n✅ Stage 2 完成！最佳 Val Dice: {best_dice:.4f}")
    print(f"   權重: {OUTPUT_WEIGHTS}")
    print(f"\n📋 下一步:")
    print(f"   python eval_public.py --weights {OUTPUT_WEIGHTS}")
    print(f"   python eval_hospital.py --weights {OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    main()
