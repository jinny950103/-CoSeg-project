"""
train_stage2_hospital.py — Stage 2: 只用醫院資料精修，衝醫院 Dice 0.85
========================================================================
Stage 1 (train_v8_1.py): 公開+醫院混合 → 學通用分割能力
Stage 2 (這支): 只用醫院資料 fine-tune → 讓預測匹配球形 GT

差異 vs Stage 1:
  - 資料：只用醫院 31 人（不含公開）
  - Val：用醫院 val（球形 GT，Val Dice 直接反映醫院表現）
  - Warm start：從 Stage 1 best
  - LR：更小（避免破壞 Stage 1 學到的位置知識）
  - Epochs：80（微調不用太多）
  - Aug：更弱（不要太擾動）
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

# === 只用醫院資料 ===
HOSPITAL_IMG_DIR  = "data/hospital_data/train/image_1024"
HOSPITAL_MASK_DIR = "data/hospital_data/train/mask_sem_1024"
JSON_PATH         = "data_split_v10_mixed.json"

OUTPUT_DIR        = "outputs"
OUTPUT_WEIGHTS    = os.path.join(OUTPUT_DIR, "coseg_hospital_best.pth")
OUTPUT_LAST       = os.path.join(OUTPUT_DIR, "coseg_hospital_last.pth")
SAM2_CHECKPOINT   = "checkpoints/sam2_hiera_large.pt"
WARMSTART_WEIGHTS = os.path.join(OUTPUT_DIR, "coseg_v8_best.pth")  # 從 Stage 1 開始

# === 超參數（比 Stage 1 保守）===
BATCH_SIZE       = 16
NUM_EPOCHS       = 80
BASE_LR          = 5e-5       # Stage 1 是 1e-4，這裡減半
WARMUP_EPOCHS    = 3
NUM_WORKERS      = 8
POSITIVE_WEIGHT  = 4.0
CONTEXT_SLICES   = 1
AUG_PROB         = 0.5        # 更弱，不要太擾動
DS_WEIGHTS       = [0.3, 0.2, 0.1]
CLDICE_WEIGHT    = 0.3        # 降低 clDice 權重（球形 GT 的骨架不重要）
PATIENCE         = 20


# ==========================================
# 🔧 Soft-clDice Loss
# ==========================================
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


# ==========================================
# 🔧 Augmentation（弱版）
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


def random_affine(image, mask):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), random.uniform(-20, 20),
                                 random.uniform(0.85, 1.15))
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    return image, mask


def augment_mild(image, mask):
    """弱 augmentation，不要太擾動精修過程"""
    if random.random() < AUG_PROB:
        image = np.flip(image, axis=1).copy(); mask = np.flip(mask, axis=1).copy()
    if random.random() < AUG_PROB * 0.5:
        image = np.flip(image, axis=0).copy(); mask = np.flip(mask, axis=0).copy()
    if random.random() < AUG_PROB:
        image, mask = random_affine(image, mask)
    if random.random() < 0.1:
        image, mask = elastic_deform(image, mask)
    if random.random() < AUG_PROB:
        image = np.clip(random.uniform(0.8, 1.2) * image + random.uniform(-15, 15), 0, 255)
    if random.random() < AUG_PROB * 0.3:
        image = np.clip(image + np.random.normal(0, random.uniform(3, 10), image.shape), 0, 255)
    return image.astype(np.float32), (mask > 0.5).astype(np.float32)


# ==========================================
# 📦 Hospital-Only Dataset
# ==========================================
class HospitalCanalDataset(Dataset):
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
        print(f"📊 掃描 ({'train' if is_train else 'val'})...")
        for patient, sn, fname in tqdm(self.entries, desc="Scanning"):
            mp = os.path.join(HOSPITAL_MASK_DIR, fname + ".npy")
            if os.path.exists(mp):
                m = np.load(mp)
                self.has_canal.append(m.sum() > 0)
            else:
                self.has_canal.append(False)
        print(f"   ✅ 正: {sum(self.has_canal)} | ❌ 負: {len(self.has_canal)-sum(self.has_canal)}")

    def get_sample_weights(self):
        return [POSITIVE_WEIGHT if h else 1.0 for h in self.has_canal]

    def _load_gray(self, patient, sn):
        if patient in self.patient_slice_map and sn in self.patient_slice_map[patient]:
            fname = self.patient_slice_map[patient][sn]
        else:
            fname = f"{patient}_slice_{sn}"
        p = os.path.join(HOSPITAL_IMG_DIR, fname + ".png")
        return cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) if os.path.exists(p) else None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        patient, sn, fname = self.entries[idx]
        c = self._load_gray(patient, sn)
        p = self._load_gray(patient, sn - CONTEXT_SLICES)
        n = self._load_gray(patient, sn + CONTEXT_SLICES)
        img = np.stack([p if p is not None else c, c, n if n is not None else c], axis=-1)
        gt = (np.load(os.path.join(HOSPITAL_MASK_DIR, fname + ".npy")) > 0).astype(np.float32)
        if self.is_train:
            img, gt = augment_mild(img, gt)
        img = (img - self.pixel_mean) / (self.pixel_std + 1e-8)
        return torch.tensor(img).permute(2, 0, 1).float(), torch.tensor(gt).unsqueeze(0).float()


# ==========================================
# 🎯 Loss
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

    hospital_train = splits["hospital_train"]
    hospital_val = splits["hospital_val"]

    print("="*60)
    print("🏥 Stage 2: 醫院資料專用精修")
    print(f"  Train: {len(hospital_train)} slices ({len(splits.get('hospital_train_patients', []))} 人)")
    print(f"  Val:   {len(hospital_val)} slices")
    print(f"  Loss:  Focal+Dice+Boundary + clDice(w={CLDICE_WEIGHT}) + DS")
    print(f"  LR:    {BASE_LR}（Stage 1 的一半）")
    print(f"  Warm start: {WARMSTART_WEIGHTS}")
    print(f"  目標: 醫院 Dice 0.85+")
    print("="*60)

    train_ds = HospitalCanalDataset(hospital_train, is_train=True)
    val_ds = HospitalCanalDataset(hospital_val, is_train=False)

    sampler = WeightedRandomSampler(train_ds.get_sample_weights(), 8000, True)
    train_dl = DataLoader(train_ds, BATCH_SIZE, sampler=sampler, drop_last=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
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
    else:
        print(f"⚠️ 找不到 {WARMSTART_WEIGHTS}")

    model = model.to(device)

    # 三組 LR（比 Stage 1 更小）
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
        {"params": enc_params, "lr": BASE_LR * 0.1},   # 5e-6
        {"params": new_params, "lr": BASE_LR * 0.5},   # 2.5e-5
        {"params": head_params, "lr": BASE_LR},         # 5e-5
    ], weight_decay=1e-4)
    scheduler = CosineWarmupScheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)
    criterion = FocalDiceBoundaryLoss()
    cldice_loss = SoftCLDiceLoss(iterations=10)

    best_dice, patience_counter = 0.0, 0
    start_epoch = 0
    if os.path.exists(OUTPUT_LAST):
        ckpt = torch.load(OUTPUT_LAST, map_location=device)
        if isinstance(ckpt, dict) and "epoch" in ckpt:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
            best_dice = ckpt["best_dice"]
            patience_counter = ckpt["patience"]
            for e_ in range(start_epoch): scheduler.step(e_)
            print(f"📂 從 epoch {start_epoch} 續跑（best: {best_dice:.4f}, patience: {patience_counter}）")

    for epoch in range(start_epoch, NUM_EPOCHS):
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

                # Soft-clDice（256×256 避免 OOM）
                pred_small = F.interpolate(pred_t, (256, 256), mode="bilinear", align_corners=False)
                mask_small = F.interpolate(masks, (256, 256), mode="nearest")
                pred_prob = torch.sigmoid(pred_small.float()).clamp(1e-6, 1-1e-6)
                cl_loss = cldice_loss(pred_prob, mask_small)

                # DS
                ds_loss = torch.tensor(0.0, device=device)
                for dp, dw in zip(ds_outs, DS_WEIGHTS):
                    dg = F.interpolate(masks, dp.shape[2:], mode='nearest')
                    ds_loss = ds_loss + dw * criterion(dp.float().clamp(-20, 20), dg)

                loss = main_loss + CLDICE_WEIGHT * cl_loss + ds_loss

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(); continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step(); optimizer.zero_grad()

            probs = torch.sigmoid(pred_t.float().detach())
            bd = compute_dice_correct(probs, masks)
            rl += loss.item() * imgs.size(0); ns += imgs.size(0)
            if bd is not None: rd += bd; dc += 1

        tl, td = rl/max(ns,1), rd/max(dc,1)

        if epoch % 1 != 0 and epoch != NUM_EPOCHS-1:
            scheduler.step(epoch)
            print(f"\nEpoch {epoch} | Train Loss: {tl:.4f} Dice: {td:.4f} | Val: skipped | LR: {scheduler.get_lr()[0]:.2e}")
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_dice": best_dice, "patience": patience_counter}, OUTPUT_LAST); continue

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

        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_dice": best_dice, "patience": patience_counter}, OUTPUT_LAST)
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

    print(f"\n✅ 完成！最佳 Val Dice: {best_dice:.4f}")
    print(f"   權重: {OUTPUT_WEIGHTS}")
    print(f"\n📋 下一步:")
    print(f"   python eval_hospital.py --weights {OUTPUT_WEIGHTS}")

if __name__ == "__main__":
    main()
