#!/usr/bin/env python3
"""
V11 前處理 Pipeline — 只產生 metadata（ROI bbox），不複製圖片
=============================================================

設計理念：
  原始 PNG 已經是 1024×1024，不需要再 resize 或轉 .npy。
  前處理只做兩件事：
    1. 掃描檔案，按病人分組
    2. 讀 GT mask 計算 ROI bounding box
  結果存成每個病人一個 metadata.json（幾 KB），不佔磁碟空間。
  CLAHE 在訓練時 on-the-fly 做。

適配的資料結構：
  A) Flat (公開 & 醫院 train):
     data_dir/image_1024/VOL_5_slice_0.png
     data_dir/mask_sem_1024/VOL_5_slice_0.npy

  B) Per-patient (醫院 eval):
     data_dir/HOSP_xxx/image_1024/slice_0.png
     data_dir/HOSP_xxx/mask_sem_1024/slice_0.npy

  C) 同一資料夾含 train+val（靠 split JSON 區分）

用法：
  # 掃描公開 train+val（同一資料夾，靠 split JSON 分）
  python preprocess_v11_spacing.py \
      --data-dir /home/u9444861/-CoSeg-project/data/public_data/train \
      --output-dir /home/u9444861/-CoSeg-project/data/v11_meta/public \
      --domain public

  # 掃描醫院 train
  python preprocess_v11_spacing.py \
      --data-dir /home/u9444861/-CoSeg-project/data/hospital_data/train \
      --output-dir /home/u9444861/-CoSeg-project/data/v11_meta/hospital_train \
      --domain hospital

  # 掃描醫院 eval
  python preprocess_v11_spacing.py \
      --data-dir /home/u9444861/-CoSeg-project/data/hospital_data/eval \
      --output-dir /home/u9444861/-CoSeg-project/data/v11_meta/hospital_eval \
      --domain hospital

  # 最後合併產生 data_split_v11.json
  python preprocess_v11_spacing.py --build-split \
      --v10-split /home/u9444861/-CoSeg-project/data_split_v10_mixed.json \
      --meta-dirs \
          /home/u9444861/-CoSeg-project/data/v11_meta/public \
          /home/u9444861/-CoSeg-project/data/v11_meta/hospital_train \
          /home/u9444861/-CoSeg-project/data/v11_meta/hospital_eval \
      --output-dir /home/u9444861/-CoSeg-project/data/v11_meta

輸出：
  output_dir/
    {patient_id}/
      metadata.json    # 幾 KB，含 ROI bbox、原始檔案路徑、前景統計
  (不複製任何圖片！)
"""

import os
import sys
import re
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    import cv2
except ImportError:
    print("需要安裝 opencv: pip install opencv-python-headless")
    sys.exit(1)


# ─────────────────────────────────────────────
# 0. CLAHE 工具函數（供外部 import）
# ─────────────────────────────────────────────

def apply_clahe(image, clip_limit=2.0, grid_size=8):
    """
    對 float32 [0,1] 灰度影像做 CLAHE。回傳 float32 [0,1]。
    可被 train_v11_roi.py import 使用。
    """
    img_u8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    result = clahe.apply(img_u8)
    return result.astype(np.float32) / 255.0


# ─────────────────────────────────────────────
# 1. 檔名解析
# ─────────────────────────────────────────────

def parse_filename(fname):
    """
    從檔名解析 patient_id 和 slice_idx。

    支援：
      VOL_5_slice_0.png      → ('VOL_5', 0)
      Patient_14_slice_123   → ('Patient_14', 123)
      HOSP_0018108102_slice_0 → ('HOSP_0018108102', 0)
      slice_0.png            → (None, 0)
    """
    stem = Path(fname).stem

    # Pattern: {ID}_slice_{N}
    m = re.match(r'^(.+)_slice_(\d+)$', stem)
    if m:
        return m.group(1), int(m.group(2))

    # Pattern: slice_{N} (per-patient dir)
    m = re.match(r'^slice_(\d+)$', stem)
    if m:
        return None, int(m.group(1))

    # Pattern: pure number
    m = re.match(r'^(\d+)$', stem)
    if m:
        return None, int(m.group(1))

    return None, None


# ─────────────────────────────────────────────
# 2. 資料結構偵測 & 掃描
# ─────────────────────────────────────────────

def detect_structure(data_dir):
    """偵測 'flat' / 'per_patient' / 'unknown'"""
    data_path = Path(data_dir)
    img_dir = data_path / 'image_1024'

    if img_dir.is_dir():
        sample = next(img_dir.glob('*.png'), None) or next(img_dir.glob('*.npy'), None)
        if sample:
            pid, _ = parse_filename(sample.name)
            return 'flat' if pid is not None else 'single_patient'

    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    for sd in subdirs[:5]:
        if (sd / 'image_1024').is_dir():
            return 'per_patient'

    return 'unknown'


def scan_flat(data_dir):
    """掃描 flat 結構 → {patient_id: [(slice_idx, img_path, mask_path), ...]}"""
    img_dir = Path(data_dir) / 'image_1024'
    mask_dir = Path(data_dir) / 'mask_sem_1024'
    patients = defaultdict(list)

    for ext in ['*.png', '*.npy']:
        for img_path in sorted(img_dir.glob(ext)):
            pid, sidx = parse_filename(img_path.name)
            if pid is None:
                continue
            mask_path = mask_dir / f'{img_path.stem}.npy'
            if not mask_path.exists():
                mask_path = mask_dir / f'{img_path.stem}.png'
            if not mask_path.exists():
                continue
            patients[pid].append((sidx, str(img_path), str(mask_path)))
        if patients:
            break

    for pid in patients:
        patients[pid].sort(key=lambda x: x[0])
    return dict(patients)


def scan_per_patient(data_dir):
    """掃描 per-patient 結構"""
    patients = {}
    for pdir in sorted(Path(data_dir).iterdir()):
        if not pdir.is_dir():
            continue
        img_dir = pdir / 'image_1024'
        mask_dir = pdir / 'mask_sem_1024'
        if not img_dir.is_dir() or not mask_dir.is_dir():
            continue

        pid = pdir.name
        slices = []
        for ext in ['*.png', '*.npy']:
            for img_path in sorted(img_dir.glob(ext)):
                _, sidx = parse_filename(img_path.name)
                if sidx is None:
                    sidx = len(slices)
                mask_path = mask_dir / f'{img_path.stem}.npy'
                if not mask_path.exists():
                    mask_path = mask_dir / f'{img_path.stem}.png'
                if mask_path.exists():
                    slices.append((sidx, str(img_path), str(mask_path)))
            if slices:
                break

        slices.sort(key=lambda x: x[0])
        if slices:
            patients[pid] = slices
    return patients


# ─────────────────────────────────────────────
# 3. ROI 偵測
# ─────────────────────────────────────────────

def compute_roi_from_mask(mask_2d, margin=32, min_size=96):
    """從 GT mask 算 ROI bbox → (y_min, y_max, x_min, x_max) or None"""
    H, W = mask_2d.shape
    ys, xs = np.where(mask_2d > 0)
    if len(ys) == 0:
        return None

    y_min = max(0, int(ys.min()) - margin)
    y_max = min(H, int(ys.max()) + margin)
    x_min = max(0, int(xs.min()) - margin)
    x_max = min(W, int(xs.max()) + margin)

    # 確保最小尺寸
    cy, cx = (y_min + y_max) // 2, (x_min + x_max) // 2
    half = max(min_size // 2, max(y_max - y_min, x_max - x_min) // 2)
    y_min = max(0, cy - half)
    y_max = min(H, cy + half)
    x_min = max(0, cx - half)
    x_max = min(W, cx + half)
    return (int(y_min), int(y_max), int(x_min), int(x_max))


def compute_roi_from_bone(image_2d, margin=32, min_size=96):
    """無 GT 時，從骨骼偵測下顎 ROI"""
    H, W = image_2d.shape
    thresh = 0.45 if image_2d.max() <= 1.5 else 300
    bone = (image_2d > thresh).astype(np.uint8)
    lower = np.zeros_like(bone)
    lower[H // 2:, :] = bone[H // 2:, :]

    ys, xs = np.where(lower > 0)
    if len(ys) == 0:
        return (H // 2, H, 0, W)

    y_min = max(0, int(ys.min()) - margin)
    y_max = min(H, int(ys.max()) + margin)
    x_min = max(0, int(xs.min()) - margin)
    x_max = min(W, int(xs.max()) + margin)

    cy, cx = (y_min + y_max) // 2, (x_min + x_max) // 2
    half = max(min_size // 2, max(y_max - y_min, x_max - x_min) // 2)
    y_min = max(0, cy - half)
    y_max = min(H, cy + half)
    x_min = max(0, cx - half)
    x_max = min(W, cx + half)
    return (int(y_min), int(y_max), int(x_min), int(x_max))


def interpolate_rois(roi_list, num_slices):
    """用最近有效 ROI 填補空缺，並做滑動窗口平滑"""
    filled = list(roi_list)
    valid = [i for i, r in enumerate(filled) if r is not None]
    if not valid:
        return [(512, 1024, 0, 1024)] * num_slices

    for i in range(num_slices):
        if filled[i] is not None:
            continue
        dists = [abs(i - vi) for vi in valid]
        nearest = valid[np.argmin(dists)]
        filled[i] = filled[nearest]

    smoothed = []
    w = 5
    for i in range(num_slices):
        s = max(0, i - w // 2)
        e = min(num_slices, i + w // 2 + 1)
        nb = [filled[j] for j in range(s, e) if filled[j] is not None]
        if nb:
            smoothed.append((
                int(np.mean([r[0] for r in nb])),
                int(np.mean([r[1] for r in nb])),
                int(np.mean([r[2] for r in nb])),
                int(np.mean([r[3] for r in nb])),
            ))
        else:
            smoothed.append(filled[i])
    return smoothed


# ─────────────────────────────────────────────
# 4. 載入輔助
# ─────────────────────────────────────────────

def load_mask(path):
    p = str(path)
    if p.endswith('.npy'):
        mask = np.load(p)
    else:
        mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
    return (mask > (127 if mask.max() > 1 else 0.5)).astype(np.uint8)


def load_image_gray(path):
    p = str(path)
    if p.endswith('.npy'):
        img = np.load(p).astype(np.float32)
    else:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = img.astype(np.float32) / 255.0
    if img.max() > 1.5:
        mn, mx = img.min(), img.max()
        img = (img - mn) / (mx - mn + 1e-7)
    return img


# ─────────────────────────────────────────────
# 5. 處理單個病人 → 只產生 metadata.json
# ─────────────────────────────────────────────

def process_patient(patient_id, slices_info, output_dir, domain,
                    roi_margin=32, min_roi_size=96):
    """
    讀每張 mask，計算 ROI，寫 metadata.json。
    不複製任何圖片。

    metadata.json 內容：
      - patient_id, domain
      - num_slices
      - slices: [{idx, img_path, mask_path}, ...]  ← 原始絕對路徑
      - per_slice_roi: [[y_min, y_max, x_min, x_max], ...]
      - global_roi
      - foreground_slices: [local indices with fg]
    """
    out_dir = Path(output_dir) / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)

    num_slices = len(slices_info)
    print(f"  {patient_id}: {num_slices} slices ... ", end='', flush=True)

    roi_list = []
    fg_slices = []
    H, W = None, None

    for local_idx, (slice_idx, img_path, mask_path) in enumerate(slices_info):
        mask = load_mask(mask_path)
        if mask is None:
            roi_list.append(None)
            continue

        if H is None:
            H, W = mask.shape[:2]

        # ROI from GT
        roi = compute_roi_from_mask(mask, margin=roi_margin, min_size=min_roi_size)

        if roi is None:
            # 嘗試從影像骨骼偵測
            img = load_image_gray(img_path)
            if img is not None:
                roi = compute_roi_from_bone(img, margin=roi_margin, min_size=min_roi_size)

        roi_list.append(roi)

        if mask.sum() > 0:
            fg_slices.append(local_idx)

    if H is None:
        print("SKIP (no valid slices)")
        return None

    # 填補 + 平滑
    roi_list = interpolate_rois(roi_list, num_slices)

    # Global ROI
    valid_rois = [r for r in roi_list if r is not None]
    if valid_rois:
        global_roi = [
            min(r[0] for r in valid_rois),
            max(r[1] for r in valid_rois),
            min(r[2] for r in valid_rois),
            max(r[3] for r in valid_rois),
        ]
    else:
        global_roi = [0, H, 0, W]

    # Metadata — 存原始檔案的絕對路徑
    slices_entries = []
    for local_idx, (slice_idx, img_path, mask_path) in enumerate(slices_info):
        slices_entries.append({
            'idx': slice_idx,
            'img': os.path.abspath(img_path),
            'mask': os.path.abspath(mask_path),
        })

    metadata = {
        'patient_id': patient_id,
        'domain': domain,
        'num_slices': num_slices,
        'height': H,
        'width': W,
        'slices': slices_entries,
        'per_slice_roi': [list(r) if r else None for r in roi_list],
        'global_roi': global_roi,
        'foreground_slices': fg_slices,
        'num_foreground_slices': len(fg_slices),
        'foreground_ratio': len(fg_slices) / num_slices if num_slices > 0 else 0,
    }

    meta_path = out_dir / 'metadata.json'
    with open(str(meta_path), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"{len(fg_slices)} fg, ROI: {global_roi}")
    return metadata


# ─────────────────────────────────────────────
# 6. Build data_split_v11.json
# ─────────────────────────────────────────────

def build_split(args):
    """
    結合 V10 的 split JSON + 前處理產生的 metadata，
    產生 data_split_v11.json。

    V10 split JSON 格式：
      {"train": ["VOL_5_slice_0.png", ...],
       "val": [...],
       "hospital_train": [...],
       "hospital_val": [...]}
    """
    # 讀 V10 split
    v10_split = {}
    if args.v10_split and os.path.exists(args.v10_split):
        with open(args.v10_split) as f:
            v10_split = json.load(f)
        print(f"讀取 V10 split: {args.v10_split}")
        for k, v in v10_split.items():
            print(f"  {k}: {len(v)} entries")

    # 從 V10 split 提取 patient IDs
    def extract_pids_from_slices(entries):
        """從 slice filename list 提取 unique patient IDs"""
        pids = set()
        for e in entries:
            if isinstance(e, str):
                pid, _ = parse_filename(e)
                if pid:
                    pids.add(pid)
                else:
                    pids.add(e)
            elif isinstance(e, dict):
                pids.add(e.get('patient_id', ''))
        return pids

    def extract_pids_from_ids(entries, prefix='VOL_'):
        """從 patient ID list (如 [5, 6, 7]) 提取，加上 prefix"""
        pids = set()
        for e in entries:
            if isinstance(e, int):
                pids.add(f'{prefix}{e}')
            elif isinstance(e, str):
                # hospital: '0018108102' → 'HOSP_0018108102'
                if prefix == 'HOSP_':
                    pids.add(f'{prefix}{e}')
                else:
                    pids.add(e)
        return pids

    # 優先用 *_patients 欄位（更準確），fallback 到 slice list
    if 'train_patients' in v10_split:
        train_pids = extract_pids_from_ids(v10_split['train_patients'], 'VOL_')
    else:
        train_pids = extract_pids_from_slices(v10_split.get('train', []))

    if 'val_patients' in v10_split:
        val_pids = extract_pids_from_ids(v10_split['val_patients'], 'VOL_')
    else:
        val_pids = extract_pids_from_slices(v10_split.get('val', []))

    if 'test_patients' in v10_split:
        test_pids = extract_pids_from_ids(v10_split['test_patients'], 'VOL_')
    else:
        test_pids = set()

    if 'hospital_train_patients' in v10_split:
        hosp_train_pids = extract_pids_from_ids(v10_split['hospital_train_patients'], 'HOSP_')
    else:
        hosp_train_pids = extract_pids_from_slices(v10_split.get('hospital_train', []))

    if 'hospital_val_patients' in v10_split:
        hosp_val_pids = extract_pids_from_ids(v10_split['hospital_val_patients'], 'HOSP_')
    else:
        hosp_val_pids = extract_pids_from_slices(v10_split.get('hospital_val', []))

    # hospital test = hospital_data/eval 裡的病人（不在 train/val 中的）
    hosp_test_pids = set()

    print(f"\nV10 patient IDs:")
    print(f"  train: {sorted(train_pids)}")
    print(f"  val: {sorted(val_pids)}")
    print(f"  test: {sorted(test_pids)}")
    print(f"  hospital_train: {len(hosp_train_pids)} patients")
    print(f"  hospital_val: {sorted(hosp_val_pids)}")

    # 掃描所有 metadata
    all_meta = {}
    for meta_dir in args.meta_dirs:
        for meta_path in sorted(Path(meta_dir).rglob('metadata.json')):
            with open(meta_path) as f:
                meta = json.load(f)
            pid = meta['patient_id']
            meta['_meta_path'] = str(meta_path)
            meta['_meta_dir'] = str(meta_path.parent)
            all_meta[pid] = meta

    print(f"\n找到 {len(all_meta)} 個病人的 metadata")

    # 分配到 splits
    def make_entries(pids, default_domain):
        entries = []
        for pid in sorted(pids):
            if pid in all_meta:
                m = all_meta[pid]
                entries.append({
                    'patient_id': pid,
                    'meta_dir': m['_meta_dir'],
                    'domain': 0 if m['domain'] == 'public' else 1,
                    'num_slices': m['num_slices'],
                    'num_fg_slices': m['num_foreground_slices'],
                })
            else:
                print(f"  [警告] {pid} 沒有 metadata")
        return entries

    split_v11 = {
        'train': make_entries(train_pids, 0),
        'val': make_entries(val_pids, 0),
        'test': make_entries(test_pids, 0),
        'hospital_train': make_entries(hosp_train_pids, 1),
        'hospital_val': make_entries(hosp_val_pids, 1),
        'hospital_test': [],
    }

    # 未被任何 split 覆蓋的病人 → test（不混入 train/val）
    assigned = train_pids | val_pids | test_pids | hosp_train_pids | hosp_val_pids
    for pid, meta in all_meta.items():
        if pid not in assigned:
            domain = 0 if meta['domain'] == 'public' else 1
            key = 'hospital_test' if domain == 1 else 'test'
            split_v11[key].append({
                'patient_id': pid,
                'meta_dir': meta['_meta_dir'],
                'domain': domain,
                'num_slices': meta['num_slices'],
                'num_fg_slices': meta['num_foreground_slices'],
            })
            print(f"  {pid} (未在 V10 split 中) → {key}")

    # 儲存
    out_path = os.path.join(args.output_dir, 'data_split_v11.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(split_v11, f, indent=2)

    print(f"\n已儲存: {out_path}")
    for k, v in split_v11.items():
        total_s = sum(e['num_slices'] for e in v)
        total_fg = sum(e['num_fg_slices'] for e in v)
        print(f"  {k}: {len(v)} patients, {total_s} slices, {total_fg} fg")


# ─────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='V11 前處理: 只產生 ROI metadata（不複製圖片）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    # 前處理模式
    parser.add_argument('--data-dir', type=str, default=None,
                        help='原始資料目錄')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='metadata 輸出目錄')
    parser.add_argument('--domain', type=str, choices=['public', 'hospital'],
                        default='public')
    parser.add_argument('--roi-margin', type=int, default=32)
    parser.add_argument('--min-roi-size', type=int, default=96)
    parser.add_argument('--patients', type=str, nargs='*', default=None,
                        help='只處理指定病人')

    # Build split 模式
    parser.add_argument('--build-split', action='store_true')
    parser.add_argument('--v10-split', type=str, default=None,
                        help='原本的 data_split_v10_mixed.json')
    parser.add_argument('--meta-dirs', type=str, nargs='*', default=None,
                        help='metadata 目錄列表')

    args = parser.parse_args()

    if args.build_split:
        if not args.meta_dirs or not args.output_dir:
            print("錯誤: --build-split 需要 --meta-dirs 和 --output-dir")
            sys.exit(1)
        build_split(args)
        return

    if not args.data_dir or not args.output_dir:
        parser.print_help()
        sys.exit(1)

    # 偵測結構
    structure = detect_structure(args.data_dir)
    print(f"結構: {structure}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Domain: {args.domain}")
    print()

    if structure == 'flat':
        patients = scan_flat(args.data_dir)
    elif structure == 'per_patient':
        patients = scan_per_patient(args.data_dir)
    else:
        print(f"無法偵測! 請檢查 {args.data_dir}")
        sys.exit(1)

    if args.patients:
        patients = {k: v for k, v in patients.items() if k in args.patients}

    print(f"找到 {len(patients)} 個病人\n")

    all_meta = []
    for pid, slices in patients.items():
        meta = process_patient(pid, slices, args.output_dir, args.domain,
                               args.roi_margin, args.min_roi_size)
        if meta:
            all_meta.append(meta)

    total_s = sum(m['num_slices'] for m in all_meta)
    total_fg = sum(m['num_foreground_slices'] for m in all_meta)

    print(f"\n{'='*50}")
    print(f"完成! {len(all_meta)} 病人, {total_s} slices, {total_fg} fg")
    print(f"Metadata 在: {args.output_dir}")
    print(f"磁碟用量: 僅 {len(all_meta)} 個 JSON 檔 (~幾 KB)")


if __name__ == '__main__':
    main()
