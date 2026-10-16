"""
expand_points_to_sphere_v3.py

演算法與原版 expand_points_to_sphere.py 相同：
每個醫師白點 → 半徑 0.75 mm 的球體，輸出 NIfTI mask。

與原版的差異（只改資料讀取與檢查，不改演算法）:
  1. label / no_label 都用 SimpleITK 讀取（依幾何位置排序，兩邊一致）；
     Enhanced CT 多幀檔被讀成 4D 時，自動 squeeze 成 3D
  2. 讀取前檢查資料夾內矩陣大小是否一致
     （避免像某些病例混了 500x500 與 503x503 而被截斷的 volume）
  3. 檢查 label 與 no_label 的 size / spacing / origin / direction 是否一致
  4. 標註點 = (label >= THRESHOLD) 且 (no_label < THRESHOLD)，
     並印出被排除的重疊體素數（目前驗證過的兩位都是 0）
  5. 輸出 3D uint8 mask，幾何沿用 no_label；
     若設定 NIFTI_DIR，會核對 mask 與 {id}_0000.nii.gz 的 shape

用法:
  python3 expand_points_to_sphere_v3.py <病歷號>       # 單一病患
  python3 expand_points_to_sphere_v3.py                # 處理 valid_patients.txt 內全部病患
"""

import os
import sys
import numpy as np
import pydicom
import SimpleITK as sitk

# ── 設定 ──────────────────────────────────────────────────────
BASE_DATA_DIR = "/home/code-server/data/nas"
OUTPUT_DIR    = "/home/code-server/focus_train/sphere_output"
VALID_LIST    = "/home/code-server/valid_patients.txt"
NIFTI_DIR     = None     # 例如 "/path/to/imagesTr"；設定後會核對 shape，None 代表跳過
THRESHOLD     = 3094     # 醫生白點 HU 閾值（SimpleITK 讀出的值已套用 slope/intercept）
RADIUS_MM     = 0.75     # 球體半徑（mm）


# ── 讀取與檢查 ────────────────────────────────────────────────
def check_uniform_matrix(folder):
    """資料夾內所有 DICOM 的 (Rows, Columns) 必須一致"""
    shapes = set()
    for f in os.listdir(folder):
        if f.endswith(".dcm"):
            ds = pydicom.dcmread(os.path.join(folder, f), stop_before_pixels=True)
            shapes.add((int(ds.Rows), int(ds.Columns)))
    if len(shapes) != 1:
        raise ValueError(f"{folder} 內矩陣大小不一致: {sorted(shapes)}")


def read3d(folder):
    """用 SimpleITK 讀取 DICOM 資料夾；多幀檔若被讀成 4D (x,y,z,1) 則 squeeze 成 3D"""
    check_uniform_matrix(folder)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(folder))
    img = reader.Execute()
    if img.GetDimension() == 4:
        if img.GetSize()[3] != 1:
            raise ValueError(f"{folder} 讀成 4D 且第 4 維不是 1: {img.GetSize()}")
        img = img[:, :, :, 0]
    return img


def check_same_geometry(a, b):
    if a.GetSize() != b.GetSize():
        raise ValueError(f"size 不一致: no_label {a.GetSize()} vs label {b.GetSize()}")
    for name in ("GetSpacing", "GetOrigin", "GetDirection"):
        va, vb = np.array(getattr(a, name)()), np.array(getattr(b, name)())
        if not np.allclose(va, vb, atol=1e-4):
            raise ValueError(f"{name[3:]} 不一致: no_label {va} vs label {vb}")


# ── 建立球體 mask（與原版相同）────────────────────────────────
def create_sphere_mask(points, shape, spacing, radius_mm):
    """
    points : (N, 3) 陣列，每列為 (z, y, x)
    shape  : (z, y, x)
    spacing: (sx, sy, sz)，SimpleITK 格式
    """
    sx, sy, sz = spacing
    mask = np.zeros(shape, dtype=np.uint8)

    rz = int(np.ceil(radius_mm / sz))
    ry = int(np.ceil(radius_mm / sy))
    rx = int(np.ceil(radius_mm / sx))

    for (z, y, x) in points:
        z_min = max(0, z - rz); z_max = min(shape[0], z + rz + 1)
        y_min = max(0, y - ry); y_max = min(shape[1], y + ry + 1)
        x_min = max(0, x - rx); x_max = min(shape[2], x + rx + 1)

        for dz in range(z_min, z_max):
            for dy in range(y_min, y_max):
                for dx in range(x_min, x_max):
                    dist = np.sqrt(
                        ((dz - z) * sz) ** 2 +
                        ((dy - y) * sy) ** 2 +
                        ((dx - x) * sx) ** 2
                    )
                    if dist <= radius_mm:
                        mask[dz, dy, dx] = 1
    return mask


# ── 單一病患 ──────────────────────────────────────────────────
def process_patient(patient_id):
    print(f"\n{'=' * 55}\n  病患: {patient_id}\n{'=' * 55}")

    label_folder    = os.path.join(BASE_DATA_DIR, patient_id, "label")
    no_label_folder = os.path.join(BASE_DATA_DIR, patient_id, "no_label")
    out_dir         = os.path.join(OUTPUT_DIR, patient_id)

    nol = read3d(no_label_folder)
    lab = read3d(label_folder)
    check_same_geometry(nol, lab)

    spacing = nol.GetSpacing()
    a = sitk.GetArrayFromImage(nol)   # (z, y, x)
    b = sitk.GetArrayFromImage(lab)
    print(f"  影像大小: {a.shape} (z, y, x) | spacing: {tuple(round(s, 4) for s in spacing)} mm")

    A = b >= THRESHOLD
    B = a >= THRESHOLD
    overlap = int((A & B).sum())
    markers = A & ~B
    n_points = int(markers.sum())
    print(f"  label >= {THRESHOLD}: {int(A.sum())} | no_label >= {THRESHOLD}: {int(B.sum())} | 重疊(已排除): {overlap}")
    if overlap > 0:
        print(f"  [警告] 有 {overlap} 個體素在 label 與 no_label 都 >= 閾值，已從標註點排除")
    if n_points == 0:
        raise ValueError("沒有找到醫生標記點")

    points = np.argwhere(markers)     # (z, y, x)
    z_idx = np.unique(points[:, 0])
    print(f"  標記點: {n_points} 個，分布在 {len(z_idx)} 層 (z {z_idx.min()} ~ {z_idx.max()})")

    mask = create_sphere_mask(points, a.shape, spacing, RADIUS_MM)
    print(f"  半徑 {RADIUS_MM} mm → {int(mask.sum())} 個 voxel")

    mask_img = sitk.GetImageFromArray(mask)
    mask_img.CopyInformation(nol)

    os.makedirs(out_dir, exist_ok=True)
    radius_str = str(RADIUS_MM).replace(".", "p")
    out_path = os.path.join(out_dir, f"{patient_id}_sphere_r{radius_str}mm.nii.gz")
    sitk.WriteImage(mask_img, out_path)
    print(f"  已儲存: {out_path}")

    if NIFTI_DIR:
        import nibabel as nib
        p = os.path.join(NIFTI_DIR, f"{patient_id}_0000.nii.gz")
        if os.path.exists(p):
            shp = nib.load(p).shape
            if len(shp) == 4 and shp[3] == 1:
                print("  [警告] 影像 NIfTI 是 4D (最後一維為 1)，訓練前需 squeeze 成 3D")
                shp = shp[:3]
            same = tuple(shp) == tuple(mask.shape[::-1])
            print(f"  與影像 NIfTI shape 一致: {same} (影像 {tuple(shp)} / mask {tuple(mask.shape[::-1])})")
        else:
            print(f"  [警告] 找不到 {p}，跳過 shape 核對")


# ── 主程式 ───────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) >= 2:
        patient_ids = [sys.argv[1]]
    else:
        with open(VALID_LIST) as f:
            patient_ids = [l.strip() for l in f if l.strip()]

    failed = []
    for pid in patient_ids:
        try:
            process_patient(pid)
        except Exception as e:
            print(f"  [錯誤] {pid}: {e}")
            failed.append(pid)

    print(f"\n完成 {len(patient_ids) - len(failed)}/{len(patient_ids)} 位")
    if failed:
        print("失敗名單:", failed)
