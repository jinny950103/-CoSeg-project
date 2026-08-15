"""
generate_stl.py — 3D STL 生成（修正版）
========================================
修正：
  1. 解析度 256 → 512（更精細的 mesh）
  2. 不用 top-K filtering（會丟掉真管）
  3. 改用 z-span filtering（保留跨越 ≥10 切片的 component）
  4. 同時生成 GT mesh 和 Prediction mesh（簡報對照用）

用法：
  python generate_stl.py
  python generate_stl.py --weights outputs/coseg_v7_best.pth --patient Patient_4
  python generate_stl.py --resolution 1024   # 最高品質（吃記憶體）
"""
import os, cv2, torch, gc, argparse
import numpy as np
import torch.nn.functional as F
import hydra
from tqdm import tqdm
from scipy.ndimage import label as scipy_label

from model_v6 import CoSegV6
from sam2.build_sam import build_sam2

PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
EVAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data/public_data/eval")
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "outputs/coseg_v7_best.pth")
SAM2_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs/stl")


def run_inference(model, patient_dir, device, resolution=512):
    """推論 + TTA，回傳指定解析度的 3D volume"""
    idir = os.path.join(patient_dir, "image_1024")
    mdir = os.path.join(patient_dir, "mask_sem_1024")
    pm = np.array([123.675, 116.280, 103.530], dtype=np.float32)
    ps = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    ifs = sorted([f for f in os.listdir(idir) if f.endswith(".png")])
    ag = {}
    for f in ifs:
        g = cv2.imread(os.path.join(idir, f), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        ag[int(f.replace(".png", "").rsplit("_", 1)[-1])] = g

    pred_list, gt_list = [], []

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for f in tqdm(ifs, desc="推論"):
            idx = int(f.replace(".png", "").rsplit("_", 1)[-1])
            c = ag[idx]
            p_ = ag.get(idx - 1, c)
            n_ = ag.get(idx + 1, c)
            img = np.stack([p_, c, n_], axis=-1)
            gt = np.load(os.path.join(mdir, f.replace(".png", ".npy"))).astype(np.float32)

            # 原圖
            t = torch.tensor((img - pm) / (ps + 1e-8)).permute(2, 0, 1).unsqueeze(0).float().to(device)
            o = model(x=t)
            s = F.interpolate(o[1], (1024, 1024), mode='bilinear', align_corners=False)
            prob = torch.sigmoid(s[:, 0, :, :])[0].cpu().numpy()

            # TTA: 水平翻轉
            img_f = np.flip(img, axis=1).copy()
            tf = torch.tensor((img_f - pm) / (ps + 1e-8)).permute(2, 0, 1).unsqueeze(0).float().to(device)
            of = model(x=tf)
            sf = F.interpolate(of[1], (1024, 1024), mode='bilinear', align_corners=False)
            pf = np.flip(torch.sigmoid(sf[:, 0, :, :])[0].cpu().numpy(), axis=1).copy()

            prob = (prob + pf) / 2.0

            # resize 到目標解析度
            pred_list.append(
                (cv2.resize(prob, (resolution, resolution), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8))
            gt_list.append(
                (cv2.resize(gt, (resolution, resolution), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8))

            del t, tf

    torch.cuda.empty_cache()
    return np.stack(pred_list, 0), np.stack(gt_list, 0)


def zspan_filter(volume, min_zspan=10):
    """
    Z-span 過濾：只保留在 z 方向跨越 ≥ min_zspan 個切片的 component
    比 top-K 好：真正的神經管跨越上百個切片，雜訊碎片只佔 1-3 個切片
    """
    labeled, n_comp = scipy_label(volume)
    if n_comp == 0:
        return volume

    clean = np.zeros_like(volume)
    kept, removed = 0, 0

    for c in range(1, n_comp + 1):
        comp_mask = (labeled == c)
        # 找這個 component 在 z 方向的跨度
        z_indices = np.where(comp_mask.any(axis=(1, 2)))[0]
        zspan = z_indices[-1] - z_indices[0] + 1 if len(z_indices) > 0 else 0

        if zspan >= min_zspan:
            clean[comp_mask] = 1
            kept += 1
        else:
            removed += 1

    print(f"  Z-span filter: 保留 {kept} 個 (z≥{min_zspan}), 移除 {removed} 個碎片")
    return clean


def volume_to_stl(volume, stl_path, smooth_iterations=10):
    """3D binary volume → STL 檔"""
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        from skimage.measure import marching_cubes_lewiner as marching_cubes

    if volume.sum() == 0:
        print(f"  ⚠️ Volume 是空的，跳過")
        return False

    # padding 讓 mesh 封閉
    padded = np.pad(volume.astype(np.float32), 1, mode='constant', constant_values=0)

    try:
        verts, faces, normals, _ = marching_cubes(padded, level=0.5)
        verts -= 1.0  # 移除 padding 偏移
    except Exception as e:
        print(f"  ⚠️ Marching cubes 失敗: {e}")
        return False

    print(f"  Mesh: {len(verts)} vertices, {len(faces)} faces")

    # 嘗試用 trimesh 平滑 + 匯出
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        if smooth_iterations > 0:
            trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iterations)
        mesh.export(stl_path)
    except ImportError:
        # 沒有 trimesh，手動寫 ASCII STL
        print("  (trimesh 未安裝，用 ASCII STL)")
        with open(stl_path, 'w') as f:
            f.write("solid canal\n")
            for face in faces:
                v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
                e1, e2 = v1 - v0, v2 - v0
                n = np.cross(e1, e2)
                norm = np.linalg.norm(n)
                if norm > 0: n = n / norm
                f.write(f"  facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
                f.write("    outer loop\n")
                for v in [v0, v1, v2]:
                    f.write(f"      vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                f.write("    endloop\n  endfacet\n")
            f.write("endsolid canal\n")

    print(f"  💾 {stl_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--patient", default=None)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--resolution", type=int, default=512, help="512 或 1024")
    parser.add_argument("--min-zspan", type=int, default=10, help="Z-span 過濾閾值")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')

    print(f"📦 載入模型: {args.weights}")
    model = CoSegV6(build_sam2("sam2_hiera_l.yaml", SAM2_CHECKPOINT, mode="train")).to(device)
    sd = torch.load(args.weights, map_location=device)
    model.load_state_dict(
        {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}, strict=True)
    model.eval()

    if args.patient:
        patients = [args.patient]
    else:
        patients = sorted([p for p in os.listdir(EVAL_DATA_DIR) if p.startswith("Patient_")])

    print(f"\n🔧 3D STL 生成（解析度 {args.resolution}×{args.resolution}）")
    print("=" * 50)

    for pid in patients:
        pdir = os.path.join(EVAL_DATA_DIR, pid)
        if not os.path.isdir(os.path.join(pdir, "image_1024")):
            continue

        print(f"\n{'='*50}")
        print(f"📊 {pid}")
        print(f"{'='*50}")

        pred_vol, gt_vol = run_inference(model, pdir, device, args.resolution)
        gc.collect()

        # === GT Mesh（乾淨，展示用）===
        print(f"\n  🟢 GT Mesh:")
        print(f"     Voxels: {int(gt_vol.sum())}")
        gt_path = os.path.join(args.output, f"{pid}_GT_canal.stl")
        volume_to_stl(gt_vol, gt_path, smooth_iterations=10)

        # === Prediction Mesh（全部保留，未過濾）===
        print(f"\n  🟡 Prediction Mesh (raw):")
        labeled, n_comp = scipy_label(pred_vol)
        print(f"     Voxels: {int(pred_vol.sum())}, Components: {n_comp}")
        raw_path = os.path.join(args.output, f"{pid}_pred_raw_canal.stl")
        volume_to_stl(pred_vol, raw_path, smooth_iterations=5)

        # === Prediction Mesh（Z-span 過濾）===
        print(f"\n  🔵 Prediction Mesh (z-span filtered):")
        pred_filtered = zspan_filter(pred_vol.copy(), min_zspan=args.min_zspan)
        labeled_f, n_f = scipy_label(pred_filtered)
        print(f"     Voxels: {int(pred_filtered.sum())}, Components: {n_f}")
        filtered_path = os.path.join(args.output, f"{pid}_pred_filtered_canal.stl")
        volume_to_stl(pred_filtered, filtered_path, smooth_iterations=10)

        del pred_vol, gt_vol, pred_filtered
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n✅ 完成！STL 檔案在: {args.output}")
    print(f"\n  每個病人 3 個 STL：")
    print(f"    *_GT_canal.stl             ← GT（最乾淨，展示 pipeline 能力）")
    print(f"    *_pred_raw_canal.stl       ← 模型預測（全部碎片）")
    print(f"    *_pred_filtered_canal.stl  ← 模型預測（z-span 過濾後）")
    print(f"\n  用 3D Slicer 或 MeshLab 開啟檢視")


if __name__ == "__main__":
    main()
