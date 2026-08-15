"""
creative_applications.py — 創新應用（3D STL + 植牙距離圖 + 軌跡分析）
======================================================================
從模型預測生成三種臨床應用：
  1. 3D STL Mesh：分割結果轉 3D 網格，可 3D 列印
  2. 植牙風險距離圖：每個 voxel 到神經管的距離，標示安全/警告/危險區
  3. 神經管軌跡分析：centerline、管徑、曲率

用法：
  python creative_applications.py
  python creative_applications.py --weights outputs/coseg_v7_best.pth --patient Patient_4
"""
import os, cv2, torch, gc, argparse
import numpy as np
import torch.nn.functional as F
import hydra
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, label as scipy_label, center_of_mass
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from model_v6 import CoSegV6
from sam2.build_sam import build_sam2

PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
EVAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data/public_data/eval")
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "outputs/coseg_v7_best.pth")
SAM2_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs/applications")


# ==========================================
# 🔮 推論
# ==========================================
def run_inference(model, patient_dir, device):
    """對一個病人跑推論，回傳 3D 預測 volume 和 GT volume"""
    idir = os.path.join(patient_dir, "image_1024")
    mdir = os.path.join(patient_dir, "mask_sem_1024")

    pm = np.array([123.675, 116.280, 103.530], dtype=np.float32)
    ps = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    ifs = sorted([f for f in os.listdir(idir) if f.endswith(".png")])
    ag = {}
    for f in ifs:
        g = cv2.imread(os.path.join(idir, f), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        ag[int(f.replace(".png", "").rsplit("_", 1)[-1])] = g

    pred_list, gt_list, img_list = [], [], []

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for f in tqdm(ifs, desc="推論"):
            idx = int(f.replace(".png", "").rsplit("_", 1)[-1])
            c = ag[idx]
            p_ = ag.get(idx - 1, c)
            n_ = ag.get(idx + 1, c)
            img = np.stack([p_, c, n_], axis=-1)
            gt = np.load(os.path.join(mdir, f.replace(".png", ".npy"))).astype(np.float32)

            img_norm = (img - pm) / (ps + 1e-8)
            t = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
            outputs = model(x=t)
            s = F.interpolate(outputs[1], (1024, 1024), mode='bilinear', align_corners=False)
            prob = torch.sigmoid(s[:, 0, :, :])[0].cpu().numpy()

            # TTA: 水平翻轉
            img_flip = np.flip(img, axis=1).copy()
            img_fn = (img_flip - pm) / (ps + 1e-8)
            tf = torch.tensor(img_fn).permute(2, 0, 1).unsqueeze(0).float().to(device)
            of = model(x=tf)
            sf = F.interpolate(of[1], (1024, 1024), mode='bilinear', align_corners=False)
            pf = np.flip(torch.sigmoid(sf[:, 0, :, :])[0].cpu().numpy(), axis=1).copy()

            prob = (prob + pf) / 2.0

            # 縮小到 256 省記憶體
            pred_list.append(cv2.resize(prob, (256, 256), interpolation=cv2.INTER_LINEAR))
            gt_list.append((cv2.resize(gt, (256, 256), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8))
            img_list.append(cv2.resize(c, (256, 256), interpolation=cv2.INTER_LINEAR))

            del t, tf

    torch.cuda.empty_cache()

    pred_vol = (np.stack(pred_list, 0) > 0.5).astype(np.uint8)
    gt_vol = np.stack(gt_list, 0).astype(np.uint8)
    img_vol = np.stack(img_list, 0)

    return pred_vol, gt_vol, img_vol


# ==========================================
# 1️⃣ 3D STL Mesh 生成
# ==========================================
def generate_3d_mesh(pred_vol, patient_id, output_dir, keep_top_n=2):
    """
    用 Marching Cubes 把分割結果轉成 3D mesh，匯出 STL
    只保留最大的 N 個 component（左右神經管）做展示用
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        from skimage.measure import marching_cubes_lewiner as marching_cubes

    print(f"\n🔧 生成 3D Mesh: {patient_id}")

    # 清理：只保留最大的 N 個 component
    labeled, n_comp = scipy_label(pred_vol)
    if n_comp == 0:
        print("  ⚠️ 沒有預測，跳過")
        return None

    sizes = np.bincount(labeled.ravel())[1:]
    top_n = min(keep_top_n, n_comp)
    top_labels = np.argsort(sizes)[::-1][:top_n] + 1

    clean = np.zeros_like(pred_vol)
    for lbl in top_labels:
        clean[labeled == lbl] = 1

    print(f"  保留 {top_n} 個最大 component（原始 {n_comp} 個）")
    print(f"  Clean volume: {int(clean.sum())} voxels")

    # 加一圈 padding 讓 marching cubes 能封閉 mesh
    padded = np.pad(clean, 1, mode='constant', constant_values=0)

    try:
        verts, faces, normals, values = marching_cubes(padded.astype(np.float32), level=0.5)
        # 移除 padding 的偏移
        verts -= 1.0
    except Exception as e:
        print(f"  ⚠️ Marching cubes 失敗: {e}")
        return None

    print(f"  Mesh: {len(verts)} vertices, {len(faces)} faces")

    # 匯出 STL
    stl_path = os.path.join(output_dir, f"{patient_id}_canal_3d.stl")
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        # 平滑
        trimesh.smoothing.filter_laplacian(mesh, iterations=5)
        mesh.export(stl_path)
        print(f"  💾 STL: {stl_path}")
    except ImportError:
        # 沒有 trimesh，手動寫 STL
        _write_stl_manual(verts, faces, normals, stl_path)
        print(f"  💾 STL (manual): {stl_path}")

    # 生成 3D 預覽圖
    preview_path = os.path.join(output_dir, f"{patient_id}_canal_3d_preview.png")
    _plot_3d_preview(verts, faces, patient_id, preview_path)

    return stl_path


def _write_stl_manual(verts, faces, normals, path):
    """不依賴 trimesh 的 STL 寫入"""
    with open(path, 'w') as f:
        f.write("solid canal\n")
        for face in faces:
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
            # 計算面法向量
            e1 = v1 - v0
            e2 = v2 - v0
            n = np.cross(e1, e2)
            norm = np.linalg.norm(n)
            if norm > 0:
                n = n / norm
            f.write(f"  facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\n")
            f.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
            f.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid canal\n")


def _plot_3d_preview(verts, faces, patient_id, save_path):
    """生成 3D 散點預覽圖"""
    fig = plt.figure(figsize=(12, 5))

    for i, (elev, azim, title) in enumerate([
        (20, -60, 'Front View'), (20, 30, 'Side View'), (80, -60, 'Top View')
    ]):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        # 用頂點的子集畫散點
        step = max(1, len(verts) // 2000)
        v = verts[::step]
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], c=v[:, 0], cmap='viridis',
                   s=1, alpha=0.6)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)

    fig.suptitle(f'{patient_id} — Mandibular Canal 3D Mesh', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 3D Preview: {save_path}")


# ==========================================
# 2️⃣ 植牙風險距離圖
# ==========================================
def generate_risk_map(pred_vol, img_vol, gt_vol, patient_id, output_dir):
    """
    計算每個 voxel 到神經管的最短距離
    生成彩色距離圖：紅=危險(<2mm), 黃=警告(2-4mm), 綠=安全(>4mm)
    """
    print(f"\n🦷 生成植牙風險距離圖: {patient_id}")

    # 用 GT 和 Pred 的聯集算距離（更保守）
    canal_mask = ((pred_vol > 0) | (gt_vol > 0)).astype(bool)

    if canal_mask.sum() == 0:
        print("  ⚠️ 沒有神經管，跳過")
        return

    # 距離場（單位是 pixel，在 256x256 下）
    dist = distance_transform_edt(~canal_mask).astype(np.float32)

    # 找含神經管的中間切片來展示
    canal_z = np.where(canal_mask.sum(axis=(1, 2)) > 0)[0]
    if len(canal_z) == 0:
        return

    # 選 3 張代表切片
    indices = [canal_z[len(canal_z)//4], canal_z[len(canal_z)//2], canal_z[3*len(canal_z)//4]]

    # 風險等級顏色
    risk_cmap = LinearSegmentedColormap.from_list('risk', [
        (0.0, '#FF0000'),   # 紅：0 pixel（管本身）
        (0.08, '#FF4400'),  # 深紅：~2mm 等效
        (0.16, '#FFAA00'),  # 橙：~4mm
        (0.3, '#FFFF00'),   # 黃：~8mm
        (0.5, '#00FF00'),   # 綠：安全
        (1.0, '#006600'),   # 深綠：很遠
    ])

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{patient_id} — Implant Risk Distance Map\n'
                 f'Red=Danger (<2mm equiv.) | Yellow=Warning | Green=Safe',
                 fontsize=14, fontweight='bold')

    for col, z in enumerate(indices):
        # 上排：CT + 距離圖疊加
        ax1 = axes[0, col]
        img_slice = img_vol[z] / max(img_vol[z].max(), 1)
        dist_slice = dist[z]
        max_dist = max(dist_slice.max(), 1)

        ax1.imshow(img_slice, cmap='gray', alpha=0.4)
        im = ax1.imshow(dist_slice / max_dist, cmap=risk_cmap, alpha=0.6, vmin=0, vmax=1)
        # 標記管的位置
        canal_contour = (canal_mask[z]).astype(np.uint8)
        if canal_contour.sum() > 0:
            contours, _ = cv2.findContours(canal_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt = cnt.squeeze()
                if len(cnt.shape) == 2:
                    ax1.plot(cnt[:, 0], cnt[:, 1], 'w-', linewidth=2)
        ax1.set_title(f'Slice {z}', fontsize=12)
        ax1.axis('off')

        # 下排：純距離圖
        ax2 = axes[1, col]
        im2 = ax2.imshow(dist_slice, cmap='hot_r', vmin=0, vmax=30)
        ax2.set_title(f'Distance (px) — Slice {z}', fontsize=12)
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Distance (pixels)')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{patient_id}_implant_risk_map.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Risk Map: {save_path}")


# ==========================================
# 3️⃣ 神經管軌跡分析
# ==========================================
def analyze_trajectory(pred_vol, gt_vol, patient_id, output_dir):
    """
    提取神經管 centerline，分析管徑和曲率
    """
    print(f"\n📐 軌跡分析: {patient_id}")

    # 用 GT 做軌跡分析（更乾淨）
    vol = gt_vol.copy()

    # 找含管的 z 範圍
    canal_z = np.where(vol.sum(axis=(1, 2)) > 0)[0]
    if len(canal_z) < 5:
        print("  ⚠️ 管太短，跳過")
        return

    # 提取每個 z 切片的重心和面積
    # 可能有左右兩條管，用 connected component 分開
    trajectories = {}  # {comp_id: [(z, cy, cx, area, radius), ...]}

    for z in canal_z:
        labeled_z, n = scipy_label(vol[z])
        for c in range(1, n + 1):
            mask_c = (labeled_z == c)
            area = mask_c.sum()
            if area < 3:  # 太小的忽略
                continue
            ys, xs = np.where(mask_c)
            cy, cx = ys.mean(), xs.mean()
            radius = np.sqrt(area / np.pi)

            # 用位置匹配到哪條管（左半 or 右半）
            side = 'L' if cx < vol.shape[2] / 2 else 'R'
            key = side
            if key not in trajectories:
                trajectories[key] = []
            trajectories[key].append((z, cy, cx, area, radius))

    if not trajectories:
        print("  ⚠️ 找不到軌跡")
        return

    # 計算曲率（相鄰三點的曲率半徑）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{patient_id} — Mandibular Canal Trajectory Analysis',
                 fontsize=14, fontweight='bold')

    colors = {'L': '#2196F3', 'R': '#FF5722'}

    for side, traj in trajectories.items():
        traj.sort(key=lambda x: x[0])  # 按 z 排序
        zs = np.array([t[0] for t in traj])
        cys = np.array([t[1] for t in traj])
        cxs = np.array([t[2] for t in traj])
        areas = np.array([t[3] for t in traj])
        radii = np.array([t[4] for t in traj])

        color = colors.get(side, '#888888')
        label = f'{"Left" if side == "L" else "Right"} Canal'

        # (1) XY 軌跡
        axes[0, 0].plot(cxs, cys, '-o', color=color, markersize=2, label=label, alpha=0.8)
        axes[0, 0].set_xlabel('X (pixels)')
        axes[0, 0].set_ylabel('Y (pixels)')
        axes[0, 0].set_title('Canal Trajectory (Axial View)')
        axes[0, 0].legend()
        axes[0, 0].set_aspect('equal')
        axes[0, 0].invert_yaxis()

        # (2) Z vs X
        axes[0, 1].plot(zs, cxs, '-', color=color, label=label, alpha=0.8)
        axes[0, 1].set_xlabel('Z (slice)')
        axes[0, 1].set_ylabel('X position')
        axes[0, 1].set_title('Canal X-position along Z')
        axes[0, 1].legend()

        # (3) 管徑沿 Z 變化
        axes[1, 0].plot(zs, radii * 2, '-', color=color, label=label, alpha=0.8)
        axes[1, 0].set_xlabel('Z (slice)')
        axes[1, 0].set_ylabel('Diameter (pixels)')
        axes[1, 0].set_title('Canal Diameter along Z')
        axes[1, 0].legend()
        axes[1, 0].axhline(y=np.median(radii * 2), color=color, linestyle='--', alpha=0.4)

        # (4) 曲率
        if len(zs) > 2:
            curvatures = []
            for i in range(1, len(zs) - 1):
                # 三點曲率
                p1 = np.array([zs[i-1], cys[i-1], cxs[i-1]])
                p2 = np.array([zs[i], cys[i], cxs[i]])
                p3 = np.array([zs[i+1], cys[i+1], cxs[i+1]])
                v1 = p2 - p1
                v2 = p3 - p2
                cross = np.linalg.norm(np.cross(v1, v2))
                denom = np.linalg.norm(v1) * np.linalg.norm(v2) * np.linalg.norm(p3 - p1)
                curv = (2 * cross / denom) if denom > 1e-8 else 0
                curvatures.append(curv)
            axes[1, 1].plot(zs[1:-1], curvatures, '-', color=color, label=label, alpha=0.8)

    axes[1, 1].set_xlabel('Z (slice)')
    axes[1, 1].set_ylabel('Curvature')
    axes[1, 1].set_title('Canal Curvature along Z')
    axes[1, 1].legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{patient_id}_trajectory_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Trajectory: {save_path}")

    # 印出統計
    for side, traj in trajectories.items():
        radii = np.array([t[4] for t in traj])
        name = "Left" if side == "L" else "Right"
        print(f"  📊 {name} Canal:")
        print(f"     Z range: {traj[0][0]} ~ {traj[-1][0]} ({len(traj)} slices)")
        print(f"     Diameter: {np.median(radii*2):.1f} px (median), {radii.min()*2:.1f}~{radii.max()*2:.1f} px")


# ==========================================
# 🚀 主程式
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--patient", default=None, help="指定病人，如 Patient_4")
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--skip-stl", action="store_true", help="跳過 STL 生成")
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

    print(f"\n🎨 生成創新應用（{len(patients)} 個病人）")
    print("=" * 50)

    for pid in patients:
        pdir = os.path.join(EVAL_DATA_DIR, pid)
        if not os.path.isdir(os.path.join(pdir, "image_1024")):
            continue

        print(f"\n{'='*50}")
        print(f"📊 {pid}")
        print(f"{'='*50}")

        # 推論
        pred_vol, gt_vol, img_vol = run_inference(model, pdir, device)
        gc.collect()

        # 1. 3D STL
        if not args.skip_stl:
            generate_3d_mesh(pred_vol, pid, args.output)

        # 2. 植牙風險距離圖
        generate_risk_map(pred_vol, img_vol, gt_vol, pid, args.output)

        # 3. 軌跡分析
        analyze_trajectory(pred_vol, gt_vol, pid, args.output)

        del pred_vol, gt_vol, img_vol
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n✅ 全部完成！輸出在: {args.output}")
    print(f"   - *_canal_3d.stl          → 3D 列印用")
    print(f"   - *_canal_3d_preview.png   → 3D 預覽圖")
    print(f"   - *_implant_risk_map.png   → 植牙風險距離圖")
    print(f"   - *_trajectory_analysis.png → 軌跡分析")


if __name__ == "__main__":
    main()
