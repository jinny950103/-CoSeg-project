"""
implant_risk_quantified.py — 量化植牙風險圖
=============================================
教授要求：
  1. 量化下顎管粗度（直徑 mm）
  2. 設定高風險範圍（管表面 + X mm）
  3. 計算風險機率（%）
  4. 熱力圖顯示具體機率值

用法：
  python implant_risk_quantified.py --weights outputs/coseg_v7_best.pth
  python implant_risk_quantified.py --weights outputs/coseg_v7_best.pth --risk_radius 2.0
"""
import os, cv2, torch, gc, argparse
import numpy as np
import torch.nn.functional as F
import hydra
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, label as scipy_label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

from model_v6 import CoSegV6
from sam2.build_sam import build_sam2

PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
EVAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data/public_data/eval")
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "outputs/coseg_v7_best.pth")
SAM2_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs/risk_maps")

# 預設 voxel spacing（mm/pixel），可從 NIfTI header 讀取覆蓋
DEFAULT_SPACING = 0.4  # mm per pixel（公開資料在 1024×1024 下約 0.4mm/px）


def predict_tta4(model, img_3ch, pm, ps, device):
    """4-view TTA，回傳機率圖"""
    def pred_one(img):
        t = torch.tensor((img - pm) / (ps + 1e-8)).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            o = model(x=t)
            s = F.interpolate(o[1], (1024, 1024), mode='bilinear', align_corners=False)
            return torch.sigmoid(s[:, 0, :, :])[0].cpu().numpy()

    p0 = pred_one(img_3ch)
    p1 = np.flip(pred_one(np.flip(img_3ch, 1).copy()), 1).copy()
    p2 = np.flip(pred_one(np.flip(img_3ch, 0).copy()), 0).copy()
    p3 = np.flip(np.flip(pred_one(np.flip(np.flip(img_3ch, 1), 0).copy()), 0), 1).copy()
    return (p0 + p1 + p2 + p3) / 4.0


def measure_canal_diameter(prob_map, spacing_mm, threshold=0.5):
    """
    量化管的粗度
    回傳：直徑（mm）、面積（mm²）、重心（y, x）
    """
    binary = (prob_map > threshold).astype(np.uint8)
    if binary.sum() == 0:
        return None

    labeled, n = scipy_label(binary)
    results = []

    for c in range(1, n + 1):
        comp = (labeled == c)
        area_px = comp.sum()
        if area_px < 3:
            continue

        # 面積 → 等效直徑
        area_mm2 = area_px * (spacing_mm ** 2)
        diameter_mm = 2.0 * np.sqrt(area_mm2 / np.pi)

        # 重心
        ys, xs = np.where(comp)
        cy, cx = ys.mean(), xs.mean()

        results.append({
            'diameter_mm': diameter_mm,
            'area_mm2': area_mm2,
            'center_y': cy,
            'center_x': cx,
            'area_px': area_px,
        })

    return results if results else None


def compute_risk_probability(prob_map, binary_mask, spacing_mm, risk_radius_mm=2.0):
    """
    計算風險機率
    - 從管表面往外 risk_radius_mm 的區域
    - 在該區域內取模型的平均信心度 = 風險機率

    回傳：
      risk_map: 每個 pixel 的風險機率（0-1）
      zone_stats: 各風險區的統計
    """
    if binary_mask.sum() == 0:
        return np.zeros_like(prob_map), {}

    # 計算離管表面的距離（mm）
    dist_px = distance_transform_edt(~binary_mask.astype(bool))
    dist_mm = dist_px * spacing_mm

    # 管內部的距離設為 0
    dist_mm[binary_mask > 0] = 0

    # 風險機率 = 模型信心度 × 距離衰減
    # 越靠近管，風險越高
    # 使用高斯衰減：risk = confidence × exp(-dist² / (2σ²))
    sigma = risk_radius_mm
    distance_weight = np.exp(-(dist_mm ** 2) / (2 * sigma ** 2))

    # 風險機率 = 距離權重（越近越高）× 模型在管附近的信心
    # 管內部：風險 = 100%
    risk_map = np.zeros_like(prob_map)
    risk_map[binary_mask > 0] = 1.0  # 管本身 = 100% 風險

    # 管外部：按距離衰減
    outside = ~binary_mask.astype(bool)
    risk_map[outside] = distance_weight[outside]

    # 各風險區統計
    zones = {
        'canal': {
            'range': '管本身',
            'mask': binary_mask > 0,
        },
        'danger': {
            'range': f'0-{risk_radius_mm}mm',
            'mask': (dist_mm > 0) & (dist_mm <= risk_radius_mm),
        },
        'warning': {
            'range': f'{risk_radius_mm}-{risk_radius_mm*2}mm',
            'mask': (dist_mm > risk_radius_mm) & (dist_mm <= risk_radius_mm * 2),
        },
        'safe': {
            'range': f'>{risk_radius_mm*2}mm',
            'mask': dist_mm > risk_radius_mm * 2,
        },
    }

    zone_stats = {}
    for name, info in zones.items():
        mask = info['mask']
        if mask.sum() > 0:
            zone_risk = risk_map[mask]
            zone_stats[name] = {
                'range': info['range'],
                'mean_risk': float(zone_risk.mean() * 100),
                'max_risk': float(zone_risk.max() * 100),
                'min_risk': float(zone_risk.min() * 100),
                'area_mm2': float(mask.sum() * spacing_mm ** 2),
                'pixel_count': int(mask.sum()),
            }

    return risk_map, zone_stats


def create_risk_figure(img_gray, prob_map, risk_map, canal_info, zone_stats,
                       patient_id, slice_idx, spacing_mm, risk_radius_mm, save_path):
    """生成量化風險圖（4 欄）"""

    binary = (prob_map > 0.5).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(28, 7))

    # 管粗度文字
    if canal_info:
        diameters = [c['diameter_mm'] for c in canal_info]
        diam_text = ' / '.join([f'{d:.2f}mm' for d in diameters])
    else:
        diam_text = 'N/A'

    fig.suptitle(f'{patient_id} — Slice {slice_idx}\n'
                 f'Canal Diameter: {diam_text} | Risk Radius: {risk_radius_mm}mm',
                 fontsize=14, fontweight='bold', y=1.02)

    # === 1. CT + 管輪廓 + 粗度標註 ===
    ax1 = axes[0]
    img_rgb = np.stack([img_gray]*3, axis=-1).astype(np.float32)
    img_rgb = img_rgb / max(img_rgb.max(), 1)
    ax1.imshow(img_rgb)

    # 畫管輪廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cnt = cnt.squeeze()
        if len(cnt.shape) == 2 and len(cnt) > 2:
            ax1.plot(np.append(cnt[:, 0], cnt[0, 0]),
                     np.append(cnt[:, 1], cnt[0, 1]),
                     'lime', linewidth=2)

    # 標註管粗度
    if canal_info:
        for c in canal_info:
            ax1.annotate(f'⌀{c["diameter_mm"]:.1f}mm',
                        (c['center_x'], c['center_y']),
                        color='yellow', fontsize=11, fontweight='bold',
                        ha='center', va='bottom',
                        path_effects=[pe.withStroke(linewidth=3, foreground='black')])

    ax1.set_title('CT + Canal Diameter', fontsize=12)
    ax1.axis('off')

    # === 2. 模型信心度熱力圖 ===
    ax2 = axes[1]
    ax2.imshow(img_rgb * 0.3)
    # 只在管附近顯示（避免全圖都是顏色）
    dist_px = distance_transform_edt(~binary.astype(bool))
    show_region = dist_px < (risk_radius_mm * 3 / spacing_mm)
    confidence_display = np.ma.masked_where(~show_region, prob_map)
    im2 = ax2.imshow(confidence_display, cmap='hot', vmin=0, vmax=1, alpha=0.8)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Model Confidence')
    ax2.set_title('Prediction Confidence', fontsize=12)
    ax2.axis('off')

    # === 3. 風險機率熱力圖（重點）===
    ax3 = axes[2]
    ax3.imshow(img_rgb * 0.3)

    # 自定義顏色：紅(高風險) → 黃 → 綠(低風險)
    risk_cmap = LinearSegmentedColormap.from_list('risk', [
        (0.0, '#006600'),    # 深綠 0%
        (0.15, '#00CC00'),   # 綠 15%
        (0.30, '#FFFF00'),   # 黃 30%
        (0.50, '#FF8800'),   # 橙 50%
        (0.70, '#FF4400'),   # 深橙 70%
        (1.0, '#FF0000'),    # 紅 100%
    ])

    risk_display = np.ma.masked_where(~show_region, risk_map)
    im3 = ax3.imshow(risk_display, cmap=risk_cmap, vmin=0, vmax=1, alpha=0.8)
    cbar = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('Risk Probability (%)')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

    # 在管的位置標注風險數值
    if canal_info:
        for c in canal_info:
            ax3.annotate(f'100%',
                        (c['center_x'], c['center_y']),
                        color='white', fontsize=10, fontweight='bold',
                        ha='center', va='center',
                        path_effects=[pe.withStroke(linewidth=3, foreground='red')])

    ax3.set_title(f'Risk Probability (radius={risk_radius_mm}mm)', fontsize=12)
    ax3.axis('off')

    # === 4. 風險區統計表 ===
    ax4 = axes[3]
    ax4.axis('off')

    table_data = []
    colors = []
    zone_colors = {
        'canal': '#FF0000',
        'danger': '#FF4400',
        'warning': '#FFAA00',
        'safe': '#00CC00',
    }
    zone_labels = {
        'canal': 'Canal\n(管本身)',
        'danger': f'Danger\n(0-{risk_radius_mm}mm)',
        'warning': f'Warning\n({risk_radius_mm}-{risk_radius_mm*2}mm)',
        'safe': f'Safe\n(>{risk_radius_mm*2}mm)',
    }

    for zone_name in ['canal', 'danger', 'warning', 'safe']:
        if zone_name in zone_stats:
            s = zone_stats[zone_name]
            table_data.append([
                zone_labels[zone_name],
                f'{s["mean_risk"]:.1f}%',
                f'{s["max_risk"]:.1f}%',
                f'{s["area_mm2"]:.1f}mm²',
            ])
            colors.append(zone_colors[zone_name])

    if table_data:
        table = ax4.table(
            cellText=table_data,
            colLabels=['Zone', 'Avg Risk', 'Max Risk', 'Area'],
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.0, 2.0)

        # 上色
        for i, color in enumerate(colors):
            table[i+1, 0].set_facecolor(color)
            table[i+1, 0].set_text_props(color='white', fontweight='bold')

    ax4.set_title('Risk Zone Statistics', fontsize=12, pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"  💾 {save_path}")


def create_summary_report(patient_id, all_canal_info, all_zone_stats, spacing_mm,
                          risk_radius_mm, save_path):
    """生成整個病人的統計報告圖"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'{patient_id} — Canal Analysis Summary\n'
                 f'Voxel spacing: {spacing_mm}mm | Risk radius: {risk_radius_mm}mm',
                 fontsize=14, fontweight='bold')

    # 收集所有切片的數據
    slice_indices = []
    diameters_left = []
    diameters_right = []
    danger_risks = []
    warning_risks = []

    for slice_idx, (canal_info, zone_stats) in sorted(all_canal_info.items()):
        if canal_info is None:
            continue
        slice_indices.append(slice_idx)

        # 分左右管
        for c in canal_info:
            if c['center_x'] < 512:
                diameters_left.append((slice_idx, c['diameter_mm']))
            else:
                diameters_right.append((slice_idx, c['diameter_mm']))

        if 'danger' in zone_stats:
            danger_risks.append((slice_idx, zone_stats['danger']['mean_risk']))
        if 'warning' in zone_stats:
            warning_risks.append((slice_idx, zone_stats['warning']['mean_risk']))

    # 1. 管徑沿 Z 軸變化
    ax1 = axes[0]
    if diameters_left:
        zl, dl = zip(*diameters_left)
        ax1.plot(zl, dl, 'b-o', markersize=2, label='Left', alpha=0.7)
    if diameters_right:
        zr, dr = zip(*diameters_right)
        ax1.plot(zr, dr, 'r-o', markersize=2, label='Right', alpha=0.7)
    ax1.set_xlabel('Slice (Z)')
    ax1.set_ylabel('Diameter (mm)')
    ax1.set_title('Canal Diameter along Z')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 中位數標線
    all_diams = [d for _, d in diameters_left + diameters_right]
    if all_diams:
        median_d = np.median(all_diams)
        ax1.axhline(y=median_d, color='gray', linestyle='--', alpha=0.5,
                    label=f'Median: {median_d:.2f}mm')
        ax1.legend()

    # 2. 風險機率沿 Z 軸
    ax2 = axes[1]
    if danger_risks:
        zd, rd = zip(*danger_risks)
        ax2.plot(zd, rd, 'r-', label=f'Danger (0-{risk_radius_mm}mm)', alpha=0.8)
    if warning_risks:
        zw, rw = zip(*warning_risks)
        ax2.plot(zw, rw, 'orange', label=f'Warning ({risk_radius_mm}-{risk_radius_mm*2}mm)', alpha=0.8)
    ax2.set_xlabel('Slice (Z)')
    ax2.set_ylabel('Mean Risk (%)')
    ax2.set_title('Risk Probability along Z')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    # 3. 統計摘要
    ax3 = axes[2]
    ax3.axis('off')

    all_diams = [d for _, d in diameters_left + diameters_right]
    all_danger = [r for _, r in danger_risks]
    all_warning = [r for _, r in warning_risks]

    summary_text = f"""
Canal Statistics
{'='*30}
Slices with canal: {len(slice_indices)}

Diameter:
  Mean:   {np.mean(all_diams):.2f} mm
  Median: {np.median(all_diams):.2f} mm
  Min:    {np.min(all_diams):.2f} mm
  Max:    {np.max(all_diams):.2f} mm
  Std:    {np.std(all_diams):.2f} mm

Risk (Danger zone 0-{risk_radius_mm}mm):
  Mean:   {np.mean(all_danger):.1f}%
  Max:    {np.max(all_danger):.1f}%

Risk (Warning zone {risk_radius_mm}-{risk_radius_mm*2}mm):
  Mean:   {np.mean(all_warning):.1f}%
  Max:    {np.max(all_warning):.1f}%
"""

    ax3.text(0.1, 0.95, summary_text, transform=ax3.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Summary: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="量化植牙風險圖")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--patient", default=None)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--risk_radius", type=float, default=2.0, help="高風險範圍（mm）")
    parser.add_argument("--spacing", type=float, default=DEFAULT_SPACING, help="mm/pixel")
    parser.add_argument("--num", type=int, default=6, help="每個病人幾張圖")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')

    print(f"📦 載入模型: {args.weights}")
    model = CoSegV6(build_sam2("sam2_hiera_l.yaml", SAM2_CHECKPOINT, mode="train")).to(device)
    sd = torch.load(args.weights, map_location=device)
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    model.load_state_dict(
        {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}, strict=True)
    model.eval()

    pm = np.array([123.675, 116.280, 103.530], dtype=np.float32)
    ps = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    if args.patient:
        patients = [args.patient]
    else:
        patients = sorted([p for p in os.listdir(EVAL_DATA_DIR) if p.startswith("Patient_")])

    print(f"\n🦷 量化植牙風險圖")
    print(f"   Risk radius: {args.risk_radius}mm")
    print(f"   Spacing: {args.spacing}mm/pixel")
    print("=" * 50)

    for pid in patients:
        idir = os.path.join(EVAL_DATA_DIR, pid, "image_1024")
        mdir = os.path.join(EVAL_DATA_DIR, pid, "mask_sem_1024")
        if not os.path.isdir(idir):
            continue

        ifs = sorted([f for f in os.listdir(idir) if f.endswith(".png")])
        if not ifs:
            continue

        ag = {}
        for f in ifs:
            g = cv2.imread(os.path.join(idir, f), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            ag[int(f.replace(".png", "").rsplit("_", 1)[-1])] = g

        print(f"\n📊 {pid}")

        # 先跑完所有推論收集數據
        all_canal_info = {}
        positive_slices = []

        for f in tqdm(ifs, desc=f"推論 {pid}"):
            idx = int(f.replace(".png", "").rsplit("_", 1)[-1])
            c = ag[idx]
            p_ = ag.get(idx - 1, c)
            n_ = ag.get(idx + 1, c)
            img_3ch = np.stack([p_, c, n_], axis=-1)

            prob = predict_tta4(model, img_3ch, pm, ps, device)
            binary = (prob > 0.5).astype(np.uint8)

            if binary.sum() > 0:
                canal_info = measure_canal_diameter(prob, args.spacing)
                risk_map, zone_stats = compute_risk_probability(
                    prob, binary, args.spacing, args.risk_radius)
                all_canal_info[idx] = (canal_info, zone_stats)
                positive_slices.append((f, idx, c, img_3ch, prob, binary,
                                       canal_info, risk_map, zone_stats))

        torch.cuda.empty_cache()

        if not positive_slices:
            print(f"  ⚠️ 沒有正樣本")
            continue

        # 選代表切片生成圖
        import random
        step = max(1, len(positive_slices) // args.num)
        selected = positive_slices[::step][:args.num]

        print(f"  生成 {len(selected)} 張風險圖...")

        for f, idx, c, img_3ch, prob, binary, canal_info, risk_map, zone_stats in selected:
            save_name = f"{pid}_risk_slice{idx}.png"
            save_path = os.path.join(args.output, save_name)
            create_risk_figure(c, prob, risk_map, canal_info, zone_stats,
                             pid, idx, args.spacing, args.risk_radius, save_path)

        # 生成統計報告
        summary_path = os.path.join(args.output, f"{pid}_risk_summary.png")
        create_summary_report(pid, all_canal_info, all_canal_info,
                            args.spacing, args.risk_radius, summary_path)

        del positive_slices
        gc.collect()

    print(f"\n✅ 完成！輸出在: {args.output}")


if __name__ == "__main__":
    main()
