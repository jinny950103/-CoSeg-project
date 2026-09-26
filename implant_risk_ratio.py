"""
implant_risk_ratio.py — 植牙風險量化（Area Ratio %）
=====================================================
教授要求：
  1. 量化下顎管粗度（直徑 mm）
  2. 設定風險範圍（0-2mm danger, 2-4mm warning）
  3. 計算高風險區佔候選植牙區的比例（%）
  4. 熱力圖 + 具體數字

輸出範例：
  Danger (0-2mm):  12.5%
  Warning (2-4mm): 18.3%
  Safe (>4mm):     69.2%

用法：
  python implant_risk_ratio.py --weights outputs/coseg_hospital_best.pth --patient HOSP_0026779709
  python implant_risk_ratio.py --weights outputs/coseg_v7_best.pth --patient Patient_1 --data_dir data/public_data/eval
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
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "outputs/coseg_hospital_best.pth")
SAM2_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs/risk_ratio")

# 風險閾值（mm）— 文獻依據：Greenstein & Tarnow 2006, 建議最小安全距離 2mm
DANGER_THRESHOLD = 2.0   # 0 ~ 2mm = 高風險
WARNING_THRESHOLD = 4.0  # 2 ~ 4mm = 警告
ROI_RADIUS_MM = 15.0     # 候選植牙區 = canal 中心 ± 15mm


def predict_tta4(model, img_3ch, pm, ps, device):
    """4-view TTA"""
    def pred(img):
        t = torch.tensor((img - pm) / (ps + 1e-8)).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            o = model(x=t)
            s = F.interpolate(o[1], (1024, 1024), mode='bilinear', align_corners=False)
            return torch.sigmoid(s[:, 0, :, :])[0].cpu().numpy()
    p0 = pred(img_3ch)
    p1 = np.flip(pred(np.flip(img_3ch, 1).copy()), 1).copy()
    p2 = np.flip(pred(np.flip(img_3ch, 0).copy()), 0).copy()
    p3 = np.flip(np.flip(pred(np.flip(np.flip(img_3ch, 1), 0).copy()), 0), 1).copy()
    return (p0 + p1 + p2 + p3) / 4.0


def compute_risk_ratio(canal_mask, spacing_mm, roi_radius_mm=15.0,
                       danger_mm=2.0, warning_mm=4.0):
    """
    計算風險區域的面積比例

    Args:
        canal_mask: 2D binary mask (H, W)
        spacing_mm: mm per pixel
        roi_radius_mm: 候選植牙區半徑
        danger_mm: 高風險距離閾值
        warning_mm: 警告距離閾值

    Returns:
        dict with area ratios and zone masks
    """
    if canal_mask.sum() == 0:
        return None

    # 距離場（pixel）→ 轉 mm
    dist_px = distance_transform_edt(~canal_mask.astype(bool))
    dist_mm = dist_px * spacing_mm

    # 定義候選植牙區 ROI = canal 中心 ± roi_radius_mm
    roi_mask = dist_mm <= roi_radius_mm

    # 排除管本身（管裡面不能植牙）
    roi_mask = roi_mask & (~canal_mask.astype(bool))

    roi_area_px = roi_mask.sum()
    if roi_area_px == 0:
        return None

    roi_area_mm2 = roi_area_px * (spacing_mm ** 2)

    # 各風險區
    danger_mask = roi_mask & (dist_mm <= danger_mm)
    warning_mask = roi_mask & (dist_mm > danger_mm) & (dist_mm <= warning_mm)
    safe_mask = roi_mask & (dist_mm > warning_mm)

    danger_area = danger_mask.sum() * (spacing_mm ** 2)
    warning_area = warning_mask.sum() * (spacing_mm ** 2)
    safe_area = safe_mask.sum() * (spacing_mm ** 2)

    # 面積比例（%）
    danger_ratio = (danger_mask.sum() / roi_area_px) * 100
    warning_ratio = (warning_mask.sum() / roi_area_px) * 100
    safe_ratio = (safe_mask.sum() / roi_area_px) * 100

    # 管粗度
    labeled, n = scipy_label(canal_mask.astype(np.uint8))
    diameters = []
    centers = []
    for c in range(1, n + 1):
        comp = (labeled == c)
        area_px = comp.sum()
        if area_px < 3:
            continue
        area_mm2 = area_px * (spacing_mm ** 2)
        diam = 2.0 * np.sqrt(area_mm2 / np.pi)
        ys, xs = np.where(comp)
        diameters.append(diam)
        centers.append((ys.mean(), xs.mean()))

    return {
        'dist_mm': dist_mm,
        'roi_mask': roi_mask,
        'danger_mask': danger_mask,
        'warning_mask': warning_mask,
        'safe_mask': safe_mask,
        'roi_area_mm2': roi_area_mm2,
        'danger_area_mm2': danger_area,
        'warning_area_mm2': warning_area,
        'safe_area_mm2': safe_area,
        'danger_ratio': float(danger_ratio),
        'warning_ratio': float(warning_ratio),
        'safe_ratio': float(safe_ratio),
        'canal_diameters': diameters,
        'canal_centers': centers,
    }


def create_risk_figure(img_gray, canal_mask, result, patient_id, slice_idx,
                       spacing_mm, save_path):
    """生成風險圖（3 欄 + 統計表）"""

    fig, axes = plt.subplots(1, 4, figsize=(28, 7))

    dr = result['danger_ratio']
    wr = result['warning_ratio']
    sr = result['safe_ratio']
    diams = result['canal_diameters']
    diam_text = ' / '.join([f'{d:.2f}mm' for d in diams]) if diams else 'N/A'

    fig.suptitle(f'{patient_id} — Slice {slice_idx}\n'
                 f'Canal Diameter: {diam_text} | '
                 f'Danger: {dr:.1f}% | Warning: {wr:.1f}% | Safe: {sr:.1f}%',
                 fontsize=14, fontweight='bold', y=1.02)

    img_rgb = np.stack([img_gray]*3, axis=-1).astype(np.float32)
    img_rgb = img_rgb / max(img_rgb.max(), 1)

    # === 1. CT + 管輪廓 + 粗度 ===
    ax1 = axes[0]
    ax1.imshow(img_rgb)
    contours, _ = cv2.findContours(canal_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cnt = cnt.squeeze()
        if len(cnt.shape) == 2 and len(cnt) > 2:
            ax1.plot(np.append(cnt[:, 0], cnt[0, 0]),
                     np.append(cnt[:, 1], cnt[0, 1]), 'lime', linewidth=2)
    for diam, (cy, cx) in zip(result['canal_diameters'], result['canal_centers']):
        ax1.annotate(f'⌀{diam:.1f}mm', (cx, cy), color='yellow', fontsize=11,
                     fontweight='bold', ha='center', va='bottom',
                     path_effects=[pe.withStroke(linewidth=3, foreground='black')])
    ax1.set_title('CT + Canal Diameter', fontsize=12)
    ax1.axis('off')

    # === 2. 距離場 + 風險區 ===
    ax2 = axes[1]
    ax2.imshow(img_rgb * 0.3)

    # 只在 ROI 內顯示
    zone_map = np.zeros((*canal_mask.shape, 3), dtype=np.float32)
    zone_map[canal_mask > 0] = [1.0, 1.0, 1.0]           # 白：管本身
    zone_map[result['danger_mask']] = [1.0, 0.0, 0.0]     # 紅：danger
    zone_map[result['warning_mask']] = [1.0, 0.7, 0.0]    # 橙：warning
    zone_map[result['safe_mask']] = [0.0, 0.8, 0.0]       # 綠：safe

    # 只在 ROI 內有顏色
    alpha = np.zeros(canal_mask.shape, dtype=np.float32)
    alpha[result['roi_mask']] = 0.6
    alpha[canal_mask > 0] = 0.9

    for c in range(3):
        blend = img_rgb[:,:,c] * 0.3 * (1 - alpha) + zone_map[:,:,c] * alpha
        zone_map[:,:,c] = blend

    ax2.imshow(zone_map)
    ax2.set_title(f'Risk Zones (ROI ±{ROI_RADIUS_MM}mm)', fontsize=12)
    ax2.axis('off')

    # 圖例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='Canal'),
        Patch(facecolor='red', label=f'Danger 0-{DANGER_THRESHOLD}mm ({dr:.1f}%)'),
        Patch(facecolor='orange', label=f'Warning {DANGER_THRESHOLD}-{WARNING_THRESHOLD}mm ({wr:.1f}%)'),
        Patch(facecolor='green', label=f'Safe >{WARNING_THRESHOLD}mm ({sr:.1f}%)'),
    ]
    ax2.legend(handles=legend_elements, loc='lower left', fontsize=9)

    # === 3. 距離熱力圖 ===
    ax3 = axes[2]
    dist_display = result['dist_mm'].copy()
    dist_display[~result['roi_mask'] & (canal_mask == 0)] = np.nan
    dist_display[canal_mask > 0] = 0

    im3 = ax3.imshow(dist_display, cmap='hot_r', vmin=0, vmax=ROI_RADIUS_MM)
    cbar = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('Distance to Canal Surface (mm)')
    ax3.set_title('Distance Map', fontsize=12)
    ax3.axis('off')

    # === 4. 統計表 ===
    ax4 = axes[3]
    ax4.axis('off')

    table_data = [
        ['Canal', f'{sum(result["canal_diameters"])/max(len(result["canal_diameters"]),1):.1f}mm ⌀',
         f'{canal_mask.sum() * spacing_mm**2:.1f} mm²', '—'],
        [f'Danger\n(0-{DANGER_THRESHOLD}mm)', f'{dr:.1f}%',
         f'{result["danger_area_mm2"]:.1f} mm²', '🔴 High'],
        [f'Warning\n({DANGER_THRESHOLD}-{WARNING_THRESHOLD}mm)', f'{wr:.1f}%',
         f'{result["warning_area_mm2"]:.1f} mm²', '🟡 Medium'],
        [f'Safe\n(>{WARNING_THRESHOLD}mm)', f'{sr:.1f}%',
         f'{result["safe_area_mm2"]:.1f} mm²', '🟢 Low'],
        ['ROI Total', '100%', f'{result["roi_area_mm2"]:.1f} mm²', ''],
    ]

    table = ax4.table(
        cellText=table_data,
        colLabels=['Zone', 'Area Ratio', 'Area', 'Risk Level'],
        loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.2)

    # 上色
    zone_colors = ['white', '#FF6666', '#FFCC66', '#66CC66', '#DDDDDD']
    for i, color in enumerate(zone_colors):
        for j in range(4):
            table[i+1, j].set_facecolor(color)

    ax4.set_title('Risk Zone Statistics', fontsize=12, pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"  💾 {save_path}")


def create_summary(patient_id, all_results, spacing_mm, save_path):
    """整個病人的 3D Volume Ratio 統計"""

    total_danger = 0
    total_warning = 0
    total_safe = 0
    total_roi = 0
    all_diameters = []
    slice_data = []

    for idx, result in sorted(all_results.items()):
        if result is None:
            continue
        total_danger += result['danger_mask'].sum()
        total_warning += result['warning_mask'].sum()
        total_safe += result['safe_mask'].sum()
        total_roi += result['roi_mask'].sum()
        all_diameters.extend(result['canal_diameters'])
        slice_data.append((idx, result['danger_ratio'], result['warning_ratio'],
                          result['safe_ratio']))

    if total_roi == 0:
        print("  ⚠️ 沒有 ROI")
        return

    # 3D Volume Ratio
    vol_danger = (total_danger / total_roi) * 100
    vol_warning = (total_warning / total_roi) * 100
    vol_safe = (total_safe / total_roi) * 100

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(f'{patient_id} — Implant Risk Summary (3D Volume)',
                 fontsize=14, fontweight='bold')

    # 1. 各切片的 danger ratio 沿 Z
    ax1 = axes[0]
    if slice_data:
        zs, drs, wrs, srs = zip(*slice_data)
        ax1.fill_between(zs, 0, drs, color='red', alpha=0.6, label='Danger')
        ax1.fill_between(zs, drs, [d+w for d,w in zip(drs,wrs)], color='orange', alpha=0.6, label='Warning')
        ax1.set_xlabel('Slice (Z)')
        ax1.set_ylabel('Area Ratio (%)')
        ax1.set_title('Risk Ratio per Slice')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. 圓餅圖
    ax2 = axes[1]
    sizes = [vol_danger, vol_warning, vol_safe]
    labels = [f'Danger\n0-{DANGER_THRESHOLD}mm\n{vol_danger:.1f}%',
              f'Warning\n{DANGER_THRESHOLD}-{WARNING_THRESHOLD}mm\n{vol_warning:.1f}%',
              f'Safe\n>{WARNING_THRESHOLD}mm\n{vol_safe:.1f}%']
    colors = ['#FF4444', '#FFAA44', '#44CC44']
    explode = (0.05, 0.02, 0)
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='', startangle=90, textprops={'fontsize': 11})
    ax2.set_title('3D Volume Risk Ratio')

    # 3. 統計摘要
    ax3 = axes[2]
    ax3.axis('off')

    summary = f"""
3D Volume Risk Ratio
{'='*35}

Danger  (0-{DANGER_THRESHOLD}mm):    {vol_danger:.1f}%
Warning ({DANGER_THRESHOLD}-{WARNING_THRESHOLD}mm):  {vol_warning:.1f}%
Safe    (>{WARNING_THRESHOLD}mm):   {vol_safe:.1f}%

Canal Statistics
{'='*35}
Slices with canal: {len(slice_data)}
Mean diameter:     {np.mean(all_diameters):.2f} mm
Median diameter:   {np.median(all_diameters):.2f} mm
Min diameter:      {np.min(all_diameters):.2f} mm
Max diameter:      {np.max(all_diameters):.2f} mm

Risk Assessment
{'='*35}
ROI: Canal ± {ROI_RADIUS_MM}mm
Danger threshold: {DANGER_THRESHOLD}mm
Warning threshold: {WARNING_THRESHOLD}mm
Spacing: {spacing_mm} mm/pixel
"""
    ax3.text(0.05, 0.95, summary, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  📊 3D Volume Risk Ratio:")
    print(f"     Danger  (0-{DANGER_THRESHOLD}mm):  {vol_danger:.1f}%")
    print(f"     Warning ({DANGER_THRESHOLD}-{WARNING_THRESHOLD}mm): {vol_warning:.1f}%")
    print(f"     Safe    (>{WARNING_THRESHOLD}mm):  {vol_safe:.1f}%")
    print(f"     Canal diameter: {np.mean(all_diameters):.2f} ± {np.std(all_diameters):.2f} mm")
    print(f"  💾 {save_path}")


def main():
    global DANGER_THRESHOLD, WARNING_THRESHOLD, ROI_RADIUS_MM
    parser = argparse.ArgumentParser(description="植牙風險量化 — Area Ratio (%)")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--patient", required=True, help="病人 ID，如 HOSP_0026779709 或 Patient_1")
    parser.add_argument("--data_dir", default=None, help="資料夾路徑（預設自動判斷公開/醫院）")
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--spacing", type=float, default=0.4, help="mm/pixel")
    parser.add_argument("--danger", type=float, default=DANGER_THRESHOLD, help="高風險閾值 mm")
    parser.add_argument("--warning", type=float, default=WARNING_THRESHOLD, help="警告閾值 mm")
    parser.add_argument("--roi", type=float, default=ROI_RADIUS_MM, help="ROI 半徑 mm")
    parser.add_argument("--num_figs", type=int, default=6, help="生成幾張切片圖")
    args = parser.parse_args()

    
    DANGER_THRESHOLD = args.danger
    WARNING_THRESHOLD = args.warning
    ROI_RADIUS_MM = args.roi

    os.makedirs(args.output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 自動判斷資料路徑
    if args.data_dir:
        patient_dir = os.path.join(args.data_dir, args.patient)
    elif args.patient.startswith("HOSP_"):
        patient_dir = os.path.join(PROJECT_ROOT, "data/hospital_data/eval", args.patient)
    else:
        patient_dir = os.path.join(PROJECT_ROOT, "data/public_data/eval", args.patient)

    idir = os.path.join(patient_dir, "image_1024")
    mdir = os.path.join(patient_dir, "mask_sem_1024")

    if not os.path.isdir(idir):
        print(f"❌ 找不到 {idir}")
        return

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

    ifs = sorted([f for f in os.listdir(idir) if f.endswith(".png")])
    ag = {}
    for f in ifs:
        g = cv2.imread(os.path.join(idir, f), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        ag[int(f.replace(".png", "").rsplit("_", 1)[-1])] = g

    print(f"\n🦷 植牙風險量化: {args.patient}")
    print(f"   Danger: 0-{DANGER_THRESHOLD}mm | Warning: {DANGER_THRESHOLD}-{WARNING_THRESHOLD}mm | ROI: ±{ROI_RADIUS_MM}mm")
    print(f"   Spacing: {args.spacing} mm/px")
    print("=" * 50)

    all_results = {}
    positive_slices = []

    with torch.no_grad():
        for f in tqdm(ifs, desc="推論"):
            idx = int(f.replace(".png", "").rsplit("_", 1)[-1])
            c = ag[idx]
            p_ = ag.get(idx - 1, c)
            n_ = ag.get(idx + 1, c)
            img_3ch = np.stack([p_, c, n_], axis=-1)

            prob = predict_tta4(model, img_3ch, pm, ps, device)
            canal_mask = (prob > 0.5).astype(np.uint8)

            if canal_mask.sum() > 0:
                result = compute_risk_ratio(canal_mask, args.spacing,
                                           ROI_RADIUS_MM, DANGER_THRESHOLD, WARNING_THRESHOLD)
                all_results[idx] = result
                positive_slices.append((f, idx, c, canal_mask, result))

    torch.cuda.empty_cache()

    if not positive_slices:
        print("⚠️ 沒有偵測到神經管")
        return

    # 選代表切片生成圖
    step = max(1, len(positive_slices) // args.num_figs)
    selected = positive_slices[::step][:args.num_figs]

    print(f"\n  生成 {len(selected)} 張風險圖...")
    for f, idx, c, canal_mask, result in selected:
        save_path = os.path.join(args.output, f"{args.patient}_risk_slice{idx}.png")
        create_risk_figure(c, canal_mask, result, args.patient, idx, args.spacing, save_path)

    # 整個病人的 3D 統計
    summary_path = os.path.join(args.output, f"{args.patient}_risk_summary.png")
    create_summary(args.patient, all_results, args.spacing, summary_path)

    gc.collect()
    print(f"\n✅ 完成！輸出在: {args.output}")


if __name__ == "__main__":
    main()
