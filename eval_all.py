"""
eval_all.py — 統一評估公開 + 醫院 eval 資料
=============================================
一次跑完所有 test 病人，分別報告公開/醫院/總體指標

用法：
  python eval_all.py --weights outputs/coseg_v8_best.pth
  python eval_all.py --weights outputs/coseg_v7_best.pth
  python eval_all.py --weights outputs/coseg_v8_best.pth --no-tta
  python eval_all.py --weights outputs/coseg_v8_best.pth --tta2
  python eval_all.py --weights outputs/coseg_v8_best.pth --eval-size 512

資料夾結構：
  data/public_data/eval/
    Patient_1/image_1024/  mask_sem_1024/
    Patient_3/image_1024/  mask_sem_1024/
    Patient_4/image_1024/  mask_sem_1024/

  data/hospital_data/eval/
    HOSP_0026779709/image_1024/  mask_sem_1024/
    HOSP_0026864677/image_1024/  mask_sem_1024/
    HOSP_0026873925/image_1024/  mask_sem_1024/
"""
import os, cv2, torch, gc, argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, binary_erosion, label as scipy_label
import hydra
from model_v6 import CoSegV6
from sam2.build_sam import build_sam2

PROJECT_ROOT = "/home/u9444861/-CoSeg-project"
PUBLIC_EVAL_DIR  = os.path.join(PROJECT_ROOT, "data/public_data/eval")
HOSPITAL_EVAL_DIR = os.path.join(PROJECT_ROOT, "data/hospital_data/eval")
DEFAULT_WEIGHTS  = os.path.join(PROJECT_ROOT, "outputs/coseg_v8_best.pth")
SAM2_CHECKPOINT  = "checkpoints/sam2_hiera_large.pt"
EVAL_SIZE, SKEL_SIZE, FULL_RES = 256, 128, 1024


# ==========================================
# 指標計算
# ==========================================
def compute_dice(p, g, s=1e-5):
    i = np.sum(p * g)
    return (2 * i + s) / (np.sum(p) + np.sum(g) + s)


def skeletonize_3d_safe(vol):
    try:
        from skimage.morphology import skeletonize_3d
        return skeletonize_3d(vol.astype(np.uint8)).astype(np.float32)
    except Exception:
        from skimage.morphology import skeletonize
        s = np.zeros_like(vol, dtype=np.float32)
        for z in range(vol.shape[0]):
            if vol[z].sum() > 0:
                s[z] = skeletonize(vol[z].astype(bool)).astype(np.float32)
        return s


def compute_cldice(pred, gt, smooth=1e-5):
    pb = (pred > 0).astype(np.uint8)
    gb = (gt > 0).astype(np.uint8)
    if pb.sum() == 0 and gb.sum() == 0: return 1.0
    if pb.sum() == 0 or gb.sum() == 0: return 0.0
    sp = skeletonize_3d_safe(pb)
    sg = skeletonize_3d_safe(gb)
    if sp.sum() == 0 or sg.sum() == 0: return 0.0
    tp = (sp * gb).sum() / (sp.sum() + smooth)
    ts = (sg * pb).sum() / (sg.sum() + smooth)
    r = float(2 * tp * ts / (tp + ts + smooth))
    del sp, sg; gc.collect()
    return r


def compute_hd95(pred, gt):
    pb = (pred > 0).astype(np.uint8)
    gb = (gt > 0).astype(np.uint8)
    if pb.sum() == 0 or gb.sum() == 0: return float('inf')
    ps = pb ^ binary_erosion(pb).astype(np.uint8)
    gs = gb ^ binary_erosion(gb).astype(np.uint8)
    if ps.sum() == 0 or gs.sum() == 0: return float('inf')
    dp = distance_transform_edt(~pb.astype(bool))
    dg = distance_transform_edt(~gb.astype(bool))
    ad = np.concatenate([dg[ps > 0], dp[gs > 0]])
    del dp, dg; gc.collect()
    return float(np.percentile(ad, 95))


def count_bp(pred, gt):
    gr = np.where(gt.sum(axis=(1, 2)) > 0)[0]
    bp = 0
    if len(gr) > 0:
        pz = pred[gr[0]:gr[-1] + 1].sum(axis=(1, 2)) > 0
        ic = False
        for i, h in enumerate(pz):
            if h and not ic:
                if i > 0: bp += 1
                ic = True
            elif not h and ic: ic = False
    return bp


# ==========================================
# 推論
# ==========================================
def predict_one(model, img_3ch, pm, ps, device):
    t = torch.tensor((img_3ch - pm) / (ps + 1e-8)).permute(2, 0, 1).unsqueeze(0).float().to(device)
    o = model(x=t)
    s = F.interpolate(o[1], (FULL_RES, FULL_RES), mode='bilinear', align_corners=False)
    return torch.sigmoid(s[:, 0, :, :])[0].cpu().numpy()


def predict_tta4(model, img, pm, ps, device):
    p0 = predict_one(model, img, pm, ps, device)
    p1 = np.flip(predict_one(model, np.flip(img, 1).copy(), pm, ps, device), 1).copy()
    p2 = np.flip(predict_one(model, np.flip(img, 0).copy(), pm, ps, device), 0).copy()
    p3 = np.flip(np.flip(predict_one(model, np.flip(np.flip(img, 1), 0).copy(), pm, ps, device), 0), 1).copy()
    return (p0 + p1 + p2 + p3) / 4.0


def predict_tta2(model, img, pm, ps, device):
    p0 = predict_one(model, img, pm, ps, device)
    p1 = np.flip(predict_one(model, np.flip(img, 1).copy(), pm, ps, device), 1).copy()
    return (p0 + p1) / 2.0


# ==========================================
# 掃描病人
# ==========================================
def scan_patients(eval_dir, prefix):
    """
    掃描 eval 資料夾，回傳 [(patient_id, image_dir, mask_dir, source), ...]
    支援兩種結構：
      - per-patient: eval_dir/PatientXX/image_1024/
      - flat: eval_dir/image_1024/ (所有病人共用)
    """
    patients = []
    if not os.path.isdir(eval_dir):
        return patients

    # 檢查是否有子資料夾（per-patient 模式）
    subdirs = sorted([d for d in os.listdir(eval_dir)
                      if os.path.isdir(os.path.join(eval_dir, d))
                      and os.path.isdir(os.path.join(eval_dir, d, "image_1024"))])

    if subdirs:
        # Per-patient 模式
        for d in subdirs:
            idir = os.path.join(eval_dir, d, "image_1024")
            mdir = os.path.join(eval_dir, d, "mask_sem_1024")
            if os.path.isdir(mdir):
                patients.append((d, idir, mdir, prefix))
    else:
        # Flat 模式：從檔名分組
        idir = os.path.join(eval_dir, "image_1024")
        mdir = os.path.join(eval_dir, "mask_sem_1024")
        if os.path.isdir(idir) and os.path.isdir(mdir):
            from collections import defaultdict
            groups = defaultdict(list)
            for f in os.listdir(idir):
                if f.endswith(".png"):
                    base = f.replace(".png", "")
                    pid = base.rsplit("_slice_", 1)[0]
                    groups[pid].append(f)
            for pid in sorted(groups.keys()):
                patients.append((pid, idir, mdir, prefix))

    return patients


# ==========================================
# 評估單一病人
# ==========================================
def eval_patient(model, pid, idir, mdir, pm, ps, device, tta_mode, eval_size):
    """回傳 dict with metrics, or None if failed"""
    # 列出此病人的所有 slice
    all_pngs = sorted([f for f in os.listdir(idir) if f.endswith(".png")])

    # 如果是 flat 模式，只取屬於此病人的檔案
    patient_pngs = [f for f in all_pngs if f.replace(".png", "").rsplit("_slice_", 1)[0] == pid]
    if not patient_pngs:
        # per-patient 模式，所有檔案都屬於此病人
        patient_pngs = all_pngs

    if not patient_pngs:
        print(f"  ⚠️ {pid}: 沒有 PNG 檔案")
        return None

    # 載入灰度圖建立 slice index 映射
    ag = {}
    for f in patient_pngs:
        base = f.replace(".png", "")
        idx = int(base.rsplit("_", 1)[-1])
        g = cv2.imread(os.path.join(idir, f), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        ag[idx] = g

    probs_list, gts_list = [], []
    for f in tqdm(patient_pngs, desc=f"推論 {pid}"):
        base = f.replace(".png", "")
        idx = int(base.rsplit("_", 1)[-1])
        c = ag[idx]
        p_ = ag.get(idx - 1, c)
        n_ = ag.get(idx + 1, c)
        img = np.stack([p_, c, n_], axis=-1)

        mask_path = os.path.join(mdir, base + ".npy")
        if not os.path.exists(mask_path):
            continue
        gt = np.load(mask_path).astype(np.float32)

        if tta_mode == "none":
            prob = predict_one(model, img, pm, ps, device)
        elif tta_mode == "2":
            prob = predict_tta2(model, img, pm, ps, device)
        else:
            prob = predict_tta4(model, img, pm, ps, device)

        probs_list.append(cv2.resize(prob, (eval_size, eval_size), interpolation=cv2.INTER_LINEAR))
        gts_list.append((cv2.resize(gt, (eval_size, eval_size), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8))

    torch.cuda.empty_cache()
    del ag

    if not probs_list:
        return None

    pv = np.stack(probs_list, 0)
    gv = np.stack(gts_list, 0).astype(np.float32)
    pb = (pv > 0.5).astype(np.float32)

    # Dice
    dice = compute_dice(pb, gv)

    # Components
    lb, nc = scipy_label(pb.astype(np.uint8))

    # Breakpoints
    bp = count_bp(pb, gv)

    # clDice
    skel_size = min(SKEL_SIZE, eval_size)
    print(f"  clDice...", end="", flush=True)
    ps2 = np.zeros((pb.shape[0], skel_size, skel_size), dtype=np.uint8)
    gs2 = np.zeros((gv.shape[0], skel_size, skel_size), dtype=np.uint8)
    for z in range(pb.shape[0]):
        ps2[z] = cv2.resize(pb[z], (skel_size, skel_size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        gs2[z] = cv2.resize(gv[z], (skel_size, skel_size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    cld = compute_cldice(ps2, gs2)
    print(f" {cld:.4f}")
    del ps2, gs2

    # HD95
    print(f"  HD95...", end="", flush=True)
    hd = compute_hd95(pb, gv)
    print(f" {hd:.2f}")

    del pv, gv, pb
    gc.collect()

    return {"p": pid, "dice": dice, "cldice": cld, "hd95": hd, "comp": nc, "bp": bp}


# ==========================================
# 主程式
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="統一評估公開 + 醫院 eval 資料")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--public-eval-dir", default=PUBLIC_EVAL_DIR)
    parser.add_argument("--hospital-eval-dir", default=HOSPITAL_EVAL_DIR)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--tta2", action="store_true", help="2-view TTA")
    parser.add_argument("--eval-size", type=int, default=EVAL_SIZE, help="指標計算解析度")
    args = parser.parse_args()

    tta_mode = "none" if args.no_tta else ("2" if args.tta2 else "4")
    tta_label = {"none": "No TTA", "2": "2-view TTA", "4": "4-view TTA"}[tta_mode]
    weight_name = os.path.basename(args.weights).replace(".pth", "")

    # === 載入模型 ===
    device = "cuda"
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')
    model = CoSegV6(build_sam2("sam2_hiera_l.yaml", SAM2_CHECKPOINT, mode="train")).to(device)
    sd = torch.load(args.weights, map_location=device)
    model.load_state_dict(
        {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()},
        strict=True
    )
    model.eval()
    del sd; gc.collect()

    pm = np.array([123.675, 116.280, 103.530], dtype=np.float32)
    ps = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    # === 掃描所有病人 ===
    public_patients = scan_patients(args.public_eval_dir, "public")
    hospital_patients = scan_patients(args.hospital_eval_dir, "hospital")
    all_patients = public_patients + hospital_patients

    print("=" * 65)
    print(f"🧠 CoSeg eval — {weight_name} ({tta_label})")
    print(f"   Eval size: {args.eval_size}")
    print(f"   公開 eval:  {len(public_patients)} patients")
    for pid, _, _, _ in public_patients:
        print(f"     - {pid}")
    print(f"   醫院 eval:  {len(hospital_patients)} patients")
    for pid, _, _, _ in hospital_patients:
        print(f"     - {pid}")
    print("=" * 65)

    if not all_patients:
        print("❌ 沒有找到任何 eval 病人")
        return

    # === 逐一評估 ===
    results_public = []
    results_hospital = []

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for pid, idir, mdir, source in all_patients:
            print(f"\n{'─'*50}")
            print(f"📋 [{source.upper()}] {pid}")
            print(f"{'─'*50}")

            r = eval_patient(model, pid, idir, mdir, pm, ps, device, tta_mode, args.eval_size)
            if r is None:
                print(f"  ⚠️ 跳過")
                continue

            r["source"] = source
            if source == "public":
                results_public.append(r)
            else:
                results_hospital.append(r)

            print(f"  📊 Dice={r['dice']:.4f} clDice={r['cldice']:.4f} HD95={r['hd95']:.2f} Comp={r['comp']} BP={r['bp']}")

    # === 總結 ===
    def print_summary(label, results):
        if not results:
            return
        ds = [r['dice'] for r in results]
        cs = [r['cldice'] for r in results]
        hs = [r['hd95'] for r in results if r['hd95'] != float('inf')]
        comps = [r['comp'] for r in results]
        bps = [r['bp'] for r in results]

        print(f"\n  [{label}] ({len(results)} patients)")
        print(f"  {'指標':<12s} {'Mean':>8s} {'Std':>8s}")
        print(f"  {'-'*30}")
        print(f"  {'Dice':<12s} {np.mean(ds):>8.4f} {np.std(ds):>8.4f}")
        print(f"  {'clDice':<12s} {np.mean(cs):>8.4f} {np.std(cs):>8.4f}")
        if hs:
            print(f"  {'HD95':<12s} {np.mean(hs):>8.2f} {np.std(hs):>8.2f}")
        print(f"  {'Components':<12s} {np.mean(comps):>8.1f} {np.std(comps):>8.1f}")
        print(f"  {'Breakpoints':<12s} {np.mean(bps):>8.1f} {np.std(bps):>8.1f}")

        print(f"\n  Per-patient:")
        for r in results:
            print(f"    {r['p']}: Dice={r['dice']:.4f} clDice={r['cldice']:.4f} "
                  f"HD95={r['hd95']:.2f} Comp={r['comp']} BP={r['bp']}")

    all_results = results_public + results_hospital

    print(f"\n{'='*65}")
    print(f"📊 評估總結 [{weight_name}] ({tta_label})")
    print(f"{'='*65}")

    print_summary("公開 Test", results_public)
    print_summary("醫院 Test", results_hospital)

    # 總體
    if results_public and results_hospital:
        print_summary("全部 (公開+醫院)", all_results)

    # 參考比較
    print(f"\n  📋 參考:")
    print(f"     v7  公開 test:  Dice=0.7874  clDice=0.8287  HD95=1.91")
    print(f"     v8  公開 test:  Dice=0.7861  clDice=0.8319  HD95=1.91")
    print(f"     DentalSeg:      Dice=0.7589  clDice=0.8577  HD95=2.16")

    if results_public:
        dp = np.mean([r['dice'] for r in results_public])
        print(f"     本次公開 test:  Dice={dp:.4f}")
    if results_hospital:
        dh = np.mean([r['dice'] for r in results_hospital])
        print(f"     本次醫院 test:  Dice={dh:.4f}")

    print(f"{'='*65}")


if __name__ == "__main__":
    main()
