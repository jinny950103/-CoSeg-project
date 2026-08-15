"""
eval_v7.py — CoSeg v7 評估（4-view TTA + 全指標）
===================================================
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
EVAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data/public_data/eval")
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "outputs/coseg_v7_best.pth")
SAM2_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
EVAL_SIZE, SKEL_SIZE, FULL_RES = 256, 128, 1024


def compute_dice(p, g, s=1e-5):
    i = np.sum(p*g); return (2*i+s)/(np.sum(p)+np.sum(g)+s)

def skeletonize_3d_safe(vol):
    try:
        from skimage.morphology import skeletonize_3d
        return skeletonize_3d(vol.astype(np.uint8)).astype(np.float32)
    except:
        from skimage.morphology import skeletonize
        s = np.zeros_like(vol, dtype=np.float32)
        for z in range(vol.shape[0]):
            if vol[z].sum() > 0: s[z] = skeletonize(vol[z].astype(bool)).astype(np.float32)
        return s

def compute_cldice(pred, gt, smooth=1e-5):
    pb, gb = (pred>0).astype(np.uint8), (gt>0).astype(np.uint8)
    if pb.sum()==0 and gb.sum()==0: return 1.0
    if pb.sum()==0 or gb.sum()==0: return 0.0
    sp, sg = skeletonize_3d_safe(pb), skeletonize_3d_safe(gb)
    if sp.sum()==0 or sg.sum()==0: return 0.0
    tp = (sp*gb).sum()/(sp.sum()+smooth); ts = (sg*pb).sum()/(sg.sum()+smooth)
    r = float(2*tp*ts/(tp+ts+smooth)); del sp, sg; gc.collect(); return r

def compute_hd95(pred, gt):
    pb, gb = (pred>0).astype(np.uint8), (gt>0).astype(np.uint8)
    if pb.sum()==0 or gb.sum()==0: return float('inf')
    ps = pb ^ binary_erosion(pb).astype(np.uint8)
    gs = gb ^ binary_erosion(gb).astype(np.uint8)
    if ps.sum()==0 or gs.sum()==0: return float('inf')
    dp = distance_transform_edt(~pb.astype(bool))
    dg = distance_transform_edt(~gb.astype(bool))
    ad = np.concatenate([dg[ps>0], dp[gs>0]]); del dp, dg; gc.collect()
    return float(np.percentile(ad, 95))

def count_bp(pred, gt):
    gr = np.where(gt.sum(axis=(1,2))>0)[0]; bp=0
    if len(gr)>0:
        pz = pred[gr[0]:gr[-1]+1].sum(axis=(1,2))>0; ic=False
        for i,h in enumerate(pz):
            if h and not ic:
                if i>0: bp+=1
                ic=True
            elif not h and ic: ic=False
    return bp

def predict_one(model, img_3ch, pm, ps, device):
    t = torch.tensor((img_3ch-pm)/(ps+1e-8)).permute(2,0,1).unsqueeze(0).float().to(device)
    o = model(x=t); s = F.interpolate(o[1], (FULL_RES,FULL_RES), mode='bilinear', align_corners=False)
    return torch.sigmoid(s[:,0,:,:])[0].cpu().numpy()

def predict_tta4(model, img, pm, ps, device):
    """4-view TTA"""
    p0 = predict_one(model, img, pm, ps, device)
    p1 = np.flip(predict_one(model, np.flip(img, 1).copy(), pm, ps, device), 1).copy()
    p2 = np.flip(predict_one(model, np.flip(img, 0).copy(), pm, ps, device), 0).copy()
    p3 = np.flip(np.flip(predict_one(model, np.flip(np.flip(img, 1), 0).copy(), pm, ps, device), 0), 1).copy()
    return (p0 + p1 + p2 + p3) / 4.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--tta2", action="store_true", help="2-view TTA（只水平翻轉）")
    args = parser.parse_args()

    device = "cuda"
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')
    model = CoSegV6(build_sam2("sam2_hiera_l.yaml", SAM2_CHECKPOINT, mode="train")).to(device)
    sd = torch.load(args.weights, map_location=device)
    model.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k,v in sd.items()}, strict=True)
    model.eval()

    pm = np.array([123.675, 116.280, 103.530], dtype=np.float32)
    ps = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    patients = sorted([p for p in os.listdir(EVAL_DATA_DIR) if p.startswith("Patient_")])
    tta = "No TTA" if args.no_tta else ("2-view" if args.tta2 else "4-view")

    print("="*60)
    print(f"🧠 CoSeg v7 eval ({tta} TTA)")
    print("="*60)

    results = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for pid in patients:
            idir = os.path.join(EVAL_DATA_DIR, pid, "image_1024")
            mdir = os.path.join(EVAL_DATA_DIR, pid, "mask_sem_1024")
            if not os.path.isdir(idir): continue
            ifs = sorted([f for f in os.listdir(idir) if f.endswith(".png")])
            if not ifs: continue

            ag = {}
            for f in ifs:
                g = cv2.imread(os.path.join(idir,f), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                ag[int(f.replace(".png","").rsplit("_",1)[-1])] = g

            probs, gts = [], []
            for f in tqdm(ifs, desc=f"推論 {pid}"):
                idx = int(f.replace(".png","").rsplit("_",1)[-1])
                c = ag[idx]; p_ = ag.get(idx-1, c); n_ = ag.get(idx+1, c)
                img = np.stack([p_, c, n_], axis=-1)
                gt = np.load(os.path.join(mdir, f.replace(".png",".npy"))).astype(np.float32)

                if args.no_tta:
                    prob = predict_one(model, img, pm, ps, device)
                elif args.tta2:
                    p0 = predict_one(model, img, pm, ps, device)
                    p1 = np.flip(predict_one(model, np.flip(img,1).copy(), pm, ps, device), 1).copy()
                    prob = (p0 + p1) / 2.0
                else:
                    prob = predict_tta4(model, img, pm, ps, device)

                probs.append(cv2.resize(prob, (EVAL_SIZE,EVAL_SIZE), interpolation=cv2.INTER_LINEAR))
                gts.append((cv2.resize(gt, (EVAL_SIZE,EVAL_SIZE), interpolation=cv2.INTER_NEAREST)>0).astype(np.uint8))

            torch.cuda.empty_cache(); del ag
            pv = np.stack(probs,0); gv = np.stack(gts,0).astype(np.float32)
            pb = (pv>0.5).astype(np.float32)

            dice = compute_dice(pb, gv)
            lb, nc = scipy_label(pb.astype(np.uint8))
            conn = (np.bincount(lb.ravel())[1:].max()/pb.sum()) if pb.sum()>0 and nc>0 else 0
            bp = count_bp(pb, gv)

            # clDice
            print(f"  clDice...", end="", flush=True)
            ps2 = np.zeros((pb.shape[0],SKEL_SIZE,SKEL_SIZE), dtype=np.uint8)
            gs2 = np.zeros((gv.shape[0],SKEL_SIZE,SKEL_SIZE), dtype=np.uint8)
            for z in range(pb.shape[0]):
                ps2[z] = cv2.resize(pb[z], (SKEL_SIZE,SKEL_SIZE), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                gs2[z] = cv2.resize(gv[z], (SKEL_SIZE,SKEL_SIZE), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            cld = compute_cldice(ps2, gs2); print(f" {cld:.4f}"); del ps2, gs2

            print(f"  HD95...", end="", flush=True)
            hd = compute_hd95(pb, gv); print(f" {hd:.2f}")

            r = {"p":pid, "dice":dice, "cldice":cld, "hd95":hd, "comp":nc, "bp":bp}
            results.append(r)
            print(f"\n  📊 {pid}: Dice={dice:.4f} clDice={cld:.4f} HD95={hd:.2f} Comp={nc} BP={bp}")
            del pv, gv, pb; gc.collect()

    # 總結
    print(f"\n{'='*60}\n📊 總結\n{'='*60}")
    ds = [r['dice'] for r in results]; cs = [r['cldice'] for r in results]
    hs = [r['hd95'] for r in results if r['hd95']!=float('inf')]
    print(f"  {'指標':<12s} {'Mean':>8s} {'Std':>8s}")
    print(f"  {'-'*30}")
    print(f"  {'Dice':<12s} {np.mean(ds):>8.4f} {np.std(ds):>8.4f}")
    print(f"  {'clDice':<12s} {np.mean(cs):>8.4f} {np.std(cs):>8.4f}")
    if hs: print(f"  {'HD95':<12s} {np.mean(hs):>8.2f} {np.std(hs):>8.2f}")
    print(f"  {'Components':<12s} {np.mean([r['comp'] for r in results]):>8.1f}")
    print(f"  {'Breakpoints':<12s} {np.mean([r['bp'] for r in results]):>8.1f}")
    print(f"\n  Per-patient:")
    for r in results:
        print(f"    {r['p']}: Dice={r['dice']:.4f} clDice={r['cldice']:.4f} HD95={r['hd95']:.2f} Comp={r['comp']} BP={r['bp']}")
    print(f"\n  📋 比較:")
    print(f"     DentalSeg:      Dice=0.7589  clDice=0.8577  HD95=2.16")
    print(f"     v5 TTA (5人):   Dice=0.7608  clDice=0.7930  HD95=3.97")
    print(f"     v6 TTA (5人):   Dice=0.7569  clDice=0.7928  HD95=3.14")
    print(f"     v6 TTA (8人):   Dice=0.7616  clDice=0.7982  HD95=5.71")
    if hs: print(f"     v7 (8人):       Dice={np.mean(ds):.4f}  clDice={np.mean(cs):.4f}  HD95={np.mean(hs):.2f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
