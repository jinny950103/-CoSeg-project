#!/usr/bin/env python3
"""Ray network that also gets the bridge: image where the wall is faint, interpolation where it is gone.

canal_raynet_exam.py showed the plain ray network beats the bridge while a faded wall still has
some signal, but falls apart (Dice ~0.5) once the wall is gone, where the bridge still gets ~0.8.
This network gets both: the section image (polar, 5 planes) plus, per direction, the bridge radius
interpolated from the nearest clear sections on both sides of the gap (this patient only) and how
far away they are. Training mixes
  - clear sections with a simulated gap around them: image faded by a random amount (down to no
    wall at all) and the bridge computed with the block hidden;
  - partial sections as they are (real faint walls, real gaps), scored on their measured directions.
Directions the strict labeller did not measure stay out of the loss. Same 5 patient folds.

    <venv>/python scripts/canal_raynet_bridge.py train --out outputs/raynet_bridge
    <venv>/python scripts/canal_raynet_bridge.py exam  --out outputs/raynet_bridge --contrast 0.35 0.15 0
"""
import argparse
from case_lists import load_cases
import glob
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
import canal_edge_gt as E  # noqa: E402
import canal_gap_fill as G  # noqa: E402
import canal_raynet as RN  # noqa: E402
from canal_raynet_exam import patches, star_dice  # noqa: E402

ANG = G.ANG
SIG_PRIOR = 0.15          # mm, width of the bridge bump in the prior channel


def bridge_prior(d, k, sp, known, max_mm):
    """Bridge radii for section k from its own centre, from the nearest known sections on both sides.

    One side only (track ends) -> that side's contour. Returns (radii, distance to nearest end mm) or None.
    """
    R, C, U, W, K = d["r_agree"], d["c"], d["u"], d["w"], d["click"]
    L = known[known < k]; Rr = known[known > k]
    a = L[-1] if len(L) and (k - L[-1]) * 0.4 <= max_mm else None
    b = Rr[0] if len(Rr) and (Rr[0] - k) * 0.4 <= max_mm else None
    if a is None and b is None:
        return None
    if a is None or b is None:
        a = b = a if b is None else b
        t = 0.0
    else:
        t = (k - a) / (b - a)
    ca = G.wall_centre(C[a], U[a], W[a], R[a], sp); cb = G.wall_centre(C[b], U[b], W[b], R[b], sp)
    c = K[k] + (1 - t) * (ca - K[a]) + t * (cb - K[b])
    rk = (1 - t) * E.fill_angles(ANG, R[a]) + t * E.fill_angles(ANG, R[b])
    o = np.array([(c - C[k]) @ U[k], (c - C[k]) @ W[k]]) * sp
    py = o[0] + np.cos(ANG) * rk; px = o[1] + np.sin(ANG) * rk
    th = np.mod(np.arctan2(px, py), 2 * np.pi); rr = np.hypot(px, py)
    order = np.argsort(th)
    dist = min(abs(k - a), abs(b - k)) * 0.4
    return np.interp(ANG, th[order], rr[order], period=2 * np.pi), dist


def prior_channels(prior, dist, r):
    """(B, 3, A, R): gaussian bump at the bridge radius, distance to the end, has-bridge flag."""
    B = len(prior)
    has = np.isfinite(prior).all(1)
    pr = np.where(has[:, None], prior, 0).astype(np.float32)
    rr = r.cpu().numpy()
    bump = np.exp(-0.5 * ((rr[None, None] - pr[:, :, None]) / SIG_PRIOR) ** 2) * has[:, None, None]
    dd = np.broadcast_to((np.where(has, dist, 0) / 8.0)[:, None, None], bump.shape)
    hh = np.broadcast_to(has[:, None, None].astype(np.float32), bump.shape)
    return np.stack([bump, dd, hh], 1).astype(np.float32)


def gauss_blur(x, sig):
    """Separable gaussian over the last three dims (planes, rows, cols), sig in voxels per dim."""
    for dim, s in zip((2, 3, 4), sig):
        n = int(np.ceil(3 * s)); k = torch.exp(-0.5 * (torch.arange(-n, n + 1, device=x.device) / s) ** 2)
        k = (k / k.sum()).view(1, 1, -1, 1, 1) if dim == 2 else (k / k.sum()).view(1, 1, 1, -1, 1) if dim == 3 \
            else (k / k.sum()).view(1, 1, 1, 1, -1)
        pad = [0, 0, 0, 0, 0, 0]; pad[2 * (4 - dim)] = pad[2 * (4 - dim) + 1] = n
        x = F.conv3d(F.pad(x, pad, mode="replicate"), k)
    return x


def fade(x, c, noise):
    """Patch version of canal_gap_fill.degrade: x (B, P, H, W) normalised, c (B,) contrast kept."""
    x5 = x[:, None]
    m = gauss_blur(x5, (2.0 / 0.4, 2.0 / 0.1, 2.0 / 0.1))
    y = m + c.view(-1, 1, 1, 1, 1) * (x5 - m) + noise * torch.randn_like(x5)
    return gauss_blur(y, (0.2 / 0.4, 0.2 / 0.1, 0.2 / 0.1))[:, 0]


def load_rays(pattern, skip):
    out = {}
    for npz in sorted(glob.glob(pattern)):
        cid, sp, sides = G.load_case(npz)
        if cid not in skip:
            out[int(cid)] = (sp, sides)
    return out


def build_samples(meta, Y, rays, max_mm, n_sim, rng):
    """Rows: (data row, prior[72], dist, simulated?)."""
    cov = np.isfinite(Y).mean(1)
    row_of = {tuple(m[:3]): j for j, m in enumerate(meta)}
    rows, pri, dst, sim = [], [], [], []
    for cid, (sp, sides) in rays.items():
        for s, d in sides.items():
            good = np.where(np.isfinite(d["r_agree"]).mean(1) >= 0.5)[0]
            for k in range(len(d["r_agree"])):
                j = row_of.get((cid, s, k))
                if j is None or cov[j] == 0:
                    continue
                if cov[j] < 0.5:                       # real partial section, real ends
                    bp = bridge_prior(d, k, sp, good, max_mm)
                    rows.append(j); sim.append(False)
                    pri.append(bp[0] if bp else np.full(72, np.nan)); dst.append(bp[1] if bp else 0)
                    continue
                for _ in range(n_sim):                 # clear section inside a simulated gap
                    Lb = int(rng.integers(1, int(max_mm / 0.4) + 1)); off = int(rng.integers(0, Lb))
                    blk = np.arange(k - off, k - off + Lb)
                    bp = bridge_prior(d, k, sp, np.setdiff1d(good, blk), max_mm)
                    rows.append(j); sim.append(True)
                    pri.append(bp[0] if bp else np.full(72, np.nan)); dst.append(bp[1] if bp else 0)
    return np.array(rows), np.array(pri, np.float32), np.array(dst, np.float32), np.array(sim)


def train(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    z = np.load(args.data); X, Y, meta = z["X"], z["Y"], z["meta"]
    rays = load_rays(args.rays, args.skip)
    rng = np.random.default_rng(args.seed)
    S_row, S_pri, S_dst, S_sim = build_samples(meta, Y, rays, args.max_mm, args.n_sim, rng)
    print(f"samples {len(S_row)}  simulated {S_sim.sum()}  real partial {(~S_sim).sum()}  "
          f"no bridge {np.mean(~np.isfinite(S_pri).all(1)):.2f}", flush=True)
    grid, r = RN.polar_grid(X.shape[-1], 0.1, dev)
    for fold in args.folds:
        torch.manual_seed(args.seed + fold)
        fo = meta[S_row, 3]
        tr = np.where(fo != fold)[0]; va = np.where(fo == fold)[0]
        net = RN.Net(X.shape[1] + 3).to(dev)
        opt = torch.optim.AdamW(net.parameters(), args.lr, weight_decay=1e-4)
        steps = args.epochs * (len(tr) // args.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
        t0 = time.time()
        for ep in range(args.epochs):
            net.train(); perm = rng.permutation(tr); tot = 0
            for k in range(0, len(perm) - args.bs + 1, args.bs):
                idx = perm[k:k + args.bs]
                x = torch.from_numpy(X[S_row[idx]]).float().to(dev); y = torch.from_numpy(Y[S_row[idx]]).to(dev)
                # simulated gaps: fade the wall by a random amount, 40% of them all the way to nothing
                c = rng.uniform(0, 1, len(idx)); c[rng.random(len(idx)) < 0.4] = 0
                c = torch.from_numpy(np.where(S_sim[idx], c, 1.0)).float().to(dev)
                x = torch.where(c.view(-1, 1, 1, 1) < 1, fade(x, c, args.noise), x)
                pc = torch.from_numpy(prior_channels(S_pri[idx], S_dst[idx], r)).to(dev)
                sh = int(rng.integers(RN.N_ANG))
                if rng.random() < 0.5:
                    x = x.flip(1)
                x = x * float(rng.uniform(0.85, 1.15)) + float(rng.uniform(-0.05, 0.05))
                xp = torch.cat([RN.to_polar(x, grid), pc], 1).roll(sh, 2); y = y.roll(sh, 1)
                lr_, lv = net(xp)
                m = torch.isfinite(y)
                tgt = torch.clamp(torch.round(torch.nan_to_num(y) / RN.DR), 0, len(r) - 1).long()
                ce = F.cross_entropy(lr_[m], tgt[m])
                rad = (lr_.softmax(-1) * r).sum(-1)
                loss = ce + 5 * (rad[m] - y[m]).abs().mean()
                opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += float(loss)
            if ep % 10 == 9 or ep == args.epochs - 1:
                print(f"fold {fold} ep {ep + 1} loss {tot / (len(perm) // args.bs):.3f} ({time.time() - t0:.0f}s)", flush=True)
        torch.save(net.state_dict(), out / f"net_fold{fold}.pt")
    (out / "args.json").write_text(json.dumps(vars(args)))


def predict(net, X, prior, dist, grid, r):
    net.eval()
    with torch.no_grad():
        x = torch.cat([RN.to_polar(torch.from_numpy(X).float().to(r.device), grid),
                       torch.from_numpy(prior_channels(prior, dist, r)).to(r.device)], 1)
        lr_, _ = net(x)
        return (lr_.softmax(-1) * r).sum(-1).cpu().numpy()


def exam(args):
    """Same blocks and fading as canal_raynet_exam.py, at several contrasts."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = json.load(open(args.splits))
    fold_of = {c[5:]: k for k, f in enumerate(folds) for c in f["val"] if c.startswith("HOSP_")}
    nets = {}
    for f in range(5):
        n_ = RN.Net(8).to(dev); n_.load_state_dict(torch.load(Path(args.out) / f"net_fold{f}.pt", map_location=dev))
        nets[f] = n_
    grid, r = RN.polar_grid(64, 0.1, dev)
    rows = []
    for cid, (sp, sides) in load_rays(args.rays, args.skip).items():
        cid = f"{cid:010d}"; rng = np.random.default_rng(int(cid))
        vol = np.asanyarray(nib.load(G.IMG.format(cid)).dataobj).astype(np.float32)
        p1, p99 = np.percentile(vol[::4, ::4, ::4], [1, 99])
        vs = ndimage.gaussian_filter(vol, 0.2 / sp)
        net = nets[fold_of[cid]]
        for s, d in sides.items():
            R = d["r_agree"]; n = len(R)
            good = np.isfinite(R).mean(1) >= 0.5; gi = np.where(good)[0]
            for L in args.L:
                i = 2
                while i + L + 2 <= n:
                    blk = np.arange(i, i + L)
                    if not good[i - 2:i + L + 2].all():
                        i += 1; continue
                    known = np.setdiff1d(gi, blk)
                    bp = [bridge_prior(d, k, sp, known, args.max_mm) for k in blk]
                    pri = np.array([b[0] for b in bp], np.float32); dst = np.array([b[1] for b in bp], np.float32)
                    for cc in args.contrast:
                        ga = argparse.Namespace(degrade_contrast=cc, degrade_noise=args.noise_hu, degrade_r_mm=4.0)
                        v = G.degrade(vs, d["c"][blk], sp, ga, rng)
                        pr = predict(net, patches(v, d, blk, sp, p1, p99), pri, dst, grid, r)
                        for j, k in enumerate(blk):
                            ok = np.isfinite(R[k]); truth = E.fill_angles(ANG, R[k])
                            rows.append({"case": cid[-4:], "L": L, "c": cc, "sec": int(k),
                                         "net_err": float(np.median(np.abs(pr[j] - R[k])[ok])),
                                         "net_dice": float(star_dice(pr[j], truth)),
                                         "bridge_err": float(np.median(np.abs(pri[j] - R[k])[ok])),
                                         "bridge_dice": float(star_dice(pri[j], truth))})
                    i += L + 10
        print(cid[-4:], len(rows), flush=True)
    (Path(args.out) / "exam.json").write_text(json.dumps(rows))
    for L in args.L:
        for cc in args.contrast:
            x = [q for q in rows if q["L"] == L and q["c"] == cc]
            med = lambda k: np.median([q[k] for q in x])
            print(f"L={L} ({L * 0.4:.0f}mm) contrast {cc:.2f}  n={len(x)}  net err {med('net_err'):.3f} Dice {med('net_dice'):.2f}"
                  f"  |  bridge err {med('bridge_err'):.3f} Dice {med('bridge_dice'):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["train", "exam"])
    ap.add_argument("--data", default="outputs/raynet/data.npz")
    ap.add_argument("--rays", default="outputs/sparse_label_v6_rays/chunk_0*/*_rays.npz")
    ap.add_argument("--splits", default="data/nnUNet_raw/Dataset504_MixedCanalClean/splits_final.json")
    ap.add_argument("--skip", nargs="*", default=load_cases())
    ap.add_argument("--out", default="outputs/raynet_bridge")
    ap.add_argument("--max_mm", type=float, default=8.0, help="bridge 兩端最遠可以離多遠")
    ap.add_argument("--n_sim", type=int, default=2, help="每張清楚截面模擬幾種缺口")
    ap.add_argument("--noise", type=float, default=0.018, help="淡化時加的雜訊(正規化後,約 40 HU)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--contrast", type=float, nargs="+", default=[0.35, 0.15, 0.0])
    ap.add_argument("--noise_hu", type=float, default=40)
    args = ap.parse_args()
    train(args) if args.mode == "train" else exam(args)


if __name__ == "__main__":
    main()
