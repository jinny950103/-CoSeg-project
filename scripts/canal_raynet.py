#!/usr/bin/env python3
"""Ray network: predict the canal wall radius in each of 72 directions from the section image.

Trained only on this hospital's images and on the radii our strict edge detection kept
(canal_sparse_label.py); directions it did not measure are left out of the loss. Same 5 patient
folds as nnU-Net 504/505/506, so every patient is predicted by a model that never saw them.

Input: the section stack (canal_raynet_data.py) resampled to polar form (72 directions x 61
radii 0..3mm) on the GPU. A small CNN with circular padding over the direction axis gives, per
direction, a distribution over the 61 radii (soft-argmax = radius) and a "wall visible" logit
(was this direction measured by the strict labeller).
The output is star-shaped around the centre by construction: one radius per direction, so it
cannot spill into a large dark area the way a voxel model can; width still varies per direction
and per section.

    <venv>/python scripts/canal_raynet.py --data outputs/raynet/data.npz --out outputs/raynet/cv
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_ANG, R_MAX, DR = 72, 3.0, 0.05


def polar_grid(n_px, px_mm, device):
    r = torch.arange(0, R_MAX + 1e-6, DR, device=device)
    th = torch.arange(N_ANG, device=device) * 2 * np.pi / N_ANG
    half = (n_px - 1) / 2 * px_mm
    # patch axes: dim -2 = u (row), dim -1 = w (col); grid_sample wants (x=col, y=row) in [-1, 1]
    yy = (torch.cos(th)[:, None] * r[None]) / half
    xx = (torch.sin(th)[:, None] * r[None]) / half
    return torch.stack([xx, yy], -1)[None], r                     # (1, A, R, 2)


class Net(nn.Module):
    def __init__(self, cin, ch=64, depth=7):
        super().__init__()
        self.layers = nn.ModuleList()
        c = cin
        for k in range(depth):
            dil = [1, 1, 2, 4, 8, 1, 1][k % 7]
            self.layers.append(nn.Conv2d(c, ch, 3, dilation=(dil, 1)))
            self.layers.append(nn.BatchNorm2d(ch))
            c = ch
        self.head_r = nn.Conv2d(ch, 1, 1)
        self.head_v = nn.Conv2d(ch, 1, 1)

    def forward(self, x):                       # x (B, C, A, R)
        h = x
        for conv, bn in zip(self.layers[::2], self.layers[1::2]):
            d = conv.dilation[0]
            h = F.pad(h, (0, 0, d, d), mode="circular")         # directions wrap round
            h = F.pad(h, (1, 1, 0, 0))                          # radius: zero pad
            h = F.relu(bn(conv(h))) + (h[:, :, d:-d, 1:-1] if h.shape[1] == conv.out_channels else 0)
        logit_r = self.head_r(h)[:, 0]                          # (B, A, R)
        logit_v = self.head_v(h)[:, 0].mean(-1)                 # (B, A)
        return logit_r, logit_v


def to_polar(x, grid):
    return F.grid_sample(x, grid.expand(x.shape[0], -1, -1, -1), align_corners=True)   # (B, C, A, R)


def predict(net, X, grid, r, device, bs=512):
    net.eval(); out_r, out_v, out_p = [], [], []
    with torch.no_grad():
        for k in range(0, len(X), bs):
            x = to_polar(torch.from_numpy(X[k:k + bs]).float().to(device), grid)
            lr, lv = net(x)
            p = lr.softmax(-1)
            out_r.append((p * r).sum(-1).cpu().numpy()); out_v.append(torch.sigmoid(lv).cpu().numpy())
            out_p.append(p.max(-1).values.cpu().numpy())
    return np.concatenate(out_r), np.concatenate(out_v), np.concatenate(out_p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--px_mm", type=float, default=0.1)
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(args.data)
    X, Y, meta = z["X"], z["Y"], z["meta"]
    grid, r = polar_grid(X.shape[-1], args.px_mm, dev)
    rbin = torch.arange(len(r), device=dev)
    res_r = np.full(Y.shape, np.nan, np.float32); res_v = np.full(Y.shape, np.nan, np.float32)
    res_p = np.full(Y.shape, np.nan, np.float32)
    for fold in args.folds:
        torch.manual_seed(args.seed + fold); rng = np.random.default_rng(args.seed + fold)
        tr = np.where(meta[:, 3] != fold)[0]; va = np.where(meta[:, 3] == fold)[0]
        net = Net(X.shape[1]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), args.lr, weight_decay=1e-4)
        steps = args.epochs * (len(tr) // args.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
        t0 = time.time()
        for ep in range(args.epochs):
            net.train(); perm = rng.permutation(tr); tot = 0
            for k in range(0, len(perm) - args.bs + 1, args.bs):
                idx = np.sort(perm[k:k + args.bs])
                x = torch.from_numpy(X[idx]).float().to(dev); y = torch.from_numpy(Y[idx]).to(dev)
                # augment: rotate the section (roll directions), reverse along the canal, intensity
                sh = int(rng.integers(N_ANG))
                if rng.random() < 0.5:
                    x = x.flip(1)
                x = x * float(rng.uniform(0.85, 1.15)) + float(rng.uniform(-0.05, 0.05))
                xp = to_polar(x, grid).roll(sh, 2); y = y.roll(sh, 1)
                lr_, lv = net(xp)
                m = torch.isfinite(y)
                tgt = torch.clamp(torch.round(torch.nan_to_num(y) / DR), 0, len(r) - 1).long()
                ce = F.cross_entropy(lr_[m], tgt[m]) if m.any() else lr_.sum() * 0
                rad = (lr_.softmax(-1) * r).sum(-1)
                l1 = (rad[m] - y[m]).abs().mean() if m.any() else 0
                bce = F.binary_cross_entropy_with_logits(lv, m.float())
                loss = ce + 5 * l1 + 0.5 * bce
                opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += float(loss)
            if ep % 10 == 9 or ep == args.epochs - 1:
                pr, _, _ = predict(net, X[va], grid, r, dev)
                mv = np.isfinite(Y[va])
                print(f"fold {fold} ep {ep + 1} loss {tot / (len(perm) // args.bs):.3f} "
                      f"val |r err| median {np.median(np.abs(pr[mv] - Y[va][mv])):.3f} mm  ({time.time() - t0:.0f}s)", flush=True)
        res_r[va], res_v[va], res_p[va] = predict(net, X[va], grid, r, dev)
        torch.save(net.state_dict(), out / f"net_fold{fold}.pt")
    tag = "".join(map(str, args.folds))
    np.savez_compressed(out / f"cv_pred_f{tag}.npz", r=res_r, v=res_v, p=res_p, meta=meta, Y=Y)
    (out / "args.json").write_text(json.dumps(vars(args)))


if __name__ == "__main__":
    main()
