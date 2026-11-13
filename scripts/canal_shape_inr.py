#!/usr/bin/env python3
"""Learned canal shape (neural implicit, NISF-style auto-decoder) to predict hidden sections.

The ray networks read the image, and the image has no answer inside a real gap, so a fade exam
does not tell how they behave there. This model never looks at the image: it learns, from the
clear sections (>= 50% of directions measured) of the training-fold patients, how the canal's
centre and wall radius vary along the track, as
    centre offset from the doctor's track (u, w mm)  = f(s, z)
    wall radius from that centre (mm)                = g(s, angle, z)
with s = position along the track (0..1) and z a small code per canal. For a new canal the
networks are frozen and only z is fitted to the sections that are visible; the hidden sections
are then read off. Width still varies per direction and along the canal; nothing comes from
other datasets, only from this hospital's patients.

Exam: hide blocks of L clear sections (as canal_raynet_exam.py), fit z on the rest of that canal,
score the hidden sections against the strict radii, next to the bridge.

    <venv>/python scripts/canal_shape_inr.py --out outputs/shape_inr
"""
import argparse
from case_lists import load_cases
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import canal_edge_gt as E  # noqa: E402
import canal_gap_fill as G  # noqa: E402
from canal_raynet_bridge import bridge_prior, load_rays  # noqa: E402
from canal_raynet_exam import star_dice  # noqa: E402

ANG = G.ANG
N_FREQ = 8


def plane(v, U, W, sp):
    return np.stack([v @ U, v @ W], -1) * sp


def contour_from(c_from, d, k, sp, cen, ang, rad):
    """Points (cen + dir(ang) * rad, in-plane mm relative to the track click) -> radii on ANG from c_from."""
    U, W = d["u"][k], d["w"][k]
    o = cen - plane(c_from - d["click"][k], U, W, sp)
    py = o[0] + np.cos(ang) * rad; px = o[1] + np.sin(ang) * rad
    th = np.mod(np.arctan2(px, py), 2 * np.pi); rr = np.hypot(px, py)
    order = np.argsort(th)
    return np.interp(ANG, th[order], rr[order], period=2 * np.pi)


def canal_samples(d, sp, use):
    """Training samples for one canal from its sections `use`: centre offsets and (angle, radius) wall points."""
    R = d["r_agree"]; n = len(R)
    cs, cy, ws = [], [], []
    for k in use:
        wc = G.wall_centre(d["c"][k], d["u"][k], d["w"][k], R[k], sp)
        cen = plane(wc - d["click"][k], d["u"][k], d["w"][k], sp)
        cs.append([k / (n - 1)]); cy.append(cen)
        ok = np.isfinite(R[k])
        rel = plane(d["c"][k] - wc, d["u"][k], d["w"][k], sp)
        py = rel[0] + np.cos(ANG[ok]) * R[k][ok]; px = rel[1] + np.sin(ANG[ok]) * R[k][ok]
        th = np.arctan2(px, py); rr = np.hypot(px, py)
        ws.append(np.stack([np.full(ok.sum(), k / (n - 1)), np.cos(th), np.sin(th), rr], 1))
    return np.array(cs, np.float32), np.array(cy, np.float32), np.concatenate(ws).astype(np.float32)


class Shape(nn.Module):
    def __init__(self, zdim=16, h=128):
        super().__init__()
        fin = 2 * N_FREQ + 1 + zdim
        self.cen = nn.Sequential(nn.Linear(fin, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))
        self.rad = nn.Sequential(nn.Linear(fin + 6, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))

    @staticmethod
    def pe(s):
        f = torch.arange(1, N_FREQ + 1, device=s.device) * np.pi
        return torch.cat([s, torch.sin(s * f), torch.cos(s * f)], -1)

    def centre(self, s, z):
        return self.cen(torch.cat([self.pe(s), z], -1))

    def radius(self, s, cth, sth, z):
        ang = torch.cat([cth, sth, 2 * cth * sth, cth ** 2 - sth ** 2, cth * (4 * cth ** 2 - 3), sth * (3 - 4 * sth ** 2)], -1)
        return nn.functional.softplus(self.rad(torch.cat([self.pe(s), ang, z], -1)))[:, 0]


def loss_on(net, z, cs, cy, ws, idx_c, idx_w):
    pc = net.centre(cs, z[idx_c])
    pr = net.radius(ws[:, :1], ws[:, 1:2], ws[:, 2:3], z[idx_w])
    return (pc - cy).abs().mean() + (pr - ws[:, 3]).abs().mean()


def to_t(*a, dev):
    return [torch.from_numpy(x).to(dev) for x in a]


def fit(net, canals, dev, steps, lr_net, lr_z, zdim, train_net):
    """canals: list of (cs, cy, ws). Returns the codes (and trains net if train_net)."""
    cs = np.concatenate([c[0] for c in canals]); cy = np.concatenate([c[1] for c in canals])
    ws = np.concatenate([c[2] for c in canals])
    ic = np.concatenate([np.full(len(c[0]), i) for i, c in enumerate(canals)])
    iw = np.concatenate([np.full(len(c[2]), i) for i, c in enumerate(canals)])
    cs, cy, ws = to_t(cs, cy, ws, dev=dev); ic, iw = to_t(ic, iw, dev=dev)
    z = torch.zeros(len(canals), zdim, device=dev, requires_grad=True)
    params = [{"params": [z], "lr": lr_z}] + ([{"params": net.parameters(), "lr": lr_net}] if train_net else [])
    opt = torch.optim.Adam(params)
    for p in net.parameters():
        p.requires_grad_(train_net)
    for _ in range(steps):
        loss = loss_on(net, z, cs, cy, ws, ic, iw) + 1e-3 * (z ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return z.detach(), float(loss)


def predict_section(net, z, d, k, sp, dev):
    n = len(d["r_agree"])
    with torch.no_grad():
        s = torch.full((len(ANG), 1), k / (n - 1), device=dev)
        zz = z[None].expand(len(ANG), -1)
        cen = net.centre(s[:1], zz[:1])[0].cpu().numpy()
        rad = net.radius(s, torch.from_numpy(np.cos(ANG)[:, None]).float().to(dev),
                         torch.from_numpy(np.sin(ANG)[:, None]).float().to(dev), zz).cpu().numpy()
    return contour_from(d["c"][k], d, k, sp, cen, ANG, rad)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rays", default="outputs/sparse_label_v6_rays/chunk_0*/*_rays.npz")
    ap.add_argument("--splits", default="data/nnUNet_raw/Dataset504_MixedCanalClean/splits_final.json")
    ap.add_argument("--skip", nargs="*", default=load_cases())
    ap.add_argument("--out", default="outputs/shape_inr")
    ap.add_argument("--zdim", type=int, default=16)
    ap.add_argument("--train_steps", type=int, default=4000)
    ap.add_argument("--fit_steps", type=int, default=300)
    ap.add_argument("--L", type=int, nargs="+", default=[10, 20, 40], help="藏起的區塊長度(截面數,0.4mm/張)")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    folds = json.load(open(args.splits))
    fold_of = {c[5:]: k for k, f in enumerate(folds) for c in f["val"] if c.startswith("HOSP_")}
    rays = load_rays(args.rays, args.skip)
    canals = []          # (cid, side, d, sp, good indices)
    for cid, (sp, sides) in rays.items():
        for s, d in sides.items():
            good = np.where(np.isfinite(d["r_agree"]).mean(1) >= 0.5)[0]
            if len(good) >= 10:
                canals.append((f"{cid:010d}", s, d, sp, good))
    rows = []
    t0 = time.time()
    for fold in range(5):
        torch.manual_seed(fold)
        tr = [c for c in canals if fold_of[c[0]] != fold]; te = [c for c in canals if fold_of[c[0]] == fold]
        net = Shape(args.zdim).to(dev)
        _, l = fit(net, [canal_samples(d, sp, g) for _, _, d, sp, g in tr], dev, args.train_steps, 1e-3, 1e-2,
                   args.zdim, True)
        torch.save(net.state_dict(), out / f"net_fold{fold}.pt")
        print(f"fold {fold} train loss {l:.3f} ({time.time() - t0:.0f}s)", flush=True)
        for cid, s, d, sp, gi in te:
            R = d["r_agree"]; n = len(R)
            good = np.isfinite(R).mean(1) >= 0.5
            for L in args.L:
                i = 2
                while i + L + 2 <= n:
                    blk = np.arange(i, i + L)
                    if not good[i - 2:i + L + 2].all():
                        i += 1; continue
                    known = np.setdiff1d(gi, blk)
                    z, _ = fit(net, [canal_samples(d, sp, known)], dev, args.fit_steps, 0, 3e-2, args.zdim, False)
                    for k in blk:
                        ok = np.isfinite(R[k]); truth = E.fill_angles(ANG, R[k])
                        pr = predict_section(net, z[0], d, k, sp, dev)
                        row = {"case": cid[-4:], "side": int(s), "L": L, "sec": int(k),
                               "inr_err": float(np.median(np.abs(pr - R[k])[ok])),
                               "inr_dice": float(star_dice(pr, truth))}
                        bp = bridge_prior(d, k, sp, known, 20.0)
                        if bp is not None:
                            row["bridge_err"] = float(np.median(np.abs(bp[0] - R[k])[ok]))
                            row["bridge_dice"] = float(star_dice(bp[0], truth))
                        rows.append(row)
                    i += L + 10
        print(f"fold {fold} exam rows {len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    (out / "exam.json").write_text(json.dumps(rows))
    for L in args.L:
        x = [q for q in rows if q["L"] == L and "bridge_err" in q]
        med = lambda k: np.median([q[k] for q in x])
        print(f"L={L} ({L * 0.4:.0f}mm) n={len(x)}  shape-INR err {med('inr_err'):.3f} Dice {med('inr_dice'):.2f}"
              f"  |  bridge err {med('bridge_err'):.3f} Dice {med('bridge_dice'):.2f}")


if __name__ == "__main__":
    main()
