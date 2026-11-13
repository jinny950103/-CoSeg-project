#!/usr/bin/env python3
"""Package the canal dataset for the doctors to check: images, labels (0-6), ITK-SNAP label
description, a per-section list (where each section is, how it was labelled) and a README.

    <venv>/python scripts/make_doctor_dataset.py --out outputs/dataset_for_doctor_20261102
"""
import argparse
from case_lists import load_cases
import csv
import glob
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import canal_gap_fill as G  # noqa: E402

LABELS = [  # value, name, rgb
    (0, "背景(量到)", (0, 0, 0)),
    (1, "管(量到)", (40, 110, 220)),
    (2, "不確定(不計分)", (150, 150, 150)),
    (3, "管(推測-前後接起來)", (240, 140, 30)),
    (4, "推測管外一圈背景", (120, 70, 20)),
    (5, "管(外插)", (220, 40, 160)),
    (6, "外插管外一圈背景", (110, 20, 80)),
]

README = """下頜管資料集（給醫師檢查用）2026-11-02
=======================================

內容
- images/<病歷號>.nii.gz      醫院 CBCT（NIfTI，0.4 mm）
- labels/<病歷號>_label.nii.gz 標註，和影像同一座標，可直接疊在影像上
- itksnap_labels.txt          ITK-SNAP 標籤顏色與名稱（Segmentation > Import Label Descriptions）
- sections.csv                每一張截面的位置和標註方式，方便直接跳到要檢查的地方
- summary.csv                 每位病人各種標註的截面比例
- patient_review_20261102.pdf 每位病人一頁：原始影像 / 醫生標註 / 我們的標註

病人數：{n_cases} 位（另 1 位原始標註上下錯開約 14 mm，未放入，待醫師確認）

標籤
  0 背景（量到）
  1 管（量到）：影像上直接抓到管壁內緣
  2 不確定：不計分
  3 管（推測）：前後看得清楚的截面之間接起來（缺口最長 16 mm）
  4 推測管外一圈 0.4 mm 背景
  5 管（外插）：從最近看得到的截面往外推（多在管的頭尾）
  6 外插管外一圈 0.4 mm 背景

截面統計（沿醫生標註的軌跡，每 0.4 mm 一張，共 {n_sec} 張）
  量到   {m:5d} 張 {mp:5.1%}
  推測   {p:5d} 張 {pp:5.1%}
  外插   {e:5d} 張 {ep:5.1%}
  合計可用 100%

可信度（把看得清楚的截面藏起來再猜，和量到的管壁比 Dice；影像完全看不到的情況）
  推測（接起來）：缺口 4 mm 0.84、8 mm 0.78、16 mm 0.80
  外插：離已知截面 0–2 mm 0.90、2–4 mm 0.80、4–8 mm 0.77、8–16 mm 0.70
  對照：只在醫生點的位置畫一個固定大小的圓，約 0.77–0.79
  → 外插離已知截面超過約 2 mm 後，不比畫圓好，特別需要醫師確認。

sections.csv 欄位
  case       病歷號
  side       0/1 兩側下頜管
  sec        沿管的截面編號（0.4 mm 一張）
  kind       measured 量到 / predicted 推測 / extrapolated 外插
  extrap_mm  外插截面離最近已知截面的距離（mm）
  i,j,k      截面中心的體素座標（ITK-SNAP 游標可直接輸入）

建議檢查方式
  每位病人在 sections.csv 依 kind 各隨機抽幾張（例如 量到 10、推測 5、外插 5），
  跳到 i,j,k 看輪廓對不對，記錄 對/錯。
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="outputs/extrap_fill/{}_extrap.nii.gz")
    ap.add_argument("--track", default="outputs/track_fill/sections.json")
    ap.add_argument("--extrap", default="outputs/extrap_fill/sections.json")
    ap.add_argument("--rays", default="outputs/sparse_label_v6_rays/chunk_0*/*_rays.npz")
    ap.add_argument("--skip", nargs="*", default=load_cases())
    ap.add_argument("--out", default="outputs/dataset_for_doctor_20261102")
    ap.add_argument("--no_images", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    for sub in ("images", "labels"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    tr = {(s["case"], s["side"], s["sec"]): s for s in json.load(open(args.track))}
    ex = {(s["case"], s["side"], s["sec"]): s for s in json.load(open(args.extrap))}
    rows, summ = [], []
    for npz in sorted(glob.glob(args.rays)):
        cid, sp, sides = G.load_case(npz)
        if cid in args.skip:
            continue
        lab = nib.load(args.labels.format(cid))
        img = nib.load(G.IMG.format(cid))
        assert lab.shape == img.shape and np.allclose(lab.affine, img.affine), cid
        shutil.copyfile(args.labels.format(cid), out / "labels" / f"{cid}_label.nii.gz")
        if not args.no_images:
            shutil.copyfile(G.IMG.format(cid), out / "images" / f"{cid}.nii.gz")
        cnt = {"measured": 0, "predicted": 0, "extrapolated": 0}
        for s, d in sides.items():
            for k in range(len(d["click"])):
                key = (cid[-4:], s, k)
                t = tr[key]["kind"]
                kind = "measured" if t == "measured" else "predicted" if t == "predicted" else "extrapolated"
                if kind == "extrapolated":
                    assert ex[key]["kind"] == "extrap", key
                cnt[kind] += 1
                i, j, kk = np.round(d["click"][k]).astype(int)
                rows.append({"case": cid, "side": s, "sec": k, "kind": kind,
                             "extrap_mm": round(ex[key]["dist_mm"], 1) if kind == "extrapolated" else "",
                             "i": i, "j": j, "k": kk})
        n = sum(cnt.values())
        summ.append({"case": cid, "sections": n, **{f"{a}_frac": round(b / n, 3) for a, b in cnt.items()}})
        print(cid[-4:], cnt, flush=True)
    with open(out / "sections.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    with open(out / "itksnap_labels.txt", "w", encoding="utf-8") as f:
        for v, name, (r, g, b) in LABELS:
            vis = 0 if v in (0, 2) else 1          # 不確定區很大,預設隱藏
            f.write(f'{v:5d} {r:5d} {g:5d} {b:5d} {vis:9.2f} {vis:2d} {vis:2d}    "{name}"\n')
    N = len(rows); c = {k: sum(r["kind"] == k for r in rows) for k in ("measured", "predicted", "extrapolated")}
    (out / "README.txt").write_text(README.format(n_cases=len(summ), n_sec=N, m=c["measured"], mp=c["measured"] / N,
                                                  p=c["predicted"], pp=c["predicted"] / N, e=c["extrapolated"],
                                                  ep=c["extrapolated"] / N), encoding="utf-8")
    print("cases", len(summ), "sections", N, c)


if __name__ == "__main__":
    main()
