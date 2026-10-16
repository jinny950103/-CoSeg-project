"""Pre-upload integrity check for the nnU-Net raw dataset.

Run this before transferring to the cluster: catching a problem here costs seconds,
catching it after a 6GB upload and an hour of preprocessing costs an afternoon.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

OK, BAD = "  [ok]  ", "  [BAD] "
problems = []


def check(cond, msg_ok, msg_bad):
    print((OK + msg_ok) if cond else (BAD + msg_bad))
    if not cond:
        problems.append(msg_bad)
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="./data/nnUNet_raw/Dataset503_MixedCanal")
    ap.add_argument("--sample", type=int, default=12, help="how many volumes to open fully")
    args = ap.parse_args()
    ds = Path(args.dataset)

    print(f"=== {ds} ===\n")

    imgs_tr = sorted(p.name for p in (ds / "imagesTr").glob("*.nii.gz"))
    lbls_tr = sorted(p.name for p in (ds / "labelsTr").glob("*.nii.gz"))
    imgs_ts = sorted(p.name for p in (ds / "imagesTs").glob("*.nii.gz"))
    lbls_ts = sorted(p.name for p in (ds / "labelsTs_eval").glob("*.nii.gz"))

    print("1. 檔名與數量")
    tr_ids = [n[: -len("_0000.nii.gz")] for n in imgs_tr]
    ts_ids = [n[: -len("_0000.nii.gz")] for n in imgs_ts]
    check(len(imgs_tr) == len(lbls_tr),
          f"imagesTr {len(imgs_tr)} = labelsTr {len(lbls_tr)}",
          f"imagesTr {len(imgs_tr)} != labelsTr {len(lbls_tr)}")
    check(len(imgs_ts) == len(lbls_ts),
          f"imagesTs {len(imgs_ts)} = labelsTs_eval {len(lbls_ts)}",
          f"imagesTs {len(imgs_ts)} != labelsTs_eval {len(lbls_ts)}")
    check(sorted(f"{i}.nii.gz" for i in tr_ids) == lbls_tr,
          "每個訓練影像都有對應 label", "訓練影像與 label 名稱對不上")
    check(all(n.endswith("_0000.nii.gz") for n in imgs_tr + imgs_ts),
          "影像檔名都有 _0000 channel 後綴", "有影像缺少 _0000 後綴")

    print("\n2. dataset.json")
    dj = json.loads((ds / "dataset.json").read_text())
    check(dj.get("numTraining") == len(imgs_tr),
          f"numTraining {dj.get('numTraining')} 與實際檔數相符",
          f"numTraining {dj.get('numTraining')} != 實際 {len(imgs_tr)}")
    check(set(dj.get("labels", {}).values()) == {0, 1},
          f"labels = {dj.get('labels')}", f"labels 定義異常: {dj.get('labels')}")
    check(dj.get("file_ending") == ".nii.gz" and "channel_names" in dj,
          "使用 nnU-Net v2 的 schema (channel_names + file_ending)",
          "dataset.json 不是 v2 schema")

    print("\n3. splits_final.json")
    splits = json.loads((ds / "splits_final.json").read_text())
    mapping = json.loads((ds / "case_id_mapping.json").read_text())
    train_set = set(tr_ids)
    all_ok = True
    for i, s in enumerate(splits):
        tr, va = set(s["train"]), set(s["val"])
        if tr & va:
            all_ok = False
            problems.append(f"fold {i}: train/val 重疊 {len(tr & va)} 個")
        if (tr | va) != train_set:
            all_ok = False
            problems.append(f"fold {i}: 覆蓋不完整 (缺 {len(train_set - (tr | va))})")
    check(all_ok, f"{len(splits)} 折都無重疊、完整覆蓋 {len(train_set)} 個訓練 case",
          "splits 有重疊或覆蓋不全")

    # subject groups must not straddle train/val
    def gkey(cid):
        src = mapping["cases"].get(cid, {}).get("image", cid)
        stem = Path(src).name
        for e in (".nii.gz", ".nii"):
            if stem.endswith(e):
                stem = stem[: -len(e)]
        m = re.match(r"^([A-Za-z]+\d+)-\d+$", stem)
        return m.group(1) if m else cid

    leaks = []
    for i, s in enumerate(splits):
        gt = {gkey(c) for c in s["train"]}
        gv = {gkey(c) for c in s["val"]}
        if gt & gv:
            leaks.append((i, sorted(gt & gv)))
    check(not leaks, "沒有 subject 群組橫跨 train/val", f"群組洩漏: {leaks[:3]}")

    hosp_per_fold = [sum(1 for c in s["val"] if c.startswith("HOSP_")) for s in splits]
    check(all(h > 0 for h in hosp_per_fold),
          f"每折 val 都有醫院病人: {hosp_per_fold}",
          f"有折次的 val 沒有醫院病人: {hosp_per_fold}")

    print("\n4. 測試集隔離")
    hosp_test = set(mapping["hospital_test_patients"])
    bleed = [p for p in hosp_test if f"HOSP_{p}" in train_set]
    check(not bleed, f"{len(hosp_test)} 位醫院測試病人都不在訓練集",
          f"測試病人出現在訓練集: {bleed}")
    check(not (set(tr_ids) & set(ts_ids)), "訓練與測試 case id 無重疊",
          f"訓練/測試重疊: {sorted(set(tr_ids) & set(ts_ids))[:5]}")

    print(f"\n5. 影像內容抽樣 ({args.sample} 個)")
    pick = tr_ids[:: max(1, len(tr_ids) // args.sample)][: args.sample]
    spacings, lab_vals, bad = Counter(), set(), 0
    for cid in pick:
        img = nib.load(ds / "imagesTr" / f"{cid}_0000.nii.gz")
        lbl = nib.load(ds / "labelsTr" / f"{cid}.nii.gz")
        sp = tuple(round(float(z), 3) for z in img.header.get_zooms()[:3])
        spacings[sp] += 1
        if img.shape != lbl.shape or not np.allclose(img.affine, lbl.affine, atol=1e-3):
            bad += 1
        lab_vals |= {float(v) for v in np.unique(np.asanyarray(lbl.dataobj))}
    check(bad == 0, "抽樣的 image/label shape 與 affine 都一致",
          f"{bad} 個抽樣案例 image/label 幾何不符")
    check(lab_vals <= {0.0, 1.0}, f"label 值只有 {sorted(lab_vals)}",
          f"label 出現非 0/1 的值: {sorted(lab_vals)}")
    check(len(spacings) == 1, f"抽樣 spacing 統一為 {list(spacings)[0]}",
          f"spacing 不一致: {dict(spacings)}")

    print("\n6. 組成")
    coh = Counter((v["cohort"], v["split"]) for v in mapping["cases"].values())
    for k in sorted(coh):
        print(f"       {k[0]:9s} {k[1]:5s}: {coh[k]}")

    print()
    if problems:
        print(f"!!! 發現 {len(problems)} 個問題,先不要上傳:")
        for p in problems:
            print(f"    - {p}")
        raise SystemExit(1)
    print("全部檢查通過,可以上傳。")


if __name__ == "__main__":
    main()
