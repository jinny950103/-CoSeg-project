#!/usr/bin/env python3
"""Package Dataset504 for upload: hospital files + metadata only, public data reused from 503.

504 differs from 503 only in its hospital cases, so the MCSTU_* files are not uploaded
again: nano4 copies them from its own 503 (scripts/nano4/10_assemble_504.sh).

To make that safe, each MCSTU file in 504 is first checked to hold the same voxels as its
503 counterpart (decompressed content compared, since gzip headers carry a timestamp),
then replaced by the 503 file so the bytes match exactly. mcstu_md5.txt lists those files
for nano4 to verify after copying.

    python scripts/package_504_upload.py
"""
import gzip
import hashlib
import shutil
import tarfile
from pathlib import Path

RAW = Path("./data/nnUNet_raw")
OLD = RAW / "Dataset503_MixedCanal"
NEW = RAW / "Dataset504_MixedCanalClean"
OUT = Path("./upload504.tar")
SUBDIRS = ("imagesTr", "labelsTr", "imagesTs", "labelsTs_eval")
META = ("dataset.json", "splits_final.json", "case_id_mapping.json", "build_report.json")
SCRIPTS = [f"scripts/nano4/{n}" for n in
           ("10_assemble_504.sh", "11_preprocess_504.sh", "12_train_504.sh", "13_predict_504.sh")]


def md5(path, decompress=False):
    h = hashlib.md5()
    opener = gzip.open if decompress else open
    with opener(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    lines, hosp = [], []
    for sub in SUBDIRS:
        for p in sorted((NEW / sub).glob("*.nii.gz")):
            rel = f"{sub}/{p.name}"
            if p.name.startswith("HOSP_"):
                hosp.append(rel)
                continue
            old = OLD / rel
            if not old.exists():
                raise SystemExit(f"503 裡沒有 {rel},不能從 503 複製")
            if md5(p, True) != md5(old, True):
                raise SystemExit(f"{rel} 的內容跟 503 不同,不能從 503 複製")
            shutil.copyfile(old, p)
            lines.append(f"{md5(p)}  {rel}")
            if len(lines) % 25 == 0:
                print(f"  公開資料比對 {len(lines)} 個...", flush=True)

    (NEW / "mcstu_md5.txt").write_bytes(("\n".join(lines) + "\n").encode())
    print(f"公開資料 {len(lines)} 個檔案與 503 內容相同")

    top = NEW.name
    with tarfile.open(OUT, "w") as tar:
        for rel in hosp:
            tar.add(NEW / rel, arcname=f"data/nnUNet_raw/{top}/{rel}")
        for m in META + ("mcstu_md5.txt",):
            tar.add(NEW / m, arcname=f"data/nnUNet_raw/{top}/{m}")
        for s in SCRIPTS:
            tar.add(s, arcname=s)
    print(f"醫院檔案 {len(hosp)} 個 + metadata + 腳本 -> {OUT} "
          f"({OUT.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
