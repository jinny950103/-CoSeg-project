#!/bin/bash
# 在 nano4 上組出 Dataset504_MixedCanalClean。不用 sbatch,直接在登入節點跑:
#   bash scripts/nano4/10_assemble_504.sh
#
# 504 和 503 的公開資料(MCSTU_*)完全相同,只換了醫院病人,所以只上傳醫院檔案和
# metadata,公開資料直接從 nano4 上的 503 複製過來,省掉約 6GB 的上傳。
# mcstu_md5.txt 是在本機用 503 的檔案算的,逐檔比對,有任何一個不同就停下來。
set -euo pipefail

ROOT=/home/u9444861/-CoSeg-project
RAW=$ROOT/data/nnUNet_raw
OLD=$RAW/Dataset503_MixedCanal
NEW=$RAW/Dataset504_MixedCanalClean
MANI=$NEW/mcstu_md5.txt

[ -f "$MANI" ] || { echo "找不到 $MANI -> 先把 Dataset504 的醫院檔案包解壓到 $RAW"; exit 1; }
[ -d "$OLD" ] || { echo "找不到 $OLD -> 503 已經刪掉了,公開資料要從本機整包重傳"; exit 1; }

echo "=== 從 503 複製公開資料 ==="
n=0
while read -r _ rel; do
  if [ ! -f "$NEW/$rel" ]; then
    mkdir -p "$(dirname "$NEW/$rel")"
    cp "$OLD/$rel" "$NEW/$rel"
    n=$((n + 1))
  fi
done < "$MANI"
echo "  複製了 $n 個檔案"

echo
echo "=== md5 比對 ==="
( cd "$NEW" && md5sum -c --quiet mcstu_md5.txt ) && echo "  公開資料 $(wc -l < "$MANI") 個檔案全部一致"

echo
echo "=== 數量 ==="
for d in imagesTr labelsTr imagesTs labelsTs_eval; do
  printf "  %-14s 共 %3d  (醫院 %2d)\n" "$d" \
    "$(ls "$NEW/$d" | wc -l)" "$(ls "$NEW/$d" | grep -c '^HOSP_')"
done
echo "  應該是:imagesTr/labelsTr 252 (醫院 34),imagesTs/labelsTs_eval 29 (醫院 11)"

echo
echo "=== 確認燒入標記的病人不在裡面 ==="
# 病歷號不進 repo:每行一個,放在同目錄 burned_cases.local.txt(gitignored)
BAD="$(cat "$(dirname "$0")/burned_cases.local.txt" 2>/dev/null)"
[ -z "$BAD" ] && echo "  (找不到 burned_cases.local.txt,略過此檢查)"
hit=0
for p in $BAD; do
  if ls "$NEW"/*/HOSP_${p}* >/dev/null 2>&1; then echo "  !!! $p 還在"; hit=1; fi
done
[ $hit = 0 ] && echo "  都不在,OK"

echo
echo "完成。下一步:sbatch scripts/nano4/11_preprocess_504.sh"
