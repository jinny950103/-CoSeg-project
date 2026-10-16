#!/bin/bash
# 診斷 nnU-Net 到底裝在哪裡、為什麼 pip list 找不到。
#   bash scripts/nano4/00b_diagnose_nnunet.sh
set -u

echo "===== 當前環境 ====="
echo "  CONDA_DEFAULT_ENV = ${CONDA_DEFAULT_ENV:-<無>}"
echo "  CONDA_PREFIX      = ${CONDA_PREFIX:-<無>}"
echo "  which python      = $(which python 2>/dev/null || echo 無)"
echo "  python 版本        = $(python --version 2>&1)"
echo "  which pip         = $(which pip 2>/dev/null || echo 無)"
echo "  pip 指向的 python  = $(pip --version 2>&1)"

echo
echo "===== 1. nnU-Net 的指令存在嗎(最關鍵) ====="
# 如果 CLI 存在,就表示有裝,只是 pip/python 對不上
for c in nnUNetv2_train nnUNetv2_plan_and_preprocess nnUNetv2_predict nnUNet_train; do
  p=$(which "$c" 2>/dev/null)
  printf '  %-32s %s\n' "$c" "${p:-<找不到>}"
done

echo
echo "===== 2. 能不能 import ====="
python - <<'PY'
for mod in ("nnunetv2", "nnunet"):
    try:
        m = __import__(mod)
        print(f"  {mod}: OK -> {m.__file__}")
        print(f"    version = {getattr(m, '__version__', '<未標示>')}")
    except Exception as e:
        print(f"  {mod}: 失敗 ({type(e).__name__}: {e})")
PY

echo
echo "===== 3. pip 看得到什麼(完整搜尋) ====="
echo "  --- 含 nn / net / unet 的套件 ---"
pip list 2>/dev/null | grep -iE "nn|net|unet|torch" | sed 's/^/    /' || echo "    (pip list 失敗)"

echo
echo "===== 4. 所有 conda 環境 ====="
conda env list 2>/dev/null | sed 's/^/  /' || echo "  conda 指令不可用"

echo
echo "===== 5. 各環境裡有沒有 nnunetv2 ====="
for envdir in "$HOME"/.conda/envs/* "$CONDA_PREFIX"/../* ; do
  [ -d "$envdir" ] || continue
  name=$(basename "$envdir")
  hit=$(ls -d "$envdir"/lib/python*/site-packages/nnunetv2 2>/dev/null | head -1)
  if [ -n "$hit" ]; then
    printf '  %-20s 有 nnunetv2 -> %s\n' "$name" "$hit"
  fi
done

echo
echo "===== 6. 從檔案系統找 ====="
echo "  (找 nnunetv2 的安裝位置,最多等 30 秒)"
timeout 30 find "$HOME" -maxdepth 7 -name "nnunetv2" -type d -not -path "*/.git/*" 2>/dev/null | head -10 | sed 's/^/    /'
echo "  (找 clone 下來的 nnUNet 原始碼)"
timeout 20 find "$HOME" -maxdepth 4 -iname "nnUNet*" -type d -not -path "*/.git/*" 2>/dev/null | head -10 | sed 's/^/    /'

echo
echo "===== 判讀 ====="
cat <<'EOF'
  情況 A:第 1 節的指令找得到,但第 2/3 節失敗
     -> nnU-Net 有裝,但 pip/python 指到別的地方。
        通常是 module load 後 PATH 變了。照第 1 節顯示的路徑反推環境即可。

  情況 B:第 5 或 6 節在「其他環境」找到 nnunetv2
     -> 裝在別的 conda env。改用那個環境,或在 coseg 裡重裝。

  情況 C:哪裡都找不到
     -> 其實沒裝成功。在 coseg 環境裡裝:
          source activate coseg
          pip install nnunetv2
        (需要連外網;若無法連外,要用離線 wheel 或 pip download 後上傳)
EOF
