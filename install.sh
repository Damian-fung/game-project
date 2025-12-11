#!/usr/bin/env bash
# 深度學習AI貪吃蛇 - 安裝依賴
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "  安裝 Python 依賴套件"
echo "========================================"
echo ""

# Find Python interpreter and show info
PYTHON_CMD=""
if command -v py >/dev/null 2>&1; then
  PYTHON_CMD="py -3"
  echo "✓ 找到 Python: $(py -3 --version 2>&1)"
  echo "✓ Python 路徑: $(py -3 -c 'import sys; print(sys.executable)' 2>&1)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
  echo "✓ 找到 Python: $(python --version 2>&1)"
  echo "✓ Python 路徑: $(python -c 'import sys; print(sys.executable)' 2>&1)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
  echo "✓ 找到 Python: $(python3 --version 2>&1)"
  echo "✓ Python 路徑: $(python3 -c 'import sys; print(sys.executable)' 2>&1)"
else
  echo "❌ 錯誤: 找不到 Python 解釋器"
  echo ""
  echo "請先安裝 Python 3.10+ 並將其加入系統 PATH"
  echo "下載地址: https://www.python.org/downloads/"
  exit 1
fi

echo ""
echo "正在安裝以下套件："
echo "  - pygame (遊戲引擎)"
echo "  - torch (深度學習框架)"
echo "  - numpy (數值計算)"
echo "  - PyOpenGL (3D 渲染)"
echo ""
echo "這可能需要幾分鐘時間，請耐心等待..."
echo "========================================"
echo ""

$PYTHON_CMD -m pip install -r "$DIR/requirements.txt"

echo ""
echo "========================================"
echo "  安裝完成！"
echo "========================================"
echo ""
read -p "按 Enter 繼續..."
