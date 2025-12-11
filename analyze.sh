#!/usr/bin/env bash
# 深度學習AI貪吃蛇 - 分析訓練數據
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "  訓練數據分析工具"
echo "========================================"
echo ""

# Find Python interpreter
if command -v py >/dev/null 2>&1; then
  py -3 "$DIR/analyze_training.py" "$@"
elif command -v python >/dev/null 2>&1; then
  python "$DIR/analyze_training.py" "$@"
elif command -v python3 >/dev/null 2>&1; then
  python3 "$DIR/analyze_training.py" "$@"
else
  echo "Error: No Python interpreter found in PATH. Please install Python or add it to PATH."
  exit 1
fi

echo ""
read -p "按 Enter 繼續..."
