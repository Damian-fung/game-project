#!/usr/bin/env bash
# 深度學習AI貪吃蛇 - 啟動遊戲（訓練模式）
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "  深度學習 AI 貪吃蛇對戰系統"
echo "========================================"
echo ""
echo "正在啟動遊戲（訓練模式）..."
echo ""

# Find Python interpreter
if command -v py >/dev/null 2>&1; then
  exec py -3 "$DIR/game.py" "$@"
elif command -v python >/dev/null 2>&1; then
  exec python "$DIR/game.py" "$@"
elif command -v python3 >/dev/null 2>&1; then
  exec python3 "$DIR/game.py" "$@"
else
  echo "Error: No Python interpreter found in PATH. Please install Python or add it to PATH."
  exit 1
fi
