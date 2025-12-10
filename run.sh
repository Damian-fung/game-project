#!/usr/bin/env bash
# 一鍵啟動 - 深度學習AI貪吃蛇
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

# Prefer Windows py launcher, then python, then python3 to avoid WindowsApps stub
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
