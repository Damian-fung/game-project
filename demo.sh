#!/usr/bin/env bash
# 深度學習AI貪吃蛇 - 演示模式
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "  深度學習 AI 貪吃蛇對戰系統"
echo "========================================"
echo ""
echo "正在啟動遊戲（演示模式）..."
echo ""

# 嘗試多種 Python 版本（優先使用較新版本）
for cmd in "py -3.13" "py -3.12" "py -3.11" "py -3.10" "py -3" python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v ${cmd%% *} >/dev/null 2>&1; then
        if $cmd -c "import pygame, torch, numpy" 2>/dev/null; then
            exec $cmd "$DIR/game.py" --demo "$@"
        fi
    fi
done

echo "Error: 找不到已安裝 pygame、torch、numpy 的 Python 環境"
echo "請執行: ./install.sh"
exit 1
