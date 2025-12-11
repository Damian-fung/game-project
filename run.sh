#!/usr/bin/env bash
# 一鍵啟動 - 深度學習AI貪吃蛇
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

# Try Python 3.11 first (known working location on this PC)
if [ -f "/c/Users/NEO/AppData/Local/Programs/Python/Python311/python.exe" ]; then
    exec "/c/Users/NEO/AppData/Local/Programs/Python/Python311/python.exe" "$DIR/game.py" "$@"
fi

# 嘗試多種 Python 命令
for cmd in "py -3.11" "py -3.10" "py -3" python3 python; do
    # 檢查命令是否存在
    if command -v ${cmd%% *} >/dev/null 2>&1; then
        # 檢測是否安裝了必需套件
        if $cmd -c "import pygame, torch, numpy" 2>/dev/null; then
            exec $cmd "$DIR/game.py" "$@"
        fi
    fi
done

# 如果都沒找到，給出清晰的錯誤訊息
echo "Error: 找不到已安裝 pygame、torch、numpy 的 Python 環境"
echo "請執行: ./install.sh"
exit 1
