#!/usr/bin/env bash
# 深度學習AI貪吃蛇 - 安裝依賴
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "  Python 環境診斷與套件安裝"
echo "========================================"
echo ""

# 診斷所有可用的 Python 環境
echo "🔍 掃描 Python 環境..."
echo ""

# 候選 Python 命令列表（優先使用已知的工作版本）
PYTHON_CANDIDATES=(
    "/c/Users/NEO/AppData/Local/Programs/Python/Python311/python.exe"
    "py -3.11"
    "py -3.10"
    "py -3"
    "python3.11"
    "python3.10"
    "python3"
    "python"
)

FOUND_PYTHONS=()
PYTHON_WITH_PACKAGES=""
BEST_PYTHON=""

# 掃描並測試所有 Python
for cmd in "${PYTHON_CANDIDATES[@]}"; do
    # 測試命令是否可執行
    if $cmd --version >/dev/null 2>&1; then
        VERSION=$($cmd --version 2>&1 | head -1)
        EXEC_PATH=$($cmd -c 'import sys; print(sys.executable)' 2>&1)
        
        # 檢查套件是否已安裝
        if $cmd -c "import pygame, torch, numpy" 2>/dev/null; then
            echo "✓ $VERSION"
            echo "  路徑: $EXEC_PATH"
            echo "  狀態: ✓ 已安裝 pygame, torch, numpy"
            PYTHON_WITH_PACKAGES="$cmd"
            BEST_PYTHON="$cmd"
        else
            echo "○ $VERSION"
            echo "  路徑: $EXEC_PATH"
            echo "  狀態: ✗ 缺少套件"
            if [ -z "$BEST_PYTHON" ]; then
                BEST_PYTHON="$cmd"
            fi
        fi
        echo ""
    fi
done

# 如果沒找到任何 Python
if [ -z "$BEST_PYTHON" ]; then
    echo "❌ 錯誤: 找不到 Python 解釋器"
    echo ""
    echo "⚠️  請先安裝 Python 3.10 或 3.11 並將其加入系統 PATH"
    echo ""
    echo "下載地址:"
    echo "  Python 3.11: https://www.python.org/downloads/release/python-3119/"
    echo "  Python 3.10: https://www.python.org/downloads/release/python-31011/"
    echo ""
    exit 1
fi

# 檢查 Python 版本是否支援
PYTHON_VERSION=$($BEST_PYTHON --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR_VERSION" -eq 3 ] && [ "$MINOR_VERSION" -ge 12 ]; then
    echo "⚠️  警告: 檢測到 Python $PYTHON_VERSION"
    echo ""
    echo "   pygame 和 torch 套件不支援 Python 3.12 或更新版本！"
    echo "   強烈建議安裝 Python 3.10 或 3.11"
    echo ""
    echo "   下載地址:"
    echo "     Python 3.11: https://www.python.org/downloads/release/python-3119/"
    echo "     Python 3.10: https://www.python.org/downloads/release/python-31011/"
    echo ""
    read -p "是否仍要繼續安裝？(可能會失敗) [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "========================================"
echo ""

# 如果已經有安裝套件的 Python，詢問是否重新安裝
if [ -n "$PYTHON_WITH_PACKAGES" ]; then
    echo "✓ 檢測到已安裝套件的 Python 環境"
    echo ""
    read -p "是否要重新安裝套件？[y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "跳過安裝。"
        exit 0
    fi
fi

# 開始安裝
PYTHON_CMD="$BEST_PYTHON"
echo "使用 Python: $($PYTHON_CMD --version 2>&1)"
echo "路徑: $($PYTHON_CMD -c 'import sys; print(sys.executable)' 2>&1)"
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
echo "  ✓ 安裝完成！"
echo "========================================"
echo ""
echo "現在可以使用以下指令啟動遊戲："
echo "  ./run.sh    - 一鍵啟動"
echo "  ./train.sh  - 訓練模式"
echo "  ./demo.sh   - 演示模式"
echo ""
