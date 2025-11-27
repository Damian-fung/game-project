# 深度學習 AI 貪吃蛇對戰系統 - 快速開始指南

## 🎯 最簡單的方式

### Windows

```bash
run.bat         # 一鍵啟動遊戲
```

### Linux/Mac/Git Bash

```bash
chmod +x *.sh   # 首次需要給予執行權限
./run.sh        # 一鍵啟動遊戲
```

---

## 📦 完整指令列表

### Windows 批次檔 (.bat)

| 指令          | 說明        | 使用場景       |
| ------------- | ----------- | -------------- |
| `run.bat`     | ⚡ 一鍵啟動 | 最快速啟動遊戲 |
| `start.bat`   | 🏃 訓練模式 | 開始 AI 訓練   |
| `demo.bat`    | 👁️ 演示模式 | 觀看訓練成果   |
| `analyze.bat` | 📊 數據分析 | 查看詳細統計   |
| `test.bat`    | 🧪 測試日誌 | 驗證訓練功能   |
| `install.bat` | 📦 安裝依賴 | 安裝所需套件   |

### Linux/Mac/Git Bash Shell 腳本 (.sh)

| 指令           | 說明        | 使用場景       |
| -------------- | ----------- | -------------- |
| `./run.sh`     | ⚡ 一鍵啟動 | 最快速啟動遊戲 |
| `./start.sh`   | 🏃 訓練模式 | 開始 AI 訓練   |
| `./demo.sh`    | 👁️ 演示模式 | 觀看訓練成果   |
| `./analyze.sh` | 📊 數據分析 | 查看詳細統計   |
| `./install.sh` | 📦 安裝依賴 | 安裝所需套件   |

---

## 💡 像使用 npm 一樣簡單！

**傳統方式**（麻煩）：

```bash
C:\Users\NEO\AppData\Local\Programs\Python\Python311\python.exe game.py
```

**現在的方式**（超簡單）：

```bash
# Windows
run.bat

# Linux/Mac/Git Bash
./run.sh
```

---

## 🚀 快速開始（3 步驟）

### 第一次使用

**步驟 1: 安裝依賴**

```bash
# Windows
install.bat

# Linux/Mac/Git Bash
./install.sh
```

**步驟 2: 啟動遊戲**

```bash
# Windows
run.bat

# Linux/Mac/Git Bash
./run.sh

python game.py
```

**步驟 3: 查看訓練結果**

```bash
# Windows
analyze.bat

# Linux/Mac/Git Bash
./analyze.sh
```

---

## 🎯 遊戲控制

| 按鍵     | 功能              |
| -------- | ----------------- |
| **ESC**  | 退出遊戲          |
| **空格** | 暫停/繼續         |
| **D**    | 切換 2D/3D 模式   |
| **T**    | 切換訓練/演示模式 |
| **S**    | 保存模型          |
| **R**    | 重置回合          |

---

## 📊 訓練數據位置

- **訓練日誌**: `training_log.csv`
- **AI 模型**: `snake_models/*.pth`
- **經驗回放**: `snake_memories/*.pkl`

---

## 💡 使用技巧

### 快速查看訓練進度

```bash
# Windows
analyze.bat

# Linux/Mac/Git Bash
./analyze.sh
```

### 清除訓練數據重新開始

```bash
# Windows
del training_log.csv
del snake_memories\*.pkl

# Linux/Mac/Git Bash
rm -f training_log.csv
rm -f snake_memories/*.pkl
```

### 備份訓練好的模型

```bash
# 複製模型文件
copy snake_models\*.pth backup\
```

---

## 🔧 故障排除

### 問題：無法執行 .bat 或 .sh 文件

**Windows (.bat)**：右鍵點擊 → 選擇「以管理員身分執行」

**Linux/Mac/Git Bash (.sh)**：

```bash
chmod +x *.sh  # 給予執行權限
```

### 問題：找不到 Python 或 python3 命令

**解決**：所有腳本已配置使用完整 Python 路徑，應該可以直接運行

### 問題：Git Bash 下執行沒反應

**解決**：

- 使用 `./run.sh` 而不是 `run.bat`
- 確保使用 `.sh` 文件而非 `.bat` 文件

### 問題：訓練數據為空

**解決**：先運行遊戲至少完成一個回合，再執行分析

---

## 💡 專業技巧

### 創建桌面快捷方式（Windows）

1. 右鍵點擊 `run.bat`
2. 選擇「發送到」→「桌面快捷方式」
3. 重命名為「AI 貪吃蛇」
4. 雙擊桌面圖標即可啟動！

### 快速查看訓練進度

```bash
# Windows
copy snake_models\*.pth backup\

# Linux/Mac/Git Bash
cp snake_models/*.pth backup/
```

---

## 📂 文件說明

| 文件               | 類型     | 說明                   |
| ------------------ | -------- | ---------------------- |
| `run.bat/sh`       | 啟動腳本 | 一鍵啟動遊戲           |
| `start.bat/sh`     | 啟動腳本 | 訓練模式啟動           |
| `demo.bat/sh`      | 啟動腳本 | 演示模式啟動           |
| `analyze.bat/sh`   | 分析腳本 | 數據分析工具           |
| `test.bat`         | 測試腳本 | 測試工具（僅 Windows） |
| `install.bat/sh`   | 安裝腳本 | 依賴安裝               |
| `training_log.csv` | 訓練數據 | CSV 格式訓練日誌       |
| `snake_models/`    | 模型目錄 | AI 神經網路模型        |
| `snake_memories/`  | 記憶目錄 | 經驗回放數據           |

---

## 📚 更多信息

詳細文檔請參閱：`PROJECT_DOCUMENTATION_zh-TW.md`

---

**快樂訓練！🚀**
