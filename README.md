# 深度學習 AI 貪吃蛇對戰系統

基於深度 Q 網路（DQN）的雙 AI 貪吃蛇對戰遊戲。兩條 AI 蛇通過強化學習持續改進策略，互相競爭食物並攻擊對方。

## ✨ 核心特色

- 🧠 **深度強化學習**：PyTorch 實現的 DQN 算法
- ⚔️ **雙 AI 對戰**：兩種不同策略的AI（激進型 vs 穩健型）互相學習對抗
- 🎮 **2D/3D 視覺化**：支援傳統 2D 與等角投影 3D 顯示模式
- 🎨 **響應式界面**：窗口大小自適應，遊戲區域與UI面板自動調整
- 📊 **訓練監控**：即時顯示學習曲線、探索率、勝率等統計數據
- 🌈 **視覺反饋**：蛇的顏色隨身體長度變化，紅色/綠色主題統一
- 💾 **自動保存**：模型與訓練數據自動持久化
- 🔍 **不同探索策略**：AI蛇1（紅色）使用 ε-greedy，AI蛇2（綠色）使用 Boltzmann 探索

---

## 🚀 快速開始

### 1. Python 版本要求

✅ **支援 Python 3.10、3.11、3.12、3.13**

- [Python 3.13 下載](https://www.python.org/downloads/) （最新版）
- [Python 3.12 下載](https://www.python.org/downloads/release/python-3120/)
- [Python 3.11 下載](https://www.python.org/downloads/release/python-3119/)
- 安裝時請勾選 **"Add Python to PATH"**

驗證安裝：

```bash
python --version  # 必須顯示 3.10.x 或 3.11.x 或 3.12.x 或 3.13.x
```

### 2. 安裝依賴

```bash
./install.sh
```

腳本會自動：

- 掃描並診斷 Python 環境
- 檢查版本相容性
- 安裝所需套件（pygame, torch, numpy, PyOpenGL）

### 3. 啟動遊戲

```bash
chmod +x *.sh      # 首次需要給執行權限
./run.sh           # 一鍵啟動
```

---

## 📦 指令列表

| 指令           | 說明                            |
| -------------- | ------------------------------- |
| `./run.sh`     | ⚡ 啟動遊戲（訓練模式）         |
| `./train.sh`   | 🏃 訓練模式（與 run.sh 相同）   |
| `./demo.sh`    | 👁️ 演示模式（載入模型，不訓練） |
| `./analyze.sh` | 📊 分析訓練數據                 |
| `./install.sh` | 📦 安裝/診斷依賴環境            |

---

## 🎮 遊戲控制

| 按鍵/操作     | 功能                                    |
| ------------- | --------------------------------------- |
| **ESC**       | 退出遊戲                                |
| **空格**      | 暫停/繼續                               |
| **D**         | 切換 2D/3D 模式                         |
| **T**         | 切換訓練/演示模式（自動重置回合）       |
| **S**         | 保存模型                                |
| **R**         | 重置回合                                |
| **窗口調整**  | 拖動窗口邊緣可自由調整大小              |
| **點擊按鈕**  | 使用UI面板的按鈕進行相同操作            |

---

## 📁 專案結構

```
game-project/
├── game.py                  # 主程式（1829 行）
├── README.md                # 本文檔
├── requirements.txt         # Python 依賴列表
├── training_log.csv         # 訓練數據日誌（CSV 格式）
│
├── run.sh                   # 啟動腳本
├── train.sh                 # 訓練腳本
├── demo.sh                  # 演示腳本
├── analyze.sh               # 數據分析腳本
├── install.sh               # 安裝/診斷腳本
│
├── analyze_training.py      # 訓練數據分析工具
├── test_training_log.py     # 測試腳本
│
├── snake_models/            # AI 模型目錄
│   ├── snake1_model.pth    # AI 蛇1（紅色-激進型）的神經網路權重
│   └── snake2_model.pth    # AI 蛇2（綠色-穩健型）的神經網路權重
│
└── snake_memories/          # 訓練記憶目錄
    └── episode_memory_*.pkl # 經驗回放數據
```

---

## 🧠 DQN 架構

### 神經網路結構

```
輸入層(11) → FC(128) → FC(128) → FC(64) → 輸出層(4)
激活函數：ReLU
輸出：4 個動作的 Q 值（上、下、左、右）
```

### 狀態空間（11 維）

1. **當前方向**（4 維 one-hot）：左、右、上、下
2. **食物相對位置**（4 維）：食物在左、右、上、下
3. **危險檢測**（3 維）：前方危險、左方危險、右方危險

### 訓練參數與AI策略

**AI蛇1（紅色 - 激進型，ε-greedy策略）**：

- 顏色：RGB(255, 100, 100) - 紅色
- 學習率：0.001（較高，快速學習）
- 探索率：1.0 → 0.01（衰減率 0.995，快速收斂）
- 折扣因子 γ：0.95
- 批次大小：32
- 記憶容量：10,000
- 策略特點：快速決策，激進進攻，適合主動出擊

**AI蛇2（綠色 - 穩健型，Boltzmann策略）**：

- 顏色：RGB(100, 255, 100) - 綠色
- 學習率：0.0008（較低，穩定學習）
- 探索率：0.8 → 0.05（衰減率 0.998，慢速收斂）
- 使用溫度參數的 Softmax 動作選擇
- 策略特點：謹慎決策，平衡風險，適合穩健防守

### 視覺設計

**顏色漸變系統**：
- 蛇的顏色深淺根據身體長度動態變化
- 2格（初始）：intensity = 0.55（較淺）
- 3格：intensity = 0.70
- 4格：intensity = 0.85
- 5格及以上：intensity = 1.0（最深/最飽和）
- 視覺效果：身體越長，顏色越深，直觀展示戰況

**統一配色**：
- 所有UI元素（狀態面板、學習曲線、圖例）使用相同的紅/綠配色
- 確保視覺一致性和識別度

### 獎勵系統

```python
基礎生存獎勵：+0.01 / 步
吃到食物：+10
攻擊成功（咬尾巴）：+5
死亡懲罰：-20
長時間未進食：-0.1（超過 100 步）

距離獎勵（新增）：
  靠近食物：+0.1
  遠離食物：-0.05
```

---

## 🎯 對戰規則

1. **初始狀態**：每條蛇開局時為2格（頭部+1身體）
2. **頭對頭碰撞**：身體較長者獲勝；同長則雙死
3. **頭對身體**：攻擊者咬掉被攻擊者尾巴一節
4. **自我碰撞**：立即死亡
5. **邊界**：環繞模式（從一側出現在另一側）
6. **模式切換**：切換訓練/演示模式時自動重置回合

---

## 📊 訓練數據

### 數據檔案

- **模型**：`snake_models/*.pth`（PyTorch 權重）
- **記憶**：`snake_memories/*.pkl`（經驗回放數據）
- **日誌**：`training_log.csv`（15 列詳細訓練數據）

### CSV 日誌欄位

```
episode, snake1_score, snake2_score, snake1_length, snake2_length,
snake1_epsilon, snake2_epsilon, snake1_loss, snake2_loss,
snake1_reward, snake2_reward, winner, total_steps,
snake1_win_rate, snake2_win_rate
```

### 分析數據

```bash
./analyze.sh  # 使用內建分析工具
```

或手動分析：

```python
import pandas as pd
df = pd.read_csv('training_log.csv')
print(df.tail(10))  # 最近 10 回合
```

---

## 💡 進階使用

### 清除訓練數據重新開始

```bash
rm -f training_log.csv
rm -f snake_memories/*.pkl
```

### 備份訓練好的模型

```bash
mkdir -p backup
cp snake_models/*.pth backup/
```

### 僅載入已訓練模型對戰

```bash
./demo.sh  # 不會更新模型，純演示
```

---

## 🔧 故障排除

### 問題：Python 版本不相容

⚠️ **重要：pygame 和 torch 套件不支援 Python 3.12 或更新版本！**

**症狀**：

- 執行 `./install.sh` 時出現編譯錯誤
- 顯示 "FileNotFoundError: [WinError 2] 系統找不到指定的檔案"
- pygame 安裝失敗

**解決方案**：

1. **安裝 Python 3.10 或 3.11**（推薦 3.11）

   - Python 3.11: https://www.python.org/downloads/release/python-3119/
   - Python 3.10: https://www.python.org/downloads/release/python-31011/
   - 下載 Windows installer (64-bit)
   - 安裝時勾選 "Add Python to PATH"

2. **驗證版本**：

   ```bash
   python --version  # 需顯示 3.10.x、3.11.x、3.12.x 或 3.13.x
   ```

3. **重新安裝套件**：
   ```bash
   ./install.sh
   ```

### 問題：無法執行 .sh 文件

```bash
chmod +x *.sh  # 給予執行權限
```

### 問題：找不到 Python 命令或多個 Python 版本

如果系統有多個 Python 版本，**請使用腳本啟動**：

```bash
# 使用腳本（推薦）
./run.sh

# 不要直接用 python
# python game.py  ❌ 可能使用錯誤的 Python 版本
```

**手動指定 Python 版本**：

```bash
# 查看已安裝套件的 Python 路徑
which python3.11  # Linux/Mac
where python  # Windows

# 使用完整路徑運行
/usr/bin/python3.11 game.py  # Linux/Mac
C:/Users/你的用戶名/AppData/Local/Programs/Python/Python311/python.exe game.py  # Windows
```

### 問題：缺少套件

```bash
./install.sh
# 或手動安裝
pip install -r requirements.txt
```

### 問題：訓練速度太慢

**解決方案**：

```python
# 改為累積 N 個回合後批次訓練
if len(self.ai1.memory) >= self.batch_size * 10:
    for _ in range(10):  # 一次訓練 10 批次
        self.ai1.replay()
```

### 問題：學習曲線不收斂

**可能原因**：

1. 學習率過高 → 降至 0.0001
2. 網路太小 → 增加層數/神經元
3. 狀態表示不足 → 使用 CNN

---

## 🎨 UI/UX 特性

### 響應式設計
- **自適應布局**：窗口大小可自由調整（最小800x600）
- **動態縮放**：遊戲網格和UI元素根據窗口大小自動調整
- **防重疊機制**：遊戲區域與UI面板保持10像素間隔
- **智能間距**：UI元素高度根據窗口大小動態計算

### 視覺效果
- **顏色漸變**：蛇的身體顏色隨長度變化（短=淺，長=深）
- **統一主題**：紅/綠配色貫穿整個界面
- **實時反饋**：學習曲線、探索率、勝率即時更新
- **3D模式**：等角投影的立體視覺效果

## 🚧 已知限制與改進方向

### 當前實現特點

1. ✅ 雙AI不同策略對戰系統
2. ✅ 響應式UI設計
3. ✅ 顏色漸變視覺反饋
4. ✅ 訓練/演示模式自動重置
5. ✅ 距離獎勵優化

### 建議改進方向

1. 🔄 **CNN 狀態表示**：處理完整棋盤圖像
2. 🔄 **優先經驗回放**：重要經驗加權採樣
3. 🔄 **Double DQN**：減少 Q 值過估計
4. 🔄 **Dueling DQN**：分離狀態價值與動作優勢
5. 🔄 **多蛇對戰**：支援 3-4 條蛇同時對戰
6. 🔄 **自我對弈**：與歷史版本模型對戰

---

## 📚 關鍵程式碼位置

| 功能               | 檔案    | 行數      |
| ------------------ | ------- | --------- |
| DQN 神經網路定義   | game.py | 64-73     |
| DQN 代理類別       | game.py | 85-151    |
| 狀態特徵提取       | game.py | 107-130   |
| 獎勵計算（含距離） | game.py | 1171-1210 |
| 距離計算與應用     | game.py | 1240-1281 |
| 碰撞檢測           | game.py | 1177-1209 |
| UI 繪製            | game.py | 1391-1492 |

---

## 📖 參考資源

- [DQN 論文](https://arxiv.org/abs/1312.5602) - Playing Atari with Deep RL
- [PyTorch 官方教學](https://pytorch.org/tutorials/)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Pygame 文檔](https://www.pygame.org/docs/)

---

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

1. Fork 本專案
2. 建立功能分支：`git checkout -b feature/amazing-feature`
3. 提交變更：`git commit -m 'Add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 開啟 Pull Request

---

## 📝 版本歷史

### v1.1.0（當前）

**新功能**：
- ✅ 響應式UI設計（窗口自適應）
- ✅ 蛇身顏色漸變效果（根據長度變化）
- ✅ 統一紅/綠配色主題
- ✅ 訓練/演示模式切換自動重置
- ✅ 初始長度調整為2格（頭+1身體）
- ✅ UI元素動態間距計算

**改進**：
- ✅ 學習曲線顏色與蛇的顏色統一
- ✅ 防止遊戲區域與UI重疊
- ✅ 最小窗口尺寸限制（800x600）
- ✅ Python 3.10-3.13 全版本支援

### v1.0.0

- ✅ 基礎 DQN 雙蛇對戰
- ✅ 2D/3D 視覺化
- ✅ 學習曲線監控
- ✅ CSV 訓練日誌
- ✅ 距離基礎獎勵系統
- ✅ 不同探索策略（ε-greedy vs Boltzmann）
- ✅ 跨平台腳本支援
- ✅ 自動環境診斷工具

### v1.2.0（規劃中）

- 🔄 優先經驗回放（PER）
- 🔄 Double DQN
- 🔄 視覺化訓練曲線（matplotlib）
- 🔄 更多AI策略選項
- 🔄 回放系統（觀看歷史對戰）

---

**最後更新**：2025-12-12  
**專案狀態**：積極開發中 🚀

**快樂訓練！🎮🐍**
