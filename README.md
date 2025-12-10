# 深度學習 AI 貪吃蛇對戰系統 - 完整專案文檔

## 專案概述

這是一個基於深度強化學習（Deep Q-Network, DQN）的雙AI貪吃蛇對戰系統，使用 Pygame 實現遊戲視覺化。兩條AI蛇會通過深度學習持續改進策略，互相競爭以獲取食物並攻擊對方。

### 核心特色
- ✅ **深度強化學習**：使用 PyTorch 實現的 DQN 算法
- ✅ **雙AI對戰**：兩條蛇同時訓練，互相學習對抗策略
- ✅ **2D/3D 視覺化**：支援傳統2D與等角投影3D顯示模式
- ✅ **即時訓練監控**：顯示學習曲線、探索率、勝率等統計數據
- ✅ **經驗回放**：保存訓練記憶供後續分析
- ✅ **模型持久化**：自動保存與載入訓練好的模型

---

## 🎯 快速開始

### 最簡單的方式

```bash
chmod +x *.sh   # 首次需要給予執行權限
./run.sh        # 一鍵啟動遊戲
```

### 或直接使用 Python

```bash
python game.py
```

---

## 📦 Shell 腳本列表

| 指令           | 說明        | 使用場景       |
| -------------- | ----------- | -------------- |
| `./run.sh`     | ⚡ 一鍵啟動 | 最快速啟動遊戲 |
| `./start.sh`   | 🏃 訓練模式 | 開始 AI 訓練   |
| `./demo.sh`    | 👁️ 演示模式 | 觀看訓練成果   |
| `./analyze.sh` | 📊 數據分析 | 查看詳細統計   |
| `./install.sh` | 📦 安裝依賴 | 安裝所需套件   |

---

## 🚀 快速開始（3 步驟）

### 第一次使用

**步驟 1: 安裝依賴**

```bash
./install.sh
# 或
pip install -r requirements.txt
```

**步驟 2: 啟動遊戲**

```bash
./run.sh
# 或
python game.py
```

**步驟 3: 查看訓練結果**

```bash
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

## 目錄結構

```
game-project/
├── game.py                        # 主程式（1564行）
├── README.md                      # 本文檔（完整說明）
├── requirements.txt               # Python 依賴套件
├── run.sh                         # 一鍵啟動腳本
├── start.sh                       # 訓練模式腳本
├── demo.sh                        # 演示模式腳本
├── analyze.sh                     # 數據分析腳本
├── install.sh                     # 依賴安裝腳本
├── __pycache__/                   # Python 快取檔案
├── snake_models/                  # AI 模型儲存目錄
│   ├── snake1_model.pth          # AI蛇1 的神經網路權重
│   └── snake2_model.pth          # AI蛇2 的神經網路權重
└── snake_memories/                # 訓練記憶儲存目錄
    └── episode_memory_*.pkl      # 訓練回合記憶檔案
```

---

## 核心系統架構

### 1. 深度學習模組

#### **DQN 神經網路**
```
架構：輸入層(11維) → 全連接(128) → 全連接(128) → 全連接(64) → 輸出層(4維)
激活函數：ReLU
輸出：4個動作的 Q 值（上、下、左、右）
```

**狀態特徵（11維）**：
1. 當前方向（4維 one-hot）：左、右、上、下
2. 食物相對位置（4維）：食物在左、右、上、下
3. 危險檢測（3維）：前方危險、左方危險、右方危險

**訓練參數**：
- 學習率：0.001
- 折扣因子 γ：0.95
- 批次大小：32
- 記憶容量：10,000
- ε-greedy 探索率：1.0 → 0.01（衰減率 0.995）

### 2. 遊戲邏輯

#### **獎勵系統**
```
基礎生存獎勵：+0.01 每步
吃到食物：+10
攻擊成功（咬掉對方尾巴）：+5
死亡懲罰：-20
長時間未進食懲罰：-0.1（超過100步無分）
```

#### **對戰規則**
1. **頭對頭碰撞**：身體較長者獲勝；同長則雙死
2. **頭對身體**：攻擊者咬掉被攻擊者尾巴一節，被攻擊者只剩一節時死亡
3. **自我碰撞**：立即死亡
4. **邊界**：環繞模式（從一側出現在另一側）

---

## 環境需求與安裝

### 環境需求

**Python 版本**：3.10+ （支援 3.13）

**必要套件**：
```
pygame       # 遊戲引擎
PyOpenGL     # 3D 渲染支援
numpy        # 數值計算
torch        # PyTorch 深度學習框架
```

### 安裝步驟

```bash
# 方式1: 使用安裝腳本
./install.sh

# 方式2: 直接安裝
pip install -r requirements.txt

# 驗證安裝
python -c "import pygame, torch, numpy; print('所有套件安裝成功')"
```

### 執行方式

#### **訓練模式（預設）**
```bash
./run.sh
# 或
python game.py
```
- 兩條AI蛇持續訓練
- 自動保存模型（每100回合）
- 記錄經驗回放（每50回合）

#### **演示模式（僅展示）**
```bash
./demo.sh
# 或
python game.py --demo
```
- 載入已訓練模型進行對戰
- 不進行學習更新
- 適合觀察訓練成果

---

## 📊 訓練數據

### 訓練數據位置

- **AI 模型**: `snake_models/*.pth`
- **經驗回放**: `snake_memories/*.pkl`
- **訓練日誌**: `training_log.csv`（如果有實作）

### 資料檔案說明

#### 模型檔案（`snake_models/`）
- **格式**：PyTorch `.pth` 檔案（state_dict）
- **內容**：神經網路權重與偏置

#### 記憶檔案（`snake_memories/`）
- **格式**：Pickle `.pkl` 檔案
- **內容**：每個檔案包含 50 回合的經驗數據
- **用途**：離線分析訓練過程、視覺化決策過程

### 分析訓練數據

```bash
./analyze.sh
```

範例程式碼：
```python
import pickle
import matplotlib.pyplot as plt

# 載入記憶檔案
with open('snake_memories/episode_memory_20251117_202955.pkl', 'rb') as f:
    data = pickle.load(f)

# 分析獎勵趨勢
rewards1 = [d['snake1_reward'] for d in data]
plt.plot(rewards1)
plt.title('Snake 1 Rewards Over Time')
plt.show()
```

---

## 💡 使用技巧

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

---

## 🔧 故障排除

### 問題：無法執行 .sh 文件

```bash
chmod +x *.sh  # 給予執行權限
```

### 問題：找不到 Python 命令

腳本已配置自動尋找 Python 解釋器（優先順序：py、python、python3）

### 問題：缺少套件

```bash
pip install pygame numpy torch
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

## 如何繼續未完成的工作

### 📋 短期任務（1-2週）

#### 1. **優化訓練效率**
- [ ] 實作優先經驗回放（Prioritized Experience Replay）
- [ ] 加入 Double DQN 算法
- [ ] 調整獎勵函數（加入距離獎勵）
- [ ] 調整探索率衰減策略

#### 2. **資料分析與視覺化**
- [ ] 讀取 `snake_memories/` 中的 `.pkl` 檔案
- [ ] 繪製詳細學習曲線（使用 matplotlib）
- [ ] 分析勝率、平均得分、平均回合長度
- [ ] 視覺化 Q 值熱力圖

#### 3. **改進 UI**
- [ ] 加入即時 FPS 顯示
- [ ] 顯示當前 Q 值最大的動作
- [ ] 加入訓練進度條

### 🎯 中期目標（1-2月）

#### 1. **升級神經網路架構**

**CNN 狀態表示範例**：
```python
class ConvDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 25 * 20, 256)
        self.fc2 = nn.Linear(256, 4)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

#### 2. **多蛇對戰系統**
- [ ] 支援 3-4 條蛇同時對戰
- [ ] 實作淘汰賽制
- [ ] 計算 ELO 等級分

#### 3. **自我對弈（Self-Play）**
```python
# 每 100 回合保存一個檢查點
if episode % 100 == 0:
    torch.save(model.state_dict(), f'checkpoints/model_{episode}.pth')

# 隨機選擇歷史版本作為對手
opponent_checkpoint = random.choice(os.listdir('checkpoints'))
opponent_model.load_state_dict(torch.load(opponent_checkpoint))
```

### 🚀 長期願景（3-6月）

1. **發表訓練好的模型**：訓練至少 10,000 回合
2. **比賽與排行榜**：建立線上對戰平台
3. **遷移學習**：應用於其他遊戲
4. **研究方向**：探索合作與競爭的平衡，撰寫學術論文

---

## 已知限制與改進方向

### 🔧 需要改進的地方

1. **狀態表示不完整**
   - ❌ 僅考慮3個方向的危險（前、左、右）
   - ✅ **建議**：改用卷積神經網路（CNN）處理完整棋盤狀態

2. **獎勵設計過於簡單**
   - ❌ 沒有距離獎勵引導蛇接近食物
   - ✅ **建議**：加入曼哈頓距離獎勵、生存時間獎勵

3. **探索率衰減過快**
   - ❌ 0.995 的衰減率約在 500 回合後降至 0.08
   - ✅ **建議**：調整為 0.999 或使用分段衰減

4. **目標網路更新頻率**
   - ❌ 每 100 步更新可能太頻繁
   - ✅ **建議**：改為每 1000 步或使用軟更新（τ=0.005）

5. **沒有優先經驗回放**
   - ❌ 所有經驗等權重採樣
   - ✅ **建議**：實作優先經驗回放（PER）

### 🚀 進階演算法建議

1. **Double DQN**：減少 Q 值過估計
2. **Dueling DQN**：分離狀態價值與動作優勢
3. **Rainbow DQN**：結合多種 DQN 改進
4. **PPO / A3C**：Actor-Critic 方法
5. **Self-Play**：保存歷史版本模型進行對戰

---

## 程式碼導覽

### 關鍵函數位置

| 功能 | 檔案 | 行數 |
|------|------|------|
| DQN 神經網路定義 | game.py | 64-73 |
| DQN 代理實作 | game.py | 85-151 |
| 狀態特徵提取 | game.py | 107-130 |
| 獎勵計算 | game.py | 1067-1087 |
| 碰撞檢測 | game.py | 1177-1209 |
| 訓練循環 | game.py | 1212-1228 |
| UI 繪製 | game.py | 1391-1492 |
| 學習曲線圖 | game.py | 1282-1349 |

---

## 參考資源

### 深度強化學習
- [Playing Atari with Deep Reinforcement Learning (DQN)](https://arxiv.org/abs/1312.5602)
- [Deep Reinforcement Learning Hands-On](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On)
- [OpenAI Spinning Up](https://spinningup.openai.com/)

### Pygame 教學
- [Pygame 官方文檔](https://www.pygame.org/docs/)
- [Real Python Pygame Tutorial](https://realpython.com/pygame-a-primer/)

### PyTorch 教學
- [PyTorch 官方教學](https://pytorch.org/tutorials/)
- [Deep Learning with PyTorch](https://pytorch.org/deep-learning-with-pytorch)

---

## 授權與貢獻

### 如何貢獻
1. Fork 本專案
2. 建立功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交變更 (`git commit -m 'Add some amazing feature'`)
4. 推送至分支 (`git push origin feature/amazing-feature`)
5. 開啟 Pull Request

---

## 版本歷史

### v1.0.0（當前版本）
- ✅ 基礎 DQN 雙蛇對戰
- ✅ 2D/3D 視覺化
- ✅ 學習曲線監控
- ✅ 模型與記憶保存

### v1.1.0（規劃中）
- 🔄 優先經驗回放
- 🔄 Double DQN
- 🔄 改進獎勵函數
- 🔄 完整的訓練分析工具

### v2.0.0（未來展望）
- 💡 CNN 狀態表示
- 💡 多蛇對戰
- 💡 Web 介面
- 💡 自我對弈系統

---

**最後更新**：2025年12月11日  
**文檔版本**：2.0.0  
**專案狀態**：積極開發中 🚀

---

**快樂訓練！🎮🐍**
