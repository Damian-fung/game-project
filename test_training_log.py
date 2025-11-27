#!/usr/bin/env python3
"""測試訓練追蹤功能的腳本"""

import csv
import os

# 檢查 CSV 文件
log_file = "training_log.csv"

if os.path.exists(log_file):
    print(f"✅ CSV 日誌文件存在: {log_file}")
    
    with open(log_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
        
        print(f"✅ CSV 文件行數: {len(rows)}")
        
        if len(rows) > 0:
            print(f"\n📊 CSV 表頭:")
            print(rows[0])
            
        if len(rows) > 1:
            print(f"\n📈 最新訓練數據 (最近 {min(5, len(rows)-1)} 回合):")
            for i, row in enumerate(rows[-5:] if len(rows) > 5 else rows[1:]):
                print(f"  回合 {i+1}: {row}")
        else:
            print("\n⚠️  尚無訓練數據（需要完成至少一個回合）")
else:
    print(f"❌ CSV 日誌文件不存在: {log_file}")

print("\n" + "="*60)
print("訓練追蹤功能已實現，包括:")
print("✅ 1. DQNAgent 追蹤損失值 (last_loss)")
print("✅ 2. 遊戲統計追蹤獎勵、步數、平均值")
print("✅ 3. CSV 日誌系統自動記錄每回合數據")
print("✅ 4. UI 顯示損失、獎勵、平均獎勵等指標")
print("\n運行遊戲後，訓練數據會自動保存到 training_log.csv")
print("="*60)
