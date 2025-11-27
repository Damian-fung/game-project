#!/usr/bin/env python3
"""
訓練數據分析腳本
使用方法: python analyze_training.py
"""

import csv
import os

def analyze_training_log(log_file="training_log.csv"):
    """分析訓練日誌並顯示統計資訊"""
    
    if not os.path.exists(log_file):
        print(f"❌ 找不到日誌文件: {log_file}")
        return
    
    with open(log_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    if not data:
        print("⚠️  日誌文件為空，請先運行遊戲訓練")
        return
    
    print("="*70)
    print("🎮 深度學習 AI 貪吃蛇訓練數據分析")
    print("="*70)
    
    total_episodes = len(data)
    print(f"\n📊 總訓練回合數: {total_episodes}")
    
    # 勝率統計
    snake1_wins = sum(1 for row in data if row['winner'] == 'AI1 勝利')
    snake2_wins = sum(1 for row in data if row['winner'] == 'AI2 勝利')
    draws = sum(1 for row in data if row['winner'] == '平手')
    
    print(f"\n🏆 勝負統計:")
    print(f"   AI蛇1 勝利: {snake1_wins} ({snake1_wins/max(1,total_episodes)*100:.1f}%)")
    print(f"   AI蛇2 勝利: {snake2_wins} ({snake2_wins/max(1,total_episodes)*100:.1f}%)")
    print(f"   平手: {draws} ({draws/max(1,total_episodes)*100:.1f}%)")
    
    # 平均分數
    avg_score1 = sum(float(row['snake1_score']) for row in data) / total_episodes
    avg_score2 = sum(float(row['snake2_score']) for row in data) / total_episodes
    
    print(f"\n📈 平均分數:")
    print(f"   AI蛇1: {avg_score1:.2f}")
    print(f"   AI蛇2: {avg_score2:.2f}")
    
    # 平均長度
    avg_len1 = sum(float(row['snake1_length']) for row in data) / total_episodes
    avg_len2 = sum(float(row['snake2_length']) for row in data) / total_episodes
    
    print(f"\n📏 平均長度:")
    print(f"   AI蛇1: {avg_len1:.2f}")
    print(f"   AI蛇2: {avg_len2:.2f}")
    
    # 平均回合步數
    avg_steps = sum(float(row['steps']) for row in data) / total_episodes
    print(f"\n⏱️  平均回合步數: {avg_steps:.1f}")
    
    # 探索率趨勢
    latest_epsilon1 = float(data[-1]['snake1_epsilon'])
    latest_epsilon2 = float(data[-1]['snake2_epsilon'])
    
    print(f"\n🔍 當前探索率:")
    print(f"   AI蛇1: {latest_epsilon1:.4f}")
    print(f"   AI蛇2: {latest_epsilon2:.4f}")
    
    # 訓練損失趨勢（最近10回合）
    recent_data = data[-10:] if len(data) > 10 else data
    avg_loss1 = sum(float(row['snake1_loss']) for row in recent_data) / len(recent_data)
    avg_loss2 = sum(float(row['snake2_loss']) for row in recent_data) / len(recent_data)
    
    print(f"\n📉 平均訓練損失 (最近 {len(recent_data)} 回合):")
    print(f"   AI蛇1: {avg_loss1:.4f}")
    print(f"   AI蛇2: {avg_loss2:.4f}")
    
    # 平均獎勵
    avg_reward1 = sum(float(row['snake1_reward']) for row in recent_data) / len(recent_data)
    avg_reward2 = sum(float(row['snake2_reward']) for row in recent_data) / len(recent_data)
    
    print(f"\n🎁 平均累積獎勵 (最近 {len(recent_data)} 回合):")
    print(f"   AI蛇1: {avg_reward1:.2f}")
    print(f"   AI蛇2: {avg_reward2:.2f}")
    
    print("\n" + "="*70)
    print("💡 提示:")
    print("   - 探索率應逐漸降低（1.0 → 0.01）")
    print("   - 損失應逐漸穩定並降低")
    print("   - 平均分數和長度應逐漸提高")
    print("="*70)

if __name__ == "__main__":
    analyze_training_log()
