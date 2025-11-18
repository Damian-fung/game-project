import pygame
import numpy as np
import random
import math
import sys
import os
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
from datetime import datetime

# 遊戲常量
GRID_WIDTH = 25
GRID_HEIGHT = 20
CELL_SIZE = 30
FPS = 10

# UI 面板寬度
UI_PANEL_WIDTH = 300

# 初始化Pygame
pygame.init()
# 設置可調整大小的視窗
screen = pygame.display.set_mode((GRID_WIDTH * CELL_SIZE + UI_PANEL_WIDTH, GRID_HEIGHT * CELL_SIZE), pygame.RESIZABLE)
pygame.display.set_caption("深度學習貪吃蛇AI對戰 - 改進版")
clock = pygame.time.Clock()

# 方向常量
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

# 全局暫停變量
paused = False

# 3D顯示模式
is_3d_mode = False

# 顏色定義
BACKGROUND = (15, 20, 30)
GRID_COLOR = (40, 50, 80)
UI_BACKGROUND = (25, 30, 45, 200)
UI_BORDER = (70, 100, 150)
TEXT_COLOR = (220, 230, 255)
HIGHLIGHT_COLOR = (100, 180, 255)
BUTTON_COLOR = (60, 100, 180)
BUTTON_HOVER = (80, 140, 220)

# 3D專用顏色
FLOOR_COLOR = (30, 40, 60)
WALL_COLOR = (50, 60, 90)
LIGHT_COLOR = (255, 255, 220, 50)

# 障礙物顏色
OBSTACLE_COLOR = (120, 80, 40)
OBSTACLE_HIGHLIGHT = (150, 110, 70)

# 深度學習模型定義
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, input_size, output_size, model_path=None):
        self.input_size = input_size
        self.output_size = output_size
        self.model = DQN(input_size, output_size)
        self.target_model = DQN(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = ReplayMemory(10000)
        self.batch_size = 32
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # 提高最小探索率
        self.epsilon_decay = 0.998  # 減慢衰減
        self.update_target_every = 100
        self.steps = 0
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.target_model.load_state_dict(self.model.state_dict())
            print(f"載入已有模型: {model_path}")
        
    def get_state(self, snake, foods, other_snake, obstacles):
        """獲取遊戲狀態作為神經網絡輸入 - 增強版，包含障礙物感知"""
        head = snake.body[0]
        
        # 方向特徵 (one-hot)
        dir_left = 1 if snake.direction == LEFT else 0
        dir_right = 1 if snake.direction == RIGHT else 0
        dir_up = 1 if snake.direction == UP else 0
        dir_down = 1 if snake.direction == DOWN else 0
        
        # 最近食物的相對位置
        closest_food = None
        min_distance = float('inf')
        for food in foods:
            dist = abs(food.position[0] - head[0]) + abs(food.position[1] - head[1])
            if dist < min_distance:
                min_distance = dist
                closest_food = food
        
        food_left = 1 if closest_food and closest_food.position[0] < head[0] else 0
        food_right = 1 if closest_food and closest_food.position[0] > head[0] else 0
        food_up = 1 if closest_food and closest_food.position[1] < head[1] else 0
        food_down = 1 if closest_food and closest_food.position[1] > head[1] else 0
        
        # 危險檢測 (前、左、右三個方向)
        danger_straight = 0
        danger_left = 0
        danger_right = 0
        
        # 計算前方位置
        front = ((head[0] + snake.direction[0]), (head[1] + snake.direction[1]))
        
        # 計算左方位置 (相對當前方向的左轉)
        if snake.direction == UP:
            left_dir = LEFT
        elif snake.direction == DOWN:
            left_dir = RIGHT
        elif snake.direction == LEFT:
            left_dir = DOWN
        else:  # RIGHT
            left_dir = UP
            
        left = ((head[0] + left_dir[0]), (head[1] + left_dir[1]))
        
        # 計算右方位置 (相對當前方向的右轉)
        if snake.direction == UP:
            right_dir = RIGHT
        elif snake.direction == DOWN:
            right_dir = LEFT
        elif snake.direction == LEFT:
            right_dir = UP
        else:  # RIGHT
            right_dir = DOWN
            
        right = ((head[0] + right_dir[0]), (head[1] + right_dir[1]))
        
        # 檢查危險
        danger_straight = self._is_dangerous(front, snake, other_snake, obstacles)
        danger_left = self._is_dangerous(left, snake, other_snake, obstacles)
        danger_right = self._is_dangerous(right, snake, other_snake, obstacles)
        
        # 新增：障礙物距離感知 (八個方向)
        obstacle_nearby = [0] * 8  # 上、右上、右、右下、下、左下、左、左上
        
        # 檢查八個方向的障礙物
        directions_8 = [
            (0, -1),   # 上
            (1, -1),   # 右上
            (1, 0),    # 右
            (1, 1),    # 右下
            (0, 1),    # 下
            (-1, 1),   # 左下
            (-1, 0),   # 左
            (-1, -1)   # 左上
        ]
        
        for i, (dx, dy) in enumerate(directions_8):
            check_pos = (head[0] + dx, head[1] + dy)
            if self._is_dangerous(check_pos, snake, other_snake, obstacles):
                obstacle_nearby[i] = 1
        
        # 組合成狀態向量 (11個基礎特徵 + 8個障礙物特徵 = 19維)
        state = [
            # 方向 (4維)
            dir_left, dir_right, dir_up, dir_down,
            # 食物位置 (4維)
            food_left, food_right, food_up, food_down,
            # 危險 (3維)
            danger_straight, danger_left, danger_right,
            # 障礙物感知 (8維)
            *obstacle_nearby
        ]
        
        return np.array(state, dtype=np.float32)
    
    def _is_dangerous(self, pos, snake, other_snake, obstacles):
        """檢查位置是否危險（撞牆、自己、對方、障礙物）"""
        # 檢查是否超出邊界
        if pos[0] < 0 or pos[0] >= GRID_WIDTH or pos[1] < 0 or pos[1] >= GRID_HEIGHT:
            return 1
            
        # 檢查是否撞到自己
        if pos in snake.body[1:]:
            return 1
            
        # 檢查是否撞到對方
        if pos in other_snake.body:
            return 1
            
        # 檢查是否撞到障礙物
        if pos in obstacles:
            return 1
            
        return 0
    
    def get_action(self, state):
        """根據當前狀態選擇動作"""
        if random.random() < self.epsilon:
            return random.randint(0, self.output_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.max(1)[1].item()
    
    def remember(self, state, action, next_state, reward, done):
        """記住經驗"""
        self.memory.push(state, action, next_state, reward, done)
    
    def replay(self):
        """從記憶中學習"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.BoolTensor(dones)
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 定期更新目標網絡
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
    
    def save_model(self, path):
        """保存模型"""
        torch.save(self.model.state_dict(), path)

class Snake:
    def __init__(self, pos, color, name, snake_id):
        self.body = [pos]
        self.direction = RIGHT
        self.color = color
        self.name = name
        self.snake_id = snake_id
        self.score = 0
        self.alive = True
        self.total_steps = 0
        self.last_action = None
        self.stuck_count = 0  # 記錄被困的幀數

    def move(self, action=None, obstacles=None):
        if not self.alive:
            return

        self.total_steps += 1
        self.last_action = action

        if action is not None:
            self._ai_move(action, obstacles)
        else:
            self._normal_move(obstacles)

    def _ai_move(self, action, obstacles):
        # action: 0 up,1 down,2 left,3 right
        if action == 0:
            new_direction = UP
        elif action == 1:
            new_direction = DOWN
        elif action == 2:
            new_direction = LEFT
        elif action == 3:
            new_direction = RIGHT
        else:
            new_direction = self.direction

        # 防止直接反向移動
        if (new_direction[0] * -1, new_direction[1] * -1) != self.direction:
            self.direction = new_direction

        self._apply_move(obstacles)

    def _normal_move(self, obstacles):
        self._apply_move(obstacles)

    def _apply_move(self, obstacles):
        head = self.body[0]
        new_pos = (
            head[0] + self.direction[0],
            head[1] + self.direction[1]
        )
        
        # 檢查新位置是否可移動（不是障礙物且不超出邊界）
        if (0 <= new_pos[0] < GRID_WIDTH and 
            0 <= new_pos[1] < GRID_HEIGHT and 
            new_pos not in obstacles):
            
            # 移動身體
            self.body.insert(0, new_pos)
            self.body.pop()
            self.stuck_count = 0  # 重置被困計數
        else:
            # 如果無法移動，增加被困計數
            self.stuck_count += 1
            
            # 如果被困太久，嘗試隨機轉向
            if self.stuck_count > 5:
                self._try_escape(obstacles)

    def _try_escape(self, obstacles):
        """嘗試逃脫被困狀態"""
        head = self.body[0]
        
        # 嘗試所有可能的方向
        possible_directions = []
        for direction in [UP, DOWN, LEFT, RIGHT]:
            if direction == (self.direction[0] * -1, self.direction[1] * -1):
                continue  # 跳過反向移動
                
            new_pos = (head[0] + direction[0], head[1] + direction[1])
            if (0 <= new_pos[0] < GRID_WIDTH and 
                0 <= new_pos[1] < GRID_HEIGHT and 
                new_pos not in obstacles):
                possible_directions.append(direction)
        
        # 如果有可移動的方向，隨機選擇一個
        if possible_directions:
            self.direction = random.choice(possible_directions)
            self.stuck_count = 0

    def grow(self):
        # 增加身體長度（複製尾端）
        self.body.append(self.body[-1])
        self.score += 1

    def check_collision(self, other_snake):
        """檢查碰撞（牆壁、自己、對方）- 移除障礙物檢查"""
        head = self.body[0]
        
        # 檢查是否撞牆
        if head[0] < 0 or head[0] >= GRID_WIDTH or head[1] < 0 or head[1] >= GRID_HEIGHT:
            return True
            
        # 檢查是否撞到自己
        if head in self.body[1:]:
            return True
            
        # 檢查是否撞到對方
        if head in other_snake.body:
            return True
            
        return False

    def draw(self, surface):
        for i, segment in enumerate(self.body):
            x, y = segment
            
            if is_3d_mode:
                self.draw_3d_segment(surface, x, y, i == 0)
            else:
                self.draw_2d_segment(surface, x, y, i == 0)
    
    def draw_2d_segment(self, surface, x, y, is_head):
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)

        if is_head:  # 頭部
            # 繪製帶陰影的頭部
            shadow_rect = pygame.Rect(x * CELL_SIZE + 2, y * CELL_SIZE + 2, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, (max(0, self.color[0]-30), max(0, self.color[1]-30), max(0, self.color[2]-30)), shadow_rect, border_radius=5)
            pygame.draw.rect(surface, self.color, rect, border_radius=5)
            
            # 繪製眼睛表示方向
            self.draw_eyes(surface, x, y)
            
            # 繪製嘴巴
            self.draw_mouth(surface, x, y)
        else:  # 身體
            # 繪製帶陰影的身體
            shadow_rect = pygame.Rect(x * CELL_SIZE + 2, y * CELL_SIZE + 2, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, (max(0, self.color[0]-30), max(0, self.color[1]-30), max(0, self.color[2]-30)), shadow_rect, border_radius=5)
            pygame.draw.rect(surface, self.color, rect, border_radius=5)
            
            # 繪製身體紋理
            inner_rect = pygame.Rect(x * CELL_SIZE + 5, y * CELL_SIZE + 5, CELL_SIZE - 10, CELL_SIZE - 10)
            pygame.draw.rect(surface, (min(255, self.color[0]+30), min(255, self.color[1]+30), min(255, self.color[2]+30)), inner_rect, border_radius=3)

    def draw_3d_segment(self, surface, x, y, is_head):
        # 等角投影參數
        iso_scale = 0.7
        iso_x = (x - y) * (CELL_SIZE * iso_scale)
        iso_y = (x + y) * (CELL_SIZE * iso_scale / 2)
        
        # 調整位置到屏幕中央
        offset_x = GRID_WIDTH * CELL_SIZE * 0.3
        offset_y = GRID_HEIGHT * CELL_SIZE * 0.1
        iso_x += offset_x
        iso_y += offset_y
        
        # 3D方塊的高度
        height = CELL_SIZE * 0.9 if is_head else CELL_SIZE * 0.7
        
        # 計算3D方塊的頂點
        points_top = [
            (iso_x, iso_y),  # 左上
            (iso_x + CELL_SIZE * iso_scale, iso_y + CELL_SIZE * iso_scale / 2),  # 右上
            (iso_x + CELL_SIZE * iso_scale, iso_y + CELL_SIZE * iso_scale / 2 + height),  # 右下
            (iso_x, iso_y + height)  # 左下
        ]
        
        # 繪製3D方塊的側面
        side_color = (
            max(0, self.color[0] - 40),
            max(0, self.color[1] - 40),
            max(0, self.color[2] - 40)
        )
        
        # 繪製頂面
        pygame.draw.polygon(surface, self.color, points_top)
        
        # 繪製側面
        side_points = [
            (iso_x + CELL_SIZE * iso_scale, iso_y + CELL_SIZE * iso_scale / 2),
            (iso_x + CELL_SIZE * iso_scale, iso_y + CELL_SIZE * iso_scale / 2 + height),
            (iso_x + CELL_SIZE * iso_scale * 1.2, iso_y + CELL_SIZE * iso_scale / 2 + height * 0.5),
            (iso_x + CELL_SIZE * iso_scale * 1.2, iso_y + CELL_SIZE * iso_scale / 2 - height * 0.5)
        ]
        pygame.draw.polygon(surface, side_color, side_points)
        
        # 繪製前面
        front_color = (
            max(0, self.color[0] - 20),
            max(0, self.color[1] - 20),
            max(0, self.color[2] - 20)
        )
        front_points = [
            (iso_x, iso_y + height),
            (iso_x + CELL_SIZE * iso_scale, iso_y + CELL_SIZE * iso_scale / 2 + height),
            (iso_x + CELL_SIZE * iso_scale * 1.2, iso_y + CELL_SIZE * iso_scale / 2 + height * 0.5),
            (iso_x + CELL_SIZE * iso_scale * 0.2, iso_y + height * 0.5)
        ]
        pygame.draw.polygon(surface, front_color, front_points)
        
        # 繪製高光效果
        highlight_points = [
            (iso_x + CELL_SIZE * iso_scale * 0.2, iso_y + CELL_SIZE * iso_scale * 0.1),
            (iso_x + CELL_SIZE * iso_scale * 0.8, iso_y + CELL_SIZE * iso_scale * 0.4),
            (iso_x + CELL_SIZE * iso_scale * 0.7, iso_y + CELL_SIZE * iso_scale * 0.5),
            (iso_x + CELL_SIZE * iso_scale * 0.1, iso_y + CELL_SIZE * iso_scale * 0.2)
        ]
        highlight_color = (
            min(255, self.color[0] + 50),
            min(255, self.color[1] + 50),
            min(255, self.color[2] + 50),
            100
        )
        
        # 創建透明表面繪製高光
        s = pygame.Surface((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.polygon(s, highlight_color, highlight_points)
        surface.blit(s, (0, 0))
        
        # 如果是頭部，繪製眼睛和嘴巴
        if is_head:
            self.draw_3d_eyes(surface, iso_x, iso_y)
            self.draw_3d_mouth(surface, iso_x, iso_y)
            
            # 繪製頭部裝飾
            self.draw_3d_decoration(surface, iso_x, iso_y)

    def draw_3d_eyes(self, surface, iso_x, iso_y):
        # 3D模式下的眼睛位置
        eye_radius = CELL_SIZE // 8
        center_x = iso_x + CELL_SIZE * 0.35
        center_y = iso_y + CELL_SIZE * 0.2
        
        if self.direction == RIGHT:
            eye1_pos = (center_x + CELL_SIZE//4, center_y - CELL_SIZE//6)
            eye2_pos = (center_x + CELL_SIZE//4, center_y + CELL_SIZE//6)
        elif self.direction == LEFT:
            eye1_pos = (center_x - CELL_SIZE//4, center_y - CELL_SIZE//6)
            eye2_pos = (center_x - CELL_SIZE//4, center_y + CELL_SIZE//6)
        elif self.direction == UP:
            eye1_pos = (center_x - CELL_SIZE//6, center_y - CELL_SIZE//4)
            eye2_pos = (center_x + CELL_SIZE//6, center_y - CELL_SIZE//4)
        else:  # DOWN
            eye1_pos = (center_x - CELL_SIZE//6, center_y + CELL_SIZE//4)
            eye2_pos = (center_x + CELL_SIZE//6, center_y + CELL_SIZE//4)

        # 繪製眼睛陰影
        pygame.draw.circle(surface, (30, 30, 30), (eye1_pos[0]+1, eye1_pos[1]+1), eye_radius)
        pygame.draw.circle(surface, (30, 30, 30), (eye2_pos[0]+1, eye2_pos[1]+1), eye_radius)
        
        # 繪製眼睛
        pygame.draw.circle(surface, (255, 255, 255), eye1_pos, eye_radius)
        pygame.draw.circle(surface, (255, 255, 255), eye2_pos, eye_radius)
        pygame.draw.circle(surface, (0, 0, 0), eye1_pos, eye_radius//2)
        pygame.draw.circle(surface, (0, 0, 0), eye2_pos, eye_radius//2)
        
        # 繪製眼睛高光
        pygame.draw.circle(surface, (255, 255, 255), (eye1_pos[0]-eye_radius//3, eye1_pos[1]-eye_radius//3), eye_radius//4)
        pygame.draw.circle(surface, (255, 255, 255), (eye2_pos[0]-eye_radius//3, eye2_pos[1]-eye_radius//3), eye_radius//4)

    def draw_3d_mouth(self, surface, iso_x, iso_y):
        center_x = iso_x + CELL_SIZE * 0.35
        center_y = iso_y + CELL_SIZE * 0.2
        
        if self.direction == RIGHT:
            mouth_points = [
                (center_x + CELL_SIZE//4, center_y),
                (center_x + CELL_SIZE//3, center_y + CELL_SIZE//10),
                (center_x + CELL_SIZE//4, center_y + CELL_SIZE//5)
            ]
        elif self.direction == LEFT:
            mouth_points = [
                (center_x - CELL_SIZE//4, center_y),
                (center_x - CELL_SIZE//3, center_y + CELL_SIZE//10),
                (center_x - CELL_SIZE//4, center_y + CELL_SIZE//5)
            ]
        elif self.direction == UP:
            mouth_points = [
                (center_x, center_y - CELL_SIZE//4),
                (center_x - CELL_SIZE//10, center_y - CELL_SIZE//3),
                (center_x + CELL_SIZE//10, center_y - CELL_SIZE//3)
            ]
        else:  # DOWN
            mouth_points = [
                (center_x, center_y + CELL_SIZE//4),
                (center_x - CELL_SIZE//10, center_y + CELL_SIZE//3),
                (center_x + CELL_SIZE//10, center_y + CELL_SIZE//3)
            ]
        
        pygame.draw.polygon(surface, (150, 50, 50), mouth_points)

    def draw_3d_decoration(self, surface, iso_x, iso_y):
        # 繪製頭部裝飾 - 角或鱗片
        center_x = iso_x + CELL_SIZE * 0.35
        center_y = iso_y + CELL_SIZE * 0.2
        
        if self.direction == RIGHT:
            # 繪製角
            horn_color = (min(255, self.color[0] + 20), min(255, self.color[1] + 20), min(255, self.color[2] + 20))
            horn_points = [
                (center_x + CELL_SIZE//3, center_y - CELL_SIZE//3),
                (center_x + CELL_SIZE//2, center_y - CELL_SIZE//2),
                (center_x + CELL_SIZE//3, center_y - CELL_SIZE//4)
            ]
            pygame.draw.polygon(surface, horn_color, horn_points)
        elif self.direction == LEFT:
            horn_color = (min(255, self.color[0] + 20), min(255, self.color[1] + 20), min(255, self.color[2] + 20))
            horn_points = [
                (center_x - CELL_SIZE//3, center_y - CELL_SIZE//3),
                (center_x - CELL_SIZE//2, center_y - CELL_SIZE//2),
                (center_x - CELL_SIZE//3, center_y - CELL_SIZE//4)
            ]
            pygame.draw.polygon(surface, horn_color, horn_points)
        elif self.direction == UP:
            # 繪製背鰭
            fin_color = (min(255, self.color[0] + 20), min(255, self.color[1] + 20), min(255, self.color[2] + 20))
            fin_points = [
                (center_x, center_y - CELL_SIZE//3),
                (center_x - CELL_SIZE//6, center_y - CELL_SIZE//2),
                (center_x + CELL_SIZE//6, center_y - CELL_SIZE//2)
            ]
            pygame.draw.polygon(surface, fin_color, fin_points)

    def draw_eyes(self, surface, x, y):
        center_x = x * CELL_SIZE + CELL_SIZE // 2
        center_y = y * CELL_SIZE + CELL_SIZE // 2
        eye_radius = CELL_SIZE // 6

        if self.direction == RIGHT:
            eye1_pos = (center_x + CELL_SIZE//4, center_y - CELL_SIZE//4)
            eye2_pos = (center_x + CELL_SIZE//4, center_y + CELL_SIZE//4)
        elif self.direction == LEFT:
            eye1_pos = (center_x - CELL_SIZE//4, center_y - CELL_SIZE//4)
            eye2_pos = (center_x - CELL_SIZE//4, center_y + CELL_SIZE//4)
        elif self.direction == UP:
            eye1_pos = (center_x - CELL_SIZE//4, center_y - CELL_SIZE//4)
            eye2_pos = (center_x + CELL_SIZE//4, center_y - CELL_SIZE//4)
        else:  # DOWN
            eye1_pos = (center_x - CELL_SIZE//4, center_y + CELL_SIZE//4)
            eye2_pos = (center_x + CELL_SIZE//4, center_y + CELL_SIZE//4)

        # 繪製眼睛陰影
        pygame.draw.circle(surface, (30, 30, 30), (eye1_pos[0]+1, eye1_pos[1]+1), eye_radius)
        pygame.draw.circle(surface, (30, 30, 30), (eye2_pos[0]+1, eye2_pos[1]+1), eye_radius)
        
        # 繪製眼睛
        pygame.draw.circle(surface, (255, 255, 255), eye1_pos, eye_radius)
        pygame.draw.circle(surface, (255, 255, 255), eye2_pos, eye_radius)
        pygame.draw.circle(surface, (0, 0, 0), eye1_pos, eye_radius//2)
        pygame.draw.circle(surface, (0, 0, 0), eye2_pos, eye_radius//2)
        
        # 繪製眼睛高光
        pygame.draw.circle(surface, (255, 255, 255), (eye1_pos[0]-eye_radius//3, eye1_pos[1]-eye_radius//3), eye_radius//4)
        pygame.draw.circle(surface, (255, 255, 255), (eye2_pos[0]-eye_radius//3, eye2_pos[1]-eye_radius//3), eye_radius//4)

    def draw_mouth(self, surface, x, y):
        center_x = x * CELL_SIZE + CELL_SIZE // 2
        center_y = y * CELL_SIZE + CELL_SIZE // 2
        
        if self.direction == RIGHT:
            pygame.draw.arc(surface, (150, 50, 50), 
                           (center_x + CELL_SIZE//6, center_y - CELL_SIZE//6, CELL_SIZE//3, CELL_SIZE//3),
                           0, math.pi/2, 2)
        elif self.direction == LEFT:
            pygame.draw.arc(surface, (150, 50, 50), 
                           (center_x - CELL_SIZE//2, center_y - CELL_SIZE//6, CELL_SIZE//3, CELL_SIZE//3),
                           math.pi/2, math.pi, 2)
        elif self.direction == UP:
            pygame.draw.arc(surface, (150, 50, 50), 
                           (center_x - CELL_SIZE//6, center_y - CELL_SIZE//2, CELL_SIZE//3, CELL_SIZE//3),
                           math.pi, 3*math.pi/2, 2)
        else:  # DOWN
            pygame.draw.arc(surface, (150, 50, 50), 
                           (center_x - CELL_SIZE//6, center_y + CELL_SIZE//6, CELL_SIZE//3, CELL_SIZE//3),
                           3*math.pi/2, 2*math.pi, 2)


class Food:
    def __init__(self):
        self.position = self.generate_position()
        self.animation_offset = random.random() * 2 * math.pi
        self.rotation = 0

    def generate_position(self, snakes=None, obstacles=None, other_foods=None):
        attempts = 0
        while attempts < 100:
            pos = (
                random.randint(0, GRID_WIDTH - 1),
                random.randint(0, GRID_HEIGHT - 1)
            )
            
            # 檢查位置是否被佔用
            valid = True
            
            # 檢查蛇
            if snakes:
                for snake in snakes:
                    if pos in snake.body:
                        valid = False
                        break
            
            # 檢查障礙物
            if obstacles and pos in obstacles:
                valid = False
                
            # 檢查其他食物
            if other_foods and pos in [f.position for f in other_foods]:
                valid = False
                
            if valid:
                return pos
                
            attempts += 1
            
        # 如果找不到合適位置，返回隨機位置
        return (
            random.randint(0, GRID_WIDTH - 1),
            random.randint(0, GRID_HEIGHT - 1)
        )

    def draw(self, surface):
        x, y = self.position
        
        if is_3d_mode:
            self.draw_3d(surface, x, y)
        else:
            self.draw_2d(surface, x, y)
    
    def draw_2d(self, surface, x, y):
        # 計算脈動效果
        pulse = math.sin(pygame.time.get_ticks() / 200 + self.animation_offset) * 0.1 + 0.9
        
        center_x = x * CELL_SIZE + CELL_SIZE // 2
        center_y = y * CELL_SIZE + CELL_SIZE // 2
        radius = CELL_SIZE // 2 * pulse
        
        # 繪製陰影
        pygame.draw.circle(surface, (180, 0, 0), (center_x + 2, center_y + 2), radius)
        
        # 繪製食物主體
        pygame.draw.circle(surface, (255, 50, 50), (center_x, center_y), radius)
        
        # 繪製高光
        highlight_radius = radius * 0.4
        highlight_pos = (center_x - radius//2, center_y - radius//2)
        pygame.draw.circle(surface, (255, 150, 150), highlight_pos, highlight_radius)
        
        # 繪製葉子
        leaf_angle = pygame.time.get_ticks() / 500
        leaf_x = center_x + math.cos(leaf_angle) * radius * 0.7
        leaf_y = center_y - math.sin(leaf_angle) * radius * 0.7
        leaf_radius = radius * 0.4
        
        leaf_points = []
        for i in range(5):
            angle = i * 2 * math.pi / 5 + leaf_angle
            point_x = leaf_x + math.cos(angle) * leaf_radius
            point_y = leaf_y + math.sin(angle) * leaf_radius
            leaf_points.append((point_x, point_y))
        
        pygame.draw.polygon(surface, (100, 200, 100), leaf_points)
    
    def draw_3d(self, surface, x, y):
        # 等角投影參數
        iso_scale = 0.7
        iso_x = (x - y) * (CELL_SIZE * iso_scale)
        iso_y = (x + y) * (CELL_SIZE * iso_scale / 2)
        
        # 調整位置到屏幕中央
        offset_x = GRID_WIDTH * CELL_SIZE * 0.3
        offset_y = GRID_HEIGHT * CELL_SIZE * 0.1
        iso_x += offset_x
        iso_y += offset_y
        
        # 計算脈動效果
        pulse = math.sin(pygame.time.get_ticks() / 200 + self.animation_offset) * 0.1 + 0.9
        
        # 3D蘋果的高度
        height = CELL_SIZE * 0.6 * pulse
        
        # 計算3D蘋果的頂點
        points_top = [
            (iso_x, iso_y),  # 左上
            (iso_x + CELL_SIZE * iso_scale, iso_y + CELL_SIZE * iso_scale / 2),  # 右上
            (iso_x + CELL_SIZE * iso_scale, iso_y + CELL_SIZE * iso_scale / 2 + height),  # 右下
            (iso_x, iso_y + height)  # 左下
        ]
        
        # 繪製3D蘋果
        pygame.draw.polygon(surface, (255, 50, 50), points_top)
        
        # 繪製側面
        side_points = [
            (iso_x + CELL_SIZE * iso_scale, iso_y + CELL_SIZE * iso_scale / 2),
            (iso_x + CELL_SIZE * iso_scale, iso_y + CELL_SIZE * iso_scale / 2 + height),
            (iso_x + CELL_SIZE * iso_scale * 1.2, iso_y + CELL_SIZE * iso_scale / 2 + height * 0.5),
            (iso_x + CELL_SIZE * iso_scale * 1.2, iso_y + CELL_SIZE * iso_scale / 2 - height * 0.5)
        ]
        side_color = (200, 40, 40)
        pygame.draw.polygon(surface, side_color, side_points)
        
        # 繪製前面
        front_color = (220, 60, 60)
        front_points = [
            (iso_x, iso_y + height),
            (iso_x + CELL_SIZE * iso_scale, iso_y + CELL_SIZE * iso_scale / 2 + height),
            (iso_x + CELL_SIZE * iso_scale * 1.2, iso_y + CELL_SIZE * iso_scale / 2 + height * 0.5),
            (iso_x + CELL_SIZE * iso_scale * 0.2, iso_y + height * 0.5)
        ]
        pygame.draw.polygon(surface, front_color, front_points)
        
        # 繪製高光
        highlight_points = [
            (iso_x + CELL_SIZE * iso_scale * 0.2, iso_y + CELL_SIZE * iso_scale * 0.1),
            (iso_x + CELL_SIZE * iso_scale * 0.8, iso_y + CELL_SIZE * iso_scale * 0.4),
            (iso_x + CELL_SIZE * iso_scale * 0.7, iso_y + CELL_SIZE * iso_scale * 0.5),
            (iso_x + CELL_SIZE * iso_scale * 0.1, iso_y + CELL_SIZE * iso_scale * 0.2)
        ]
        highlight_color = (255, 150, 150, 100)
        
        # 創建透明表面繪製高光
        s = pygame.Surface((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.polygon(s, highlight_color, highlight_points)
        surface.blit(s, (0, 0))
        
        # 繪製蘋果梗
        stem_start = (iso_x + CELL_SIZE * iso_scale * 0.5, iso_y)
        stem_end = (iso_x + CELL_SIZE * iso_scale * 0.5, iso_y - CELL_SIZE * 0.2)
        pygame.draw.line(surface, (139, 69, 19), stem_start, stem_end, 3)
        
        # 繪製葉子
        leaf_angle = pygame.time.get_ticks() / 500
        leaf_center = (iso_x + CELL_SIZE * iso_scale * 0.6, iso_y - CELL_SIZE * 0.1)
        leaf_radius = CELL_SIZE * 0.15
        
        leaf_points = []
        for i in range(5):
            angle = i * 2 * math.pi / 5 + leaf_angle
            point_x = leaf_center[0] + math.cos(angle) * leaf_radius
            point_y = leaf_center[1] + math.sin(angle) * leaf_radius
            leaf_points.append((point_x, point_y))
        
        pygame.draw.polygon(surface, (100, 200, 100), leaf_points)
        
        # 繪製蘋果光暈
        glow_radius = CELL_SIZE * 0.8 * pulse
        glow_center_x = iso_x + CELL_SIZE * iso_scale / 2
        glow_center_y = iso_y + CELL_SIZE * iso_scale / 4
        
        s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        for r in range(int(glow_radius), 0, -2):
            alpha = int(50 * (1 - r/glow_radius))
            pygame.draw.circle(s, (255, 100, 100, alpha), (glow_radius, glow_radius), r)
        surface.blit(s, (glow_center_x - glow_radius, glow_center_y - glow_radius))


class Obstacle:
    def __init__(self, x, y):
        self.position = (x, y)
        self.animation_offset = random.random() * 2 * math.pi

    def draw(self, surface):
        x, y = self.position
        
        if is_3d_mode:
            self.draw_3d(surface, x, y)
        else:
            self.draw_2d(surface, x, y)
    
    def draw_2d(self, surface, x, y):
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        
        # 繪製陰影
        shadow_rect = pygame.Rect(x * CELL_SIZE + 2, y * CELL_SIZE + 2, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, (max(0, OBSTACLE_COLOR[0]-30), max(0, OBSTACLE_COLOR[1]-30), max(0, OBSTACLE_COLOR[2]-30)), shadow_rect, border_radius=3)
        
        # 繪製障礙物主體
        pygame.draw.rect(surface, OBSTACLE_COLOR, rect, border_radius=3)
        
        # 繪製紋理
        texture_rect = pygame.Rect(x * CELL_SIZE + 5, y * CELL_SIZE + 5, CELL_SIZE - 10, CELL_SIZE - 10)
        pygame.draw.rect(surface, OBSTACLE_HIGHLIGHT, texture_rect, border_radius=2)
        
        # 繪製裂縫效果
        for i in range(3):
            crack_start = (x * CELL_SIZE + random.randint(5, CELL_SIZE-5), 
                          y * CELL_SIZE + random.randint(5, CELL_SIZE-5))
            crack_end = (x * CELL_SIZE + random.randint(5, CELL_SIZE-5), 
                        y * CELL_SIZE + random.randint(5, CELL_SIZE-5))
            pygame.draw.line(surface, (80, 50, 20), crack_start, crack_end, 1)
    
    def draw_3d(self, surface, x, y):
        # 等角投影參數
        iso_scale = 0.7
        iso_x = (x - y) * (CELL_SIZE * iso_scale)
        iso_y = (x + y) * (CELL_SIZE * iso_scale / 2)
        
        # 調整位置到屏幕中央
        offset_x = GRID_WIDTH * CELL_SIZE * 0.3
        offset_y = GRID_HEIGHT * CELL_SIZE * 0.1
        iso_x += offset_x
        iso_y += offset_y
        
        # 3D障礙物的高度
        height = CELL_SIZE * 1.0
        
        # 計算3D障礙物的頂點
        points_top = [
            (iso_x, iso_y),  # 左上
            (iso_x + CELL_SIZE * iso_scale, iso_y + CELL_SIZE * iso_scale / 2),  # 右上
            (iso_x + CELL_SIZE * iso_scale, iso_y + CELL_SIZE * iso_scale / 2 + height),  # 右下
            (iso_x, iso_y + height)  # 左下
        ]
        
        # 繪製3D障礙物的頂面
        pygame.draw.polygon(surface, OBSTACLE_COLOR, points_top)
        
        # 繪製側面
        side_points = [
            (iso_x + CELL_SIZE * iso_scale, iso_y + CELL_SIZE * iso_scale / 2),
            (iso_x + CELL_SIZE * iso_scale, iso_y + CELL_SIZE * iso_scale / 2 + height),
            (iso_x + CELL_SIZE * iso_scale * 1.2, iso_y + CELL_SIZE * iso_scale / 2 + height * 0.5),
            (iso_x + CELL_SIZE * iso_scale * 1.2, iso_y + CELL_SIZE * iso_scale / 2 - height * 0.5)
        ]
        side_color = (
            max(0, OBSTACLE_COLOR[0] - 30),
            max(0, OBSTACLE_COLOR[1] - 30),
            max(0, OBSTACLE_COLOR[2] - 30)
        )
        pygame.draw.polygon(surface, side_color, side_points)
        
        # 繪製前面
        front_color = (
            max(0, OBSTACLE_COLOR[0] - 15),
            max(0, OBSTACLE_COLOR[1] - 15),
            max(0, OBSTACLE_COLOR[2] - 15)
        )
        front_points = [
            (iso_x, iso_y + height),
            (iso_x + CELL_SIZE * iso_scale, iso_y + CELL_SIZE * iso_scale / 2 + height),
            (iso_x + CELL_SIZE * iso_scale * 1.2, iso_y + CELL_SIZE * iso_scale / 2 + height * 0.5),
            (iso_x + CELL_SIZE * iso_scale * 0.2, iso_y + height * 0.5)
        ]
        pygame.draw.polygon(surface, front_color, front_points)
        
        # 繪製裂縫紋理
        for i in range(3):
            crack_start_x = iso_x + random.randint(5, int(CELL_SIZE * iso_scale) - 5)
            crack_start_y = iso_y + random.randint(5, int(CELL_SIZE * iso_scale / 2) - 5)
            crack_end_x = iso_x + random.randint(5, int(CELL_SIZE * iso_scale) - 5)
            crack_end_y = iso_y + random.randint(5, int(CELL_SIZE * iso_scale / 2) - 5)
            
            pygame.draw.line(surface, (80, 50, 20), 
                           (crack_start_x, crack_start_y),
                           (crack_end_x, crack_end_y), 2)


class AttackEffect:
    """簡單的攻擊動畫效果 — 在格子上顯示短暫粒子/閃爍"""
    def __init__(self, pos, life_frames=10):
        self.pos = pos
        self.life = life_frames
        self.max_life = life_frames
        self.particles = []
        self.create_particles()

    def create_particles(self):
        # 創建粒子效果
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            size = random.uniform(2, 5)
            life = random.randint(5, self.max_life)
            self.particles.append({
                'angle': angle,
                'speed': speed,
                'size': size,
                'life': life,
                'x': 0,
                'y': 0
            })

    def update(self):
        self.life -= 1
        for p in self.particles:
            p['life'] -= 1
            p['x'] += math.cos(p['angle']) * p['speed']
            p['y'] += math.sin(p['angle']) * p['speed']

    def draw(self, surface):
        x, y = self.pos
        
        if is_3d_mode:
            self.draw_3d(surface, x, y)
        else:
            self.draw_2d(surface, x, y)
    
    def draw_2d(self, surface, x, y):
        center_x = x * CELL_SIZE + CELL_SIZE // 2
        center_y = y * CELL_SIZE + CELL_SIZE // 2
        
        # 繪製爆炸效果
        for p in self.particles:
            if p['life'] > 0:
                alpha = int(255 * p['life'] / self.max_life)
                color = (255, 200 + random.randint(0, 55), 50, alpha)
                pos_x = center_x + p['x']
                pos_y = center_y + p['y']
                
                s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
                surface.blit(s, (pos_x - p['size'], pos_y - p['size']))
    
    def draw_3d(self, surface, x, y):
        # 等角投影參數
        iso_scale = 0.7
        iso_x = (x - y) * (CELL_SIZE * iso_scale)
        iso_y = (x + y) * (CELL_SIZE * iso_scale / 2)
        
        # 調整位置到屏幕中央
        offset_x = GRID_WIDTH * CELL_SIZE * 0.3
        offset_y = GRID_HEIGHT * CELL_SIZE * 0.1
        iso_x += offset_x
        iso_y += offset_y
        
        center_x = iso_x + CELL_SIZE * iso_scale / 2
        center_y = iso_y + CELL_SIZE * iso_scale / 4
        
        # 繪製爆炸效果
        for p in self.particles:
            if p['life'] > 0:
                alpha = int(255 * p['life'] / self.max_life)
                color = (255, 200 + random.randint(0, 55), 50, alpha)
                pos_x = center_x + p['x']
                pos_y = center_y + p['y']
                
                s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
                surface.blit(s, (pos_x - p['size'], pos_y - p['size']))
        
        # 繪製3D爆炸立方體
        if self.life > self.max_life * 0.7:
            explosion_height = CELL_SIZE * 0.8 * (self.life / self.max_life)
            
            # 計算3D爆炸立方體的頂點
            points_top = [
                (iso_x, iso_y),  # 左上
                (iso_x + CELL_SIZE * iso_scale, iso_y + CELL_SIZE * iso_scale / 2),  # 右上
                (iso_x + CELL_SIZE * iso_scale, iso_y + CELL_SIZE * iso_scale / 2 + explosion_height),  # 右下
                (iso_x, iso_y + explosion_height)  # 左下
            ]
            
            # 繪製爆炸立方體
            explosion_color = (255, 150, 50, int(200 * (self.life / self.max_life)))
            s = pygame.Surface((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.polygon(s, explosion_color, points_top)
            surface.blit(s, (0, 0))


class Game:
    def __init__(self, training_mode=True):
        self.snake1 = Snake((5, 10), (50, 200, 50), "AI蛇1", 0)
        self.snake2 = Snake((GRID_WIDTH-6, 10), (50, 50, 200), "AI蛇2", 1)
        # 初始身體都給 3 格，避免剛開始超短容易被扣光
        self.snake1.body = [(5,10),(4,10),(3,10)]
        self.snake2.body = [(GRID_WIDTH-6,10),(GRID_WIDTH-7,10),(GRID_WIDTH-8,10)]

        # 創建兩個食物
        self.foods = [Food(), Food()]
        
        # 創建障礙物
        self.obstacles = self.create_obstacles()
        
        self.running = True
        self.font = pygame.font.SysFont('simhei', 20)
        self.small_font = pygame.font.SysFont('simhei', 16)
        self.title_font = pygame.font.SysFont('simhei', 24, bold=True)

        # 創建深度學習AI代理 - 輸入維度增加到19（包含障礙物感知）
        self.input_size = 19  # 狀態特徵數量 (11基礎 + 8障礙物)
        self.output_size = 4  # 動作數量
        
        # 確保模型保存目錄存在
        self.model_dir = "snake_models"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # 加載或創建AI代理
        self.ai1 = DQNAgent(self.input_size, self.output_size, 
                           os.path.join(self.model_dir, "snake1_model.pth"))
        self.ai2 = DQNAgent(self.input_size, self.output_size, 
                           os.path.join(self.model_dir, "snake2_model.pth"))
        
        # 訓練模式
        self.training_mode = training_mode
        
        # 遊戲統計
        self.game_stats = {
            "episode": 0,
            "wins": [0, 0],  # [蛇1勝利次數, 蛇2勝利次數]
            "total_food": 0,
            "training_episodes": 0,
            "max_steps": 0
        }

        # 攻擊動畫效果列表
        self.attack_effects = []
        
        # 回合記憶
        self.episode_memory = []
        
        # 確保記憶保存目錄存在
        self.memory_dir = "snake_memories"
        if not os.path.exists(self.memory_dir):
            os.makedirs(self.memory_dir)

        # 按鈕狀態
        self.buttons = {
            "3d_toggle": {"rect": pygame.Rect(0, 0, 0, 0), "hover": False},
            "train_toggle": {"rect": pygame.Rect(0, 0, 0, 0), "hover": False},
            "reset": {"rect": pygame.Rect(0, 0, 0, 0), "hover": False},
            "save": {"rect": pygame.Rect(0, 0, 0, 0), "hover": False}
        }
        
        # 3D環境元素
        self.floor_tiles = []
        self.walls = []
        self.lights = []
        self.init_3d_environment()

    def create_obstacles(self):
        """創建障礙物 - 減少障礙物數量，讓遊戲更容易"""
        obstacles = []
        
        # 創建較少的障礙物，避免遊戲太快結束
        obstacle_positions = [
            # 中間障礙物群組 (2個方塊)
            (GRID_WIDTH//2, GRID_HEIGHT//2),
            (GRID_WIDTH//2 + 1, GRID_HEIGHT//2),
            
            # 左上角障礙物
            (GRID_WIDTH//4, GRID_HEIGHT//4),
            
            # 右下角障礙物
            (GRID_WIDTH*3//4, GRID_HEIGHT*3//4),
        ]
        
        for pos in obstacle_positions:
            obstacles.append(Obstacle(pos[0], pos[1]))
            
        return obstacles

    def init_3d_environment(self):
        # 初始化3D環境 - 地板瓷磚
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                self.floor_tiles.append((x, y))
        
        # 初始化3D環境 - 牆壁
        for x in range(GRID_WIDTH):
            self.walls.append((x, -1))  # 上邊界
            self.walls.append((x, GRID_HEIGHT))  # 下邊界
        for y in range(GRID_HEIGHT):
            self.walls.append((-1, y))  # 左邊界
            self.walls.append((GRID_WIDTH, y))  # 右邊界
            
        # 初始化3D環境 - 光源
        self.lights = [
            {"pos": (GRID_WIDTH//4, GRID_HEIGHT//4), "radius": GRID_WIDTH * 0.8, "intensity": 0.3},
            {"pos": (GRID_WIDTH*3//4, GRID_HEIGHT*3//4), "radius": GRID_WIDTH * 0.6, "intensity": 0.2}
        ]

    def draw_3d_environment(self, surface):
        # 繪製3D地板
        for x, y in self.floor_tiles:
            iso_scale = 0.7
            iso_x = (x - y) * (CELL_SIZE * iso_scale)
            iso_y = (x + y) * (CELL_SIZE * iso_scale / 2)
            
            # 調整位置到屏幕中央
            offset_x = GRID_WIDTH * CELL_SIZE * 0.3
            offset_y = GRID_HEIGHT * CELL_SIZE * 0.1
            iso_x += offset_x
            iso_y += offset_y
            
            # 計算地板瓷磚顏色 - 棋盤格效果
            if (x + y) % 2 == 0:
                floor_color = FLOOR_COLOR
            else:
                floor_color = (
                    min(255, FLOOR_COLOR[0] + 10),
                    min(255, FLOOR_COLOR[1] + 10),
                    min(255, FLOOR_COLOR[2] + 10)
                )
            
            # 繪製地板瓷磚
            points = [
                (iso_x, iso_y),
                (iso_x + CELL_SIZE * iso_scale, iso_y + CELL_SIZE * iso_scale / 2),
                (iso_x + CELL_SIZE * iso_scale, iso_y + CELL_SIZE * iso_scale / 2 + CELL_SIZE * 0.1),
                (iso_x, iso_y + CELL_SIZE * 0.1)
            ]
            pygame.draw.polygon(surface, floor_color, points)
            
            # 繪製地板瓷磚邊框
            pygame.draw.polygon(surface, (floor_color[0] - 10, floor_color[1] - 10, floor_color[2] - 10), points, 1)
        
        # 繪製3D牆壁
        for x, y in self.walls:
            iso_scale = 0.7
            iso_x = (x - y) * (CELL_SIZE * iso_scale)
            iso_y = (x + y) * (CELL_SIZE * iso_scale / 2)
            
            # 調整位置到屏幕中央
            offset_x = GRID_WIDTH * CELL_SIZE * 0.3
            offset_y = GRID_HEIGHT * CELL_SIZE * 0.1
            iso_x += offset_x
            iso_y += offset_y
            
            # 牆壁高度
            wall_height = CELL_SIZE * 1.2
            
            # 繪製牆壁
            points_top = [
                (iso_x, iso_y),
                (iso_x + CELL_SIZE * iso_scale, iso_y + CELL_SIZE * iso_scale / 2),
                (iso_x + CELL_SIZE * iso_scale, iso_y + CELL_SIZE * iso_scale / 2 + wall_height),
                (iso_x, iso_y + wall_height)
            ]
            pygame.draw.polygon(surface, WALL_COLOR, points_top)
            
            # 繪製牆壁側面
            side_points = [
                (iso_x + CELL_SIZE * iso_scale, iso_y + CELL_SIZE * iso_scale / 2),
                (iso_x + CELL_SIZE * iso_scale, iso_y + CELL_SIZE * iso_scale / 2 + wall_height),
                (iso_x + CELL_SIZE * iso_scale * 1.2, iso_y + CELL_SIZE * iso_scale / 2 + wall_height * 0.5),
                (iso_x + CELL_SIZE * iso_scale * 1.2, iso_y + CELL_SIZE * iso_scale / 2 - wall_height * 0.5)
            ]
            side_color = (WALL_COLOR[0] - 20, WALL_COLOR[1] - 20, WALL_COLOR[2] - 20)
            pygame.draw.polygon(surface, side_color, side_points)
            
            # 繪製牆壁前面
            front_points = [
                (iso_x, iso_y + wall_height),
                (iso_x + CELL_SIZE * iso_scale, iso_y + CELL_SIZE * iso_scale / 2 + wall_height),
                (iso_x + CELL_SIZE * iso_scale * 1.2, iso_y + CELL_SIZE * iso_scale / 2 + wall_height * 0.5),
                (iso_x + CELL_SIZE * iso_scale * 0.2, iso_y + wall_height * 0.5)
            ]
            front_color = (WALL_COLOR[0] - 10, WALL_COLOR[1] - 10, WALL_COLOR[2] - 10)
            pygame.draw.polygon(surface, front_color, front_points)
        
        # 繪製3D光源效果
        for light in self.lights:
            x, y = light["pos"]
            iso_scale = 0.7
            iso_x = (x - y) * (CELL_SIZE * iso_scale)
            iso_y = (x + y) * (CELL_SIZE * iso_scale / 2)
            
            # 調整位置到屏幕中央
            offset_x = GRID_WIDTH * CELL_SIZE * 0.3
            offset_y = GRID_HEIGHT * CELL_SIZE * 0.1
            iso_x += offset_x
            iso_y += offset_y
            
            # 繪製光源光暈
            radius = light["radius"] * iso_scale
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            for r in range(int(radius), 0, -5):
                alpha = int(light["intensity"] * 100 * (1 - r/radius))
                pygame.draw.circle(s, (LIGHT_COLOR[0], LIGHT_COLOR[1], LIGHT_COLOR[2], alpha), 
                                 (radius, radius), r)
            surface.blit(s, (iso_x - radius, iso_y - radius))

    def handle_events(self):
        global paused, is_3d_mode
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    self.reset_episode()
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_t:
                    self.training_mode = not self.training_mode
                    print(f"訓練模式: {'開啟' if self.training_mode else '關閉'}")
                elif event.key == pygame.K_s:
                    self.save_models()
                    print("模型已保存")
                elif event.key == pygame.K_d:  # 切換3D模式
                    is_3d_mode = not is_3d_mode
                    print(f"3D模式: {'開啟' if is_3d_mode else '關閉'}")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 檢查是否點擊了按鈕
                mouse_pos = pygame.mouse.get_pos()
                for button_name, button_data in self.buttons.items():
                    if button_data["rect"].collidepoint(mouse_pos):
                        if button_name == "3d_toggle":
                            is_3d_mode = not is_3d_mode
                            print(f"3D模式: {'開啟' if is_3d_mode else '關閉'}")
                        elif button_name == "train_toggle":
                            self.training_mode = not self.training_mode
                            print(f"訓練模式: {'開啟' if self.training_mode else '關閉'}")
                        elif button_name == "reset":
                            self.reset_episode()
                        elif button_name == "save":
                            self.save_models()
                            print("模型已保存")
            elif event.type == pygame.MOUSEMOTION:
                # 檢查鼠標懸停
                mouse_pos = pygame.mouse.get_pos()
                for button_name, button_data in self.buttons.items():
                    button_data["hover"] = button_data["rect"].collidepoint(mouse_pos)

    def calculate_reward(self, snake, other_snake, action, prev_state, new_state, done):
        """計算強化學習的獎勵 - 增強版，考慮障礙物避開"""
        reward = 0
        
        # 基礎生存獎勵（鼓勵長時間生存）
        reward += 0.1  # 增加生存獎勵
        
        # 吃到食物的獎勵
        for food in self.foods:
            if snake.body[0] == food.position:
                reward += 15  # 增加食物獎勵
        
        # 攻擊成功的獎勵（讓對方縮短）
        head = snake.body[0]
        for segment in other_snake.body[1:]:
            if head == segment and other_snake.alive:
                reward += 8  # 增加攻擊獎勵
        
        # 避開障礙物的獎勵（在障礙物附近但仍安全）
        head = snake.body[0]
        near_obstacle = False
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            if (head[0] + dx, head[1] + dy) in [obs.position for obs in self.obstacles]:
                near_obstacle = True
                break
        
        if near_obstacle and snake.alive:
            reward += 1  # 在障礙物附近但仍安全的小獎勵
        
        # 死亡的懲罰
        if not snake.alive:
            reward -= 30  # 增加死亡懲罰
        
        # 長時間沒有吃到食物的懲罰
        if snake.total_steps > 200 and snake.score == 0:  # 增加步數限制
            reward -= 0.05
        
        return reward

    def update(self):
        # 記住回合開始時誰還活著（用於計分判斷）
        prev_alive = (self.snake1.alive, self.snake2.alive)
        
        # 獲取當前狀態
        state1 = self.ai1.get_state(self.snake1, self.foods, self.snake2, [obs.position for obs in self.obstacles])
        state2 = self.ai2.get_state(self.snake2, self.foods, self.snake1, [obs.position for obs in self.obstacles])
        
        # AI選擇動作
        action1 = self.ai1.get_action(state1)
        action2 = self.ai2.get_action(state2)
        
        # 更新AI蛇1 - 傳入障礙物列表
        if self.snake1.alive:
            self.snake1.move(action1, [obs.position for obs in self.obstacles])

            # 檢查碰撞（牆壁、自己、對方）- 移除障礙物檢查
            if self.snake1.check_collision(self.snake2):
                self.snake1.alive = False

        # 更新AI蛇2 - 傳入障礙物列表
        if self.snake2.alive:
            self.snake2.move(action2, [obs.position for obs in self.obstacles])

            # 檢查碰撞（牆壁、自己、對方）- 移除障礙物檢查
            if self.snake2.check_collision(self.snake1):
                self.snake2.alive = False

        # 檢查吃食物
        for snake in [self.snake1, self.snake2]:
            if snake.alive:
                for food in self.foods:
                    if snake.body[0] == food.position:
                        snake.grow()
                        # 重新產生食物（避免生成在任一蛇身上或障礙物上）
                        food.position = food.generate_position(
                            [self.snake1, self.snake2], 
                            [obs.position for obs in self.obstacles],
                            [f for f in self.foods if f != food]
                        )
                        self.game_stats["total_food"] += 1

        # 獲取新狀態
        new_state1 = self.ai1.get_state(self.snake1, self.foods, self.snake2, [obs.position for obs in self.obstacles])
        new_state2 = self.ai2.get_state(self.snake2, self.foods, self.snake1, [obs.position for obs in self.obstacles])
        
        # 計算獎勵
        reward1 = self.calculate_reward(self.snake1, self.snake2, action1, state1, new_state1, not self.snake1.alive)
        reward2 = self.calculate_reward(self.snake2, self.snake1, action2, state2, new_state2, not self.snake2.alive)
        
        # 記錄經驗（如果處於訓練模式）
        if self.training_mode:
            self.ai1.remember(state1, action1, new_state1, reward1, not self.snake1.alive)
            self.ai2.remember(state2, action2, new_state2, reward2, not self.snake2.alive)
            
            # 記錄回合數據
            self.episode_memory.append({
                'snake1_state': state1,
                'snake1_action': action1,
                'snake1_reward': reward1,
                'snake1_done': not self.snake1.alive,
                'snake2_state': state2,
                'snake2_action': action2,
                'snake2_reward': reward2,
                'snake2_done': not self.snake2.alive,
                'episode': self.game_stats["episode"]
            })

        # ========== 新的蛇之間碰撞邏輯（含扣尾巴與攻擊動畫） ==========
        # 先處理頭對頭
        if self.snake1.alive and self.snake2.alive:
            head1 = self.snake1.body[0]
            head2 = self.snake2.body[0]

            # 頭對頭碰撞
            if head1 == head2:
                if len(self.snake1.body) > len(self.snake2.body):
                    self.snake2.alive = False
                elif len(self.snake2.body) > len(self.snake1.body):
                    self.snake1.alive = False
                else:
                    # 同長 → 雙死
                    self.snake1.alive = False
                    self.snake2.alive = False

            # 蛇1頭撞蛇2身（包含頭都算，撞頭上面已處理）
            for segment in self.snake2.body[1:]:
                if head1 == segment and self.snake2.alive:
                    # 攻擊動畫：在被撞位置顯示
                    self.attack_effects.append(AttackEffect(segment, life_frames=12))
                    # 扣尾巴或死亡
                    if len(self.snake2.body) > 1:
                        self.snake2.body.pop()
                        # 攻擊者加1分（可改）
                        self.snake1.score += 1
                    else:
                        self.snake2.alive = False

            # 蛇2頭撞蛇1身
            for segment in self.snake1.body[1:]:
                if head2 == segment and self.snake1.alive:
                    self.attack_effects.append(AttackEffect(segment, life_frames=12))
                    if len(self.snake1.body) > 1:
                        self.snake1.body.pop()
                        self.snake2.score += 1
                    else:
                        self.snake1.alive = False

        # 判定誰死亡（本次更新造成的死亡）並更新勝利統計
        # 若同時死亡則視為平手（不加分）
        now_alive = (self.snake1.alive, self.snake2.alive)
        # 如果蛇1 剛死（prev alive True -> now False）且蛇2 現在活著 -> 蛇2 +1 勝
        if prev_alive[0] and not now_alive[0] and now_alive[1]:
            self.game_stats["wins"][1] += 1
        # 如果蛇2 剛死且蛇1 現在活著 -> 蛇1 +1 勝
        if prev_alive[1] and not now_alive[1] and now_alive[0]:
            self.game_stats["wins"][0] += 1

        # 更新最大步數記錄
        current_steps = max(self.snake1.total_steps, self.snake2.total_steps)
        self.game_stats["max_steps"] = max(self.game_stats["max_steps"], current_steps)

        # 如果兩條蛇都死了，視為回合結束（不計勝場），重新開始
        if not self.snake1.alive and not self.snake2.alive:
            self.end_episode("平手")
            return

        # 如果只有一條蛇存活（另一條死），記錄勝利並重置
        if self.snake1.alive and not self.snake2.alive:
            self.end_episode("AI1 勝利")
            return
        if self.snake2.alive and not self.snake1.alive:
            self.end_episode("AI2 勝利")
            return

        # 更新並剔除攻擊動畫
        for effect in list(self.attack_effects):
            effect.update()
            if effect.life <= 0:
                self.attack_effects.remove(effect)

    def end_episode(self, result):
        """結束當前回合並進行訓練"""
        print(f"回合 {self.game_stats['episode']} 完成 - {result}")
        print(f"最大步數: {self.game_stats['max_steps']}")
        
        # 如果是訓練模式，進行學習
        if self.training_mode:
            self.ai1.replay()
            self.ai2.replay()
            
            # 定期保存記憶
            if self.game_stats["episode"] % 50 == 0:
                self.save_episode_memory()
        
        # 定期保存模型
        if self.game_stats["episode"] % 100 == 0:
            self.save_models()
        
        self.game_stats["episode"] += 1
        self.game_stats["training_episodes"] += 1
        self.reset_episode()

    def save_episode_memory(self):
        """保存回合記憶到文件"""
        if not self.episode_memory:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"episode_memory_{timestamp}.pkl"
        filepath = os.path.join(self.memory_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.episode_memory, f)
        
        print(f"回合記憶已保存: {filepath}")
        self.episode_memory = []  # 清空當前記憶

    def save_models(self):
        """保存AI模型"""
        self.ai1.save_model(os.path.join(self.model_dir, "snake1_model.pth"))
        self.ai2.save_model(os.path.join(self.model_dir, "snake2_model.pth"))
        print("AI模型已保存")

    def draw(self):
        # 繪製背景
        screen.fill(BACKGROUND)

        # 如果是3D模式，繪製3D環境
        if is_3d_mode:
            self.draw_3d_environment(screen)
        else:
            # 繪製2D網格
            for x in range(0, GRID_WIDTH * CELL_SIZE, CELL_SIZE):
                pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, GRID_HEIGHT * CELL_SIZE), 1)
            for y in range(0, GRID_HEIGHT * CELL_SIZE, CELL_SIZE):
                pygame.draw.line(screen, GRID_COLOR, (0, y), (GRID_WIDTH * CELL_SIZE, y), 1)

        # 繪製遊戲元素
        for food in self.foods:
            food.draw(screen)
            
        # 繪製障礙物
        for obstacle in self.obstacles:
            obstacle.draw(screen)
            
        self.snake1.draw(screen)
        self.snake2.draw(screen)

        # 繪製攻擊效果（在蛇與食物之後，方便顯示）
        for effect in self.attack_effects:
            effect.draw(screen)

        # 繪製UI
        self.draw_ui()

        pygame.display.flip()

    def draw_ui(self):
        # 計算UI面板位置
        ui_panel_x = GRID_WIDTH * CELL_SIZE
        
        # 繪製UI面板背景
        ui_bg = pygame.Surface((UI_PANEL_WIDTH, GRID_HEIGHT * CELL_SIZE), pygame.SRCALPHA)
        ui_bg.fill(UI_BACKGROUND)
        screen.blit(ui_bg, (ui_panel_x, 0))
        
        # 標題
        title_text = self.title_font.render("深度學習AI貪吃蛇大戰", True, HIGHLIGHT_COLOR)
        screen.blit(title_text, (ui_panel_x + UI_PANEL_WIDTH // 2 - title_text.get_width() // 2, 10))

        # AI狀態面板
        for i, snake in enumerate([self.snake1, self.snake2]):
            y_pos = 50 + i * 100

            # 繪製AI狀態背景
            ai_bg = pygame.Surface((UI_PANEL_WIDTH - 20, 90), pygame.SRCALPHA)
            ai_bg.fill((snake.color[0]//4, snake.color[1]//4, snake.color[2]//4, 150))
            screen.blit(ai_bg, (ui_panel_x + 10, y_pos))
            pygame.draw.rect(screen, snake.color, (ui_panel_x + 10, y_pos, UI_PANEL_WIDTH - 20, 90), 2, border_radius=5)

            # AI標題
            ai_title = self.font.render(f"{snake.name}", True, snake.color)
            screen.blit(ai_title, (ui_panel_x + 20, y_pos + 5))

            # 分數和長度
            score_text = self.small_font.render(f"分數: {snake.score}", True, TEXT_COLOR)
            length_text = self.small_font.render(f"長度: {len(snake.body)}", True, TEXT_COLOR)
            screen.blit(score_text, (ui_panel_x + 20, y_pos + 30))
            screen.blit(length_text, (ui_panel_x + 20, y_pos + 50))

            # 狀態
            status = "存活" if snake.alive else "死亡"
            status_color = (0, 255, 0) if snake.alive else (255, 0, 0)
            status_text = self.small_font.render(f"狀態: {status}", True, status_color)
            screen.blit(status_text, (ui_panel_x + 120, y_pos + 30))
            
            # AI信息
            epsilon = self.ai1.epsilon if i == 0 else self.ai2.epsilon
            ai_info = self.small_font.render(f"探索率: {epsilon:.3f}", True, (200, 200, 100))
            screen.blit(ai_info, (ui_panel_x + 120, y_pos + 50))

        # 中央信息面板
        center_bg = pygame.Surface((UI_PANEL_WIDTH - 20, 120), pygame.SRCALPHA)
        center_bg.fill((40, 45, 60, 200))
        screen.blit(center_bg, (ui_panel_x + 10, 250))
        pygame.draw.rect(screen, UI_BORDER, (ui_panel_x + 10, 250, UI_PANEL_WIDTH - 20, 120), 2, border_radius=5)

        # 回合信息
        episode_text = self.font.render(f"回合: {self.game_stats['episode']}", True, TEXT_COLOR)
        screen.blit(episode_text, (ui_panel_x + UI_PANEL_WIDTH // 2 - episode_text.get_width() // 2, 260))
        
        # 訓練信息
        train_status = "訓練中" if self.training_mode else "演示模式"
        train_color = (0, 255, 0) if self.training_mode else (255, 255, 0)
        train_text = self.font.render(train_status, True, train_color)
        screen.blit(train_text, (ui_panel_x + UI_PANEL_WIDTH // 2 - train_text.get_width() // 2, 285))
        
        # 3D模式信息
        mode_status = "3D模式" if is_3d_mode else "2D模式"
        mode_color = (100, 200, 255) if is_3d_mode else (200, 200, 200)
        mode_text = self.font.render(mode_status, True, mode_color)
        screen.blit(mode_text, (ui_panel_x + UI_PANEL_WIDTH // 2 - mode_text.get_width() // 2, 310))
        
        # 最大步數
        steps_text = self.small_font.render(f"最大步數: {self.game_stats['max_steps']}", True, TEXT_COLOR)
        screen.blit(steps_text, (ui_panel_x + UI_PANEL_WIDTH // 2 - steps_text.get_width() // 2, 335))

        # 控制面板
        controls_bg = pygame.Surface((UI_PANEL_WIDTH - 20, 140), pygame.SRCALPHA)
        controls_bg.fill(UI_BACKGROUND)
        screen.blit(controls_bg, (ui_panel_x + 10, 390))
        pygame.draw.rect(screen, UI_BORDER, (ui_panel_x + 10, 390, UI_PANEL_WIDTH - 20, 140), 2, border_radius=5)

        # 按鈕
        button_width = UI_PANEL_WIDTH - 40
        button_height = 30
        button_y = 400
        
        # 3D切換按鈕
        self.buttons["3d_toggle"]["rect"] = pygame.Rect(ui_panel_x + 20, button_y, button_width, button_height)
        button_color = BUTTON_HOVER if self.buttons["3d_toggle"]["hover"] else BUTTON_COLOR
        pygame.draw.rect(screen, button_color, self.buttons["3d_toggle"]["rect"], border_radius=5)
        pygame.draw.rect(screen, UI_BORDER, self.buttons["3d_toggle"]["rect"], 2, border_radius=5)
        button_text = self.small_font.render("切換3D模式", True, TEXT_COLOR)
        screen.blit(button_text, (ui_panel_x + 20 + (button_width - button_text.get_width()) // 2, button_y + 7))

        # 訓練模式切換按鈕
        self.buttons["train_toggle"]["rect"] = pygame.Rect(ui_panel_x + 20, button_y + 40, button_width, button_height)
        button_color = BUTTON_HOVER if self.buttons["train_toggle"]["hover"] else BUTTON_COLOR
        pygame.draw.rect(screen, button_color, self.buttons["train_toggle"]["rect"], border_radius=5)
        pygame.draw.rect(screen, UI_BORDER, self.buttons["train_toggle"]["rect"], 2, border_radius=5)
        button_text = self.small_font.render("切換訓練模式", True, TEXT_COLOR)
        screen.blit(button_text, (ui_panel_x + 20 + (button_width - button_text.get_width()) // 2, button_y + 47))

        # 重置按鈕
        self.buttons["reset"]["rect"] = pygame.Rect(ui_panel_x + 20, button_y + 80, button_width, button_height)
        button_color = BUTTON_HOVER if self.buttons["reset"]["hover"] else BUTTON_COLOR
        pygame.draw.rect(screen, button_color, self.buttons["reset"]["rect"], border_radius=5)
        pygame.draw.rect(screen, UI_BORDER, self.buttons["reset"]["rect"], 2, border_radius=5)
        button_text = self.small_font.render("重置回合", True, TEXT_COLOR)
        screen.blit(button_text, (ui_panel_x + 20 + (button_width - button_text.get_width()) // 2, button_y + 87))

        # 保存按鈕
        self.buttons["save"]["rect"] = pygame.Rect(ui_panel_x + 20, button_y + 120, button_width, button_height)
        button_color = BUTTON_HOVER if self.buttons["save"]["hover"] else BUTTON_COLOR
        pygame.draw.rect(screen, button_color, self.buttons["save"]["rect"], border_radius=5)
        pygame.draw.rect(screen, UI_BORDER, self.buttons["save"]["rect"], 2, border_radius=5)
        button_text = self.small_font.render("保存模型", True, TEXT_COLOR)
        screen.blit(button_text, (ui_panel_x + 20 + (button_width - button_text.get_width()) // 2, button_y + 127))

        # 控制提示
        controls = [
            "ESC: 退出遊戲",
            "空格: 暫停/繼續",
            "R: 重置回合",
            "T: 切換訓練模式",
            "S: 保存模型",
            "D: 切換3D模式"
        ]

        for i, control in enumerate(controls):
            control_text = self.small_font.render(control, True, TEXT_COLOR)
            screen.blit(control_text, (ui_panel_x + 20, 540 + i * 20))

        # 勝利統計
        wins_text = self.small_font.render(
            f"勝利: AI1 {self.game_stats['wins'][0]} - AI2 {self.game_stats['wins'][1]}",
            True, TEXT_COLOR
        )
        screen.blit(wins_text, (ui_panel_x + UI_PANEL_WIDTH // 2 - wins_text.get_width() // 2, 490))

    def reset_episode(self):
        # 重置蛇（保留初始三格設計）
        self.snake1 = Snake((5, 10), (50, 200, 50), "AI蛇1", 0)
        self.snake2 = Snake((GRID_WIDTH - 6, 10), (50, 50, 200), "AI蛇2", 1)
        self.snake1.body = [(5,10),(4,10),(3,10)]
        self.snake2.body = [(GRID_WIDTH-6,10),(GRID_WIDTH-7,10),(GRID_WIDTH-8,10)]
        
        # 重置食物
        self.foods = [Food(), Food()]
        for food in self.foods:
            food.position = food.generate_position(
                [self.snake1, self.snake2], 
                [obs.position for obs in self.obstacles],
                [f for f in self.foods if f != food]
            )
            
        self.attack_effects.clear()

    # 添加缺失的 run 方法
    def run(self):
        print("深度學習AI貪吃蛇對戰開始!")
        print("按 ESC 退出, R 重置, 空格 暫停, T 切換訓練模式, S 保存模型, D 切換3D模式")
        print("注意：現在有兩個蘋果和障礙物，蛇碰到障礙物不會死亡!")

        while self.running:
            self.handle_events()

            if not paused:
                self.update()

            self.draw()
            clock.tick(FPS)

        # 退出前保存
        self.save_models()
        self.save_episode_memory()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    # 可以通過命令行參數控制是否訓練模式
    training_mode = True
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        training_mode = False
    
    game = Game(training_mode=training_mode)
    game.run()