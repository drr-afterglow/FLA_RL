import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib import animation
from utils import get_robot_state, get_ray_cast
import torch
import random
from env import generate_obstacles_grid, generate_tunnel_obstacles, generate_matrix_obstacles
from agent import Agent

# === Set random seed ===
SEED = 0 
random.seed(SEED)
np.random.seed(SEED)

# === Device ===
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# === Constants (必须与训练时 config.yaml 保持一致) ===
MAP_HALF_SIZE = 20
OBSTACLE_REGION_MIN = -15
OBSTACLE_REGION_MAX = 15
MIN_RADIUS = 0.5
MAX_RADIUS = 1.0
MAX_RAY_LENGTH = 4.0
DT = 0.1
GOAL_REACHED_THRESHOLD = 0.5 # 训练时是 0.5
HRES_DEG = 10.0
VFOV_ANGLES_DEG = [-10.0, 0.0, 10.0, 20.0]
GRID_DIV = 7

# [关键参数] 与训练时的 safety 部分保持一致
COST_GAMMA = 1.0 
SAFETY_MARGIN = 0.15  # 你最后一次微调使用的参数
ROBOT_RADIUS = 0.1    # 假设机器人半径

# === Setup ===
# 场景1 障碍物网格
# # 生成障碍物 (x, y, radius)
obstacles = generate_obstacles_grid(GRID_DIV, OBSTACLE_REGION_MIN, OBSTACLE_REGION_MAX, MIN_RADIUS, MAX_RADIUS)

# 简单的场景初始化
robot_vel = np.array([0.0, 0.0])
goal = np.array([5.0, 18.0])
robot_pos = np.array([0.0, -18.0])
start_pos = robot_pos.copy()
target_dir = goal - robot_pos 


#场景2 隧道
# obstacles = generate_tunnel_obstacles(OBSTACLE_REGION_MIN, OBSTACLE_REGION_MAX, num_obstacles=25)
# # 配合隧道，建议把起点和终点设在两头，而不是随机
# robot_vel = np.array([0.0, 0.0])
# robot_pos = np.array([-18.0, 0.0]) # 左边起点
# goal = np.array([18.0, 0.0])       # 右边终点
# start_pos = robot_pos.copy()
# target_dir = goal - robot_pos

#场景3 矩阵障碍
# obstacles = generate_matrix_obstacles(OBSTACLE_REGION_MIN, OBSTACLE_REGION_MAX, rows=5, cols=6)

# # 矩阵模式适合上下穿越
# robot_vel = np.array([0.0, 0.0])
# robot_pos = np.array([0.0, -18.0]) # 下方起点
# goal = np.array([0.0, 18.0])       # 上方终点
# start_pos = robot_pos.copy()
# target_dir = goal - robot_pos

trajectory = []

# === NavRL Agent ===
agent = Agent(device=device)

# === CBF Calculation Logic (从 env.py 移植) ===
def calculate_cbf_index(robot_pos, robot_vel, obstacles):
    """
    计算当前的 CBF 安全指数 h(x)
    h(x) = d_dot + gamma * (d - r_safe)
    如果 h < 0，说明违规
    """
    min_h = 100.0
    dangerous_obs = None
    
    for ox, oy, r in obstacles:
        # 1. 计算相对位置和距离
        obs_pos = np.array([ox, oy])
        rel_pos = robot_pos - obs_pos # 向量：障碍物 -> 机器人
        dist = np.linalg.norm(rel_pos)
        
        # 训练中的 r_safe = margin + r_drone + r_obs
        r_safe = r + ROBOT_RADIUS + SAFETY_MARGIN
        
        # 只计算附近的障碍物
        if dist > r_safe + 3.0:
            continue

        # 2. 计算距离变化率 d_dot
        # d_dot = (rel_pos · rel_vel) / dist
        # rel_vel = v_robot - v_obs (这里 v_obs=0)
        d_dot = np.dot(rel_pos, robot_vel) / (dist + 1e-6)
        
        # 3. CBF 公式
        # 注意：这里 rel_pos 是 robot - obs，这代表远离的方向
        # 如果 robot 冲向障碍物，np.dot 会是负数，d_dot < 0
        h = d_dot + COST_GAMMA * (dist - r_safe)
        
        if h < min_h:
            min_h = h
            dangerous_obs = (ox, oy, r)
            
    return min_h, dangerous_obs

# === Visualization setup ===
fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor('#fefcfb') 
ax.set_facecolor('#fdf6e3')
ax.set_xlim(-MAP_HALF_SIZE, MAP_HALF_SIZE)
ax.set_ylim(-MAP_HALF_SIZE, MAP_HALF_SIZE)
ax.set_aspect('equal')
ax.set_title(f"NavRL Demo (Gamma={COST_GAMMA}, Margin={SAFETY_MARGIN})")

robot_dot, = ax.plot([], [], 'o', markersize=8, color="royalblue" , label='Robot', zorder=5)
goal_dot, = ax.plot([], [], marker='*', markersize=15, color='red', linestyle='None', label='Goal')
start_dot, = ax.plot([], [], marker='s', markersize=8, color='navy', label='Start', linestyle='None', zorder=3)
trajectory_line, = ax.plot([], [], '-', linewidth=1.5, color="lime", label='Trajectory')
ray_lines = [ax.plot([], [], 'r--', linewidth=0.5)[0] for _ in range(int(360 / HRES_DEG))]

# 增加一个显示 h(x) 的文本框
h_text = ax.text(-18, 18, "h(x): Init", fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

ax.legend(loc='upper left')

# 绘制障碍物
for obs in obstacles:
    # 绘制实体
    ax.add_patch(Circle((obs[0], obs[1]), obs[2], color='gray', zorder=2))
    # 绘制安全边界 (虚线)
    ax.add_patch(Circle((obs[0], obs[1]), obs[2] + SAFETY_MARGIN + ROBOT_RADIUS, color='red', fill=False, linestyle='--', alpha=0.3))


# === Simulation update ===
def update(frame):
    global robot_pos, robot_vel, goal, trajectory, target_dir, start_pos

    # Goal reach check
    to_goal = goal - robot_pos
    dist = np.linalg.norm(to_goal)
    if dist < GOAL_REACHED_THRESHOLD:
        print("Goal Reached!")
        h_text.set_text("GOAL REACHED!")
        h_text.set_color("green")
        # ani.event_source.stop() # 可选：停止
        return []

    # Get robot internal states
    robot_state = get_robot_state(robot_pos, goal, robot_vel, target_dir, device=device)

    # Get static obstacle representations
    static_obs_input, range_matrix, ray_segments = get_ray_cast(robot_pos, obstacles, max_range=MAX_RAY_LENGTH,
                                                       hres_deg=HRES_DEG,
                                                       vfov_angles_deg=VFOV_ANGLES_DEG,
                                                       start_angle_deg=np.degrees(np.arctan2(target_dir[1], target_dir[0])),
                                                       device=device)
    # Get dynamic obstacle representations
    dyn_obs_input = torch.zeros((1, 1, 5, 10), dtype=torch.float, device=device)

    # Target direction in tensor
    target_dir_tensor = torch.tensor(np.append(target_dir[:2], 0.0), dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)

    # ================= 核心：Agent 决策 =================
    # 这里我们不做任何干预，完全信任训练好的 Agent
    # 如果训练得好，它输出的 velocity 应该自动满足 CBF 约束
    velocity = agent.plan(robot_state, static_obs_input, dyn_obs_input, target_dir_tensor)
    
    # 清洗数据
    if isinstance(velocity, torch.Tensor):
        velocity = velocity.detach().cpu().numpy().flatten()
    velocity = np.array(velocity, dtype=np.float64)
    # ==================================================

    # ================= 验证：CBF 实时监测 =================
    # 计算当前状态的安全指数
    h_val, dangerous_obs = calculate_cbf_index(robot_pos, velocity, obstacles)
    
    h_text.set_text(f"CBF h(x): {h_val:.2f}")
    
    if h_val < 0:
        # 如果 h < 0，说明当前速度违反了安全约束（Agent 犯错了）
        h_text.set_backgroundcolor('red')
        robot_dot.set_color('red') # 机器人变红报警
        # print(f"Warning: Constraint Violation! h={h_val:.2f}")
    else:
        # 安全
        h_text.set_backgroundcolor('white')
        robot_dot.set_color('royalblue')
    # ===================================================

    # ---Visualizaton update---
    robot_dot.set_data([robot_pos[0]], [robot_pos[1]])
    start_dot.set_data([start_pos[0]], [start_pos[1]])
    goal_dot.set_data([goal[0]], [goal[1]])
    
    trajectory.append(robot_pos.copy())
    trajectory_np = np.array(trajectory)
    
    if not np.isnan(trajectory_np).any():
        trajectory_line.set_data(trajectory_np[:, 0], trajectory_np[:, 1])
    
    # Update rays
    for i, line in enumerate(ray_lines):
        if i < len(ray_segments):
            line.set_data([ray_segments[i][0][0], ray_segments[i][1][0]],
                          [ray_segments[i][0][1], ray_segments[i][1][1]])

    # Update simulation states
    robot_pos += velocity * DT
    robot_vel = velocity.copy()

    return [robot_dot, goal_dot, trajectory_line, start_dot, h_text] + ray_lines

ani = animation.FuncAnimation(fig, update, frames=600, interval=50, blit=False)
# print("开始保存 GIF，渲染可能需要几分钟...")
# ani.save('nav_demo.gif', writer='pillow', fps=20)
# print("GIF 保存成功！")
plt.show()