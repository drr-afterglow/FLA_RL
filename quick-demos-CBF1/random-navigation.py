import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib import animation
from utils import get_robot_state, get_ray_cast
import torch
import random
from env import generate_obstacles_grid, sample_free_start, sample_free_goal
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

# === Constants ===
MAP_HALF_SIZE = 16
OBSTACLE_REGION_MIN = -15
OBSTACLE_REGION_MAX = 15
MIN_RADIUS = 0.3
MAX_RADIUS = 0.5
MAX_RAY_LENGTH = 4.0
DT = 0.1
GOAL_REACHED_THRESHOLD = 0.3
HRES_DEG = 10.0
VFOV_ANGLES_DEG = [-10.0, 0.0, 10.0, 20.0]
GRID_DIV = 10

# [新增] CBF 相关参数 (与训练时保持一致)
COST_GAMMA = 1.0 
SAFETY_MARGIN = 0.15  # 你微调后的参数
ROBOT_RADIUS = 0.1    # 假设机器人半径

# === Setup ===
obstacles = generate_obstacles_grid(GRID_DIV, OBSTACLE_REGION_MIN, OBSTACLE_REGION_MAX, MIN_RADIUS, MAX_RADIUS)
robot_vel = np.array([0.0, 0.0])
# 动态生成起点和终点
goal = sample_free_goal(obstacles, obstacle_region_min=OBSTACLE_REGION_MIN, obstacle_region_max=OBSTACLE_REGION_MAX)
robot_pos = sample_free_start(obstacles, goal, obstacle_region_min=OBSTACLE_REGION_MIN, obstacle_region_max=OBSTACLE_REGION_MAX)
start_pos = robot_pos.copy()
target_dir = goal - robot_pos 
trajectory = []

# === NavRL Agent ===
agent = Agent(device=device)

# === [新增] CBF 计算逻辑 ===
def calculate_cbf_index(robot_pos, robot_vel, obstacles):
    """
    计算当前的 CBF 安全指数 h(x)
    h(x) = d_dot + gamma * (d - r_safe)
    """
    min_h = 100.0
    
    for ox, oy, r in obstacles:
        obs_pos = np.array([ox, oy])
        rel_pos = robot_pos - obs_pos 
        dist = np.linalg.norm(rel_pos)
        
        # 安全半径 = 障碍物半径 + 机器人半径 + 安全余量
        r_safe = r + ROBOT_RADIUS + SAFETY_MARGIN
        
        if dist > r_safe + 3.0: continue # 忽略远处的

        d_dot = np.dot(rel_pos, robot_vel) / (dist + 1e-6)
        h = d_dot + COST_GAMMA * (dist - r_safe)
        
        if h < min_h:
            min_h = h
            
    return min_h


# === Visualization setup ===
fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor('#fefcfb')  
ax.set_facecolor('#fdf6e3')         
ax.set_xlim(-MAP_HALF_SIZE, MAP_HALF_SIZE)
ax.set_ylim(-MAP_HALF_SIZE, MAP_HALF_SIZE)
ax.set_aspect('equal')
ax.set_title(f"NavRL Dynamic (CBF Monitored)\nGamma={COST_GAMMA}, Margin={SAFETY_MARGIN}")

robot_dot, = ax.plot([], [], 'o', markersize=6, color="royalblue" , label='Robot', zorder=5)
velocity_arrow = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1.5, width=0.005, color='purple', zorder=4)
goal_dot, = ax.plot([], [], marker='*', markersize=15, color='red', linestyle='None', label='Goal')
start_dot, = ax.plot([], [], marker='s', markersize=8, color='navy', label='Start', linestyle='None', zorder=3)
trajectory_line, = ax.plot([], [], '-', linewidth=1.5, color="lime", label='Trajectory')
ray_lines = [ax.plot([], [], 'r--', linewidth=0.5)[0] for _ in range(int(360 / HRES_DEG))]

# [新增] h(x) 显示面板
h_text = ax.text(-MAP_HALF_SIZE + 1, MAP_HALF_SIZE - 2, "Initializing...", fontsize=12, bbox=dict(facecolor='white', alpha=0.9))

ax.legend(loc='upper left')

# Store obstacle patches for color updates
obstacle_patches = []
for obs in obstacles:
    # 绘制实心障碍物
    patch = Circle((obs[0], obs[1]), obs[2], color='gray', zorder=2)
    ax.add_patch(patch)
    obstacle_patches.append(patch)
    
    # [新增] 绘制安全边界 (红色虚线)
    # 这表示 Agent 认为的“绝对禁区”边界
    safe_bound = Circle((obs[0], obs[1]), obs[2] + SAFETY_MARGIN + ROBOT_RADIUS, 
                        color='red', fill=False, linestyle='--', alpha=0.3, zorder=1)
    ax.add_patch(safe_bound)

perception_wedge = Wedge(center=(0, 0), r=MAX_RAY_LENGTH, theta1=0, theta2=120,
                         color='cyan', alpha=0.2)
ax.add_patch(perception_wedge)


# === Simulation update ===
def update(frame):
    global robot_pos, robot_vel, goal, trajectory, target_dir, start_pos

    # Goal reach check
    to_goal = goal - robot_pos
    dist = np.linalg.norm(to_goal)
    if dist < GOAL_REACHED_THRESHOLD:
        # [动态重置逻辑]
        start_pos = goal.copy() # 上一个终点变成新的起点
        goal[:] = sample_free_goal(obstacles, obstacle_region_min=OBSTACLE_REGION_MIN, obstacle_region_max=OBSTACLE_REGION_MAX)
        trajectory = [] # 清空轨迹
        velocity = np.array([0.0, 0.0])
        target_dir = goal - robot_pos 
        
        # 到达时显示绿色
        h_text.set_text("GOAL REACHED!")
        h_text.set_backgroundcolor('lightgreen')
        return [robot_dot, goal_dot, trajectory_line, perception_wedge, start_dot, velocity_arrow, h_text] + ray_lines


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

    # Output the planned velocity
    velocity = agent.plan(robot_state, static_obs_input, dyn_obs_input, target_dir_tensor)
    
    # [数据清洗]
    if isinstance(velocity, torch.Tensor):
        velocity = velocity.detach().cpu().numpy().flatten()
    velocity = np.array(velocity, dtype=np.float64)

    # === [新增] CBF 实时监测 ===
    h_val = calculate_cbf_index(robot_pos, velocity, obstacles)
    
    h_text.set_text(f"CBF h(x): {h_val:.2f}")
    if h_val < 0:
        h_text.set_backgroundcolor('salmon') # 违规报警
        robot_dot.set_color('red')
    else:
        h_text.set_backgroundcolor('white')  # 安全
        robot_dot.set_color('royalblue')
    # ==========================

    # ---Visualizaton update---
    robot_dot.set_data([robot_pos[0]], [robot_pos[1]])
    start_dot.set_data([start_pos[0]], [start_pos[1]])
    goal_dot.set_data([goal[0]], [goal[1]])
    trajectory.append(robot_pos.copy())
    trajectory_np = np.array(trajectory)
    
    if len(trajectory) > 0:
        trajectory_line.set_data(trajectory_np[:, 0], trajectory_np[:, 1])

    # [感知扇区更新逻辑]
    perception_center = robot_pos.copy()
    direction_angle_deg = np.degrees(np.arctan2(target_dir[1], target_dir[0]))
    cover_degree = 180
    start_angle = direction_angle_deg - cover_degree
    end_angle = direction_angle_deg + cover_degree

    perception_wedge.set_center((perception_center[0], perception_center[1]))
    perception_wedge.set_theta1(start_angle)
    perception_wedge.set_theta2(end_angle)

    # [障碍物高亮逻辑]
    for patch, (ox, oy, r) in zip(obstacle_patches, obstacles):
        dx, dy = ox - robot_pos[0], oy - robot_pos[1]
        dist_to_robot = np.hypot(dx, dy)
        angle_to_obs = np.degrees(np.arctan2(dy, dx))
        angle_diff = (angle_to_obs - direction_angle_deg + 180) % 360 - 180

        if abs(angle_diff) <= cover_degree and dist_to_robot <= MAX_RAY_LENGTH + r:
            patch.set_color('orange')  # 感知范围内变橙色
        else:
            patch.set_color('gray')    # 范围外灰色

    # [RayCast 线条更新]
    v0_idx = VFOV_ANGLES_DEG.index(0.0) 
    for i, (line, seg) in enumerate(zip(ray_lines, ray_segments)):
        ray_angle_deg = direction_angle_deg + i * HRES_DEG
        angle_diff = (ray_angle_deg - direction_angle_deg + 180) % 360 - 180
        ray_range = range_matrix[i, v0_idx]

        if abs(angle_diff) <= cover_degree and ray_range < MAX_RAY_LENGTH:
            line.set_data([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]])
            line.set_visible(True)
            line.set_color('blue' if i == 0 else 'red')
            line.set_linewidth(1.0)
        else:
            line.set_visible(False)

    velocity_arrow.set_offsets([robot_pos])
    velocity_arrow.set_UVC(robot_vel[0], robot_vel[1])
    # ---Visualizaton update end---

    # Update simulation states
    robot_pos += velocity * DT
    robot_vel = velocity.copy()

    return [robot_dot, goal_dot, trajectory_line, perception_wedge, start_dot, velocity_arrow, h_text] + ray_lines

# 增加帧数，让它多跑几个来回
ani = animation.FuncAnimation(fig, update, frames=1000, interval=20, blit=False)
plt.show()