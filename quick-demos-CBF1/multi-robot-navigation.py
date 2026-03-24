import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation
import torch
import random
from env import generate_obstacles_grid
from utils import get_robot_state, get_ray_cast, get_dyn_obs_state
from agent import Agent
from matplotlib import cm

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
MAP_HALF_SIZE = 20
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
NUM_ROBOTS = 8

# [新增] CBF 参数 (与训练保持一致)
COST_GAMMA = 1.0
SAFETY_MARGIN = 0.15
ROBOT_RADIUS = 0.1

# === Setup ===
obstacles = generate_obstacles_grid(GRID_DIV, OBSTACLE_REGION_MIN, OBSTACLE_REGION_MAX, MIN_RADIUS, MAX_RADIUS)

# 均匀分布起点
robot_xs = np.linspace(OBSTACLE_REGION_MIN + 3, OBSTACLE_REGION_MAX - 3, NUM_ROBOTS)
robot_positions = [np.array([x, -18.0]) for x in robot_xs]
robot_velocities = [np.zeros(2) for _ in range(NUM_ROBOTS)]
# 目标在正对面
goals = [np.array([x, 18.0]) for x in robot_xs]
target_dirs = [goals[i] - robot_positions[i] for i in range(len(goals))]
trajectories = [[p.copy()] for p in robot_positions]

# 状态追踪：记录每个机器人是否已到达
finished_status = [False] * NUM_ROBOTS

# === NavRL Agent ===
agent = Agent(device=device)

# === [复用] CBF 计算逻辑 ===
def calculate_cbf_index(robot_pos, robot_vel, obstacles):
    """
    计算单个机器人的 CBF 安全指数 h(x)
    """
    min_h = 100.0
    
    for ox, oy, r in obstacles:
        obs_pos = np.array([ox, oy])
        rel_pos = robot_pos - obs_pos 
        dist = np.linalg.norm(rel_pos)
        
        # 安全半径
        r_safe = r + ROBOT_RADIUS + SAFETY_MARGIN
        
        if dist > r_safe + 3.0: continue 

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
ax.set_title(f"Multi-Robot Swarm (CBF Monitored)\nRobots={NUM_ROBOTS}, Margin={SAFETY_MARGIN}")

# 绘制障碍物及安全边界
for obs in obstacles:
    # 实体
    ax.add_patch(Circle((obs[0], obs[1]), obs[2], color='gray', zorder=2))
    # [新增] 安全边界
    ax.add_patch(Circle((obs[0], obs[1]), obs[2] + SAFETY_MARGIN + ROBOT_RADIUS, 
                        color='red', fill=False, linestyle='--', alpha=0.3, zorder=1))

# [新增] 全局状态面板
h_text = ax.text(-MAP_HALF_SIZE + 1, MAP_HALF_SIZE - 2, "Initializing...", fontsize=12, bbox=dict(facecolor='white', alpha=0.9, boxstyle='round'))

# 初始化机器人绘图元素
color_map = cm.get_cmap('tab10', NUM_ROBOTS) 
robot_colors = [color_map(i) for i in range(NUM_ROBOTS)]

robot_dots = [ax.plot([], [], 'o', color=robot_colors[i], markersize=6, zorder=5)[0] for i in range(NUM_ROBOTS)]
trajectory_lines = [ax.plot([], [], '-', linewidth=1.0, alpha=0.6, color=robot_colors[i])[0] for i in range(NUM_ROBOTS)]
goal_dots = [ax.plot(goal[0], goal[1], '*', color=robot_colors[i], markersize=10, alpha=0.5)[0] for i, goal in enumerate(goals)]
start_positions = [pos.copy() for pos in robot_positions]
start_dots = [ax.plot(pos[0], pos[1], 's', markersize=6, color=robot_colors[i], alpha=0.5)[0] for i, pos in enumerate(start_positions)]


# === Simulation update ===
def update(frame):
    global target_dirs
    artists = []
    
    # 本帧所有机器人的最低安全指数
    swarm_min_h = 100.0
    active_robots_count = 0

    for i in range(NUM_ROBOTS):
        # 如果该机器人已经到达，跳过计算，只重绘静态位置
        if finished_status[i]:
            robot_dots[i].set_data([robot_positions[i][0]], [robot_positions[i][1]])
            artists.append(robot_dots[i])
            continue

        pos = robot_positions[i]
        vel = robot_velocities[i]
        goal = goals[i]

        to_goal = goal - pos
        dist = np.linalg.norm(to_goal)
        
        # 检查到达
        if dist < GOAL_REACHED_THRESHOLD:
            finished_status[i] = True
            # 到达后稍微把颜色变淡，表示“退场”
            robot_dots[i].set_alpha(0.3)
            continue

        active_robots_count += 1
        target_dir = target_dirs[i]

        # 1. 准备输入数据
        robot_state = get_robot_state(pos, goal, vel, target_dir, device=device)

        static_obs_input, range_matrix, _ = get_ray_cast(
            pos, obstacles, max_range=MAX_RAY_LENGTH,
            hres_deg=HRES_DEG,
            vfov_angles_deg=VFOV_ANGLES_DEG,
            start_angle_deg=np.degrees(np.arctan2(target_dir[1], target_dir[0])),
            device=device
        )
        
        target_tensor = torch.tensor(np.append(target_dir[:2], 0.0), dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)

        # 动态障碍物 (Robot-Robot 感知)
        # 注意：这里需要传入所有机器人的位置，Agent 会自动感知邻居
        dyn_obs_input = get_dyn_obs_state(pos, vel, robot_positions, robot_velocities, target_tensor, device=device)

        # 2. Agent 决策
        velocity = agent.plan(robot_state, static_obs_input, dyn_obs_input, target_tensor)

        # 3. 数据清洗
        if isinstance(velocity, torch.Tensor):
            velocity = velocity.detach().cpu().numpy().flatten()
        velocity = np.array(velocity, dtype=np.float64)

        # === [新增] CBF 监测 ===
        # 计算该机器人的安全指数
        h_val = calculate_cbf_index(pos, velocity, obstacles)
        
        # 更新集群最低分
        if h_val < swarm_min_h:
            swarm_min_h = h_val
        
        # 个体颜色警告：如果违规，变红并变大；否则恢复原色
        if h_val < 0:
            robot_dots[i].set_color('red')
            robot_dots[i].set_markersize(10) # 放大，突出显示
        else:
            robot_dots[i].set_color(robot_colors[i])
            robot_dots[i].set_markersize(6)
        # =====================

        # 更新物理状态
        robot_positions[i] += velocity * DT
        robot_velocities[i] = velocity.copy()
        trajectories[i].append(robot_positions[i].copy())

        # 绘图更新
        robot_dots[i].set_data([robot_positions[i][0]], [robot_positions[i][1]])
        trajectory_lines[i].set_data(*zip(*trajectories[i]))

        artists += [robot_dots[i], trajectory_lines[i]]

    artists += start_dots
    
    # === [新增] 更新全局面板 ===
    if active_robots_count > 0:
        h_text.set_text(f"Swarm Min h(x): {swarm_min_h:.2f}\nActive Robots: {active_robots_count}")
        if swarm_min_h < 0:
            h_text.set_backgroundcolor('salmon') # 集群报警
        else:
            h_text.set_backgroundcolor('white')
    else:
        h_text.set_text("ALL GOALS REACHED!")
        h_text.set_backgroundcolor('lightgreen')
        
    artists.append(h_text)
    
    return artists

# 增加帧数以展示完整过程
ani = animation.FuncAnimation(fig, update, frames=500, interval=20, blit=False)
plt.show()