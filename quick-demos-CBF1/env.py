import numpy as np
import random

# Generate grid-based obstacles
def generate_obstacles_grid(grid_div, region_min, region_max, min_radius, max_radius, min_clearance=1.0):
    cell_size = (region_max - region_min) / grid_div
    obstacles = []

    for i in range(grid_div):
        for j in range(grid_div):
            for _ in range(10):  # Try up to 10 times per cell
                radius = random.uniform(min_radius, max_radius)
                margin = radius + 0.2
                x = np.random.uniform(region_min + i * cell_size + margin,
                                      region_min + (i + 1) * cell_size - margin)
                y = np.random.uniform(region_min + j * cell_size + margin,
                                      region_min + (j + 1) * cell_size - margin)

                # Check clearance from existing obstacles
                too_close = False
                for ox, oy, oradius in obstacles:
                    dist = np.hypot(x - ox, y - oy)
                    min_dist = radius + oradius + min_clearance
                    if dist < min_dist:
                        too_close = True
                        break

                if not too_close:
                    obstacles.append((x, y, radius))
                    break  # Accepted, go to next cell

    return obstacles

# Sample collision-free start
def sample_free_start(obstacles, goal, obstacle_region_min, obstacle_region_max, min_clearance=1.0):
    while True:
        x = np.random.uniform(obstacle_region_min, obstacle_region_max)
        y = np.random.uniform(obstacle_region_min, obstacle_region_max)

        # Ensure clearance from obstacles
        too_close = False
        for ox, oy, r in obstacles:
            if np.hypot(x - ox, y - oy) <= r + min_clearance:
                too_close = True
                break

        # Ensure not too close to the goal
        if np.hypot(x - goal[0], y - goal[1]) < 3.0:
            too_close = True

        if not too_close:
            return np.array([x, y])

# Sample collision-free goal
def sample_free_goal(obstacles, obstacle_region_min, obstacle_region_max):

    while True:
        x = np.random.uniform(obstacle_region_min, obstacle_region_max)
        y = np.random.uniform(obstacle_region_min, obstacle_region_max)
        safe = True
        for ox, oy, r in obstacles:
            if np.hypot(x - ox, y - oy) <= r + 1.5:
                safe = False
                break
        if safe:
            return np.array([x, y])

def generate_tunnel_obstacles(region_min, region_max, num_obstacles=20):
    """
    生成一个长廊/隧道地形：上下有墙，中间有随机障碍
    """
    obstacles = []
    
    # 1. 构建上下实心墙壁 (用密集的圆模拟)
    wall_radius = 0.8
    step = wall_radius * 1.5
    
    # 上墙 (y = 8) 和 下墙 (y = -8)
    # 根据 region_min/max 动态调整范围
    x_range = np.arange(region_min, region_max + step, step)
    
    for x in x_range:
        obstacles.append((x, 10.0, wall_radius))  # 上墙 y=10
        obstacles.append((x, -10.0, wall_radius)) # 下墙 y=-10

    # 2. 中间随机撒点 (限制在 y [-8, 8] 之间)
    count = 0
    while count < num_obstacles:
        r = random.uniform(0.4, 0.7)
        x = random.uniform(region_min + 2, region_max - 2)
        y = random.uniform(-8.0, 8.0)
        
        # 简单防重叠
        overlap = False
        for ox, oy, or_ in obstacles:
            if np.hypot(x-ox, y-oy) < r + or_ + 0.5:
                overlap = True
                break
        if not overlap:
            obstacles.append((x, y, r))
            count += 1
            
    return obstacles

def generate_matrix_obstacles(region_min, region_max, rows=5, cols=6):
    """
    生成一个整齐的矩阵阵列，测试穿缝能力
    """
    obstacles = []
    
    # 计算间距
    x_space = np.linspace(region_min + 4, region_max - 4, cols)
    y_space = np.linspace(region_min + 4, region_max - 4, rows)
    
    for x in x_space:
        for y in y_space:
            # 加入一点点随机扰动，让它看起来不那么死板
            noise_x = random.uniform(-0.3, 0.3)
            noise_y = random.uniform(-0.3, 0.3)
            r = 0.6  # 固定半径
            obstacles.append((x + noise_x, y + noise_y, r))
            
    return obstacles