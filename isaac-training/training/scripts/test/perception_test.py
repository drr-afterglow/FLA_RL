from omni.isaac.kit import SimulationApp
sim_app = SimulationApp({"headless": True}) 

import sys
import os
from datetime import datetime
import hydra
import torch
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from env import NavigationEnv  

def visualize_observation(obs_dict, step, save_dir, env_id=0):
    """
    可视化：当前归一化深度、上一时刻归一化深度、归一化残差、真值图
    """
    lidar = obs_dict.get("lidar")
    if lidar is None:
        raise RuntimeError("Observation does not contain 'lidar' channel")
    lidar = lidar[env_id].detach().cpu().numpy() 

    gt = obs_dict.get("neuflow_gt")
    if gt is None:
        raise RuntimeError("Observation does not contain 'neuflow_gt' channel")
    gt = gt[env_id].detach().cpu().numpy()    

    # Lidar Channels: [0]=Current, [1]=Previous(Aligned), [2]=Residual, [3]=Sin, [4]=Cos
    depth_curr_norm = lidar[0]
    depth_prev_norm = lidar[1]
    residual_norm   = lidar[2]
    
    mask_gt = gt[3]
    
    # 如果你想看光流速度模长，可以使用下面这行替代 mask_gt：
    # vel_gt_mag = np.sqrt(gt[0]**2 + gt[1]**2 + gt[2]**2)
    # mask_gt = vel_gt_mag 

    # -----------------------------
    # 绘图布局 (1行 x 4列) 
    # -----------------------------
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle(f"Step {step:04d} - Env {env_id}", fontsize=16)

    # 1. 当前归一化深度 (Current Depth Norm)
    im1 = axes[0].imshow(depth_curr_norm, cmap='plasma', vmin=0, vmax=1.0)
    axes[0].set_title("Current Depth (Norm)")
    axes[0].axis('off') 
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. 上一时刻归一化深度 (Previous Depth Norm)
    im2 = axes[1].imshow(depth_prev_norm, cmap='plasma', vmin=0, vmax=1.0)
    axes[1].set_title("Prev/Pred Depth (Norm)")
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. 归一化残差 (Residual Norm)
    # 残差通常很小，vmin/vmax 可以根据情况调整，这里设为 0-1 覆盖全量程，
    # 或者设为 0-0.1 来增强对比度看细节
    im3 = axes[2].imshow(residual_norm, cmap='inferno', vmin=0, vmax=0.2) 
    axes[2].set_title("Residual (Norm)")
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. 真值图 (GT Mask)
    im4 = axes[3].imshow(mask_gt, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title("GT Mask (Dynamic)")
    axes[3].axis('off')

    plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

    save_path = os.path.join(save_dir, f"step_{step:04d}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

@hydra.main(version_base=None, config_path="../../cfg", config_name="env_test")
def main(cfg: DictConfig):

    print("Initialize Environment...")
    env = NavigationEnv(cfg)
    
    base_dir = "output_debug_frames"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(save_run_dir, exist_ok=True)
    print(f"Visualization frames will be saved to: {os.path.abspath(save_run_dir)}")

    print("Resetting Environment...")
    tensordict = env.reset()
    
    num_steps = 20
    print(f"Running for {num_steps} steps with Hover actions...")
    
    try:
        for i in range(num_steps):

            hover_action_value = -0.4446
            actions = torch.ones(env.num_envs, 4, device=env.device) * hover_action_value
            tensordict["agents", "action"] = actions
            tensordict = env.step(tensordict)  
            obs = tensordict["agents", "observation"]
            
            visualize_observation(obs, i, save_run_dir, env_id=0)
            
            if i % 5 == 0:
                print(f"Step {i}: Saved visualization to {save_run_dir}")

    except Exception as e:
        print(f"Error during loop: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("Closing environment...")
        env.close()
        sim_app.close()
        print("Done!")

if __name__ == "__main__":
    main()