from omni.isaac.kit import SimulationApp
sim_app = SimulationApp({"headless": True})

import os
import sys
import hydra
from omegaconf import DictConfig
import wandb
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from env import NavigationEnv
from omni_drones.utils.torchrl import RenderCallback
@hydra.main(config_path="../../cfg", config_name="env_test", version_base=None)
def main(cfg: DictConfig):
    """
    1. 验证场景创建
    2. 验证 LiDAR 传感器数据流
    3. 验证仿真循环稳定性
    4. 录制调试视频
    """
    run = wandb.init(
        project=cfg.wandb.get('project', 'navigation-rl'),
        entity=cfg.wandb.get('entity', None),
        name="env-verification",
        mode=cfg.wandb.get('mode', 'online'),
    )
    cfg.headless = True  

    env = NavigationEnv(cfg)
    env.enable_render(True)
    render_callback = RenderCallback(interval=1) 
    obs = env.reset()
   
    print("运行仿真环境...")
    total_steps = int(10.0 / cfg.sim.dt)
    step_count = 0
    
    try:
        for step in range(total_steps):

            hover_action_value=-0.4446
            actions = torch.ones(env.num_envs, 4, device=env.device)*hover_action_value
            obs["agents"]["action"] = actions
            tensordict = env.step(obs)
            # env.drone.set_velocities(torch.zeros(env.num_envs, 6, device=env.device), env.drone.indices)
            obs = tensordict  
            render_callback(env)
            
           #----进度---
            if (step + 1) % 100 == 0:
                print(f"  进度: {step + 1}/{total_steps} 步")
        
        print(f"环境测试完成")

    except Exception as e:
        print(f"仿真循环发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    # env visulize
    print("环境运行情况")
    try:
        video_array = render_callback.get_video_array(axes="t c h w")
        fps = min(int(1.0 / cfg.sim.dt), 60)
        wandb.log({
            "environment_verification_video": wandb.Video(
                video_array, 
                fps=fps, 
                format="mp4"
            ),
        })
        print("video success!")
    except Exception as e:
        print(f"视频生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("over close env")
    env.close()
    wandb.finish()
    sim_app.close()

if __name__ == "__main__":
    main()