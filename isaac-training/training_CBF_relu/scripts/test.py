import os
import hydra
import datetime
import wandb
import torch
from omegaconf import OmegaConf
from omni.isaac.kit import SimulationApp

# 1. 启动仿真器
# 必须最先执行，headless=True 也能录视频，只要开启 enable_render
FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")

@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    # ================= 配置调整 =================
    cfg.headless = True      # AutoDL 服务器必须为 True
    
    # [关键技巧] 视频录制不需要 512 个环境，那样显存压力大且视频看不清
    # 建议设为 4 或 9，这样 WandB 会生成一个 2x2 或 3x3 的网格视频，非常清晰
    cfg.env.num_envs = 4     
    
    # 开启反锯齿，让视频质量更好 (如果显存不够可以设为 0)
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})

    # ================= WandB 初始化 =================
    run_name = f"VIDEO_TEST_{datetime.datetime.now().strftime('%m%d_%H%M')}"
    wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        job_type="video_evaluation", # 标记为视频评估
        mode="online"
    )
    print(f"[NavRL Video]: WandB Initialized. Run Name: {run_name}")

    # ================= 导入模块 =================
    # 必须在 SimulationApp 启动后导入
    from ppo import PPO
    from env import NavigationEnv
    from omni_drones.controllers import LeePositionController
    from omni_drones.utils.torchrl.transforms import VelController
    from torchrl.envs.transforms import TransformedEnv, Compose
    from torchrl.envs.utils import ExplorationType
    from utils import evaluate

    # ================= 环境搭建 =================
    env = NavigationEnv(cfg)
    
    # [核心步骤 1] 强制开启渲染管线
    # 这对应你 train.py 中的 env.enable_render(True)
    # 这会让 Isaac Sim 的相机开始工作
    print("[NavRL Video]: Enabling Rendering Pipeline...")
    env.enable_render(True)
    
    transforms = []
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    transforms.append(vel_transform)
    
    transformed_env = TransformedEnv(env, Compose(*transforms))
    transformed_env.set_seed(cfg.seed)

    # ================= 策略加载 =================
    policy = PPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)
    
    # 修改为你的真实权重路径
    checkpoint_path = "/root/NavRL/isaac-training/training/scripts/wandb/run-20251211_151704-pgg1e1wn/files/checkpoint_final.pt"
    
    if os.path.exists(checkpoint_path):
        print(f"[NavRL Video]: Loading checkpoint from {checkpoint_path}")
        policy.load_state_dict(torch.load(checkpoint_path, map_location=cfg.device))
    else:
        print(f"!!! Checkpoint not found: {checkpoint_path}")
        sim_app.close()
        return

    # ================= 开始评估与录制 =================
    print("[NavRL Video]: Starting evaluation loop (Wait for video generation)...")
    
    # 切换到评估模式 (根据 train.py 逻辑)
    env.eval() 
    
    # [核心步骤 2] 调用你现有的 evaluate 函数
    # 它内部应该包含了录制逻辑和 WandB Video 对象的生成
    # 我们使用 MEAN (确定性策略) 来展示最佳效果
    eval_info = evaluate(
        env=transformed_env, 
        policy=policy,
        seed=cfg.seed, 
        cfg=cfg,
        exploration_type=ExplorationType.MEAN 
    )
    
    print("[NavRL Video]: Evaluation done.")

    # ================= 数据上传 =================
    print("\n" + "="*30)
    print("[NavRL Video]: Results & Video Upload")
    print("="*30)
    
    # 打印结果并上传
    # 如果 evaluate 返回的字典里包含 WandB Video 对象，wandb.log 会自动识别并上传
    for k, v in eval_info.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v}")
        else:
            # 这里的 Object 很可能就是 wandb.Video 对象
            print(f"{k}: [Media Object] (Likely Video)")
            
    wandb.log(eval_info)
    
    print("="*30)
    print(f"[NavRL Video]: Check your WandB dashboard (Media Section) for the video!")

    # 这里的 reset 只是为了清理状态，不是必须的
    env.train()
    env.reset()

    wandb.finish()
    sim_app.close()

if __name__ == "__main__":
    main()