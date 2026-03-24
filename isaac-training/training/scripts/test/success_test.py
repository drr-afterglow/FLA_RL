"""
测试脚本：分批次评估训练好的策略网络 (Batch Evaluation)
每次并行 200 个环境，运行 5 轮，共计 1000 个测试样本。
这样既能保证统计意义，又能避免物理引擎过载。
"""
import os
import hydra
import torch
import wandb
import datetime
import numpy as np
from omegaconf import DictConfig
from omni.isaac.kit import SimulationApp
from torchrl.envs.transforms import TransformedEnv, Compose
from torchrl.envs.utils import ExplorationType

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")

@hydra.main(config_path="../../cfg", config_name="train", version_base=None)
def main(cfg):
    
    # ---------------------- 配置区域 ----------------------
    # 权重路径
    checkpoint_path = "/root/NavRL/isaac-training/training/scripts/wandb/run-20251223_165015-syjto24q/files/checkpoint_900.pt" 
    
    # 测试参数设置
    BATCH_SIZE = 200        # 每次并行环境数量 (根据显存调整，建议 64-200)
    NUM_BATCHES = 5         # 测试轮数
    TOTAL_ENVS = BATCH_SIZE * NUM_BATCHES
    
    cfg.env.num_envs = BATCH_SIZE # 强制覆盖配置文件中的环境数量
    cfg.headless = True           # 测试通常不需要GUI，如需观察设为False
    # -----------------------------------------------------

    # 初始化仿真
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})
    
    # 延迟导入以避免 Isaac Sim 初始化前的冲突
    from env import NavigationEnv
    from ppo import PPO
    from omni_drones.controllers import LeePositionController
    from omni_drones.utils.torchrl.transforms import VelController
    from utils import evaluate
    
    # 初始化 WandB
    run = wandb.init(
        project=cfg.wandb.project,
        name=f"test_batch/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
        config={
            "checkpoint": checkpoint_path, 
            "batch_size": BATCH_SIZE,
            "num_batches": NUM_BATCHES,
            "total_envs": TOTAL_ENVS
        },
        mode=cfg.wandb.mode,
    )
    
    print(f"[Test] 初始化环境 (Batch Size: {BATCH_SIZE})...")
    # 创建环境
    env = NavigationEnv(cfg)
    transforms = []
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    transforms.append(vel_transform)
    transformed_env = TransformedEnv(env, Compose(*transforms))
    
    # 加载策略
    policy = PPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)
    if os.path.exists(checkpoint_path):
        # 增加 map_location 以防设备不匹配
        loaded_dict = torch.load(checkpoint_path, map_location=cfg.device)
        policy.load_state_dict(loaded_dict)
        print(f"[Test] 成功加载权重: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"找不到权重文件: {checkpoint_path}")

    # 用于存储每一轮的结果
    metrics_history = {
        "success_rate": [],
        "collision_rate": [],
        "timeout_rate": [], # truncated
        "avg_return": [],
        "avg_length": []
    }

    print(f"\n[Test] 开始分批评估 (共 {NUM_BATCHES} 轮，每轮 {BATCH_SIZE} 个环境)...")
    print("=" * 60)

    # ---------------------- 循环评估 ----------------------
    env.enable_render(not cfg.headless) # 如果非 headless 模式，开启渲染
    env.eval() # 切换到评估模式

    for batch_idx in range(NUM_BATCHES):
        # 1. 动态设置随机种子：确保每一轮的场景生成都不一样
        current_seed = cfg.seed + batch_idx * 100 
        transformed_env.set_seed(current_seed)
        print(f"-> Batch {batch_idx + 1}/{NUM_BATCHES} (Seed: {current_seed}) running...")

        # 2. 执行评估
        # 注意：evaluate 内部通常会处理 reset，但传入 seed 确保随机性
        with torch.no_grad():
            eval_info = evaluate(
                env=transformed_env,
                policy=policy,
                seed=current_seed,
                cfg=cfg,
                # exploration_type=ExplorationType.MEAN
                exploration_type=ExplorationType.RANDOM
            )

        # 3. 提取指标
        # 注意：这里根据你提供的代码一中的 key 格式进行提取
        success = eval_info.get('eval/stats.reach_goal', 0)
        collision = eval_info.get('eval/stats.collision', 0)
        timeout = eval_info.get('eval/stats.truncated', 0)
        ret = eval_info.get('eval/stats.return', 0)
        length = eval_info.get('eval/stats.episode_len', 0)

        # 4. 记录到历史列表
        metrics_history["success_rate"].append(success)
        metrics_history["collision_rate"].append(collision)
        metrics_history["timeout_rate"].append(timeout)
        metrics_history["avg_return"].append(ret)
        metrics_history["avg_length"].append(length)

        # 5. 上传当前轮次数据到 WandB
        # 使用 batch_idx 作为 step，方便在图表中看趋势
        wandb_log_data = {
            "batch_idx": batch_idx + 1,
            "batch/success_rate": success,
            "batch/collision_rate": collision,
            "batch/timeout_rate": timeout,
            "batch/return": ret,
            "batch/episode_len": length
        }
        run.log(wandb_log_data)
        
        print(f"   Result: Success={success:.1%}, Collision={collision:.1%}, Len={length:.1f}")

    # ---------------------- 最终统计 ----------------------
    print("=" * 60)
    print(f"评估完成! 总样本数: {TOTAL_ENVS}")
    print("-" * 60)
    
    # 计算均值
    final_stats = {k: np.mean(v) for k, v in metrics_history.items()}
    
    # 终端输出最终结果
    print(f"最终平均指标 (Average over {NUM_BATCHES} batches):")
    print(f"  成功率 (Success Rate):    {final_stats['success_rate']:.2%}")
    print(f"  碰撞率 (Collision Rate):  {final_stats['collision_rate']:.2%}")
    print(f"  超时率 (Timeout Rate):    {final_stats['timeout_rate']:.2%}")
    print(f"  平均回报 (Avg Return):    {final_stats['avg_return']:.4f}")
    print(f"  平均步长 (Avg Length):    {final_stats['avg_length']:.2f}")
    print("=" * 60)

    # 上传最终汇总数据到 wandb
    run.log({
        "final/success_rate": final_stats['success_rate'],
        "final/collision_rate": final_stats['collision_rate'],
        "final/timeout_rate": final_stats['timeout_rate'],
        "final/avg_return": final_stats['avg_return'],
        "final/avg_length": final_stats['avg_length']
    })

    wandb.finish()
    sim_app.close()
    print("[Test] Done.")

if __name__ == "__main__":
    main()