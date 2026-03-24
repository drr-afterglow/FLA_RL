import torch
import os
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from tensordict.tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from ppo import PPO
from dataclasses import dataclass, field

# === 必须加入 Config 定义，否则 PPO 初始化会报错 ===
# 因为你的 ppo.py 依赖 cfg 参数
@dataclass
class FeatureExtractorConfig:
    learning_rate: float = 3e-4 # 微调时的参数
    dyn_obs_num: int = 5

@dataclass
class ActorConfig:
    learning_rate: float = 3e-4
    clip_ratio: float = 0.1
    action_limit: float = 2.0 

@dataclass
class CriticConfig:
    learning_rate: float = 3e-4
    clip_ratio: float = 0.1

@dataclass
class AlgoConfig:
    feature_extractor: FeatureExtractorConfig = field(default_factory=FeatureExtractorConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    
    # 这里的参数不影响推理，但 PPO 初始化需要
    entropy_loss_coefficient: float = 1e-3
    training_frame_num: int = 128 
    training_epoch_num: int = 4
    num_minibatches: int = 32

cfg = AlgoConfig()
# ===============================================

class Agent:
    def __init__(self, device):
        self.device = device
        self.policy = self.init_model()

    def init_model(self):
        # 保持与 env.py 中的 spec 一致
        observation_dim = 8
        observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device), 
                    "lidar": UnboundedContinuousTensorSpec((1, 36, 4), device=self.device),
                    "direction": UnboundedContinuousTensorSpec((1, 3), device=self.device),
                    "dynamic_obstacle": UnboundedContinuousTensorSpec((1, 5, 10), device=self.device),
                }),
            }).expand(1)
        }, shape=[1], device=self.device)

        action_dim = 3
        action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": UnboundedContinuousTensorSpec((action_dim,), device=self.device), 
            })
        }).expand(1, action_dim).to(self.device)

        # 初始化 PPO (传入我们伪造的 cfg)
        # 注意：你需要修改 PPO 类，让它接受外部传入的 cfg，或者直接在这里 patch
        # 假设 ppo.py 里的 PPO 类初始化是 PPO(cfg, obs_spec, act_spec, device)
        # 如果 ppo.py 里写死了 cfg = AlgoConfig()，那这里可能不需要传 cfg，视你的 ppo.py 而定
        # 根据你刚才发的 ppo.py，它初始化是 def __init__(self, observation_spec, ...): 
        # 并没有传 cfg，而是用的全局 cfg。所以这里直接调：
        
        policy = PPO(observation_spec, action_spec, self.device)

        # === 关键：加载你的 SOTA 权重 ===
        # 请修改为你存放 checkpoint_final.pt 的真实路径
        # 最好是绝对路径
        checkpoint_path = "/home/amo/UAV/FLA_RL/quick-demos-CBF1/ckpts/checkpoint_final.pt"
        
        print(f"Loading model from: {checkpoint_path}")
        if os.path.exists(checkpoint_path):
            policy.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        else:
            print("Error: Checkpoint not found! Using random weights.")
            
        return policy.eval() # 设为评估模式
    
    def plan(self, robot_state, static_obs_input, dyn_obs_input, target_dir):
        # 构造 TensorDict
        obs = TensorDict({
            "agents": TensorDict({
                "observation": TensorDict({
                    "state": robot_state,
                    "lidar": static_obs_input,
                    "direction": target_dir,
                    "dynamic_obstacle": dyn_obs_input,
                })
            })
        }, device=self.device)

        # 使用确定性策略 (MEAN)
        with set_exploration_type(ExplorationType.MEAN):
            output = self.policy(obs)
            # 获取 action_normalized
            # 注意：env.py 里把 normalized action 转为了 world action
            # 这里 demo 需要自己转吗？
            # 看 ppo.py 的 __call__，它返回了 "action" (已经是 world scale)
            # 所以直接取 "action" 即可
            velocity = output["agents", "action"][0][0].detach().cpu().numpy()[:2] 
            
        return velocity