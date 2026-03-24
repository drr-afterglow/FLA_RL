import torch
import torch.nn as nn
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from utils import ValueNorm, make_mlp, GAE, IndependentBeta, BetaActor, vec_to_world
from dataclasses import dataclass, field

# ================= 配置部分 (保持不变) =================
@dataclass
class FeatureExtractorConfig:
    learning_rate: float = 3e-5
    dyn_obs_num: int = 5 

@dataclass
class ActorConfig:
    learning_rate: float = 3e-5
    clip_ratio: float = 0.1
    action_limit: float = 2.0 

@dataclass
class CriticConfig:
    learning_rate: float = 3e-5
    clip_ratio: float = 0.1

@dataclass
class AlgoConfig:
    feature_extractor: FeatureExtractorConfig = field(default_factory=FeatureExtractorConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    
    entropy_loss_coefficient: float = 1e-3
    training_frame_num: int = 128
    training_epoch_num: int = 4
    num_minibatches: int = 32
    
    safety_target_cost: float = 0.08
    safety_init_nu: float = 3.0
    safety_nu_lr: float = 0.1

cfg = AlgoConfig()

# ================= [修正] LagrangianPID 类 =================
class LagrangianPID(nn.Module):
    def __init__(self, target_cost, init_nu, lr, dt=0.016, device="cpu"):
        super().__init__()
        self.target_cost = target_cost
        self.lr = lr
        self.dt = dt
        
        # [核心修正] nu 必须注册在这里，才能匹配 checkpoint 中的 "lagrangian_controller.nu"
        self.register_buffer("nu", torch.tensor(init_nu, device=device))
        
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, current_cost):
        error = current_cost - self.target_cost
        
        # Proportional term (P) - 简化逻辑，保持训练一致性
        delta = self.lr * error
        
        # 更新 buffer 中的 nu
        new_nu = torch.clamp(self.nu + delta, min=0.0, max=300.0)
        self.nu.fill_(new_nu)
        
        return new_nu

# ================= PPO 类 =================
class PPO(TensorDictModuleBase):
    def __init__(self, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        
        # [核心修正] 删除这里的 register_buffer("nu", ...)
        # 改为通过 lagrangian_controller 管理 nu
        
        self.target_cost = cfg.safety_target_cost
        self.lagrangian_controller = LagrangianPID(
            target_cost=self.target_cost,
            init_nu=0.1, # 初始值会被 checkpoint 覆盖
            lr=cfg.safety_nu_lr,
            device=device
        )

        # Feature extractor for LiDAR
        feature_extractor_network = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(), 
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128), nn.LayerNorm(128),
        ).to(self.device)
        
        # Dynamic obstacle information extractor
        dynamic_obstacle_network = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),
            make_mlp([128, 64])
        ).to(self.device)

        # Feature extractor
        self.feature_extractor = TensorDictSequential(
            TensorDictModule(feature_extractor_network, [("agents", "observation", "lidar")], ["_cnn_feature"]),
            TensorDictModule(dynamic_obstacle_network, [("agents", "observation", "dynamic_obstacle")], ["_dynamic_obstacle_feature"]),
            CatTensors(["_cnn_feature", ("agents", "observation", "state"), "_dynamic_obstacle_feature"], "_feature", del_keys=False), 
            TensorDictModule(make_mlp([256, 256]), ["_feature"], ["_feature"]),
        ).to(self.device)

        # Actor network
        self.n_agents, self.action_dim = action_spec.shape
        self.actor = ProbabilisticActor(
            TensorDictModule(BetaActor(self.action_dim), ["_feature"], ["alpha", "beta"]),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")], 
            distribution_class=IndependentBeta,
            return_log_prob=True
        ).to(self.device)

        # Critic network
        self.critic = TensorDictModule(
            nn.LazyLinear(1), ["_feature"], ["state_value"] 
        ).to(self.device)
        self.value_norm = ValueNorm(1).to(self.device)

        # Loss related
        self.gae = GAE(0.99, 0.95) 
        self.critic_loss_fn = nn.HuberLoss(delta=10) 

        # Optimizer
        self.feature_extractor_optim = torch.optim.Adam(self.feature_extractor.parameters(), lr=cfg.feature_extractor.learning_rate)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.actor.learning_rate)

        # Dummy Input
        dummy_input = observation_spec.zero()
        self.__call__(dummy_input)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        self.actor.apply(init_)
        self.critic.apply(init_)

    def __call__(self, tensordict):
        self.feature_extractor(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)

        actions = (2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit) - self.cfg.actor.action_limit
        actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"])
        tensordict["agents", "action"] = actions_world
        return tensordict
    
    # 辅助函数 (保持结构一致性)
    def apply_safety_reward(self, batch):
        if "cbf_cost" in batch["agents"].keys():
             cbf_cost = batch["agents"]["cbf_cost"]
             raw_reward = batch["next", "agents", "reward"]
             # [核心修正] 使用 controller 里的 nu
             new_reward = raw_reward - self.lagrangian_controller.nu * cbf_cost
             batch["next", "agents", "reward"] = new_reward
        return batch

    def update_lagrangian(self, batch):
        if "cbf_cost" in batch["agents"].keys():
            current_cost = batch["agents"]["cbf_cost"].mean()
            # [核心修正] step 函数内部更新 buffer
            new_nu = self.lagrangian_controller.step(current_cost)
            return {"safety/nu": self.lagrangian_controller.nu.item(), "safety/cost_mean": current_cost.item()}
        return {}