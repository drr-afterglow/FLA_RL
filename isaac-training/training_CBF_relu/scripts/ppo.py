import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from utils import ValueNorm, make_mlp, IndependentNormal, Actor, GAE, make_batch, IndependentBeta, BetaActor, vec_to_world

class LagrangianPID(nn.Module):
    def __init__(self, target_cost=0.02, init_nu=1.0, lr=0.05, device="cuda"):
        super().__init__()
        self.target_cost = target_cost
        self.lr = lr
        # nu 是可学习参数，但不参与 PPO 的 autograd，手动更新
        self.register_buffer("nu", torch.tensor(init_nu, device=device))

    def update(self, current_cost):
        # 对偶梯度上升 (Dual Gradient Ascent)
        with torch.no_grad():
            diff = current_cost - self.target_cost
            self.nu.data += self.lr * diff
            self.nu.data.clamp_(min=0.0,max=20.0) # 保证非负
        return self.nu.item()


class PPO(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        
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

        # Actor network，输出action_normalized
        self.n_agents, self.action_dim = action_spec.shape
        self.actor = ProbabilisticActor(
            TensorDictModule(BetaActor(self.action_dim), ["_feature"], ["alpha", "beta"]),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")], 
            distribution_class=IndependentBeta,
            return_log_prob=True
        ).to(self.device)

        # Critic network，输出state_value
        self.critic = TensorDictModule(
            nn.LazyLinear(1), ["_feature"], ["state_value"] 
        ).to(self.device)
        self.value_norm = ValueNorm(1).to(self.device)#定义价值函数归一化模块，用于稳定训练过程中的价值估计归一化

        # Loss related
        self.gae = GAE(0.99, 0.95) # generalized adavantage esitmation
        self.critic_loss_fn = nn.HuberLoss(delta=10) # huberloss (L1+L2): https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html

        # Optimizer
        self.feature_extractor_optim = torch.optim.Adam(self.feature_extractor.parameters(), lr=cfg.feature_extractor.learning_rate)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic.learning_rate)

        # Dummy Input for nn lazymodule
        dummy_input = observation_spec.zero()
        # print("dummy_input: ", dummy_input)


        self.__call__(dummy_input)

        # Initialize network
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        self.actor.apply(init_)
        self.critic.apply(init_)

        # ================= [CBF] =================
        # 从 cfg.algo 中读取 safety 配置
        # 使用 getattr 防止旧 yaml 报错
        safety_cfg = getattr(cfg, "safety", None) 
        
        # 判断开关
        self.use_safety = getattr(safety_cfg, "enabled", False)

        if self.use_safety:
            self.lagrangian_controller = LagrangianPID(
                target_cost=getattr(safety_cfg, "target_cost_limit", 0.02),
                init_nu=getattr(safety_cfg, "init_nu", 1.0),
                lr=getattr(safety_cfg, "nu_learning_rate", 0.05),
                device=device
            )
        else:
            self.lagrangian_controller = None

    def __call__(self, tensordict):
        self.feature_extractor(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)

        # Cooridnate change: transform local to world
        actions = (2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit) - self.cfg.actor.action_limit
        actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"])
        tensordict["agents", "action"] = actions_world
        return tensordict

    def apply_safety_reward(self, batch):
        if not self.use_safety:
            return batch

        # 确保检查 keys 存在，避免报错
        if "agents" in batch.keys() and "cbf_cost" in batch["agents"].keys():
            reward_task = batch["agents"]["reward"]
            cbf_cost = batch["agents"]["cbf_cost"]
            
            nu = self.lagrangian_controller.nu.detach()
            reward_total = reward_task - nu * cbf_cost
            
            batch["agents"]["reward"] = reward_total
            batch["agents"]["reward_task"] = reward_task.clone() # 备份原始奖励
            
        return batch

    def update_lagrangian(self, batch):
        """
        在 PPO 更新后调整 nu
        """
        if not self.use_safety:
            return 0.0, 0.0

        if "cbf_cost" in batch["agents"].keys():
            mean_cost = batch["agents"]["cbf_cost"].mean()
            curr_nu = self.lagrangian_controller.update(mean_cost)
            return curr_nu, mean_cost.item()
        
        return 0.0, 0.0



    def train(self, tensordict):

        if self.use_safety:
            self.apply_safety_reward(tensordict["next"])

        # tensordict: (num_env, num_frames, dim), batchsize = num_env * num_frames
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_tensordict = torch.vmap(self.feature_extractor)(next_tensordict)
            next_values = self.critic(next_tensordict)["state_value"]
        rewards = tensordict["next", "agents", "reward"] # Reward obtained by state transition
        dones = tensordict["next", "terminated"] # Whether the next states are terminal states

        values = tensordict["state_value"] # This is calculated stored when we called forward to obtain actions
        values = self.value_norm.denormalize(values) # denomalize values based on running mean and var of return
        next_values = self.value_norm.denormalize(next_values)

        # calculate GAE: Generalized Advantage Estimation
        adv, ret = self.gae(rewards, dones, values, next_values)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        self.value_norm.update(ret) # update running mean and var for return
        ret = self.value_norm.normalize(ret)  # normalize return
        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        # Training
        infos = []
        for epoch in range(self.cfg.training_epoch_num):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))

        infos = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])

        result_log = {k: v.item() for k, v in infos.items()}
        result_log["reward/reward_total"] = tensordict["next", "agents", "reward"].mean().item()
        result_log["reward/reward_task"] = tensordict["next", "agents", "reward_task"].mean().item()
        
        #更新拉格朗日乘子
        if self.use_safety:
            # 同样使用 next 中的 cost 数据
            curr_nu, mean_cost = self.update_lagrangian(tensordict["next"])
            result_log["safety/nu"] = curr_nu
            result_log["safety/cost_mean"] = mean_cost

            if "cbf_cost" in tensordict["next", "agents"].keys():
                Hazardous_rate = (tensordict["next", "agents", "cbf_cost"] > 1e-4).float().mean().item()
                result_log["safety/Hazardous_rate"] = Hazardous_rate
                result_log["reward/CBF_penalty"] = curr_nu * mean_cost
           
        return result_log

    
    def _update(self, tensordict): # tensordict shape (batch_size, )
        self.feature_extractor(tensordict)

        # Get action from the current policy
        action_dist = self.actor.get_dist(tensordict) # this does an actor forward to get "loc" and "scale" and use them to build multivariate normal distribution
        log_probs = action_dist.log_prob(tensordict[("agents", "action_normalized")]) # based on the gaussian, we can calculate the log prob of the action from the current policy

        # Entropy Loss
        action_entropy = action_dist.entropy()
        entropy_loss = -self.cfg.entropy_loss_coefficient * torch.mean(action_entropy)

        # Actor Loss
        advantage = tensordict["adv"] # the advantage is calculated based on GAE in hte previous step
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = advantage * ratio
        surr2 = advantage * ratio.clamp(1.-self.cfg.actor.clip_ratio, 1.+self.cfg.actor.clip_ratio)
        actor_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim 

        # Critic Loss 
        b_value = tensordict["state_value"]
        ret = tensordict["ret"] # Return G
        value = self.critic(tensordict)["state_value"] 
        value_clipped = b_value + (value - b_value).clamp(-self.cfg.critic.clip_ratio, self.cfg.critic.clip_ratio) # this guarantee that critic update is clamped
        critic_loss_clipped = self.critic_loss_fn(ret, value_clipped)
        critic_loss_original = self.critic_loss_fn(ret, value)
        critic_loss = torch.max(critic_loss_clipped, critic_loss_original)

        # Total Loss
        loss = entropy_loss + actor_loss + critic_loss

        # Optimize
        self.feature_extractor_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()#反向传播计算梯度

        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), max_norm=5.) # to prevent gradient growing too large
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), max_norm=5.)
        self.feature_extractor_optim.step()
        self.actor_optim.step()
        self.critic_optim.step()
        explained_var = 1 - F.mse_loss(value, ret) / ret.var()
        return TensorDict({
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy_loss,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var
        }, [])