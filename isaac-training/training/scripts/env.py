import torch
import einops
import time
import numpy as np
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
import omni.isaac.orbit.sim as sim_utils
from omni_drones.robots.drone import MultirotorBase
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni_drones.utils.torch import euler_to_quaternion, quat_axis
from omni_drones.utils.torch import quat_rotate, quat_rotate_inverse
from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.core.utils.viewports import set_camera_view
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg
from utils import construct_input
from perception import PerceptionModule
from perception import sphere_lidar_hits, compute_rayhitsdir

class NavigationEnv(IsaacEnv):

    # In one step:
    # 1. _pre_sim_step (apply action) -> step isaac sim
    # 2. _post_sim_step (update lidar)
    # 3. increment progress_buf
    # 4. _compute_state_and_obs (get observation and states, update stats)
    # 5. _compute_reward_and_done (update reward and calculate returns)

    def __init__(self, cfg):

        print("[Navigation Environment]: Initializing Env...")
        # ----------------params-----------------------

        # 1.LiDAR params:
        self.lidar_range = cfg.sensor.lidar_range
        self.lidar_hfov = cfg.sensor.lidar_hfov
        self.lidar_vfov = (max(-89., cfg.sensor.lidar_vfov[0]), min(89., cfg.sensor.lidar_vfov[1]))
        self.lidar_vbeams = cfg.sensor.lidar_vbeams
        self.lidar_hbeams = cfg.sensor.lidar_hbeams
        self.lidar_hres = 360 / self.lidar_hbeams
        self.H = cfg.sensor.lidar_vbeams
        self.W = cfg.sensor.lidar_hbeams


        #2.reward params
        self.vel_min = 0.2
        self.vel_max = 3.0
        self.fly_height = 2.5
        self.height_bound = 1.0
        self.safety_dis = 0.5

        # 2.1 state normalization params (for policy input stability)
        self.state_vel_scale = 1.2 * self.vel_max
        self.state_ang_vel_scale = 2.0 * torch.pi
        max_dx = 48.0
        max_dy = 48.0
        max_dz = 2.0
        self.max_start_goal_distance = float(np.sqrt(max_dx**2 + max_dy**2 + max_dz**2))
        self.state_dist_scale = 0.5 * self.max_start_goal_distance

        #3.curriculum params
        self.curriculum_enabled = getattr(cfg, "curriculum", {}).get("enabled", False) if hasattr(cfg, "curriculum") else False
        self.curriculum_phases = []
        if self.curriculum_enabled:
            for p in cfg.curriculum.phases:
                self.curriculum_phases.append({
                    "frame": float(p["frame"]),
                    "active_dyn_obs": int(p["active_dyn_obs"]),
                    "dyn_vel_range": list(p["dyn_vel_range"]),
                    "vel_min": float(p["vel_min"]),
                    "vel_max": float(p["vel_max"]),
                })
            self.curriculum_phases.sort(key=lambda x: x["frame"])
        # Number of dynamic obstacles that are actively moving (controlled by curriculum)
        self.active_dyn_obs = getattr(cfg.env_dyn, "num_obstacles", 0)
        # Mutable copy of dynamic obstacle velocity range (curriculum may update this)
        self.dyn_vel_range = list(cfg.env_dyn.vel_range)

        super().__init__(cfg, cfg.headless)
        
        #-----------------Initialization----------------

        # 1.Drone Initialization
        self.drone.initialize()
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        # 2.Dynamic Obstacle Initialization
        if hasattr(self, "dynamic_obstacles") and self.dynamic_obstacles is not None:
            self.dynamic_obstacles._initialize_impl()

        # 3.LiDAR Intialization
        ray_caster_cfg = RayCasterCfg(
            prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=True,
            pattern_cfg=patterns.BpearlPatternCfg(
                horizontal_res=self.lidar_hres, 
                vertical_ray_angles=torch.linspace(*self.lidar_vfov, self.lidar_vbeams) 
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
            
        )
        self.lidar = RayCaster(ray_caster_cfg)
        self.lidar._initialize_impl()
        self.ray_hits_dir = compute_rayhitsdir(
            self.device, 
            self.num_envs, 
            self.lidar_hfov, 
            self.lidar_vfov,
            self.lidar_hbeams, 
            self.lidar_vbeams
            )

        # 4.Perception intialization
        self.perception = PerceptionModule(cfg, self.num_envs, self.device)

        # 5.simulation dt for acceleration calculation
        self.sim_dt = cfg.sim.dt * cfg.sim.substeps

        # 6.uav start position and target position intialization
        with torch.device(self.device):
            self.target_pos = torch.zeros(self.num_envs, 1, 3)
            self.target_dir = torch.zeros(self.num_envs, 1, 3)
            self.height_range = torch.zeros(self.num_envs, 1, 2)
            self.prev_drone_vel_w = torch.zeros(self.num_envs, 1 , 3)
            self.prev_drone_vel_b = torch.zeros(self.num_envs, 1, 3)
            self.last_dis2goal = None
            self.start_pos = torch.zeros(self.num_envs, 1, 3)

    # env design include obstacle light ground uav
    def _design_scene(self):

        # 1.Initialize a drone in prim /World/envs/envs_0
        drone_model = MultirotorBase.REGISTRY[self.cfg.drone.model_name] 
        cfg = drone_model.cfg_cls(force_sensor=False)
        self.drone = drone_model(cfg=cfg)
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]

        # 2.lighting
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(1.0, 1.0, 1.0), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.95, 0.95, 1.0), intensity=1000.0),
        )
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)
        
        # 3.Ground Plane 
        cfg_ground = sim_utils.CuboidCfg(
            size=(150.0, 150.0, 0.1), 
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.1, 0.1), 
                metallic=0.1,
                roughness=1.0
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, 
                disable_gravity=True
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True 
            )
        )
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.06))

        # 4.Map range
        self.map_range = [20.0, 20.0, 4.5]

        # 5.Static obstacle terrain
        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            env_spacing=0.0,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=10,
                size=(self.map_range[0]*2, self.map_range[1]*2), 
                border_width=5.0,
                num_rows=1, 
                num_cols=1, 
                horizontal_scale=0.1,
                vertical_scale=0.1,
                slope_threshold=0.75,
                use_cache=False,
                color_scheme="none",
                sub_terrains={
                    "obstacles": HfDiscreteObstaclesTerrainCfg(
                        horizontal_scale=0.1,
                        vertical_scale=0.1,
                        border_width=0.0,
                        num_obstacles=self.cfg.env.num_obstacles,
                        obstacle_height_mode="range",
                        obstacle_width_range=(0.4, 1.0),
                        obstacle_height_range=[ 2.0, 4.0, 4.0, 4.5 ],
                        obstacle_height_probability=[0.1, 0.3, 0.6],
                        platform_width=0.0,
                    ),
                },
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.15, 0.15), 
                metallic=0.1,
                roughness=0.6,
                opacity=1.0                     
            ),
            # visual_material=None,
            max_init_terrain_level=None,
            collision_group=-1,
            debug_vis=True,
        )

        terrain_importer = TerrainImporter(terrain_cfg)

        # 6.Dynamic obstacle terrain
        num_obs = self.cfg.env_dyn.num_obstacles
        
        if num_obs == 0:
            # ensure attributes exist even when there are no dynamic obstacles
            self.dyn_obs_list = []
            self.dyn_obs_state = torch.zeros((0, 13), dtype=torch.float, device=self.device)
            self.dyn_obs_goal = torch.zeros((0, 3), dtype=torch.float, device=self.device)
            self.dyn_obs_origin = torch.zeros((0, 3), dtype=torch.float, device=self.device)
            self.dyn_obs_vel = torch.zeros((0, 3), dtype=torch.float, device=self.device)
            self.dyn_obs_step_count = 0
            self.dyn_obs_size = torch.zeros((0, 3), dtype=torch.float, device=self.device)
            self.dynamic_obstacles = None
            return

        self.dyn_obs_list = []
        self.dyn_obs_state = torch.zeros((num_obs, 13), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_state[:, 3] = 1. # Quaternion w=1
        self.dyn_obs_goal = torch.zeros((num_obs, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_origin = torch.zeros((num_obs, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_vel = torch.zeros((num_obs, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_step_count = 0 
        self.dyn_obs_size = torch.zeros((num_obs, 3), dtype=torch.float, device=self.device) 

        def check_pos_validity(prev_pos_list, curr_pos, radius, other_radii):
            for i, prev_pos in enumerate(prev_pos_list):
                min_dist = radius + other_radii[i] + 1.0 
                if (np.linalg.norm(curr_pos - prev_pos) <= min_dist):
                    return False
            return True

        prev_pos_list = [] 
        prev_radius_list = []
        r_min, r_max = self.cfg.env_dyn.get("size_range", [0.3, 0.6])

        for i in range(num_obs):
            radius = np.random.uniform(r_min, r_max)
            start_time = time.time()
            while True:
                ox = np.random.uniform(low=-self.map_range[0], high=self.map_range[0])
                oy = np.random.uniform(low=-self.map_range[1], high=self.map_range[1])
                oz = np.random.uniform(low=1.0, high=self.map_range[2]) 
                curr_pos = np.array([ox, oy, oz])
                
                if len(prev_pos_list) == 0:
                    valid = True
                else:
                    valid = check_pos_validity(prev_pos_list, curr_pos, radius, prev_radius_list)
                if (time.time() - start_time > 0.2): 
                    valid = True 
                if valid:
                    prev_pos_list.append(curr_pos)
                    prev_radius_list.append(radius)
                    break
            origin = [ox, oy, oz]
            self.dyn_obs_origin[i] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)     
            self.dyn_obs_state[i, :3] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)
            self.dyn_obs_size[i] = torch.tensor([radius, radius, radius], dtype=torch.float, device=self.device)

            prim_utils.create_prim(f"/World/Obstacles/Obs_{i}", "Xform", translation=origin)
            spawn_cfg = sim_utils.SphereCfg(
                radius=radius,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                    disable_gravity=True, 
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                # enable collision so the drone cannot pass through dynamic obstacles
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True), 
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.0), metallic=0.1, roughness=0.5), 
            )
            spawn_cfg.func(f"/World/Obstacles/Obs_{i}/Sphere", spawn_cfg)
        
        # 7.Create a RigidObject to manage all dynamic obstacles
        all_obs_cfg = RigidObjectCfg(
        prim_path="/World/Obstacles/Obs_.*/Sphere", 
        init_state=RigidObjectCfg.InitialStateCfg(),
        )
        self.dynamic_obstacles = RigidObject(cfg=all_obs_cfg)

    # Dynamic obstacle movement update
    def move_dynamic_obstacle(self):
        dist_to_goal = torch.norm(self.dyn_obs_state[:, :3] - self.dyn_obs_goal, dim=1)
        need_new_goal = (dist_to_goal < 0.5) | (self.dyn_obs_step_count == 0)

        if need_new_goal.any():
            num_new = need_new_goal.sum()

            local_range_xy = 6.0 
            local_range_z = 3.0 
            random_offset = (torch.rand(num_new, 3, device=self.device) * 2 - 1)
            
            random_offset[:, 0] *= local_range_xy 
            random_offset[:, 1] *= local_range_xy 
            random_offset[:, 2] *= local_range_z  
            
            new_goals = self.dyn_obs_origin[need_new_goal] + random_offset
            new_goals[:, 0] = new_goals[:, 0].clamp(-self.map_range[0], self.map_range[0])
            new_goals[:, 1] = new_goals[:, 1].clamp(-self.map_range[1], self.map_range[1])
            new_goals[:, 2] = new_goals[:, 2].clamp(0.5, self.map_range[2]) 
        
            self.dyn_obs_goal[need_new_goal] = new_goals

        if not hasattr(self, 'current_speed_norms'):
             v_min, v_max = self.dyn_vel_range
             self.current_speed_norms = torch.full((self.dyn_obs_vel.size(0), 1), (v_max+v_min)/2, device=self.device)

        if (self.dyn_obs_step_count % int(2.0/self.cfg.sim.dt) == 0):
            v_min, v_max = self.dyn_vel_range
            self.current_speed_norms = (torch.rand(self.dyn_obs_vel.size(0), 1, device=self.device) * (v_max - v_min)) + v_min

        # Only move the first `active_dyn_obs` obstacles; freeze the rest
        n_active = min(self.active_dyn_obs, self.dyn_obs_vel.size(0))

        dir_vec = self.dyn_obs_goal - self.dyn_obs_state[:, :3]
        dir_norm = torch.norm(dir_vec, dim=1, keepdim=True).clamp(min=1e-6)
        self.dyn_obs_vel = self.current_speed_norms * (dir_vec / dir_norm)

        # Freeze inactive obstacles
        if n_active < self.dyn_obs_vel.size(0):
            self.dyn_obs_vel[n_active:] = 0.0

        
        self.dyn_obs_state[:, :3] += self.dyn_obs_vel * self.cfg.sim.dt
        self.dyn_obs_state[:, 2] = self.dyn_obs_state[:, 2].clamp(0.5, self.map_range[2])
        self.dyn_obs_state[:, 7:10] = self.dyn_obs_vel  

        self.dynamic_obstacles.write_root_state_to_sim(self.dyn_obs_state)
        self.dyn_obs_step_count += 1

    # ---- Curriculum Learning ----
    def update_curriculum(self, total_frames: int):
        """Update environment difficulty based on training progress.
        
        Call this from train.py each training iteration.
        Linearly interpolates parameters between curriculum phases.
        """
        if not self.curriculum_enabled or len(self.curriculum_phases) == 0:
            return

        phases = self.curriculum_phases
        # Find current phase bracket
        phase_idx = 0
        for j, p in enumerate(phases):
            if total_frames >= p["frame"]:
                phase_idx = j

        curr = phases[phase_idx]
        if phase_idx < len(phases) - 1:
            nxt = phases[phase_idx + 1]
            alpha = (total_frames - curr["frame"]) / max(nxt["frame"] - curr["frame"], 1.0)
            alpha = min(max(alpha, 0.0), 1.0)
        else:
            alpha = 1.0
            nxt = curr  # already at the last phase

        def _lerp(a, b, t):
            return a + (b - a) * t

        # Interpolate parameters
        self.active_dyn_obs = int(_lerp(curr["active_dyn_obs"], nxt["active_dyn_obs"], alpha))
        self.vel_min = _lerp(curr["vel_min"], nxt["vel_min"], alpha)
        self.vel_max = _lerp(curr["vel_max"], nxt["vel_max"], alpha)

        # Update dynamic obstacle velocity range (used by move_dynamic_obstacle)
        new_vmin = _lerp(curr["dyn_vel_range"][0], nxt["dyn_vel_range"][0], alpha)
        new_vmax = _lerp(curr["dyn_vel_range"][1], nxt["dyn_vel_range"][1], alpha)
        self.dyn_vel_range = [new_vmin, new_vmax]

    # observation_spec,action_spec,reward_spec,done_spec,stats_spec,info_spec,RL的输入输出标准，在super().__init__()中调用
    def _set_specs(self):
        observation_dim = 13  
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device), 
                    "lidar_static": UnboundedContinuousTensorSpec((1, self.H, self.W), device=self.device),
                    "lidar_dynamic": UnboundedContinuousTensorSpec((3, self.H, self.W), device=self.device),
                }),
            }).expand(self.num_envs)
        }, shape=[self.num_envs], device=self.device)
        
        # Action Spec（四个电机的推力命令）
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec, # number of motor
            })
        }).expand(self.num_envs).to(self.device)
        
        # Reward Spec
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1,)),
                # "cbf_cost": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)

        # Done Spec
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device) 


        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "reach_goal": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)

        #定义额外的信息空间，包括无人机状态信息：位置3、四元数4、线速度3、角速度3，共13维
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
        }).expand(self.num_envs).to(self.device)

        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()



     # 1.1 重置指定环境实例env_ids的目标点位置，train时随机生成，eval时均匀分布
    
    def reset_target(self, env_ids: torch.Tensor):
        if (self.training):
            # decide which side
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
            shifts = torch.tensor([[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]], dtype=torch.float, device=self.device)
            mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)


            # generate random positions
            target_pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
            target_pos[:, 0, 2] = heights# height
            target_pos = target_pos * selected_masks + selected_shifts
            
            # apply target pos
            self.target_pos[env_ids] = target_pos   
        else:
            self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            self.target_pos[:, 0, 1] = -24.
            self.target_pos[:, 0, 2] = 2.            


     # 1.重置物理状态、目标位置、起始位置、统计信息
    
    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        self.reset_target(env_ids)

        # ---reset drone position---
        if (self.training):
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
            shifts = torch.tensor([[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]], dtype=torch.float, device=self.device)
            mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)

            # generate random positions
            pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
            pos[:, 0, 2] = heights# height
            pos = pos * selected_masks + selected_shifts
        else:
            pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
            pos[:, 0, 1] = 24.
            pos[:, 0, 2] = 2.
        
        # Coordinate change: after reset, the drone's target direction should be changed
        self.target_dir[env_ids] = self.target_pos[env_ids] - pos

        # record start positions for downstream boundary check
        self.start_pos[env_ids] = pos

        # Coordinate change: after reset, the drone's facing direction should face the current goal
        rpy = torch.zeros(len(env_ids), 1, 3, device=self.device)
        diff = self.target_pos[env_ids] - pos
        facing_yaw = torch.atan2(diff[..., 1], diff[..., 0])
        rpy[..., 2] = facing_yaw

        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(pos, rot, env_ids)# set the drone position and orientation in the world frame
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)# set the drone linear and angular velocity to zero
        self.prev_drone_vel_w[env_ids] = 0.
        self.prev_drone_vel_b[env_ids] = 0.
        self.height_range[env_ids, 0, 0] = torch.min(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])
        self.height_range[env_ids, 0, 1] = torch.max(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])

        self.stats[env_ids] = 0.  # reset stats

        pos, rot = self.drone.get_world_poses(clone=True)
        self.perception.reset(env_ids, pos, rot)
   
    # 2.apply action
    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")] 
        self.drone.apply_action(actions) 
        if (self.cfg.env_dyn.num_obstacles != 0):
            self.move_dynamic_obstacle()

    # 3.update lidar and dynamic obstacles
    def _post_sim_step(self, tensordict: TensorDictBase):
        self.lidar.update(self.dt)
    
    # 4.get current states/observation
    def _compute_state_and_obs(self):

        #--------------------------------I：drone state------------------------
        self.root_state = self.drone.get_state(env_frame=False) 
        self.info["drone_state"][:] = self.root_state[..., :13] 
        pos_w = self.root_state[..., :3]
        vel_w = self.root_state[..., 7:10]
        ang_vel_w = self.root_state[..., 10:13]
        quat_w = self.root_state[..., 3:7]
        vel_b = quat_rotate_inverse(quat_w, vel_w)
        ang_vel_b = quat_rotate_inverse(quat_w, ang_vel_w)
        rpos_w = self.target_pos - pos_w
        distance = rpos_w.norm(dim=-1, keepdim=True)
        target_unit_w = rpos_w / distance.clamp(1e-6)
        target_unit_b = quat_rotate_inverse(quat_w, target_unit_w)
        if self.last_dis2goal is None:
            self.last_dis2goal = distance.squeeze(-1).clone()
        dis2goal = distance.squeeze(-1)
        dist_diff = self.last_dis2goal - dis2goal
        self.last_dis2goal = dis2goal.clone()

        # ----------- I: perception input data include static and dyn--------------
        lidar_pos_w = self.lidar.data.pos_w
        lidar_quat_w = self.lidar.data.quat_w
        ray_hits_w = self.lidar.data.ray_hits_w
        num_rays = self.ray_hits_dir.shape[1]
        lidar_quat_expanded = lidar_quat_w.unsqueeze(1).expand(-1, num_rays, -1)
        ray_hits_dir_w = quat_rotate(lidar_quat_expanded, self.ray_hits_dir)

        #static clamp according to lidar_range
        dist_static = torch.norm(ray_hits_w - lidar_pos_w.unsqueeze(1), dim=-1, keepdim=True)
        mask_invalid = (dist_static >= self.lidar_range) | torch.isinf(dist_static)
        truncated_static_w = lidar_pos_w.unsqueeze(1) + ray_hits_dir_w * self.lidar_range
        ray_hits_w_clamp = torch.where(mask_invalid, truncated_static_w, ray_hits_w)

        #dynamic obstacle raycasting
        if self.cfg.env_dyn.num_obstacles > 0:
            dyn_obs_pos = self.dyn_obs_state[:, :3].to(ray_hits_dir_w.device).float()
            dyn_obs_radius = self.dyn_obs_size[:, 0].to(ray_hits_dir_w.device).float()
            dyn_obs_vel = self.dyn_obs_state[..., 7:10].to(ray_hits_dir_w.device).float()
            ray_hits_w_dyn, is_dyn_hit, dyn_hit_idx = sphere_lidar_hits(
                self.lidar_range, 
                dyn_obs_pos, 
                dyn_obs_radius,
                pos_w.squeeze(1),   # (B,1,3) → (B,3)  sphere_lidar_hits expects 2-D
                ray_hits_dir_w
            )
        else:
            ray_hits_w_dyn = lidar_pos_w.unsqueeze(1) + ray_hits_dir_w * self.lidar_range
            is_dyn_hit = torch.zeros(ray_hits_w_dyn.shape[:2], dtype=torch.bool, device=self.device)
            dyn_hit_idx = torch.zeros(ray_hits_w_dyn.shape[:2], dtype=torch.long, device=self.device)
            dyn_obs_vel = torch.zeros((1, 3), device=self.device)


        obs_static_norm, obs_dyn_norm = self.perception.process(
            ray_hits_w_clamp,
            ray_hits_w_dyn,
            is_dyn_hit,
            dyn_hit_idx,
            dyn_obs_vel,
            pos_w,
            quat_w,
            vel_w
        )


        # state: body linear vel, body angular vel, target dir in body, distance, previous body vel
        vel_b_norm = (vel_b / self.state_vel_scale).clamp(-1.0, 1.0)
        ang_vel_b_norm = (ang_vel_b / self.state_ang_vel_scale).clamp(-1.0, 1.0)
        prev_vel_b_norm = (self.prev_drone_vel_b / self.state_vel_scale).clamp(-1.0, 1.0)
        # Use tanh to keep long-range goal distance bounded without losing monotonicity.
        distance_norm = torch.tanh(distance / self.state_dist_scale)

        drone_state = torch.cat([
            vel_b_norm,
            ang_vel_b_norm,
            target_unit_b,
            distance_norm,
            prev_vel_b_norm,
        ], dim=-1).squeeze(1)

        obs = {
            "lidar_static": obs_static_norm,
            "lidar_dynamic": obs_dyn_norm,
            "state": drone_state,
        }

        #-----------------------reward calculation--------------------
        height = pos_w[..., 2]
        touch_goal_mask = dis2goal <= 3.
        vel_direction = rpos_w / distance.clamp_min(1e-6)
        vel_magnitude = vel_w.norm(dim=-1)
        last_vel_w = self.prev_drone_vel_w
        
        [beta_vel, vel_limit] = [2., 1.2 * self.vel_max]
        [vel_set_min, vel_set_max] = [self.vel_min, self.vel_max]

        reward_vel,penalty_smooth = self._compute_state_reward(
            beta_vel, vel_set_min, vel_set_max, vel_magnitude, 
            vel_w, last_vel_w, touch_goal_mask)

        penalty_height = torch.zeros(self.num_envs, 1, device=self.cfg.device)
        penalty_height[self.drone.pos[..., 2] > (self.height_range[..., 1] + 0.5)] = ( (self.drone.pos[..., 2] - self.height_range[..., 1] - 0.5)**2 )[self.drone.pos[..., 2] > (self.height_range[..., 1] + 0.5)]
        penalty_height[self.drone.pos[..., 2] < (self.height_range[..., 0] - 0.5)] = ( (self.height_range[..., 0] - 0.5 - self.drone.pos[..., 2])**2 )[self.drone.pos[..., 2] < (self.height_range[..., 0] - 0.5)]
        
        reward_goal = self._compute_goal_reward(vel_w, vel_direction,
                                                self.last_dis2goal, dis2goal,
                                                touch_goal_mask)
        
        # Safety reward: merge static and dynamic depth maps, take min, compute safety reward
        depth_static_real = obs_static_norm * self.lidar_range          
        depth_dyn_real = obs_dyn_norm[:, 0:1, :, :] * self.lidar_range  
        depth_merged = torch.min(depth_static_real, depth_dyn_real)     
        reward_safety = self._compute_safety_reward(depth_merged)

        # Success bonus
        reach_goal = (distance.squeeze(-1) < 0.5)
        reward_success = reach_goal.float() * 100.0  # (B, 1)
        time_penalty = -0.2

        [k_v, k_a, k_h, k_g, k_s] = [0.3, 0.2, 0.1, 1.0, 0.9]
        self.reward = (k_v * reward_vel + k_a * penalty_smooth - k_h * penalty_height
                       + k_g * reward_goal + k_s * reward_safety + reward_success + time_penalty)


        # -----------------collision or reach goal condition----

        # collision condition
        min_dist_static = obs_static_norm.reshape(self.num_envs, -1).min(dim=1)[0] * self.lidar_range
        static_collision = (min_dist_static < 0.3).unsqueeze(1)
        min_dist_dyn = obs_dyn_norm[:, 0, :, :].reshape(self.num_envs, -1).min(dim=1)[0] * self.lidar_range
        dynamic_collision = (min_dist_dyn < 0.3).unsqueeze(1)

        # geometric body-overlap check for dynamic obstacles (ray-based check can miss glancing hits)
        if self.cfg.env_dyn.num_obstacles > 0:
            drone_pos = pos_w.squeeze(1)                           # (B, 3)
            dyn_centers = self.dyn_obs_state[:, :3].to(drone_pos.device)
            dyn_radii = self.dyn_obs_size[:, 0].to(drone_pos.device)
            # distance from drone center to obstacle surface
            dist_to_surface = torch.cdist(drone_pos, dyn_centers) - dyn_radii  # (B, num_obs)
            dyn_body_collision = dist_to_surface.min(dim=1, keepdim=True)[0] < 0.05
        else:
            dyn_body_collision = torch.zeros_like(static_collision)
        
        collision = static_collision | dynamic_collision | dyn_body_collision 

        #bound condition
        below_bound = self.drone.pos[..., 2] < 0.2
        above_bound = self.drone.pos[..., 2] > 4.5

        #xy bound
        xy_misbehave = self.get_bound_misbehave(self.root_state[..., :2].squeeze(1), 
                                                    self.start_pos[..., :2].squeeze(1),
                                                    self.target_pos[..., :2].squeeze(1))

        self.terminated = xy_misbehave | below_bound | above_bound | collision | reach_goal
        self.truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1) 
 
        self.prev_drone_vel_w = vel_w.clone()
        self.prev_drone_vel_b = vel_b.clone()

        # # -----------------Training Stats-----------------
        self.stats["return"] += self.reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["reach_goal"] = reach_goal.float()
        self.stats["collision"] = collision.float()
        self.stats["truncated"] = self.truncated.float()

        return TensorDict({
            "agents": TensorDict(
                {
                    "observation": obs,
                }, 
                [self.num_envs]
            ),
            "stats": self.stats.clone(),
            "info": self.info
        }, self.batch_size)
   
    # 5.get reward and done signal
    def _compute_reward_and_done(self):
        reward = self.reward
        terminated = self.terminated
        truncated = self.truncated

        return TensorDict(
            {
                "agents": {
                    "reward": reward,
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )

    def _compute_state_reward(self, beta_vel, vel_set_min, vel_set_max, vel_magnitude,
                              vel, last_vel, touch_goal_mask):
        reward_vel = torch.log(torch.exp(- beta_vel * (torch.clamp(vel_set_min - vel_magnitude, min = 0.)
                                         + torch.clamp(vel_magnitude - vel_set_max, min = 0.))) + 1.)
        reward_vel[touch_goal_mask] = torch.log(torch.exp(- beta_vel * torch.clamp(
                                         vel_magnitude[touch_goal_mask] - vel_set_max, min = 0.)) + 1.)
        penalty_smooth = -(vel-last_vel).norm(dim=-1)
        return reward_vel, penalty_smooth

    def _compute_goal_reward(self, vel_vector, vel_direction, last_dis2goal, dis2goal, touch_goal_mask):
        reward_goal_dir = (vel_vector * vel_direction).sum(-1).clip(max=2.0)
        reward_goal_dis_base = (torch.exp(last_dis2goal - dis2goal) - 1.) * 10.
        scale = torch.clamp((dis2goal - 0.5) / (3.0 - 0.5), min=0.2, max=1.0)
        reward_goal_dis = reward_goal_dis_base * scale
        reward_goal = reward_goal_dir + reward_goal_dis

        return reward_goal
    
    def _compute_safety_reward(self, lidar_scan):
        proximity = self.lidar_range - lidar_scan  
        proximity_flat = proximity.reshape(self.num_envs, -1)  

        close_mask = lidar_scan.reshape(self.num_envs, -1) < (self.lidar_range * 0.3)  # (B, H*W)
        obs_count = close_mask.sum(dim=1)  # (B,)
        violation = torch.clamp(proximity_flat - (self.lidar_range - self.safety_dis), min=0.)  # (B, H*W)

        mean_violation = torch.where(
            obs_count > 0,
            (violation * close_mask).sum(dim=1) / obs_count.clamp(min=1),
            torch.zeros(self.num_envs, device=self.device)
        )  # (B,)

        reward_safety = -mean_violation
        reward_safety = reward_safety.reshape(self.num_envs, 1)
        return reward_safety
    
    def get_bound_misbehave(self, drone_pos, start_pos, target_pos):
        A = target_pos[:, 1] - start_pos[:, 1]
        B = start_pos[:, 0] - target_pos[:, 0] 
        C = target_pos[:, 0] * start_pos[:, 1] - start_pos[:, 0] * target_pos[:, 1]
        distance = torch.abs(A * drone_pos[:, 0] + B * drone_pos[:, 1] + C) / torch.sqrt(A**2 + B**2)
        mask = distance > 8.0
        bound_misbehave = mask.unsqueeze(1)

        return bound_misbehave

