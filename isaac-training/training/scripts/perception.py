import torch
import numpy as np
from omni_drones.utils.torch import quat_rotate, quat_rotate_inverse

class PerceptionModule:

    def __init__(self, cfg, num_envs, device):

        self.device = device
        self.num_envs = num_envs
        self.d_max = cfg.sensor.lidar_range
        
        self.H = cfg.sensor.lidar_vbeams
        self.W = cfg.sensor.lidar_hbeams
        
        # Use config FOV values (match env.py lidar configuration)
        v_min = max(-89., cfg.sensor.lidar_vfov[0])
        v_max = min(89.,  cfg.sensor.lidar_vfov[1])
        self.fov_v_deg = (v_min, v_max)
        self.fov_h_deg = (-180.0, 180.0)
        
        self.fov_v_rad = torch.tensor([x * torch.pi / 180.0 for x in self.fov_v_deg], device=device)
        self.fov_h_rad = torch.tensor([x * torch.pi / 180.0 for x in self.fov_h_deg], device=device)

        # Angular resolution
        self.res_h = (self.fov_h_rad[1] - self.fov_h_rad[0]) / self.W
        self.res_v = (self.fov_v_rad[1] - self.fov_v_rad[0]) / max(self.H - 1, 1)

        # sim to real M-detector noise param
        self.vel_noise_std = 0.2
        self.pos_noise_std = 0.05
        # Max possible relative velocity = max obstacle speed + max drone speed
        drone_vel_max = 5.0  # matches env vel_max
        self.DYN_MAX_VEL = max(cfg.env_dyn.vel_range) + drone_vel_max


    def _transform_to_body(self, ray_hits_w, pos, quat):

        B = ray_hits_w.shape[0]
        p_world_flat = ray_hits_w.view(B, -1, 3)
        N = p_world_flat.shape[1]
        p_rel_flat = p_world_flat - pos 
        
        if quat.ndim == 2:
            quat = quat.unsqueeze(1)
        quat_expanded = quat.expand(B, N, 4)
            
        p_body_flat = quat_rotate_inverse(quat_expanded, p_rel_flat)
        
        return p_body_flat

    def _project_to_panorama(self, points, value_type='depth', mask=None, feature=None):

        B, N, _ = points.shape
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        d = torch.norm(points, dim=-1)
        
        theta_h = torch.atan2(y, x)
        theta_v = torch.asin((z / d.clamp(min=1e-6)).clamp(-1.0, 1.0))

        u = ((theta_h - self.fov_h_rad[0]) / self.res_h).long().clamp(0, self.W - 1)
        v = ((theta_v - self.fov_v_rad[0]) / self.res_v).long().clamp(0, self.H - 1)

        valid = (d > 0.1) & (d < self.d_max)
        if mask is not None:
            valid = valid & mask

        flat_idx = v * self.W + u  # (B, N)

        if value_type == 'depth':
            scatter_vals = torch.where(valid, d, torch.full_like(d, float('inf')))
            img = torch.full((B, self.H * self.W), self.d_max, device=self.device)
            img.scatter_reduce_(1, flat_idx, scatter_vals, reduce="amin", include_self=True)
            img = img.clamp(max=self.d_max)
        else:
            # Feature projection: scatter valid feature values, dump invalid to an extra bin
            scatter_vals = torch.where(valid, feature, torch.zeros_like(feature))
            extended_img = torch.zeros((B, self.H * self.W + 1), device=self.device)
            safe_flat_idx = torch.where(valid, flat_idx, torch.full_like(flat_idx, self.H * self.W))
            extended_img.scatter_(1, safe_flat_idx, scatter_vals)
            img = extended_img[:, :self.H * self.W]

        return img.view(B, 1, self.H, self.W)

    def reset(self, env_ids, pos, rot):
        """Reset perception state for specified environments."""
        pass

    def process(self, hits_static_w,hits_dyn_w,is_dyn_hit,dyn_hit_idx,all_dyn_vel_w, pos_w, quat_w,vel_w):
        
        B, N, _ = hits_static_w.shape
        #static
        p_static_body = self._transform_to_body(hits_static_w, pos_w, quat_w)
        depth_static = self._project_to_panorama(p_static_body, value_type='depth')
        #dynamic
        p_dyn_body = self._transform_to_body(hits_dyn_w, pos_w, quat_w)
        safe_idx = dyn_hit_idx.clamp(0, max(all_dyn_vel_w.shape[0] - 1, 0))
        hit_obj_vel = all_dyn_vel_w[safe_idx]  # (B, N, 3)
        rel_vel_w = hit_obj_vel - vel_w  # (B, N, 3) - (B, 1, 3) broadcast

        rel_vel_w = rel_vel_w + torch.randn_like(rel_vel_w) * self.vel_noise_std#加入noise M-detector

        quat_expanded = quat_w.expand(-1, N, -1) if quat_w.dim() == 3 else quat_w.unsqueeze(1).expand(-1, N, -1)
        rel_vel_body = quat_rotate_inverse(quat_expanded, rel_vel_w)
        dist_dyn = torch.norm(p_dyn_body, dim=-1, keepdim=True).clamp(min=1e-6)
        r_hat = p_dyn_body / dist_dyn
        
        v_rad_val = (rel_vel_body * r_hat).sum(dim=-1) # (B, N)
        v_tan_vec = rel_vel_body - v_rad_val.unsqueeze(-1) * r_hat
        v_tan_val = torch.norm(v_tan_vec, dim=-1)
        depth_dyn = self._project_to_panorama(
            p_dyn_body, value_type='depth', mask=is_dyn_hit
        )
        img_rad = self._project_to_panorama(
            p_dyn_body, value_type='feature', feature=v_rad_val, mask=is_dyn_hit
        )
        img_tan = self._project_to_panorama(
            p_dyn_body, value_type='feature', feature=v_tan_val, mask=is_dyn_hit
        )

        norm_depth_static = depth_static / self.d_max

        norm_depth_dyn = depth_dyn / self.d_max
        norm_v_rad = (img_rad / self.DYN_MAX_VEL).clamp(-1.0, 1.0)
        norm_v_tan = (img_tan / self.DYN_MAX_VEL).clamp(0.0, 1.0)
        norm_obs_dyn = torch.cat([
            norm_depth_dyn,
            norm_v_rad,
            norm_v_tan,
        ], dim=1
        )

        return norm_depth_static, norm_obs_dyn


def compute_rayhitsdir(device, num_envs, h_fov, v_fov, h_num, v_num):
    if h_fov == 360:
        horizontal_angles = torch.linspace(0, h_fov, h_num + 1, device=device)
        horizontal_angles = horizontal_angles[:h_num]
    else:
        horizontal_angles = torch.linspace(0, h_fov, h_num, device=device)
    vertical_angles = torch.linspace(v_fov[0], v_fov[1], v_num, device=device) 
    horizontal_radians = horizontal_angles * torch.pi/180 
    vertical_radians = vertical_angles * torch.pi/180
    horizontal_grid, vertical_grid = torch.meshgrid(horizontal_radians, vertical_radians)
    directions = torch.stack((
        torch.cos(vertical_grid) * torch.cos(horizontal_grid),
        torch.cos(vertical_grid) * torch.sin(horizontal_grid),
        torch.sin(vertical_grid) 
    ), dim=-1) 
    ray_hits_dir = directions.reshape(h_num * v_num, -1).unsqueeze(0).expand(num_envs, -1, -1)

    return ray_hits_dir

def sphere_lidar_hits(lidar_range, spheres_pos, spheres_radius, pos_w, ray_hits_dir):

    P = pos_w.unsqueeze(1).unsqueeze(2)
    D = ray_hits_dir.unsqueeze(2)
    C = spheres_pos.unsqueeze(0).unsqueeze(0)
    R = spheres_radius.view(1, 1, -1, 1)
    L = C - P 
    t_ca = torch.sum(L * D, dim=-1, keepdim=True)
    d_sq = torch.sum(L * L, dim=-1, keepdim=True) - t_ca**2
    radius_sq = R**2
    discriminant = radius_sq - d_sq
    t_hc = torch.sqrt(torch.clamp(discriminant, min=0))
    t0 = t_ca - t_hc
    valid_hit_obs = (discriminant >= 0) & (t0 > 0) & (t0 < lidar_range)
    t0_vals = t0.squeeze(-1)
    t0_vals[~valid_hit_obs.squeeze(-1)] = float('inf')
    min_t, hit_idx = torch.min(t0_vals, dim=-1)
    is_hit = min_t < float('inf')

    #未击中的射线，距离设为雷达最大测距
    final_t = torch.where(is_hit, min_t, torch.tensor(lidar_range, device=pos_w.device))
    hits = pos_w.unsqueeze(1) + final_t.unsqueeze(-1) * ray_hits_dir

    return hits, is_hit, hit_idx


