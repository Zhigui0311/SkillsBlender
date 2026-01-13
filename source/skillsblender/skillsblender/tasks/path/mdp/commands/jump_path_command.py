from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from collections.abc import Sequence

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers #？作用是什么？
from isaaclab.terrains import TerrainImporter #不用引用地形生成的吗？

# from isaaclab.utils.math import (quat_apply_inverse,
#                                 wrap_to_pi,
#                                 yaw_quat, 
#                                 quat_from_euler_xyz,
#                                 euler_xyz_from_quat,
#                                 random_yaw_orientation
#                                 )

from skillsblender.skillsblender.tasks.path.mdp.commands.path_command import PathCommand
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv  
    from .path_command_cfg import PathCommandCfg, JumpPathCommandCfg                                      

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR



# class JumpPathCommand(PathCommand):
#     cfg: JumpPathCommandCfg
    
#     def __init__(self, cfg, env):
#         super().__init__(cfg, env)
        
#     def _generate_trajectory(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         """Generate  position trajectory and yaw trajectory with jump over terrain gaps."""
        
        
#         num_batch = len(env_ids)
#         start_pos = self.robot.data.root_pos_w[env_ids].clone()
        
#         # 1. 确定搜索方向 (假设沿当前机器人朝向或 X 轴)
#         # 这里为了简单采用 X 轴方向，你可以根据机器人 heading 旋转方向向量
#         search_dir = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(num_batch, 1)
        
#         # 2. 地形扫描：寻找沟壑边缘
#         # 我们利用 self._env.scene.terrain 获取高度数据
#         gap_start_x = torch.zeros(num_batch, device=self.device)
#         gap_end_x = torch.zeros(num_batch, device=self.device)
        
#         # 模拟沿射线扫描高度 (Scan loop)
#         steps = int(self.cfg.jump_params.scan_dist / self.cfg.jump_params.scan_step)
#         current_heights = start_pos[:, 2] # 初始高度
        
#         found_start = torch.zeros(num_batch, dtype=torch.bool, device=self.device)
#         found_end = torch.zeros(num_batch, dtype=torch.bool, device=self.device)

#         for i in range(steps):
#             dist = i * self.cfg.jump_params.scan_step
#             scan_pos = start_pos[:, :2] + search_dir[:, :2] * dist
            
#             # 查询该点地形高度 (Isaac Lab API)
#             # 注意：如果地形是平的，此处高度恒定，逻辑会自动跳过抛物线
#             h = self._env.scene.terrain.get_height_at(scan_pos)
            
#             # 识别高度骤降 (沟壑开始)
#             is_gap_drop = (h - current_heights < self.cfg.jump_params.gap_threshold)
#             gap_start_mask = is_gap_drop & (~found_start)
#             if gap_start_mask.any():
#                 gap_start_x[gap_start_mask] = dist - self.cfg.jump_params.takeoff_buffer
#                 found_start |= gap_start_mask
            
#             # 识别高度回升 (沟壑结束)
#             is_gap_rise = (h - current_heights > -0.05) & found_start
#             gap_end_mask = is_gap_rise & (~found_end)
#             if gap_end_mask.any():
#                 gap_end_x[gap_end_mask] = dist + self.cfg.jump_params.landing_buffer
#                 found_end |= gap_end_mask

#         # 如果没找到沟壑，则生成平地终点
#         total_dist = torch.where(found_end, gap_end_x + 1.0, torch.tensor(5.0, device=self.device))
#         end_pos = start_pos + search_dir * total_dist.unsqueeze(1)

#         # 3. 插值生成 3D 轨迹
#         alpha = self.t_alpha.view(1, -1, 1)
#         pos_trajectory = start_pos.unsqueeze(1) + (end_pos.unsqueeze(1) - start_pos.unsqueeze(1)) * alpha
        
#         # 4. 叠加抛物线 (仅在扫描到的沟壑区间 [gap_start, gap_end] 内)
#         jump_h = torch.empty(num_batch, device=self.device).uniform_(*self.cfg.jump_params.jump_height_range)
        
#         # 归一化跳跃区进度
#         start_ratio = (gap_start_x / total_dist).unsqueeze(1)
#         end_ratio = (gap_end_x / total_dist).unsqueeze(1)
        
#         t_jump = (alpha.squeeze(-1) - start_ratio) / (end_ratio - start_ratio + 1e-6)
#         t_jump = torch.clamp(t_jump, 0.0, 1.0)
        
#         # 抛物线: z = 4 * H * t * (1-t)
#         arc = jump_h.unsqueeze(1) * 4 * t_jump * (1 - t_jump)
#         # Mask: 只有在找到的沟壑范围内才抬高高度
#         mask = (alpha.squeeze(-1) > start_ratio) & (alpha.squeeze(-1) < end_ratio) & found_end.unsqueeze(1)
        
#         pos_trajectory[..., 2] += torch.where(mask, arc, torch.zeros_like(arc))
        
#         # 5. 计算航向
#         yaw_trajectory = self._yaw_along_curve(pos_trajectory)
        
#         return pos_trajectory, yaw_trajectory
    
    
# source/skillsblender/skillsblender/tasks/path/mdp/commands/path_command.py

class JumpPathCommand(PathCommand):
    """Command that senses terrain to generate parabolic jump trajectories."""
    cfg: JumpPathCommandCfg

    def _generate_trajectory(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_batch = len(env_ids)
        start_pos = self.robot.data.root_pos_w[env_ids].clone()
        
        # 1. Terrain Scanning (Privileged Perspective)
        # We scan ahead of the robot to find where the ground drops
        gap_start_dist = torch.zeros(num_batch, device=self.device)
        gap_end_dist = torch.zeros(num_batch, device=self.device)
        has_gap = torch.zeros(num_batch, dtype=torch.bool, device=self.device)
        
        base_h = start_pos[:, 2]
        steps = int(self.cfg.jump_params.scan_dist / self.cfg.jump_params.scan_step)

        for i in range(steps):
            d = i * self.cfg.jump_params.scan_step
            check_xy = start_pos[:, :2].clone()
            check_xy[:, 0] += d # Assuming forward along X
            
            # Access terrain height samples
            h = self._env.scene.terrain.get_height_at(check_xy)
            
            is_gap = (h - base_h) < self.cfg.jump_params.gap_threshold
            
            # Identify gap start
            start_mask = is_gap & (~has_gap)
            if start_mask.any():
                gap_start_dist[start_mask] = d - self.cfg.jump_params.takeoff_margin
                has_gap |= start_mask
            
            # Identify gap end
            end_mask = (~is_gap) & has_gap & (gap_end_dist == 0)
            if end_mask.any():
                gap_end_dist[end_mask] = d + self.cfg.jump_params.landing_margin

        # 2. Plan 3D Path
        total_len = torch.where(has_gap & (gap_end_dist > 0), gap_end_dist + 1.0, torch.tensor(5.0, device=self.device))
        end_pos = start_pos.clone()
        end_pos[:, 0] += total_len
        
        alpha = self.t_alpha.view(1, -1, 1)
        pos_traj = start_pos.unsqueeze(1) + (end_pos.unsqueeze(1) - start_pos.unsqueeze(1)) * alpha
        
        # 3. Apply Parabolic Arc over detected gap
        dist_at_wp = alpha.squeeze(-1) * total_len.unsqueeze(1)
        t_jump = (dist_at_wp - gap_start_dist.unsqueeze(1)) / (gap_end_dist.unsqueeze(1) - gap_start_dist.unsqueeze(1) + 1e-6)
        t_jump = torch.clamp(t_jump, 0.0, 1.0)
        
        arc_h = self.cfg.jump_params.jump_height * 4 * t_jump * (1 - t_jump)
        jump_mask = (dist_at_wp > gap_start_dist.unsqueeze(1)) & (dist_at_wp < gap_end_dist.unsqueeze(1)) & has_gap.unsqueeze(1)
        
        pos_traj[..., 2] += torch.where(jump_mask, arc_h, torch.zeros_like(arc_h))
        yaw_traj = self._yaw_along_curve(pos_traj)
        
        return pos_traj, yaw_traj