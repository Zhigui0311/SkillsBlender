from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from collections.abc import Sequence

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers 
from isaaclab.terrains import TerrainImporter 

from skillsblender.tasks.path.mdp.commands.path_command import PathCommand
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv 
    from .path_command_cfg import PathCommandCfg, JumpPathCommandCfg                                      

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

class JumpPathCommand(PathCommand):
    """Command that senses terrain to generate parabolic jump trajectories."""
    cfg: JumpPathCommandCfg

    def _generate_trajectory(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate position and yaw trajectories with parabolic arcs over terrain gaps.

        This function:
        1. Scans terrain ahead to detect gaps
        2. Plans a path that includes jumps over detected gaps
        3. Generates parabolic arcs for jump segments
        4. Computes appropriate yaw angles along the path
        """
        num_batch = len(env_ids)
        start_pos = self.robot.data.root_pos_w[env_ids].clone()

        # Clamp start position to env bounds
        xy_min, xy_max = self._get_env_xy_bounds(env_ids)
        if xy_min is not None:
            start_pos[:, :2] = torch.max(torch.min(start_pos[:, :2], xy_max), xy_min)

        # 1. Terrain Scanning (Detect gaps by scanning forward)
        gap_start_dist = torch.zeros(num_batch, device=self.device)
        gap_end_dist = torch.zeros(num_batch, device=self.device)
        has_gap = torch.zeros(num_batch, dtype=torch.bool, device=self.device)

        base_h = start_pos[:, 2]
        steps = int(self.cfg.jump_params.scan_dist / self.cfg.jump_params.scan_step)

        for i in range(steps):
            d = i * self.cfg.jump_params.scan_step
            check_xy = start_pos[:, :2].clone()
            check_xy[:, 0] += d  # Scan forward along X-axis

            # Clamp to env bounds
            if xy_min is not None:
                check_xy = torch.max(torch.min(check_xy, xy_max), xy_min)

            # Query terrain height at this position
            # Note: terrain might not have a direct get_height_at method, so we use a safe approach
            try:
                h = self._env.scene.terrain.terrain_mesh.sample_points(check_xy.unsqueeze(1))[:, 0, 2]
            except:
                # If terrain sampling fails, assume flat ground
                h = base_h

            # Detect gap start (height drops)
            is_gap = (h - base_h) < self.cfg.jump_params.gap_threshold

            # Mark gap start
            start_mask = is_gap & (~has_gap)
            if start_mask.any():
                gap_start_dist[start_mask] = d - self.cfg.jump_params.takeoff_margin
                has_gap |= start_mask

            # Mark gap end (height recovers)
            end_mask = (~is_gap) & has_gap & (gap_end_dist == 0)
            if end_mask.any():
                gap_end_dist[end_mask] = d + self.cfg.jump_params.landing_margin

        # 2. Plan 3D Path
        # Total distance: extend a bit beyond gap end if gap exists, otherwise use default length
        total_len = torch.where(
            has_gap & (gap_end_dist > 0),
            gap_end_dist + 1.0,
            torch.tensor(5.0, device=self.device)
        )

        end_pos = start_pos.clone()
        end_pos[:, 0] += total_len

        # Clamp end position to env bounds
        if xy_min is not None:
            end_pos[:, :2] = torch.max(torch.min(end_pos[:, :2], xy_max), xy_min)

        # 3. Generate linear interpolation as base trajectory
        alpha = self.t_alpha.view(1, -1, 1)
        pos_traj = start_pos.unsqueeze(1) + (end_pos.unsqueeze(1) - start_pos.unsqueeze(1)) * alpha

        # 4. Apply Parabolic Arc over detected gaps
        # Calculate distance of each waypoint from start
        dist_at_wp = alpha.squeeze(-1) * total_len.unsqueeze(1)

        # Normalize position within jump segment [0, 1]
        t_jump = (dist_at_wp - gap_start_dist.unsqueeze(1)) / \
                 (gap_end_dist.unsqueeze(1) - gap_start_dist.unsqueeze(1) + 1e-6)
        t_jump = torch.clamp(t_jump, 0.0, 1.0)

        # Parabolic arc: h = 4 * H * t * (1-t) reaches max H at t=0.5
        arc_h = self.cfg.jump_params.jump_height * 4 * t_jump * (1 - t_jump)

        # Apply arc only within jump segment
        jump_mask = (dist_at_wp > gap_start_dist.unsqueeze(1)) & \
                   (dist_at_wp < gap_end_dist.unsqueeze(1)) & \
                   has_gap.unsqueeze(1)

        pos_traj[..., 2] += torch.where(jump_mask, arc_h, torch.zeros_like(arc_h))

        # 5. Compute yaw along the path
        yaw_traj = self._yaw_along_curve(pos_traj)

        return pos_traj, yaw_traj

    def _update_command(self):
        """
        Override: 更新航点索引 - 跳跃任务需要特殊处理空中情况。

        关键改进：
        - 检测机器人是否在空中
        - 在空中时只考虑XY平面距离，避免航点推进过快
        - 在地面时使用正常的3D距离判断
        """
        robot_pos = self.robot.data.root_pos_w[:, :3]  # (N, 3)
        robot_height = robot_pos[:, 2]

        # 检测是否在空中（高于站立高度+阈值）
        # GO2站立高度约0.34m，离地0.15m以上认为在空中
        standing_height = 0.34
        air_threshold = 0.15
        is_in_air = robot_height > (standing_height + air_threshold)

        # 获取当前目标航点
        target_pos_cur = self.pos_path_w[torch.arange(self.num_envs), self.current_waypoints_index]  # (N, 3)

        # 计算到目标的距离
        # 在空中时只考虑XY平面，在地面时考虑3D距离
        dis_xy = torch.norm(target_pos_cur[:, :2] - robot_pos[:, :2], dim=-1)  # XY平面距离
        dis_3d = torch.norm(target_pos_cur - robot_pos, dim=-1)  # 3D距离

        # 选择距离判断方式
        dis_to_target = torch.where(is_in_air, dis_xy, dis_3d)

        # 到达阈值判断
        reach_threshold = self.cfg.ranges.waypoint_reach_threshold
        reached = dis_to_target < reach_threshold

        # 更新航点索引
        self.current_waypoints_index = torch.where(
            reached,
            torch.clamp(self.current_waypoints_index + 1, max=self.num_waypoints - 1),
            self.current_waypoints_index
        )

        # 检查是否到达终点
        target_pos_final = self.pos_path_w[torch.arange(self.num_envs), -1]
        goal_dis = torch.norm(target_pos_final - robot_pos, dim=-1)
        goal_reached = (goal_dis < reach_threshold) & (self.current_waypoints_index >= self.num_waypoints - 1)
        self.goal_reached = self.goal_reached | goal_reached

        # 更新观测切片
        self.obs_slices = self._get_cur_slices(torch.arange(self.num_envs, device=self.device))

        # 到达终点时，将观测切片清零
        obs_slices = self.obs_slices.clone()
        if torch.any(goal_reached):
            obs_slices[goal_reached] = 0.0
        self.obs_slices = obs_slices